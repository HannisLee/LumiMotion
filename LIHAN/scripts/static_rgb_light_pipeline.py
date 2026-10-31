"""可续查的Stage1完成后全流程：RGB选基线、估计、冻结及一次独立评测。"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import importlib.metadata
from LIHAN.scripts.static_rgb_light_prepare import sha,write

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--external-candidates',type=Path,help='已审计的无GT候选清单；复用已有DEV结果，从基线门禁继续')
    a=p.parse_args(); control=a.control.resolve(); model=a.model.resolve();source=control/'source_all'
    logdir=control/'logs';logdir.mkdir(exist_ok=True)
    state=dict(pid=os.getpid(),status='WAITING_TRAINING',started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def status(s,**kwargs):
        state.update(status=s,**kwargs);write(control/'pipeline_state.json',state);print(s,kwargs,flush=True)
    def run(label,entry,args,bootstrap=True):
        cmd=[sys.executable]
        if bootstrap:cmd += [str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),entry]
        else:cmd += ['-m',entry]
        cmd += list(map(str,args))
        write(logdir/(label+'_command.json'),cmd)
        status(label,command=cmd,log=str(logdir/(label+'.log')))
        with (logdir/(label+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONUNBUFFERED='1'))
            state['child_pid']=child.pid;write(control/'pipeline_state.json',state)
            code=child.wait()
        if code:raise RuntimeError(label+'失败，exit='+str(code))
    try:
        if a.external_candidates:
            receipt=json.loads(a.external_candidates.read_text())
            if receipt.get('GT_read') is not False or receipt.get('audit_complete') is not True:
                raise RuntimeError('外部候选必须完成无GT谱系审计')
            candidates=[]
            for item in receipt['candidates']:
                if item.get('eligible') is not True:continue
                result=json.loads(Path(item['metrics_path']).read_text())
                if result.get('gt_used') is not False:raise RuntimeError('候选指标必须声明未使用GT')
                candidates.append(result)
            if not candidates:raise RuntimeError('无合格候选')
        else:
            while True:
                record=model/'rgb_only_stage1.json'
                if record.exists() and json.loads(record.read_text()).get('status')=='FINISHED':break
                proc=control/'training_process.json'
                if proc.exists() and json.loads(proc.read_text()).get('status','').startswith('FAILED'):
                    raise RuntimeError('训练失败；不得假设checkpoint完整')
                status('WAITING_TRAINING');time.sleep(30)
            candidates=[]
            for iteration in [10000,20000,35000]:
                out=control/f'baseline_dev_{iteration}'
                run(f'DEV_{iteration}','LIHAN.scripts.static_rgb_light_extract',[
                    '--source',source,'--model',model,'--iteration',iteration,'--output',out,'--action','validate'])
                candidates.append(json.loads((out/'metrics.json').read_text()))
        peak=max(r['metrics']['foreground_psnr'] for r in candidates)
        near=[r for r in candidates if r['metrics']['foreground_psnr']>=peak-.1]
        best=min(near,key=lambda r:(-r['metrics']['normal_depth_cosine'],-r['metrics']['predicted_alpha_coverage'],r['metrics']['lpips'],-r['iteration']))
        iteration=best['iteration'];model=Path(best['model']).resolve()
        deform_enabled=False
        if a.external_candidates:
            matching=[item for item in receipt['candidates'] if item.get('eligible') is True and
                      Path(json.loads(Path(item['metrics_path']).read_text())['model']).resolve()==model and
                      json.loads(Path(item['metrics_path']).read_text())['iteration']==iteration]
            if len(matching)!=1:raise RuntimeError('候选model/iteration身份不唯一')
            deform_enabled=matching[0].get('deform_enabled',False)
        write(control/'baseline_selection.json',dict(selected_iteration=iteration,model=str(model),gt_used=False,
              rule='DEV18固定RGB推断前景mask>0.5、不腐蚀、不依预测alpha缩分母的PSNR主选；差距0.1dB内按GS与模型深度法线cos、预测覆盖、全图LPIPS、较晚迭代依次破同值；不读GT',
              candidates=candidates,selected=best,deform_enabled=deform_enabled))
        status('BASELINE_REVIEW_REQUIRED',selected_iteration=iteration)
        while not (control/'baseline_review.json').exists():time.sleep(15)
        review=json.loads((control/'baseline_review.json').read_text())
        if review.get('iteration')!=iteration or review.get('proceed') is not True or (a.external_candidates and review.get('model')!=str(model)):
            raise RuntimeError('基线目检未通过；不得冻结并评测GT')
        extraction=control/'extraction'
        run('EXTRACT','LIHAN.scripts.static_rgb_light_extract',['--source',source,'--model',model,
            '--iteration',iteration,'--output',extraction,'--action','extract','--max-gaussians','30000']+(['--deform'] if deform_enabled else []))
        fitting=control/'fitting'
        run('FIT','LIHAN.scripts.static_rgb_light_fit',['--observations',extraction/'observations.npz','--output',fitting],False)
        status('METHOD_REVIEW_REQUIRED')
        while not (control/'method_review.json').exists():time.sleep(15)
        review=json.loads((control/'method_review.json').read_text())
        if review.get('proceed') is not True or review.get('GT_read') is not False:
            raise RuntimeError('无GT对应/百分位/残差/稳定性目检未通过，不得冻结并评测GT')
        status('FREEZING')
        shutil.copy2(fitting/'predictions.npz',control/'predictions.npz')
        pyfiles=[ROOT/'LIHAN/scripts'/('static_rgb_light_'+name+'.py') for name in ['prepare','stage1','extract','fit','evaluate','pipeline']]
        assets=pyfiles+[model/f'point_cloud/iteration_{iteration}/point_cloud.ply',model/f'deform/iteration_{iteration}/deform.pth',
                         control/'source_train/rgb_only_manifest.json',fitting/'method_config.json',control/'baseline_selection.json',
                         extraction/'observations.npz',extraction/'extraction.json']
        if a.external_candidates:
            assets.append(a.external_candidates.resolve())
            for item in receipt['candidates']:
                assets.append(Path(item['metrics_path']).resolve())
                if item.get('audit_path'):assets.append(Path(item['audit_path']).resolve())
        snapshot=control/'frozen_code';snapshot.mkdir(exist_ok=False)
        for path in pyfiles:shutil.copy2(path,snapshot/path.name)
        dependencies=[]
        for folder in ['arguments','scene','utils','gaussian_renderer','scripts']:
            dependencies += sorted((ROOT/folder).rglob('*.py'))
        dependencies += [ROOT/'LIHAN/scripts/bootstrap_1011.py',ROOT/'LIHAN/scripts/static_rgb_light_legacy_validate.py']
        provenance=json.loads((extraction/'extraction.json').read_text())['rasterizer_provenance']
        dependencies += [ROOT/path for path in provenance['source_files']]
        dependencies=sorted(set(dependencies))
        backend_copy=snapshot/'backend_extension.so'
        shutil.copy2(provenance['extension_path'],backend_copy)
        if sha(backend_copy)!=provenance['extension_sha256']:raise RuntimeError('原生CUDA扩展身份变动')
        code_receipt=dict(files={str(path.relative_to(ROOT)):sha(path) for path in dependencies},
                          packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()],key=lambda d:d['name'].lower()),
                          python=sys.version,server=os.uname().nodename)
        write(control/'code_environment_receipt.json',code_receipt)
        for path in dependencies:
            dst=snapshot/'dependencies'/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
        assets += dependencies+[backend_copy,control/'code_environment_receipt.json',control/'baseline_review.json',control/'method_review.json',
                                source/'rgb_only_manifest.json',source/'transforms_all.json',control/'preparation_commands.json']
        manifest=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GT_read_before_freeze=False,
              final_method='E4',coordinate_system='world',direction_convention='surface_to_light',frames=list(range(1,151)),
              predictions_sha256=sha(control/'predictions.npz'),files={str(path.relative_to(ROOT)):sha(path) for path in assets},
              baseline_model=str(model),baseline_iteration=iteration,deform_enabled=deform_enabled,
              statement='所有选择仅基于RGB与冻结Stage1几何；GT独立一次评测后不得修改预测或参数')
        write(control/'freeze_manifest.json',manifest)
        run('FREEZE_CHECK','LIHAN.scripts.static_rgb_light_evaluate',['--freeze-manifest',control/'freeze_manifest.json',
            '--predictions',control/'predictions.npz','--output',control/'evaluation','--validate-freeze-only'],False)
        run('GT_EVALUATION','LIHAN.scripts.static_rgb_light_evaluate',['--freeze-manifest',control/'freeze_manifest.json',
            '--predictions',control/'predictions.npz','--output',control/'evaluation','--allow-gt-evaluation'],False)
        status('EVALUATED_PENDING_VISUAL_REVIEW')
    except Exception as e:
        status('FAILED',error=str(e));raise


if __name__=='__main__':main()
