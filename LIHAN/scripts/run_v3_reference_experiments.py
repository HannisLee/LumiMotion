"""参考图片/历史loss三组真实训练与全部终点冻结后统一评价。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

PYTHON='/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控'
PARENT=ROOT/'output/1019-59-LH-dynamic-jumpingjacks-ELDL-V3-N00原生Stage1/model_mlp'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def main(args):
    assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path(PYTHON).resolve()
    control=args.control.resolve();snapshot=control/'snapshot_v1'
    assert control.is_relative_to(ROOT/'output') and not (control/'state.json').exists()
    for rel,digest in read(snapshot/'snapshot_manifest.json')['files'].items():assert sha(snapshot/rel)==digest,rel
    used=int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used<1024,'首次启动使用空闲GPU，禁止干预其他任务'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),DYNAMIC_V3_REPO_ROOT=str(ROOT),PYTHONUNBUFFERED='1',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    state=dict(status='STARTING',pid=os.getpid(),gpu=args.gpu,started_unix=time.time(),jobs={},
        source_sha256=sha(__file__),snapshot_manifest_sha256=sha(snapshot/'snapshot_manifest.json'),
        budget_per_model=8100,preselected=['I0','H0','I1'],test_or_gt_training=False)
    def persist():state['updated_unix']=time.time();write(control/'state.json',state)
    persist()
    def command(module,params,target=snapshot):
        return [PYTHON,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(target),module]+list(map(str,params))
    def start(name,cmd,folder,expected):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        log_path=folder/(name+'.log');assert not log_path.exists(),'禁止覆盖日志/实验'
        log=log_path.open('x');proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        rec=dict(status='RUNNING',pid=proc.pid,command=cmd,shell=shlex.join(cmd),log=str(log_path),folder=str(folder),
            expected=str(expected),started_unix=time.time(),environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','DYNAMIC_V3_REPO_ROOT','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        state['jobs'][name]=rec;persist();write(folder/(name+'_command.json'),rec)
        if not (folder/'README.md').exists():
            (folder/'README.md').write_text('# 参考损失实验\n\n完整训练/渲染命令逐条保存于*_command.json。当前调用：\n\n```bash\n'+shlex.join(cmd)+
              '\n```\n\nsource：'+str(OLD/'data_train_v1')+'；父model：'+str(PARENT)+'；iteration35000。当前model为本目录model_mlp。'
              '\n\n日志与命令同目录；量化指标、代表图/视频、RGB/Alpha/法线/Depth-Mesh/静动态四类目检待生成后更新。最终结论PENDING，进程启动不是PASS。\n')
        return proc,log,rec
    def finish(name,task):
        proc,log,rec=task;code=proc.wait();log.close();rec.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'FAILED',finished_unix=time.time());persist()
        write(Path(rec['folder'])/(name+'_command.json'),rec)
        with (Path(rec['folder'])/'README.md').open('a') as f:f.write('\n'+name+'退出码：'+str(code)+'。质量结论仍需终点评价与目检。\n')
        return code==0 and Path(rec['expected']).is_file()
    def run(name,cmd,folder,expected):assert finish(name,start(name,cmd,folder,expected)),name+'失败，保留现场'
    recipes=[('I0','image','I0图片损失Stage2'),('H0','historical','H0历史损失Stage2'),('I1','joint','I1联合校准Stage2')]
    experiments={key:ROOT/f'output/1020-{number:02d}-LH-dynamic-jumpingjacks-ELDL-V3-{label}' for number,(key,recipe,label) in enumerate(recipes,5)}
    def train(recipe,folder,smoke=False):
        params=['--source',OLD/'data_train_v1','--parent',PARENT,'--lighting',control/'lighting.json',
            '--stats',OLD/'input_audit_v1/train_statistics.json','--output',folder,'--recipe',recipe]
        if smoke:params.append('--smoke')
        return command('LIHAN.scripts.train_v3_reference_loss',params)
    try:
        state['status']='SMOKE_THREE_RECIPES';persist()
        for number,(key,recipe,label) in enumerate(recipes,2):
            folder=ROOT/f'output/smoke_test/1020-{number:02d}-LH-dynamic-jumpingjacks-ELDL-V3-{key}参考损失20步'
            run('smoke_'+key,train(recipe,folder,True),folder,folder/'smoke_reload.json')
            assert read(folder/'training_finished.json')['actual_optimizer_updates']==20
            rows=[json.loads(line) for line in (folder/'train_audit.jsonl').read_text().splitlines()]
            for r in rows:
                if r['normal_opacity_protected'] and r['gradient_audit']:
                    assert r['gradient_audit']['normal']['opacity'] is None or r['gradient_audit']['normal']['opacity']['norm']==0
            geom=next(r for r in rows if r['step']==9)
            assert geom['gradient_audit']['rgb']['xyz']['norm']>0 and geom['gradient_audit']['normal']['xyz']['norm']>0
        state['status']='THREE_FORMAL_8100_RUNNING';persist();running={};failed=[]
        for key,recipe,label in recipes:
            folder=experiments[key];assert not folder.exists(),'正式目录已存在，禁止覆盖'
            running[key]=start('train_'+key,train(recipe,folder),folder,folder/'training_finished.json')
        while running:
            for key,task in list(running.items()):
                if task[0].poll() is None:continue
                if not finish('train_'+key,task):failed.append(key)
                del running[key]
            if running:time.sleep(10)
        assert not failed,'正式训练失败：'+str(failed)
        state['status']='DEV_AND_FREEZE_ALL_THREE_BEFORE_TEST';persist();entries=[]
        for key,recipe,label in recipes:
            folder=experiments[key];assert read(folder/'training_finished.json')['actual_optimizer_updates']==8100
            run('dev_'+key,command('LIHAN.scripts.dynamic_v3_dev',['score','--source',OLD/'data_dev_v1',
                '--model',folder/'model_mlp','--kind','stage2','--iteration',8100,'--output',folder/'dev_8100_v1'],ROOT),folder,folder/'dev_8100_v1/summary.json')
            freeze_folder=folder/'eval_only/freeze_v1'
            cmd=[PYTHON,'-m','LIHAN.scripts.dynamic_v3_evaluate','--action','freeze','--source',str(OLD/'eval_only/unified_source_v3_v3/source.json'),
                '--model',str(folder/'model_mlp'),'--iteration','8100','--stage','stage2','--output',str(freeze_folder),'--confirm-frozen',
                '--track-protocol','persistent_vertices_v1','--correspondence',str(OLD/'eval_only/1019-03-新动态GT持久顶点对应审计/correspondence.json')]
            run('freeze_'+key,cmd,folder,freeze_folder/'freeze.json')
            entries.append(dict(name=key,model=str(folder/'model_mlp'),iteration=8100,freeze=str(freeze_folder/'freeze.json'),freeze_sha256=sha(freeze_folder/'freeze.json')))
        write(control/'evaluation_lock.json',dict(entries=entries,all_three_frozen_before_test=True,checkpoint_selection='无：三组预先固定8100终点全部报告'))
        state['status']='UNIFIED_EVALUATION';persist();summary={}
        for entry in entries:
            key=entry['name'];folder=experiments[key];evaluation=folder/f'eval_only/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-{key}统一评价'
            run('evaluate_'+key,command('LIHAN.scripts.dynamic_v3_evaluate',['--action','run','--freeze',entry['freeze'],'--output',evaluation],ROOT),folder,evaluation/'COMPLETED.json')
            metric=read(evaluation/'metrics.json');normal=metric['summary']['foreground']['normal']
            summary[key]=dict(evaluation=str(evaluation),normal={k:v['missing180_frame_mean_deg'] for k,v in normal.items()},
                normal_composite=sum(normal[k]['missing180_frame_mean_deg'] for k in ['gs_render','depth_normal','mesh'])/3,
                rgb_foreground=metric['rgb']['foreground'],metrics_sha256=sha(evaluation/'metrics.json'),manual_review='PENDING')
            with (folder/'README.md').open('a') as f:f.write('\n8100真实更新、DEV及统一评价完成；完整渲染命令与图像/视频/指标见'+str(evaluation/'README.md')+'。四类目检与科学PASS/FAILED仍PENDING。\n')
        write(control/'quantitative_summary.json',summary)
        state.update(status='READY_FOR_MANUAL_REVIEW',finished_unix=time.time());persist()
    except Exception as error:
        state.update(status='FAILED',error=repr(error),finished_unix=time.time());persist();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
    main(p.parse_args())
