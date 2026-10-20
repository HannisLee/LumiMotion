"""从动态DL V2原生Stage1迁移阴影投影估光及ELDL H0配方，保存独立完整结果。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(os.environ.get('DYNAMIC_V3_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
PYTHON = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
RAW = ROOT/'data/LH-data/danamic/LH-dynamic-jumpingjacks_DL_V2'
NATIVE = ROOT/'output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段'
PARENT = NATIVE/'stage1_frozen'
REFERENCE = ROOT/'output/1020-06-LH-dynamic-jumpingjacks-ELDL-V3-H0历史损失Stage2'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p = Path(p); tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def intensity(args):
    """固定新投影方向，只由本数据TRAIN拟合共享光强；不迁移ELDL的物理数值。"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel, load_train_cameras, sh2_basis
    from LIHAN.scripts.dynamic_v3_estimate_light import censored_frame_loss
    from utils.no_gt_guard import install_no_gt_guard
    out = args.experiment.resolve(); source = out/'data_train_v1'
    stats_path = out/'input_audit_v1/train_statistics.json'
    projection = out/'projection_v1'; result = read(projection/'ambient_contrast/result.json')
    freeze = read(projection/'freeze.json')
    assert not result['gt_read'] and 'directions' in result
    assert sha(projection/'ambient_contrast/result.json') == freeze['artifact_sha256']['ambient_contrast/result.json']
    assert not (out/'lighting.json').exists()
    install_no_gt_guard([source, PARENT, projection, stats_path, out])
    torch.set_num_threads(4); torch.manual_seed(1019)
    stats = read(stats_path)
    initial = dict(times=torch.tensor(result['times']), directions=torch.tensor(result['directions']), intensity=math.pi)
    model = DynamicSurfaceModel(PARENT, 35000, initial, stats, 'A')
    cameras = load_train_cameras(source)
    cached = torch.load(projection/'observations/observations.pt', map_location='cpu', weights_only=False)
    selected = np.random.default_rng(1019).choice(len(cached['ids']), min(2048,len(cached['ids'])), replace=False)
    ids = cached['ids'][selected].cuda()
    normals = cached['world_normals'][:,selected].cuda()
    target = cached['observed'][:,selected].cuda(); saturated = cached['saturated'][:,selected].cuda()
    valid = cached['valid'][:,selected].cuda().float()
    assert int(valid.sum()) > 1000
    direct = []
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            camera.load2device(); geometry = model.effective(camera)
            model.visibility.build(geometry)
            direction = model.radiance.lights(camera.fid)
            visible = model.visibility.trace(geometry.get_xyz[ids], direction[None])[:,0]
            direct.append(visible*(normals[index]*direction).sum(-1).clamp_min(0))
            camera.load2device('cpu')
    shading = torch.stack(direct).detach()
    rho_raw = torch.nn.Parameter(torch.zeros(len(ids),3,device='cuda'))
    raw_D = torch.nn.Parameter(torch.tensor(math.log((math.pi-.1)/(20-math.pi)),device='cuda'))
    coefficients = torch.nn.Parameter(torch.zeros(9,3,device='cuda'))
    basis = sh2_basis(normals); env_scale = float(model.radiance.environment_scale)
    optimizer = torch.optim.Adam([rho_raw, raw_D, coefficients], lr=.01)
    history = []
    for step in range(900):
        optimizer.zero_grad(set_to_none=True)
        rho = .03+.94*(rho_raw-rho_raw.median().detach()).sigmoid()
        D = .1+19.9*raw_D.sigmoid()
        env = env_scale*F.softplus(basis@(.5*coefficients.tanh())+math.log(math.expm1(1.)))
        prediction = rho[None]/math.pi*(env+D*shading[...,None])
        photo = censored_frame_loss(prediction,target,saturated,valid).mean()
        loss = photo+.0001*(rho_raw.square().mean()+coefficients.square().mean())
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
        if step in (0,199,399,599,899):
            row = dict(step=step+1,photo=float(photo.detach()),intensity=float(D.detach()))
            history.append(row); print(json.dumps(row),flush=True)
    bound = [source/'no_gt_manifest.json',out/'input_audit_v1/radiometry_v1.json',stats_path,
             PARENT/'cfg_args',PARENT/'point_cloud/iteration_35000/point_cloud.ply',
             PARENT/'deform/iteration_35000/deform.pth',projection/'freeze.json',
             projection/'ambient_contrast/result.json',Path(__file__)]
    write(out/'lighting.json',dict(format_version=1,status='DL_V2_TRAIN_PROJECTION_FIXED_DIRECTIONS',
        source_split='TRAIN',gt_read=False,smoke=False,source=str(source),parent=str(PARENT),parent_iteration=35000,
        train_frames=result['train_frames'],times=result['times'],directions=result['directions'],
        intensity=float((.1+19.9*raw_D.sigmoid()).detach()),global_intensity=True,per_frame_exposure=False,
        intensity_initialization='本数据TRAIN固定投影方向，2048对应点，900步共享rho/受限SH/全局D；只迁移D标量，正式rho与SH重置',
        direction_lr=0.,direction_source=str(projection/'ambient_contrast/result.json'),
        source_sha256={str(p):sha(p) for p in bound},fit_history=history))


def direction_audit(args):
    """训练终点冻结后只读GT灯向；误差不反馈训练。"""
    import importlib.util
    import numpy as np
    import torch
    out = args.experiment.resolve(); lighting = read(out/'lighting.json')
    gt_path = RAW/'lights.json'; data = read(gt_path)
    gt = []
    for n in range(1,151):
        item = data[f'{n:04d}']; v = np.asarray(item['light_dir_world'],np.float64)
        assert np.isfinite(v).all() and np.linalg.norm(v)>0
        gt.append(v/np.linalg.norm(v))
    gt = np.asarray(gt)
    spec = importlib.util.spec_from_file_location('dl_v2_table_only',ROOT/'scene/stage2_photometric.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    table = module.DirectionalLightTable(torch.tensor(lighting['times']),torch.tensor(lighting['directions']),False)
    with torch.no_grad():
        pred = torch.stack([table(n/150) for n in range(1,151)]).double().numpy()
    pred /= np.linalg.norm(pred,axis=-1,keepdims=True)
    error = np.degrees(np.arctan2(np.linalg.norm(np.cross(pred,gt),axis=-1),(pred*gt).sum(-1)))
    manifest = read(out/'data_train_v1/no_gt_manifest.json')
    splits = {k.upper():manifest[k+'_frames'] for k in ('train','dev','test')}; splits['ALL']=list(range(1,151))
    summary = {}
    for key,frames in splits.items():
        a = error[np.asarray(frames)-1]
        summary[key] = dict(count=len(a),p50_deg=float(np.quantile(a,.5)),p95_deg=float(np.quantile(a,.95)),
                            mean_deg=float(a.mean()),max_deg=float(a.max()))
    destination = out/'eval_only/direction_audit_v1'; destination.mkdir(parents=True,exist_ok=False)
    write(destination/'audit.json',dict(status='EVALUATED_ONLY',optimizer_updates=0,gt_used_for_training=False,
        gt_sha256=sha(gt_path),lighting_sha256=sha(out/'lighting.json'),summary=summary,
        per_frame=[dict(frame=n,error_deg=float(error[n-1]),pred=pred[n-1].tolist(),gt=gt[n-1].tolist()) for n in range(1,151)]))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def supervise(args):
    assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path(PYTHON).resolve()
    out = args.experiment.resolve(); assert out.is_relative_to(ROOT/'output')
    assert out.exists() and not (out/'state.json').exists()
    snapshot = out/'snapshot_v1'; snapshot.mkdir(exist_ok=False)
    for name in ('scene','utils','arguments','gaussian_renderer','assets','scripts'):
        shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (snapshot/'LIHAN').mkdir(); shutil.copytree(ROOT/'LIHAN/scripts',snapshot/'LIHAN/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    write(snapshot/'snapshot_manifest.json',dict(files={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}))
    old_snapshot = ROOT/'output/1020-04-LH-dynamic-jumpingjacks-ELDL-V3-参考损失三组总控/snapshot_v1'
    for rel in ['LIHAN/scripts/train_v3_reference_loss.py','LIHAN/scripts/dynamic_v3_surface.py',
                'LIHAN/scripts/dynamic_v3_train.py','scene/static_dl_censored.py']:
        assert sha(snapshot/rel)==sha(old_snapshot/rel),rel+'：迁移核心必须与H0相同'
    projection_freeze=read(ROOT/'output/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-T05动态投影估光/freeze.json')
    projection_source=ROOT/'LIHAN/scripts/dynamic_v3_projection.py'
    assert sha(projection_source)==projection_freeze['input_sha256'][str(projection_source)]
    hashes = {}
    for rel in ['cfg_args','point_cloud/iteration_35000/point_cloud.ply','deform/iteration_35000/deform.pth']:
        assert sha(PARENT/rel)==sha(NATIVE/'model_mlp'/rel)
        hashes[str(PARENT/rel)] = sha(PARENT/rel)
    write(out/'reference_binding.json',dict(dataset='dlv2',reference=str(REFERENCE),parent=str(PARENT),
        stage1_iteration=35000,parent_sha256=hashes,stage2_updates=8100,seed=1019,
        target_recipe='historical',time_rule='frame/150',reference_config=read(REFERENCE/'training_finished.json')['config'],
        inherited_components='loss、优化器、学习率、reset、参数开放日程；不复制ELDL灯向或光强',
        core_sha256={rel:sha(snapshot/rel) for rel in ['LIHAN/scripts/train_v3_reference_loss.py','LIHAN/scripts/dynamic_v3_projection.py']}))
    used = int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used<1024,'首次启动需要空闲GPU'
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),DYNAMIC_V3_REPO_ROOT=str(ROOT),PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    state = dict(status='PREPARED',pid=os.getpid(),gpu=args.gpu,jobs={},started_unix=time.time(),actual_updates=0,
                 fixed_budget=8100,gt_training=False,checkpoint_selection='预登记唯一8100终点；不按TEST/GT选择')
    def persist():
        state['updated_unix']=time.time(); write(out/'state.json',state)
    def command(module,params,target=snapshot):
        return [PYTHON,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(target),module]+list(map(str,params))
    commands = {}
    def job(name,cmd,expected):
        assert not (out/(name+'.log')).exists()
        commands[name]=cmd; write(out/'commands.json',commands)
        state['status']='RUNNING_'+name.upper(); record=dict(command=cmd,shell=shlex.join(cmd),started_unix=time.time())
        state['jobs'][name]=record; persist()
        with (out/'README.md').open('a') as f:
            f.write('\n### '+name+'完整命令\n\n```bash\n'+shlex.join(cmd)+'\n```\n\n日志：'+name+'.log\n')
        with (out/(name+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record.update(pid=process.pid,status='RUNNING'); persist(); rc=process.wait()
        record.update(exit_code=rc,status='PROCESS_FINISHED' if rc==0 else 'FAILED',finished_unix=time.time()); persist()
        assert rc==0 and Path(expected).is_file(),name+'失败；保留现场'
    persist()
    try:
        train_source=out/'data_train_v1'; stats=out/'input_audit_v1/train_statistics.json'
        job('prepare_train',[PYTHON,'-m','LIHAN.scripts.dynamic_v3_prepare','--source',str(RAW),'--output',str(train_source)],train_source/'no_gt_manifest.json')
        job('radiometry',[PYTHON,'-m','LIHAN.scripts.dynamic_v3_radiometry','--source',str(RAW),'--manifest',str(train_source/'no_gt_manifest.json'),
            '--output',str(out/'input_audit_v1/radiometry_v1.json')],out/'input_audit_v1/radiometry_v1.json')
        job('prepare_dev',[PYTHON,'-m','LIHAN.scripts.dynamic_v3_dev','prepare','--raw',str(RAW),'--train-manifest',str(train_source/'no_gt_manifest.json'),
            '--radiometry',str(out/'input_audit_v1/radiometry_v1.json'),'--stats',str(stats),'--output',str(out/'data_dev_v1')],out/'data_dev_v1/dev_manifest.json')
        # 原投影入口以cwd作为输出根；在仓库根执行，源码仍逐文件绑定原冻结SHA。
        job('projection',command('LIHAN.scripts.dynamic_v3_projection',['--source',train_source,'--parent',PARENT,'--stats',stats,'--output',out/'projection_v1'],ROOT),out/'projection_v1/freeze.json')
        job('intensity',command('LIHAN.scripts.reproduce_dynamic_dl_v2_h0',['--action','intensity','--experiment',out]),out/'lighting.json')
        job('train',command('LIHAN.scripts.train_v3_reference_loss',['--source',train_source,'--parent',PARENT,'--lighting',out/'lighting.json',
            '--stats',stats,'--output',out,'--recipe','historical']),out/'training_finished.json')
        finished=read(out/'training_finished.json');assert finished['actual_optimizer_updates']==8100
        assert finished['config']==read(REFERENCE/'training_finished.json')['config']
        state['actual_updates']=8100;persist()
        job('dev',command('LIHAN.scripts.dynamic_v3_dev',['score','--source',out/'data_dev_v1','--model',out/'model_mlp',
            '--kind','stage2','--iteration',8100,'--output',out/'dev_8100_v1'],ROOT),out/'dev_8100_v1/summary.json')
        evaluation_source=out/'eval_only/source_v1'
        job('prepare_evaluation',[PYTHON,'-m','LIHAN.scripts.dynamic_v3_evaluate','--action','prepare','--dataset','dlv2',
            '--output',str(evaluation_source)],evaluation_source/'source.json')
        entries=[]
        for label,stage,model,step in [('S1','stage1',PARENT,35000),('H0','stage2',out/'model_mlp',8100)]:
            folder=out/f'eval_only/freeze_{label}_v1'
            job('freeze_'+label,[PYTHON,'-m','LIHAN.scripts.dynamic_v3_evaluate','--action','freeze','--source',str(evaluation_source/'source.json'),
                '--model',str(model),'--iteration',str(step),'--stage',stage,'--output',str(folder),'--confirm-frozen'],folder/'freeze.json')
            entries.append(dict(label=label,freeze=str(folder/'freeze.json'),sha256=sha(folder/'freeze.json')))
        write(out/'evaluation_lock.json',dict(entries=entries,both_frozen_before_test=True,selection='预定8100终点；仅同数据Stage1与Stage2对比'))
        job('direction_audit',[PYTHON,'-m','LIHAN.scripts.reproduce_dynamic_dl_v2_h0','--action','direction-audit','--experiment',str(out)],out/'eval_only/direction_audit_v1/audit.json')
        for entry in entries:
            label=entry['label'];folder=out/f'eval_only/1021-01-LH-dynamic-jumpingjacks_DL_V2-{label}统一评价'
            job('evaluate_'+label,command('LIHAN.scripts.dynamic_v3_evaluate',['--action','run','--freeze',entry['freeze'],'--output',folder],ROOT),folder/'COMPLETED.json')
        summary={}
        for entry in entries:
            label=entry['label'];folder=out/f'eval_only/1021-01-LH-dynamic-jumpingjacks_DL_V2-{label}统一评价'
            metric=read(folder/'metrics.json')
            summary[label]=dict(evaluation=str(folder),normal_composite=metric['normal_composite_fixed_denominator_mae'],
                normal=metric['summary']['foreground']['normal'],rgb=metric['rgb']['foreground'],metrics_sha256=sha(folder/'metrics.json'))
        write(out/'quantitative_summary.json',summary)
        state.update(status='READY_FOR_MANUAL_REVIEW',finished_unix=time.time());persist()
        with (out/'README.md').open('a') as f:f.write('\n8100更新及同协议Stage1/Stage2完整评价完成。量化见quantitative_summary.json，四类实际目检与科学结论仍PENDING。\n')
    except Exception as exc:
        state.update(status='FAILED',error=repr(exc),finished_unix=time.time());persist()
        with (out/'README.md').open('a') as f:f.write('\n最终结论：FAILED（执行异常）'+repr(exc)+'；全部现场保留。\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['supervise','intensity','direction-audit'],default='supervise')
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
    a=p.parse_args()
    {'supervise':supervise,'intensity':intensity,'direction-audit':direction_audit}[a.action](a)
