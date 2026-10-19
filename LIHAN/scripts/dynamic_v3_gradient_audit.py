"""仅TRAIN只读核验正式物理模型的实际求导；不执行optimizer.step。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import sys

import torch

from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel, load_train_cameras
from LIHAN.scripts.dynamic_v3_train import common_geometry, photo_objective
from LIHAN.scripts.dynamic_v3_tracks import ps_geometry_loss
from utils.no_gt_guard import install_no_gt_guard


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','stats','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--ps-by-gate',action='store_true',help='只读分解模型动态/静态目标的实际PS梯度')
    parser.add_argument('--frame-policy',choices=('fixed','dynamic-valid'),default='fixed')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    source,folder,stats_path,out=[getattr(args,n).resolve() for n in ('source','model','stats','output')]
    assert socket.gethostname()=='ushas'
    assert Path(sys.executable).resolve()==Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve()
    assert out.is_relative_to(root/'output/smoke_test')
    out.mkdir(parents=True,exist_ok=True)
    assert all(p.name in {'run.log','launch.json','README.md'} for p in out.iterdir()),'已有审计产物，禁止覆盖'
    assert not (out/'verification.json').exists()
    torch.set_num_threads(4)
    marker=read(folder/'dynamic_v3_stage2.json')
    assert Path(marker['source']).resolve()==source and Path(marker['stats']).resolve()==stats_path
    manifest=read(source/'no_gt_manifest.json')
    assert all(n in manifest['train_frames'] for n in (1,81,149))
    assert args.ps_by_gate or args.frame_policy=='fixed'
    source_files={str(source/'no_gt_manifest.json'):sha(source/'no_gt_manifest.json'),str(stats_path):sha(stats_path)}
    for path,digest in marker['source_sha256'].items():
        assert sha(path)==digest,path
        source_files[path]=digest
    for name in ('dynamic_v3_surface.py','dynamic_v3_train.py','dynamic_v3_tracks.py','dynamic_v3_ps_math.py'):
        recorded=[p for p in marker['source_sha256'] if Path(p).name==name]
        assert len(recorded)==1 and sha(Path(__file__).with_name(name))==marker['source_sha256'][recorded[0]],name
        source_files[str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
    receipt_path=folder/f'dynamic_v3/iteration_{args.step}/receipt.json'
    receipt=read(receipt_path)
    for relative,digest in receipt['checkpoint_sha256'].items():
        path=(folder/relative).resolve();assert path.is_relative_to(folder) and sha(path)==digest
        source_files[str(path)]=digest
    source_files[str(receipt_path)]=sha(receipt_path)
    state=torch.load(folder/f'dynamic_v3/iteration_{args.step}/state.pt',map_location='cpu',weights_only=False)
    targets_path=Path(state['ps_reference']).resolve() if state['ps_reference'] else None
    if targets_path:source_files[str(targets_path)]=sha(targets_path)
    observations_path=targets_path.parent.parent/'observations/observations.pt' if args.ps_by_gate and targets_path else None
    if observations_path:source_files[str(observations_path)]=sha(observations_path)
    install_no_gt_guard([source,folder,stats_path,out]+([targets_path] if targets_path else [])+
        ([observations_path] if observations_path else []))
    stats=read(stats_path);model=DynamicSurfaceModel.load(folder,args.step)
    targets=torch.load(targets_path,map_location='cpu',weights_only=False) if targets_path else None
    gates=None;eligible_dynamic_frames=[];frame_ids=[1,81,149]
    if args.ps_by_gate:
        assert targets is not None and observations_path is not None,'需真实已保存PS目标与观测缓存'
        observations=torch.load(observations_path,map_location='cpu',weights_only=False)
        assert torch.equal(observations['ids'],targets['ids'])
        gates=observations['dynamic'].bool()
        supported=gates&targets['reliable'].bool()&(targets['confidence']>0)
        eligible_dynamic_frames=[frame for index,frame in enumerate(targets['frame_numbers'])
            if (targets['valid'][index]&supported).any()]
        if args.frame_policy=='dynamic-valid':
            assert eligible_dynamic_frames,'没有可靠动态目标的有效TRAIN观测；不改变门槛或扫描模型'
            frame_ids=sorted(set(eligible_dynamic_frames[i] for i in (0,len(eligible_dynamic_frames)//2,len(eligible_dynamic_frames)-1)))
    assert all(frame in manifest['train_frames'] for frame in frame_ids)
    cameras=load_train_cameras(source,frame_ids=frame_ids)
    assert len(cameras)==len(frame_ids)
    names=['xyz','rotation','scale','opacity','rho','D','SH','light']
    params=[model.pc._xyz,model.pc._rotation,model.pc._scaling,model.pc._opacity,
        model.radiance.raw_rho,model.radiance.raw_intensity,
        model.radiance.environment_coefficients,model.radiance.lights.raw_directions]
    for name,p in model.deform.deform.named_parameters():
        if not name.startswith('gaussian_color'):
            names.append('deformation.'+name);params.append(p)
    for p in params:p.requires_grad_(True)
    saved_lrs={group['name']:group['lr'] for group in state['optimizer']['param_groups']}
    rows=[];failures=[]
    for camera in cameras:
        camera.load2device()
        for protected in (False,True):
            package=model.render(camera,protect_opacity=protected)
            assert not package['surface_components']['visibility'].requires_grad
            photo=photo_objective(package,camera,stats);normal,_=common_geometry(package,camera)
            objectives={'photo':photo,'normal':normal}
            info=None
            if targets is not None:
                objectives['ps'],info=ps_geometry_loss(model,camera,package,targets)
            gradients={};gate_ps={}
            if gates is not None:
                denominator=max(info['weight_sum'],1.)
                for label,mask in [('dynamic',gates),('static',~gates)]:
                    masked=dict(targets,confidence=targets['confidence']*mask*targets['reliable'])
                    value,gate_info=ps_geometry_loss(model,camera,package,masked)
                    # 两组沿用完整PS分母，贡献可相加；不重新规范稀少动态目标的权重。
                    objectives['ps_'+label]=value*max(gate_info['weight_sum'],1.)/denominator
                    gate_ps[label]=gate_info
                difference=float((objectives['ps_dynamic']+objectives['ps_static']-objectives['ps']).detach().abs())
                assert difference<1e-6,'分组贡献未复原完整PS'
            for objective,value in objectives.items():
                gs=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) if value.requires_grad else [None]*len(params)
                gradients[objective]={name:None if g is None else dict(norm=float(g.norm()),
                    maximum=float(g.abs().max()),finite=bool(torch.isfinite(g).all())) for name,g in zip(names,gs)}
                if any(g is not None and not g['finite'] for g in gradients[objective].values()):
                    failures.append(f'{camera.image_name_train_light}/{protected}/{objective}:非有限梯度')
            if protected:
                for objective in ('normal','ps','ps_dynamic','ps_static'):
                    g=gradients.get(objective,{}).get('opacity')
                    if g is not None and g['maximum']!=0.:
                        failures.append(f'{camera.image_name_train_light}/{objective}:保护期仍更新opacity')
                g=gradients['photo']['opacity']
                if g is None or g['norm']==0.:failures.append(camera.image_name_train_light+':RGB/Alpha不更新opacity')
            row=dict(frame=camera.image_name_train_light,protected=protected,
                objectives={k:float(v.detach()) for k,v in objectives.items()},gradients=gradients,ps_diagnostics=info,
                ps_by_gate=gate_ps,visibility_detached=True)
            rows.append(row)
            print(json.dumps(dict(frame=row['frame'],protected=protected,
                photo_gradients={k:row['gradients']['photo'][k] for k in names[:8]},
                protected_normal_opacity=row['gradients']['normal']['opacity']),ensure_ascii=False),flush=True)
            del package,photo,normal,objectives,value,gs
        camera.load2device('cpu')
    observed={name:any(row['gradients']['photo'][name] is not None and row['gradients']['photo'][name]['norm']>0 for row in rows) for name in names[:8]}
    for name in ('xyz','rotation','scale','opacity','rho','D','light'):
        if not observed[name]:failures.append(name+':三TRAIN帧未观察到非零photo梯度')
    if model.path=='A' and not observed['SH']:failures.append('SH:路径A未观察到非零photo梯度')
    if model.path=='B' and observed['SH']:failures.append('路径B意外使用SH')
    dynamic_nonzero={name:any(row['gradients'].get('ps_dynamic',{}).get(name) is not None and
        row['gradients']['ps_dynamic'][name]['norm']>0 for row in rows) for name in ('xyz','rotation','scale')}
    if args.ps_by_gate and not all(dynamic_nonzero.values()):failures.append('所选TRAIN观测未证明动态分组XYZ/rotation/scale均有非零PS贡献')
    for path,digest in source_files.items():assert sha(path)==digest,'只读核验期间输入改变：'+path
    argv=list(getattr(sys,'orig_argv',[sys.executable]+sys.argv))
    command=' '.join(shlex.quote(k+'='+os.environ[k]) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS') if k in os.environ)+' '+shlex.join(argv)
    report=dict(status='FAILED_GRADIENT_AUDIT' if failures else 'PASS_TRAIN_DIFFERENTIABILITY_ONLY',
        gt_read=False,test_read=False,optimizer_step_executed=False,weights_unchanged=True,
        source=str(source),model=str(folder),iteration=args.step,path=model.path,
        saved_optimizer_lrs=saved_lrs,training_ps_weight=state['config']['ps_weight'],
        diagnostic_requires_grad='临时启用几何、材质、SH、灯向和非颜色形变参数，只计算梯度，不更新模型；不等于训练已启用这些组',
        light_gradient='真实全场景BVH重算但V停止梯度；只核验固定V的近似方向导数，不声称完整遮挡梯度',
        ps_reference=str(targets_path) if targets_path else None,photo_nonzero_on_any_frame=observed,
        frame_policy=args.frame_policy,frames=frame_ids,eligible_dynamic_train_frames=eligible_dynamic_frames,
        ps_by_gate=args.ps_by_gate,dynamic_ps_nonzero_on_any_frame=dynamic_nonzero if args.ps_by_gate else None,
        gate_limitation='按模型gate分组，非GT人体/运动标签；分组贡献共享完整PS分母。若训练ps_weight为0，只是反事实诊断，未作为训练约束',
        failures=failures,rows=rows,source_sha256=source_files,command=command,
        code_sha256=sha(__file__),model_accuracy_evaluated=False)
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text('# 正式检查点TRAIN只读求导核验\n\n完整命令：\n\n```bash\n'+command+
        '\n```\n\nsource：'+str(source)+'；model：'+str(folder)+'；iteration：'+str(args.step)+
        '。\n\n日志：run.log；定量：verification.json。TRAIN帧'+str(frame_ids)+'，策略'+args.frame_policy+'，各核验普通与opacity保护前向。'
        '不执行优化器更新，前后检查点SHA一致。临时打开诊断求导不表示训练中启用了该组，实际学习率另列。'
        '\n\n本入口不保存渲染/图片/视频；四类目检均未执行，不能冒称验收通过。'
        '\n\n最终结论：'+('FAILED' if failures else 'PASS')+'（仅TRAIN实际求导；不证明准确率、动态PS收益或真实灯向精度）。\n')
    print(json.dumps(dict(status=report['status'],failures=failures,observed=observed),ensure_ascii=False),flush=True)
    if failures:raise RuntimeError('求导核验失败，保留全部结果：'+str(failures))


if __name__=='__main__':main()
