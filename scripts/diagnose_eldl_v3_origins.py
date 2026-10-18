"""同全BVH下旧虚拟平面起点与真实surfel中心的零优化透射率诊断。"""
import argparse
import math
import os
from pathlib import Path
import shlex
import socket
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts.diagnose_eldl_v3_occlusion import trace,describe

DOC=ROOT/'LIHAN/DOC/报告/1019-07-ELDL-V3真实surfel起点诊断预登记.md'


def plan(args):
    args.output.mkdir(parents=True,exist_ok=False)
    for folder in ['logs','arrays','figures']:(args.output/folder).mkdir()
    v.write(args.output/'plan.json',dict(code_sha256=v.sha(Path(__file__)),validator_sha256=v.sha(Path(v.__file__)),
        tracer_helper_sha256=v.sha(ROOT/'scripts/diagnose_eldl_v3_occlusion.py'),protocol_sha256=v.sha(DOC),
        paths={k:str(getattr(args,k)) for k in ['cache','fit','model','source','output']},iteration=args.iteration,
        status='PLANNED_ZERO_OPTIMIZATION',internal_validation_passed=False,scientific_thresholds=None))
    (args.output/DOC.name).write_bytes(DOC.read_bytes())
    cmd=[str(v.PYTHON),str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),'scripts.diagnose_eldl_v3_origins','--action','run']
    for key in ['cache','fit','model','source','output']:cmd+=['--'+key,str(getattr(args,key))]
    cmd+=['--iteration',str(args.iteration)]
    v.write(args.output/'command.json',dict(argv=cmd,shell=shlex.join(cmd)))
    (args.output/'README.md').write_text('# 真实surfel起点零优化诊断\n\n2026-10-19，ushas。无GT/外部test，不拟合，不迁移rho做RGB验收，不放行资产。\n\n完整命令（显式GPU环境）：\n\n```bash\n'+shlex.join(cmd)+'\n```\n\nsource：`'+str(args.source)+'`；model：`'+str(args.model)+'`；iteration：'+str(args.iteration)+'。输出arrays/figures，日志logs，统计summary.json。\n\n固定fit1/81/149、held61/71/79透射率图；不生成模型四类视频。RGB、Alpha、Normal、分离的质量验收均不适用；本诊断仅位置、法线符号及V稳定性。\n\n最终结论PENDING，internal_validation_passed=false。\n')


def figures(out,origins,selected,data):
    import cv2
    selected=selected.numpy();valid=data['valid'].numpy()[:,selected]
    def tile(values,mask):
        whole=np.full(4096,np.nan);whole[selected[mask]]=values[mask]
        image=cv2.applyColorMap((np.nan_to_num(whole).reshape(64,64).clip(0,1)*255).round().astype(np.uint8),cv2.COLORMAP_TURBO)
        image[np.isnan(whole).reshape(64,64)]=80
        return cv2.resize(image,(192,192),interpolation=cv2.INTER_NEAREST)
    for split,frames in v.REPRESENTATIVES.items():
        for frame in frames:
            i=v.TRAIN.index(frame);rows=[]
            for kind in ['virtual','actual']:
                rows.append(np.concatenate([tile(origins[kind][v.scale_key(scale)][i].numpy(),valid[i]) for scale in v.SCALES],axis=1))
            rows.append(np.concatenate([tile((origins['virtual'][v.scale_key(scale)][i]-origins['actual'][v.scale_key(scale)][i]).abs().numpy(),valid[i]) for scale in v.SCALES],axis=1))
            cv2.imwrite(str(out/f'{split}_{frame:04d}_visibility.png'),np.concatenate(rows,axis=0))


@torch.no_grad()
def run(args):
    record=v.read(args.output/'plan.json')
    for key,path in [('code_sha256',Path(__file__)),('validator_sha256',Path(v.__file__)),('tracer_helper_sha256',ROOT/'scripts/diagnose_eldl_v3_occlusion.py')]:
        if v.sha(path)!=record[key]:raise ValueError('源码与预登记计划不同。')
    if record['paths']!={k:str(getattr(args,k)) for k in ['cache','fit','model','source','output']}:raise ValueError('输入路径与计划不同。')
    if (args.output/'RUN_STARTED.json').exists():raise FileExistsError('禁止覆盖已有执行。')
    v.write(args.output/'RUN_STARTED.json',dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    v.install_guard(args)
    try:
        data,state,summary,result,hashes=v.load_inputs(args)
        if summary['selected_init']!=3:raise ValueError('本诊断固定selected3。')
        if 'receiver_normals' in data:raise ValueError('需要原虚拟plane缓存作为共同点基准。')
        from scene.gaussian_model import GaussianModel
        from scene.static_dl_photometric import StaticVisibility
        from scripts.probe_dynamic_dl_selfcal import read_config
        from utils.general_utils import build_rotation
        from scipy.spatial import cKDTree
        import diff_surfel_rasterization as backend
        if getattr(backend,'provenance',{}).get('method')!='repository_lumimotion_cuda_rasterizer':raise ValueError('需要原生bootstrap。')
        cfg=read_config(args.model/'cfg_args');pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
        pc.load_ply(str(args.model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'));pc.active_sh_degree=min(args.iteration//1000,cfg.sh_degree)
        opacity=pc.get_opacity.clone();normal=F.normalize(data['normal'].cuda(),dim=0);origin=data['origin'].cuda();basis=data['basis'].cuda()
        all_normal=build_rotation(pc.get_rotation)[:,:,2];height=(pc.get_xyz-origin)@normal
        pool=torch.where((opacity[:,0]>.5)&(height.abs()<.005*float(data['extent']))&
            torch.isfinite(all_normal).all(-1)&((all_normal@normal).abs()>math.cos(math.radians(10))))[0]
        if not len(pool):raise ValueError('真实surfel候选为空。')
        grid_uv=((data['points'].cuda()-origin)@basis.T).cpu().numpy()
        pool_uv=((pc.get_xyz[pool]-origin)@basis.T).cpu().numpy()
        distance,nearest=cKDTree(pool_uv).query(grid_uv,k=1,workers=4)
        receiver_ids=pool.cpu().numpy()[nearest]
        steps=np.array([np.linalg.norm(grid_uv[1]-grid_uv[0]),np.linalg.norm(grid_uv[64]-grid_uv[0])])
        radius=float(steps.max());candidate=distance<=radius;seen=set()
        for i,identity in enumerate(receiver_ids):
            if not candidate[i]:continue
            if int(identity) in seen:candidate[i]=False
            else:seen.add(int(identity))
        fit_support=(data['valid'][data['fit']].sum(0)>=10).numpy()
        selected=torch.as_tensor(np.flatnonzero(candidate&fit_support),dtype=torch.long)
        if not len(selected):raise ValueError('共同支持映射为空。')
        selected_ids=torch.as_tensor(receiver_ids[selected.numpy()],dtype=torch.long,device='cuda')
        virtual=data['points'][selected].cuda();actual=pc.get_xyz[selected_ids].clone()
        raw_normal=all_normal[selected_ids];plane_sign=torch.where((raw_normal@normal)>=0,1.,-1.)
        oriented=raw_normal*plane_sign[:,None]
        transforms=v.read(args.source/'transforms_train.json')['frames']
        cameras={r['source_frame']:torch.tensor(r['transform_matrix'],dtype=torch.float32,device='cuda')[:3,3] for r in transforms}
        fit_ids=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]
        camera_centers=torch.stack([cameras[v.TRAIN[i]] for i in fit_ids])
        dots=((camera_centers[:,None]-actual[None])*oriented[None]).sum(-1).cpu()
        support=data['valid'][data['fit']][:,selected]
        disagreement=support&(dots<0);nearzero=support&(dots.abs()<=1e-6)
        mapping=dict(pool_count=len(pool),original_grid_points=4096,unique_nearest_ids_all=len(np.unique(receiver_ids)),
            distance_valid_before_dedup=int((distance<=radius).sum()),unique_distance_valid_before_fit=int(candidate.sum()),
            selected_common_points=len(selected),grid_axis_steps=steps.tolist(),nearest_xy_limit=radius,
            fit_facing_supported_observations=int(support.sum()),fit_facing_sign_disagreements=int(disagreement.sum()),
            fit_facing_nearzero=int(nearzero.sum()),
            nearest_xy=describe(torch.tensor(distance[selected.numpy()])),
            actual_signed_height=describe((actual-origin)@normal),
            displacement_3d=describe((actual-virtual).norm(dim=-1)),
            original_bias=float(data['shadow_bias']),sho_contact_confusion='UNKNOWN；没有GT语义',
            dedup_rule='距离有效节点按原格点升序保留首个同ID，再与原fit支持>=10相交',
            facing_scope='原缓存valid支持的fit相机；不声称已重新验证actual点的新相机可见性')
        torch.save(dict(grid_receiver_ids=torch.tensor(receiver_ids),matching_distance=torch.tensor(distance),
            candidate_support=torch.tensor(candidate),original_fit_support=torch.tensor(fit_support),
            selected_grid_indices=selected,selected_gaussian_ids=selected_ids.cpu(),virtual_points=virtual.cpu(),actual_points=actual.cpu(),
            raw_normal=raw_normal.cpu(),plane_oriented_normal=oriented.cpu(),fit_camera_dot=dots,
            fit_camera_support=support,fit_sign_disagreement=disagreement),args.output/'mapping.pt')
        v.write(args.output/'mapping.json',mapping)
        directions=state['directions'].cuda();builder=StaticVisibility(bias=float(data['shadow_bias']),update_interval=1);builder(pc,directions[:1],args.iteration)
        values={};statistics={};mask=data['valid'][:,selected]
        for kind,points in [('virtual',virtual),('actual',actual)]:
            values[kind]={};statistics[kind]={}
            for scale in v.SCALES:
                key=v.scale_key(scale);vis=trace(pc,points,directions,opacity,float(data['shadow_bias'])*scale,args.batch_directions)
                values[kind][key]=vis
                torch.save(dict(visibility=vis,valid=mask,grid_indices=selected,gaussian_ids=selected_ids.cpu(),kind=kind,bias_scale=scale),args.output/'arrays'/f'{kind}_bias_{key}.pt')
                print(f'{kind} bias {scale}，共同点{len(selected)}',flush=True)
            for key,vis in values[kind].items():
                delta=(vis-values[kind]['1']).abs();split_metrics={}
                for split,frame_mask in [('fit',data['fit']),('held',~data['fit'])]:
                    valid=mask&frame_mask[:,None]
                    split_metrics[split]=dict(visibility=describe(vis[valid]),change_from_center=describe(delta[valid]),
                        visibility_lt_point1=float((vis[valid]<.1).float().mean()),visibility_gt_point9=float((vis[valid]>.9).float().mean()))
                statistics[kind][key]=split_metrics
        if not torch.equal(opacity,pc.get_opacity):raise ValueError('生产全BVH opacity被修改。')
        figures(args.output/'figures',values,selected,data)
        hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__));hashes[str(ROOT/'scripts/diagnose_eldl_v3_occlusion.py')]=v.sha(ROOT/'scripts/diagnose_eldl_v3_occlusion.py')
        v.check_hashes(hashes)
        v.write(args.output/'summary.json',dict(status='DIAGNOSTIC_EXECUTED_NO_PRECISION_ACCEPTANCE',internal_validation_passed=False,
            gt_read=False,external_test_read=False,parameters_refit=False,rgb_rho_transfer_evaluated=False,
            full_geometry_preserved=True,original_bias=float(data['shadow_bias']),selected_init=3,mapping=mapping,
            statistics=statistics,input_sha256=hashes,manual_review='PENDING',
            figure_legend='行：旧virtual、实际center、两者绝对差；列：bias0.5/1/2。色阶0～1，灰为原valid外。',
            next_scope='只可支持新cache设计，须重新投影/估光并返回全生产BVH内部验收，不能直接放行'))
        with (args.output/'README.md').open('a') as f:f.write('\n零优化位置/透射率工程执行完成，实际目检PENDING；无精度PASS，不迁移rho，不导出资产。\n')
    except Exception as exc:
        v.write(args.output/'FAILED.json',dict(status='FAILED_DIAGNOSTIC_EXECUTION',error=str(exc),retained=True,internal_validation_passed=False))
        with (args.output/'README.md').open('a') as f:f.write('\n最终结论FAILED（工程执行异常，全部保留）。\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['plan','run'],required=True)
    for key in ['cache','fit','model','source','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000);p.add_argument('--batch-directions',type=int,default=16);args=p.parse_args()
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():p.error('必须ushas/lumimotion-ushas。')
    for key in ['cache','fit','model','source','output']:setattr(args,key,getattr(args,key).resolve())
    if not args.output.is_relative_to(ROOT/'output'):p.error('所有输出必须在output。')
    if args.action=='plan':plan(args);return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('需要协调并显式指定GPU。')
    run(args)


if __name__=='__main__':main()
