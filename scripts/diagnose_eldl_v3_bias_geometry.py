"""几何推导bias候选的固定支持V/能量/接触风险诊断，不优化、不放行。"""
import argparse
import math
import os
from pathlib import Path
import shlex
import socket
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts.diagnose_eldl_v3_occlusion import trace,describe
DOC=ROOT/'LIHAN/DOC/报告/1019-14-ELDL-V3几何尺度bias候选零优化预登记.md'


def plan(args):
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['logs','arrays','figures']:(args.output/name).mkdir()
    record=dict(status='PLANNED_DIAGNOSTIC_ONLY',paths={k:str(getattr(args,k)) for k in ['cache','plane_cache','fit','source','model','reference','output']},
        iteration=args.iteration,code_sha256=v.sha(Path(__file__)),validator_sha256=v.sha(Path(v.__file__)),
        helper_sha256=v.sha(ROOT/'scripts/diagnose_eldl_v3_occlusion.py'),protocol_sha256=v.sha(DOC),
        center_bias_extent_fraction=.01,scales=v.SCALES,selected_by='geometry_band_width_only_no_held_selection',
        internal_validation_passed=False,scientific_acceptance_thresholds=None)
    v.write(args.output/'plan.json',record);(args.output/DOC.name).write_bytes(DOC.read_bytes())
    cmd=[str(v.PYTHON),str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),'scripts.diagnose_eldl_v3_bias_geometry','--action','run']
    for key in ['cache','plane_cache','fit','source','model','reference','output']:cmd+=['--'+key.replace('_','-'),str(getattr(args,key))]
    cmd+=['--iteration',str(args.iteration)]
    v.write(args.output/'command.json',dict(argv=cmd,shell=shlex.join(cmd)))
    (args.output/'README.md').write_text('# 几何尺度bias候选零优化诊断\n\n2026-10-19，ushas。候选0.01extent预先由2h确定，不按held选择；不优化、不放宽原门槛、不导出。\n\n完整命令（协调后显式GPU环境）：\n\n```bash\n'+shlex.join(cmd)+'\n```\n\nsource：`'+str(args.source)+'`；model：`'+str(args.model)+'`；iteration：'+str(args.iteration)+'。输出arrays/figures，日志logs，统计summary.json。固定fit1/81/149；仅平面V/预测诊断，无模型四类视频；RGB/Alpha/Normal/分离的完整几何验收不适用。\n\n原窄带h只是几何近面定义；沿L偏移的法向抬升需看dot，不能保证半档跨全层。接触风险仅预测几何代理，无鞋GT语义。\n\n最终结论PENDING，internal_validation_passed=false。\n')


@torch.no_grad()
def run(args):
    record=v.read(args.output/'plan.json')
    for key,path in [('code_sha256',Path(__file__)),('validator_sha256',Path(v.__file__)),('helper_sha256',ROOT/'scripts/diagnose_eldl_v3_occlusion.py')]:
        if v.sha(path)!=record[key]:raise ValueError('源码与计划不同。')
    if record['paths']!={k:str(getattr(args,k)) for k in ['cache','plane_cache','fit','source','model','reference','output']}:raise ValueError('路径与计划不同。')
    if (args.output/'RUN_STARTED.json').exists():raise FileExistsError('禁止覆盖已有执行。')
    v.write(args.output/'RUN_STARTED.json',dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([args.cache.parent,args.plane_cache,args.plane_cache.parent/'manifest.json',args.fit,args.source,args.model,args.reference,args.output])
    try:
        data,state,summary,result,hashes=v.load_inputs(args)
        plane_meta=v.read(args.plane_cache.parent/'manifest.json')
        if plane_meta['gt_read'] or v.sha(args.plane_cache)!=plane_meta['cache_sha256']:
            raise ValueError('旧原格点几何参考不是已绑定TRAIN缓存。')
        plane_data=torch.load(args.plane_cache,map_location='cpu',weights_only=False)
        if not torch.equal(plane_data['basis'],data['basis']) or plane_data['extent']!=data['extent']:
            raise ValueError('旧格点几何与实际receiver缓存不是相同平面。')
        hashes[str(args.plane_cache)]=v.sha(args.plane_cache)
        hashes[str(args.plane_cache.parent/'manifest.json')]=v.sha(args.plane_cache.parent/'manifest.json')
        if 'receiver_normals' not in data or summary['selected_init']!=3:raise ValueError('本次仅实际surfel/selected3。')
        reference=v.read(args.reference/'summary.json')
        from scene.gaussian_model import GaussianModel
        from scene.static_dl_photometric import StaticVisibility
        from scripts.probe_dynamic_dl_selfcal import read_config
        from utils.general_utils import build_rotation
        from scipy.spatial import cKDTree
        import diff_surfel_rasterization as backend
        if getattr(backend,'provenance',{}).get('method')!='repository_lumimotion_cuda_rasterizer':raise ValueError('需要原生bootstrap。')
        cfg=read_config(args.model/'cfg_args');pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
        pc.load_ply(str(args.model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'));pc.active_sh_degree=min(args.iteration//1000,cfg.sh_degree)
        opacity=pc.get_opacity.clone();points=data['points'].cuda();directions=state['directions'].cuda()
        extent=float(data['extent']);half_width=.005*extent;new_center=.01*extent;old_center=float(data['shadow_bias'])
        builder=StaticVisibility(bias=old_center,update_interval=1);builder(pc,directions[:1],args.iteration)
        normal=data['normal'].cuda();origin=data['origin'].cuda();basis=data['basis'].cuda()
        height=(pc.get_xyz-origin)@normal;all_normals=build_rotation(pc.get_rotation)[:,:,2]
        receiver=(height.abs()<half_width)&((all_normals@normal).abs()>math.cos(math.radians(10)))&torch.isfinite(all_normals).all(-1)
        low_caster=(~receiver)&(height.abs()<=4*half_width)&(opacity[:,0]>=pc.alpha_min)
        low_ids=torch.where(low_caster)[0]
        receiver_uv=((points-origin)@basis.T).cpu().numpy();low_uv=((pc.get_xyz[low_ids]-origin)@basis.T).cpu().numpy()
        # 网格步长来自缓存格点生成的xy边界；真实点已移动，不能用相邻真实点距离替代原规则网格步长。
        original_uv=((plane_data['points'].cuda()-origin)@basis.T).cpu().numpy()
        radius=float(max(np.linalg.norm(original_uv[1]-original_uv[0]),np.linalg.norm(original_uv[64]-original_uv[0])))
        if len(low_ids):distance,nearest=cKDTree(low_uv).query(receiver_uv,k=1,workers=4);near=torch.tensor(distance<=radius)
        else:distance=np.full(len(points),np.inf);nearest=np.zeros(len(points),int);near=torch.zeros(len(points),dtype=torch.bool)
        torch.save(dict(low_caster_ids=low_ids.cpu(),receiver_near_low_caster=near,nearest_low_caster_distance=torch.tensor(distance),
            nearest_low_caster_id=low_ids.cpu()[torch.tensor(nearest)] if len(low_ids) else torch.full((len(points),),-1),
            low_caster_height=height[low_ids].cpu()),args.output/'contact_proxy.pt')
        valid=data['valid'];cosine=v.receiver_cosines(data,state['directions'])
        values={'old':{},'candidate':{}};statistics={};metrics={}
        for scale in v.SCALES:
            key=v.scale_key(scale);old_path=args.reference/'arrays'/f'exact_bias_{key}.pt'
            if v.sha(old_path)!=reference['output_sha256'][str(old_path)]:raise ValueError('旧exact SHA变化。')
            stored=torch.load(old_path,map_location='cpu',weights_only=False)
            if not torch.equal(stored['valid'],valid) or not torch.equal(stored['directions'],state['directions']):raise ValueError('旧exact不是同valid/方向。')
            values['old'][key]=stored['visibility'];hashes[str(old_path)]=v.sha(old_path)
            vis=trace(pc,points,directions,opacity,new_center*scale,args.batch_directions);values['candidate'][key]=vis
            prediction=state['albedo'][None]*(state['environment'][None,None]/math.pi+(result['direct_intensity']/math.pi)*vis[...,None]*cosine[...,None])
            metric,delta,_=v.score(prediction,data);metrics[key]=metric
            torch.save(dict(visibility=vis,prediction=prediction,censored_residual=delta,valid=valid,actual_bias=new_center*scale),args.output/'arrays'/f'candidate_bias_{key}.pt')
            v.plane_figures(args.output/'figures',key,prediction,delta,data)
            print(f'候选bias={new_center*scale:.10f}，固定支持V诊断完成',flush=True)
        for kind in values:
            statistics[kind]={}
            for key,vis in values[kind].items():
                dv=vis-values[kind]['1'];energy=state['albedo'][None]*cosine[...,None]*dv[...,None]*(result['direct_intensity']/math.pi)
                item={}
                for split,frame_mask in [('fit',data['fit']),('held',~data['fit'])]:
                    mask=valid&frame_mask[:,None]
                    item[split]=dict(visibility=describe(vis[mask]),absolute_V_change=describe(dv[mask].abs()),
                        rho_cos_deltaV_rms=float(torch.sqrt(energy[mask].double().square().mean())),
                        rho_cos_deltaV_absolute=describe(energy[mask].abs()),valid_points=int(mask.sum()))
                statistics[kind][key]=item
        contact={}
        old=values['old']['1'];new=values['candidate']['1']
        for name,pointmask in [('near_low_caster',near),('other_receivers',~near),('all_receivers',torch.ones_like(near))]:
            mask=valid&pointmask[None];count=int(mask.sum())
            contact[name]=dict(point_count=int(pointmask.sum()),observations=count,
                old_mean_V=float(old[mask].mean()) if count else None,candidate_mean_V=float(new[mask].mean()) if count else None,
                old_dark_to_candidate_open_fraction=float(((old[mask]<.5)&(new[mask]>.9)).float().mean()) if count else None,
                center_V_change=describe((new-old)[mask]))
        lifts={}
        plane_dot=state['directions']@data['normal'];local_dot=state['directions']@data['receiver_normals'].T
        for scale in v.SCALES:
            key=v.scale_key(scale);bias=new_center*scale
            lifts[key]=dict(actual_bias=bias,plane_normal_lift=describe((bias*plane_dot[:,None]).expand_as(valid)[valid]),
                receiver_normal_lift=describe((bias*local_dot)[valid]),
                plane_lift_below_full_band_width_fraction=float(((bias*plane_dot[:,None]).expand_as(valid)[valid]<2*half_width).float().mean()))
        if not torch.equal(opacity,pc.get_opacity):raise ValueError('全生产opacity被改动。')
        hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__));v.check_hashes(hashes)
        report=dict(status='DIAGNOSTIC_EXECUTED_NO_ACCEPTANCE',internal_validation_passed=False,gt_read=False,external_test_read=False,
            parameters_refit=False,bias_selected_by_held=False,old_center_bias=old_center,candidate_center_bias=new_center,band_half_width=half_width,
            candidate_source='2倍原几何近面窄带半宽，非held扫描',statistics=statistics,secondary_fixed_parameter_censored_metrics=metrics,
            contact_proxy=dict(definition='低高度caster abs(height)<=4h，近邻xy<=原网格最大轴步长代理',
                radius=radius,low_caster_count=len(low_ids),low_caster_heights=describe(height[low_ids]),subsets=contact,
                shoe_semantics='UNKNOWN_NOT_GT_SHOES',limitations='近邻只是接触风险代理，可包含平台边缘；不能把阴影变化称主体/鞋标签正确'),
            normal_lifts=lifts,input_sha256=hashes,manual_review='PENDING',
            next_step='是否重新拟合由主任务基于几何/fit阴影证据决定；仍须原25%/15°生产全BVH验收')
        v.write(args.output/'summary.json',report)
        with (args.output/'README.md').open('a') as f:f.write('\n工程诊断执行完成，统计summary.json；图像实际目检PENDING，没有精度PASS，不导出。\n')
    except Exception as exc:
        v.write(args.output/'FAILED.json',dict(status='FAILED_DIAGNOSTIC_EXECUTION',error=str(exc),retained=True,internal_validation_passed=False))
        with (args.output/'README.md').open('a') as f:f.write('\n最终结论FAILED（工程异常，全部保留）。\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['plan','run'],required=True)
    for key in ['cache','plane_cache','fit','source','model','reference','output']:p.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000);p.add_argument('--batch-directions',type=int,default=16);args=p.parse_args()
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():p.error('必须ushas/lumimotion-ushas。')
    for key in ['cache','plane_cache','fit','source','model','reference','output']:setattr(args,key,getattr(args,key).resolve())
    if not args.output.is_relative_to(ROOT/'output'):p.error('输出必须位于output。')
    if args.action=='plan':plan(args);return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('需协调并显式指定GPU。')
    run(args)


if __name__=='__main__':main()
