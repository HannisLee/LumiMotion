"""固定失败v1的遮挡来源诊断；同BVH/ID，仅opacity掩膜，不重新拟合。"""
import argparse
import hashlib
import json
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

DOC=ROOT/'LIHAN/DOC/报告/1019-06-ELDL-V3遮挡来源分解预登记.md'
MODES=['full','receiver_only','caster_only']


def describe(tensor):
    a=tensor.detach().cpu().double().flatten();a=a[torch.isfinite(a)]
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
        quantiles={str(q):float(torch.quantile(a,q)) for q in [0.,.01,.1,.5,.9,.99,1.]} if len(a) else {})


@torch.no_grad()
def trace(pc,points,directions,opacity,bias,batch):
    values=[]
    for start in range(0,len(directions),batch):
        light=directions[start:start+batch]
        rays_d=light[None].expand(len(points),-1,-1).contiguous()
        package=pc.trace((points[:,None]+rays_d*bias).contiguous(),rays_d,
            features=torch.zeros(len(pc.get_xyz),4,device='cuda'),xyz=pc.get_xyz,
            scales=pc.get_scaling,rotation=pc.get_rotation,opacity=opacity,use_zeros=True)
        visibility=(1-package['alpha']).clamp(0,1).T
        if not torch.isfinite(visibility).all():raise ValueError('mask透射率有非有限值。')
        values.append(visibility.cpu())
    return torch.cat(values)


def plan(args):
    args.output.mkdir(parents=True,exist_ok=False)
    for sub in ['logs','arrays','figures']:(args.output/sub).mkdir()
    config=dict(status='PLANNED_DIAGNOSTIC_NO_ACCEPTANCE_THRESHOLD',
        paths={k:str(getattr(args,k)) for k in ['cache','fit','model','source','reference','output']},
        iteration=args.iteration,code_sha256=v.sha(Path(__file__)),validator_sha256=v.sha(Path(v.__file__)),
        protocol_sha256=v.sha(DOC),bias_scales=v.SCALES,modes=MODES,
        receiver=dict(absolute_plane_distance_lt_extent=.005,absolute_normal_dot_gt=math.cos(math.radians(10)),
            strict_boundaries=True,invalid_normals='caster',shoe_contact_confusion='UNKNOWN_NOT_ZERO'),
        internal_validation_passed=False,training_asset_exported=False)
    v.write(args.output/'plan.json',config);(args.output/DOC.name).write_bytes(DOC.read_bytes())
    cmd=[str(v.PYTHON),str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),'scripts.diagnose_eldl_v3_occlusion','--action','run']
    for key in ['cache','fit','model','source','reference','output']:cmd+=['--'+key,str(getattr(args,key))]
    cmd+=['--iteration',str(args.iteration)]
    v.write(args.output/'command.json',dict(argv=cmd,shell=shlex.join(cmd),gpu='显式CUDA_VISIBLE_DEVICES，与主任务协调'))
    (args.output/'README.md').write_text('# 遮挡来源固定分解\n\n2026-10-19，ushas。无GT/外部test，固定v1 selected3，禁止重拟合或放行训练资产。\n\n完整命令（执行前固定GPU环境）：\n\n```bash\n'+shlex.join(cmd)+'\n```\n\nsource：`'+str(args.source)+'`；model：`'+str(args.model)+'`；iteration：'+str(args.iteration)+'。输出arrays/figures；日志logs；统计summary.json与partition.json。\n\n平面图各行1观察、2预测、3原残差、4删失残差、5原valid、6逐通道饱和；残差色阶0～0.2，灰为无效。无模型四类视频，RGB仅平面诊断、Alpha仅缓存支持、Normal仅预测法线分区，语义分离不适用。\n\n代表fit1/81/149，held61/71/79。实际目检PENDING，最终结论PENDING；没有科学验收门槛，internal_validation_passed=false。\n')


@torch.no_grad()
def run(args):
    config=v.read(args.output/'plan.json')
    if config['code_sha256']!=v.sha(Path(__file__)) or config['validator_sha256']!=v.sha(Path(v.__file__)):
        raise ValueError('诊断/验证器源码与计划不同。')
    if config['paths']!={k:str(getattr(args,k)) for k in ['cache','fit','model','source','reference','output']}:
        raise ValueError('路径与固定计划不同。')
    if (args.output/'RUN_STARTED.json').exists():raise FileExistsError('禁止覆盖执行结果。')
    v.write(args.output/'RUN_STARTED.json',dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),internal_validation_passed=False))
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([args.cache.parent,args.fit,args.model,args.source,args.reference,args.output])
    try:
        data,state,summary,result,hashes=v.load_inputs(args)
        if summary['selected_init']!=3:raise ValueError('本诊断只登记v1 selected3。')
        reference=v.read(args.reference/'summary.json')
        if reference['internal_validation_passed'] or not reference['exact_bvh']:
            raise ValueError('需要原未放行的完整精确BVH结果。')
        for path,expected in reference['input_sha256'].items():
            path=Path(path)
            if path.suffix in ('.pt','.ply','.pth') and str(path) in hashes and hashes[str(path)]!=expected:
                raise ValueError('诊断与精确基准输入不同。')
        import diff_surfel_rasterization as backend
        if getattr(backend,'provenance',{}).get('method')!='repository_lumimotion_cuda_rasterizer':raise ValueError('需要原生bootstrap。')
        from scene.gaussian_model import GaussianModel
        from scene.static_dl_photometric import StaticVisibility
        from scripts.probe_dynamic_dl_selfcal import read_config
        from utils.general_utils import build_rotation
        import surfel_tracer.raytracer as tracer_wrapper
        wrapper=Path(tracer_wrapper.__file__).resolve()
        repo_wrapper=ROOT/'submodules/surfel_tracer/surfel_tracer/raytracer.py'
        if v.sha(wrapper)!=v.sha(repo_wrapper):raise ValueError('安装BVH wrapper与核查仓库版本不同。')
        for file in [wrapper,repo_wrapper,ROOT/'submodules/surfel_tracer/src/optix/gaussiantrace_forward.cu',
                     ROOT/'submodules/surfel_tracer/src/optix/gaussiantrace_intersection_test.cu',Path(__file__).resolve()]:
            hashes[str(file)]=v.sha(file)
        cfg=read_config(args.model/'cfg_args')
        pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
        pc.load_ply(str(args.model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'))
        pc.active_sh_degree=min(args.iteration//1000,cfg.sh_degree)
        original_opacity=pc.get_opacity.clone();points=data['points'].cuda();directions=state['directions'].cuda()
        origin=data['origin'].cuda();plane_normal=F.normalize(data['normal'].cuda(),dim=0)
        normals=build_rotation(pc.get_rotation)[:,:,2]
        norm=normals.norm(dim=-1);normal_valid=torch.isfinite(normals).all(-1)&(norm>.9)&(norm<1.1)
        normal_unit=F.normalize(torch.nan_to_num(normals),dim=-1)
        dot=(normal_unit@plane_normal).abs();signed=(pc.get_xyz-origin)@plane_normal
        threshold=.005*float(data['extent']);cosine=math.cos(math.radians(10))
        receiver=(signed.abs()<threshold)&normal_valid&(dot>cosine)
        caster=~receiver
        ids=torch.arange(len(receiver),device='cuda')
        torch.save(dict(ids=ids.cpu(),receiver=receiver.cpu(),caster=caster.cpu(),signed_height=signed.cpu(),
            absolute_normal_dot=dot.cpu(),normal_world=normal_unit.cpu(),opacity=original_opacity.cpu(),
            plane_normal=plane_normal.cpu(),origin=origin.cpu()),args.output/'receiver_partition.pt')
        partition=dict(total=len(receiver),extent=float(data['extent']),plane_distance_threshold=threshold,
            receiver_count=int(receiver.sum()),caster_count=int(caster.sum()),receiver_fraction=float(receiver.float().mean()),
            invalid_normals=int((~normal_valid).sum()),ambiguous_distance_band_count=int(((signed.abs()>=threshold*.8)&(signed.abs()<=threshold*1.2)).sum()),
            ambiguous_angle_8_to_12deg_count=int(((dot>=math.cos(math.radians(12)))&(dot<=math.cos(math.radians(8)))).sum()),
            shoe_contact_confusion='UNKNOWN；分区没有GT语义，鞋底/接触平行面可能混入receiver，不声称零',groups={})
        for mode,mask in [('receiver',receiver),('caster',caster)]:
            partition['groups'][mode]=dict(signed_height=describe(signed[mask]),absolute_height=describe(signed[mask].abs()),
                normal_abs_dot=describe(dot[mask]),normal_axis_angle_deg=describe(torch.rad2deg(torch.acos(dot[mask].clamp(0,1)))),
                opacity=describe(original_opacity[mask]))
        v.write(args.output/'partition.json',partition)
        builder=StaticVisibility(bias=float(data['shadow_bias']),update_interval=1);builder(pc,directions[:1],args.iteration)
        gs_ids=pc.gaussian_tracer.gs_idxs.clone()
        zero=trace(pc,points,directions[:3],torch.zeros_like(original_opacity),float(data['shadow_bias']),args.batch_directions)
        zero_error=float((zero-1).abs().max())
        if zero_error>1e-7:raise ValueError('opacity全零没有给出单位透射率，mask语义不成立。')
        outputs={};metrics={};identity={};array_hashes={}
        for mode in MODES:
            opacity=original_opacity if mode=='full' else original_opacity*(receiver if mode=='receiver_only' else caster)[:,None]
            mode_fig=args.output/'figures'/mode;mode_fig.mkdir()
            outputs[mode]={};metrics[mode]={}
            for scale in v.SCALES:
                key=v.scale_key(scale)
                vis=trace(pc,points,directions,opacity,float(data['shadow_bias'])*scale,args.batch_directions)
                if mode=='full':
                    refpath=args.reference/'arrays'/f'exact_bias_{key}.pt'
                    if v.sha(refpath)!=reference['output_sha256'][str(refpath)]:raise ValueError('精确基准数组SHA变化。')
                    old=torch.load(refpath,map_location='cpu',weights_only=False)
                    identity[key]=float((vis-old['visibility']).abs().max())
                    if identity[key]>2e-6:raise ValueError('full与既有精确BVH不一致，停止归因。')
                    hashes[str(refpath)]=v.sha(refpath)
                c=(state['directions']@data['normal']).clamp_min(0)
                prediction=state['albedo'][None]*(state['environment'][None,None]/math.pi+
                    (result['direct_intensity']/math.pi)*vis[...,None]*c[:,None,None])
                metric,delta,_=v.score(prediction,data);metrics[mode][key]=metric
                path=args.output/'arrays'/f'{mode}_bias_{key}.pt'
                torch.save(dict(visibility=vis,prediction=prediction,censored_residual=delta,mode=mode,bias_scale=scale),path)
                array_hashes[str(path)]=v.sha(path);outputs[mode][key]=vis
                v.plane_figures(mode_fig,key,prediction,delta,data)
                print(json.dumps(dict(mode=mode,bias=scale,fit=metric['fit']['frame_mean'],held=metric['held']['frame_mean']),ensure_ascii=False),flush=True)
        if not torch.equal(gs_ids,pc.gaussian_tracer.gs_idxs) or not torch.equal(original_opacity,pc.get_opacity):
            raise ValueError('原BVH ID或opacity被修改。')
        sensitivity={};valid=data['valid'];fit=data['fit']
        for mode in MODES:
            sensitivity[mode]={}
            for key,vis in outputs[mode].items():
                diff=(vis-outputs[mode]['1']).abs();item={}
                for name,frame_mask in [('fit',fit),('held',~fit)]:
                    mask=valid&frame_mask[:,None]
                    item[name]=dict(visibility=describe(vis[mask]),absolute_change_from_center=describe(diff[mask]),
                        visibility_lt_point1_fraction=float((vis[mask]<.1).float().mean()),
                        visibility_gt_point9_fraction=float((vis[mask]>.9).float().mean()))
                sensitivity[mode][key]=item
        product={}
        for key in outputs['full']:
            error=(outputs['full'][key]-outputs['receiver_only'][key]*outputs['caster_only'][key]).abs()
            product[key]=describe(error[valid])
        v.check_hashes(hashes)
        v.write(args.output/'summary.json',dict(status='DIAGNOSTIC_EXECUTED_NO_SCIENTIFIC_ACCEPTANCE',
            internal_validation_passed=False,training_asset_exported=False,gt_read=False,external_test_read=False,
            parameters_refit=False,selected_init=3,full_identity_max_abs=identity,zero_opacity_max_abs_error=zero_error,
            gaussian_ids_preserved=True,partition=partition,metrics=metrics,sensitivity=sensitivity,
            product_residual=product,product_note='透射率提前终止/饱和会破坏严格乘积分解，不能据此当加性贡献。',
            source_sha256=hashes,array_sha256=array_hashes,manual_review='PENDING',
            receiver_exclusion_scope='仅代理差异诊断；将来估光排receiver仍须回到生产全BVH验收，不可直接放行'))
        with (args.output/'README.md').open('a') as f:f.write('\n工程执行完成；full重现与全零opacity语义通过，科学验收不适用。主任务目检PENDING，internal_validation_passed=false，原v1失败结论不变。\n')
    except Exception as exc:
        v.write(args.output/'FAILED.json',dict(status='FAILED_DIAGNOSTIC_EXECUTION',error=str(exc),retained=True,internal_validation_passed=False))
        with (args.output/'README.md').open('a') as f:f.write('\n最终结论：FAILED（工程诊断异常，完整保留）。\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',required=True,choices=['plan','run'])
    for key in ['cache','fit','model','source','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000);p.add_argument('--batch-directions',type=int,default=16)
    args=p.parse_args()
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():p.error('必须ushas/lumimotion-ushas。')
    for key in ['cache','fit','model','source','reference','output']:setattr(args,key,getattr(args,key).resolve())
    if not args.output.is_relative_to(ROOT/'output'):p.error('所有输出必须在output。')
    if args.action=='plan':plan(args);return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('需要显式协调GPU。')
    run(args)


if __name__=='__main__':main()
