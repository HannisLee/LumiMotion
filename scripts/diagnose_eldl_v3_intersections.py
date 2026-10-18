"""冻结TRAIN射线的局部面片/三角壳诊断；不改生产追踪或训练支持。"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v


def triangle_hits(origin,direction,triangles):
    """双精度Moller-Trumbore前面求交，不声称复刻OptiX边缘舍入。"""
    origin=origin.to(triangles);direction=direction.to(triangles)
    e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
    p=torch.cross(direction.expand_as(e2),e2,dim=-1)
    det=(e1*p).sum(-1)
    inv=torch.where(det>0,1/det,torch.zeros_like(det))
    s=origin-triangles[:,0];u=(s*p).sum(-1)*inv
    q=torch.cross(s,e1,dim=-1);w=(direction*q).sum(-1)*inv
    t=(e2*q).sum(-1)*inv
    ok=(det>0)&(u>=0)&(w>=0)&(u+w<=1)&torch.isfinite(t)
    return t,ok


def selftest():
    tri=torch.tensor([[[0.,0,0],[1,0,0],[0,1,0]]],dtype=torch.float64)
    t,hit=triangle_hits(torch.tensor([.2,.2,1.]),torch.tensor([0.,0.,-1.]),tri)
    assert hit.item() and abs(t.item()-1)<1e-12
    _,hit=triangle_hits(torch.tensor([.2,.2,-1.]),torch.tensor([0.,0.,1.]),tri)
    assert not hit.item()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','fit','model','source','validation','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():raise RuntimeError('需要ushas环境')
    selftest();a.output.mkdir(parents=True,exist_ok=False)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache.parent,a.fit,a.model,a.source,a.validation,a.output])
    data,state,fit,result,hashes=v.load_inputs(a)
    hashes[str(a.validation/'summary.json')]=v.sha(a.validation/'summary.json')
    reference={}
    for key in ('0p5','1','2'):
        path=a.validation/'arrays'/('exact_bias_'+key+'.pt')
        hashes[str(path)]=v.sha(path);reference[key]=torch.load(path,map_location='cpu',weights_only=False)['visibility']
    from scene.gaussian_model import GaussianModel
    from scene.static_dl_photometric import StaticVisibility
    from scripts.probe_dynamic_dl_selfcal import read_config
    from utils.general_utils import build_rotation,build_scaling_rotation
    cfg=read_config(a.model/'cfg_args');pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    pc.load_ply(str(a.model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply'))
    pc.active_sh_degree=min(a.iteration//1000,cfg.sh_degree)
    bias=float(data['shadow_bias']);StaticVisibility(bias=bias,update_interval=1)(pc,state['directions'][:1].cuda(),a.iteration)
    xyz=pc.get_xyz;opacity=pc.get_opacity[:,0];rot=build_rotation(pc.get_rotation)
    # 完全复用生产float32壳顶点构造，再用双精度做独立小射线求交。
    scale3=torch.cat([pc.get_scaling,torch.full_like(pc.get_scaling[:,:1],1e-6)],-1)
    transform=build_scaling_rotation(scale3,pc.get_rotation)
    radius=(2*(pc.get_opacity/pc.alpha_min).log().clamp_min(0)).sqrt()[:,None]
    vertices=radius*(pc.unit_icosahedron_vertices[None]@transform.transpose(-1,-2))+xyz[:,None]
    triangles=vertices[:,pc.unit_icosahedron_faces].reshape(-1,3,3).double()
    faces_per=len(pc.unit_icosahedron_faces)
    normal=rot[:,:,2].double();ru=(rot[:,:,0]/pc.get_scaling[:,0:1]).double();rv=(rot[:,:,1]/pc.get_scaling[:,1:2]).double()
    mean=xyz.double();op=opacity.double();plane_n=data['normal'].cuda().double();plane_o=data['origin'].cuda().double()
    height=(mean-plane_o)@plane_n
    receiver=(height.abs()<.005*data['extent'])&((normal@plane_n).abs()>np.cos(np.deg2rad(10)))
    rays=[]
    for frame in (1,81,149):
        row=v.TRAIN.index(frame);valid=data['valid'][row];delta=(reference['0p5'][row]-reference['1'][row]).abs()
        available=torch.where(valid)[0].tolist()
        sensitive=sorted(available,key=lambda i:(-float(delta[i]),i))[:8]
        used=set(sensitive)
        for ident in sensitive:
            rays.append(dict(frame=frame,row=row,point=ident,kind='sensitive'))
            stable=[i for i in available if i not in used and float(delta[i])<=1e-3]
            other=min(stable,key=lambda i:(float((data['points'][i]-data['points'][ident]).square().sum()),i))
            used.add(other);rays.append(dict(frame=frame,row=row,point=other,kind='nearby_stable'))
    v.write(a.output/'selection.json',dict(rule='每固定fit帧按|V半-V中|降序/ID升序前8，再为每条选最近未用|ΔV|<=.001有效点；仅诊断采样，不改支持',rays=rays))
    results=[]
    for ordinal,ray in enumerate(rays):
        x=data['points'][ray['point']].cuda().double();light=state['directions'][ray['row']].cuda().double()
        # 三角壳所有进入事件；原点可在壳内，背面剔除保留生产语义。
        ts,hit=triangle_hits(x,light,triangles)
        ids=torch.where(hit&(ts>=-bias)&(ts<100+2*bias))[0]
        ids=ids[torch.argsort(ts[ids],stable=True)];gs=ids//faces_per;tabs=ts[ids]
        dot=(normal*light).sum(-1);true_t=((mean-x)*normal).sum(-1)/dot
        true_pos=x+true_t[:,None]*light-mean
        true_uv=torch.stack([(true_pos*ru).sum(-1),(true_pos*rv).sum(-1)],-1)
        alpha_true=(op*torch.exp(-.5*true_uv.square().sum(-1))).clamp_max(.99)
        analytic=(alpha_true>=pc.alpha_min)&torch.isfinite(true_t)&(true_t>=-bias)&(true_t<100+2*bias)
        ellipse_ids=torch.where(analytic)[0]
        ellipse_ids=ellipse_ids[torch.argsort(true_t[ellipse_ids],stable=True)]
        ray_result={**ray,'receiver_gaussian_id':int(data['receiver_ids'][ray['point']]),'x':x.tolist(),'direction':light.tolist(),'scales':{}}
        for key,factor in [('0p5',.5),('1',1.),('2',2.)]:
            b=bias*factor;o=x+b*light
            native_origin=data['points'][ray['point']].cuda()+state['directions'][ray['row']].cuda()*b
            raw=pc.trace(native_origin[None,None].contiguous(),light.float()[None,None].contiguous(),
                         features=torch.zeros(len(xyz),4,device='cuda'),xyz=xyz,scales=pc.get_scaling,rotation=pc.get_rotation,opacity=pc.get_opacity,use_zeros=True)
            native=float(1-raw['alpha'].item());expected=float(reference[key][ray['row'],ray['point']])
            keep=tabs-b>np.finfo(np.float32).eps;g=gs[keep];t= tabs[keep]
            local_dot=dot[g];d=-((o-mean[g])*normal[g]).sum(-1)*local_dot/local_dot.square().clamp_min(1e-6)
            pos=o+d[:,None]*light-mean[g]
            alpha=(op[g]*torch.exp(-.5*((pos*ru[g]).sum(-1).square()+(pos*rv[g]).sum(-1).square()))).clamp_max(.99)
            events=[];trans=1.;negative_weight=0.;seen=set();duplicate=0
            for j in range(len(g)):
                gid=int(g[j]);al=float(alpha[j]);valid_alpha=al>=pc.alpha_min
                weight=trans*al if valid_alpha and trans>.03 else 0.
                if weight:
                    if gid in seen:duplicate+=1
                    seen.add(gid)
                    if float(d[j])<0:negative_weight+=weight
                    trans*=1-al
                events.append(dict(gs_idx=gid,triangle_idx=int(ids[keep][j]),triangle_t_absolute=float(t[j]),
                    plane_t_absolute=float(true_t[gid]),native_plane_distance=float(d[j]),denominator_clamped=bool(local_dot[j].square()<1e-6),
                    alpha=float(alpha[j]),weight=weight,opacity=float(op[gid]),height=float(height[gid]),
                    receiver_group=bool(receiver[gid]),center=mean[gid].tolist(),normal=normal[gid].tolist(),
                    scales=pc.get_scaling[gid].tolist(),between_half_and_center=bool(bias*.5<=true_t[gid]<bias)))
            surrogate=0. if trans<.03 else trans
            ell=1.
            for gid in ellipse_ids[true_t[ellipse_ids]>b+np.finfo(np.float32).eps]:
                if ell<=.03:break
                ell*=1-float(alpha_true[gid])
            ell=0. if ell<.03 else ell
            ray_result['scales'][key]=dict(native_V=native,reference_V=expected,reference_difference=abs(native-expected),
                triangle_surrogate_V=surrogate,triangle_surrogate_error=abs(surrogate-native),analytic_ellipse_V=ell,
                negative_plane_weight=negative_weight,duplicate_contributing_events=duplicate,events=events)
        # 独立解析事件包括native三角壳可能没有收集的面片，便于核对支持差异。
        ray_result['analytic_local_events']=[dict(gs_idx=int(g),t=float(true_t[g]),alpha=float(alpha_true[g]),
             height=float(height[g]),receiver_group=bool(receiver[g]),normal=normal[g].tolist(),scales=pc.get_scaling[g].tolist())
             for g in ellipse_ids if float(true_t[g])<=bias*2]
        v.write(a.output/f'ray_{ordinal:03d}.json',ray_result);results.append(ray_result)
        print(json.dumps(dict(ray=ordinal,frame=ray['frame'],point=ray['point'],kind=ray['kind'],
            values={k:{n:q[n] for n in ('native_V','triangle_surrogate_V','analytic_ellipse_V','negative_plane_weight','duplicate_contributing_events')} for k,q in ray_result['scales'].items()})),flush=True)
    compact=[{**{k:r[k] for k in ('frame','point','kind','receiver_gaussian_id')},'scales':{k:{n:q[n] for n in q if n!='events'} for k,q in r['scales'].items()}} for r in results]
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__));v.check_hashes(hashes)
    summary=dict(status='DIAGNOSTIC_COMPLETED_NO_TRAINING_ACCEPTANCE',gt_read=False,external_test_read=False,
       parameters_refit=False,full_geometry_preserved=True,internal_validation_passed=False,ray_count=len(results),
       original_bias=bias,reference_max_abs_error=max(q['reference_difference'] for r in compact for q in r['scales'].values()),
       triangle_surrogate_max_abs_error=max(q['triangle_surrogate_error'] for r in compact for q in r['scales'].values()),
       limitations='双精度Moller前面求交不复刻OptiX舍入、16命中分页和重复边缘事件；解析椭圆不等价三角壳。没有修改生产native。',
       rays=compact,input_sha256=hashes)
    v.write(a.output/'summary.json',summary)


if __name__=='__main__':main()
