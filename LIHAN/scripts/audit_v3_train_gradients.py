"""固定TRAIN样本的实际加权损失梯度诊断，不执行优化器更新。"""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from LIHAN.scripts.train_v3_gt_plane import objective
from utils.general_utils import build_rotation
from utils.no_gt_guard import install_no_gt_guard

FRAMES=[1,31,59,81,111,149]

def stats(g):
    return dict(l2=float(g.norm()),rms=float(g.square().mean().sqrt()),numel=g.numel(),finite=bool(torch.isfinite(g).all()))

def cosine(a,b):
    denom=a.norm()*b.norm()
    return None if float(denom)<1e-30 else float((a*b).sum()/denom)

def main(a):
    for k in ['source','model','support','output']:setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=True);assert not (a.output/'audit.json').exists()
    install_no_gt_guard([a.source,a.model,a.support,a.output]);torch.set_num_threads(4)
    model=DynamicSurfaceModel.load(a.model,a.iteration);cfg=model.config
    params=[model.pc._xyz,model.pc._rotation,model.pc._scaling,model.pc._opacity]
    deformation=[p for n,p in model.deform.deform.named_parameters() if not n.startswith('gaussian_color')]
    for p in params+deformation:p.requires_grad_(True)
    gate=model.pc.get_binary_feature().detach().reshape(-1)
    groups={'static_gate_lt01':gate<.1,'dynamic_gate_gt09':gate>.9,'transition':(gate>=.1)&(gate<=.9)}
    d=np.load(a.support);idx=torch.tensor(d['indices'],device='cuda');normal=F.normalize(torch.tensor(d['normal'],device='cuda',dtype=torch.float32),dim=0);origin=torch.tensor(d['origin'],device='cuda',dtype=torch.float32)
    rows=[];cameras=load_train_cameras(a.source);assert set(FRAMES)<=set(int(c.image_name_train_light.split('_')[-1]) for c in cameras)
    for camera in cameras:
        frame=int(camera.image_name_train_light.split('_')[-1])
        if frame not in FRAMES:continue
        camera.load2device();package=model.render(camera,direct_weight=.5,protect_opacity=True)
        _,terms,_=objective(package,camera,'historical',cfg['normal_weight']);current=package['surface_components']['geometry']
        axes=build_rotation(current.get_rotation)[idx,:,2]
        terms['plane_normal']=(1-(axes*normal).sum(-1).square()).clamp_min(0).mean()
        dist=((current.get_xyz[idx]-origin)*normal).sum(-1)
        terms['plane_position']=F.smooth_l1_loss(dist/cfg['plane_tau'],torch.zeros_like(dist),beta=1.)
        weights={'rgb':1.,'alpha':.1,'normal':cfg['normal_weight'],'distortion':1000.,'plane_normal':cfg['plane_normal_weight'],'plane_position':cfg['plane_position_weight']}
        vectors={};record=dict(frame=frame,losses={},gradient_stats={},cosines={})
        for name,w in weights.items():
            value=terms[name]*w;record['losses'][name]=dict(raw=float(terms[name].detach()),weight=w,weighted=float(value.detach()))
            grads=torch.autograd.grad(value,params+deformation,retain_graph=True,allow_unused=True)
            per={}
            for field,p,g in zip(['xyz','rotation','scale','opacity'],params,grads[:4]):
                g=torch.zeros_like(p) if g is None else g
                for group,mask in groups.items():
                    if int(mask.sum()):per[field+'/'+group]=g[mask].detach().reshape(-1).cpu()
            per['deform/shared']=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).cpu() for p,g in zip(deformation,grads[4:])])
            vectors[name]=per;record['gradient_stats'][name]={k:stats(v) for k,v in per.items()}
        for first,second in [('rgb','normal'),('rgb','distortion'),('rgb','plane_normal'),('normal','plane_normal'),('rgb','plane_position')]:
            record['cosines'][first+'__'+second]={key:cosine(v,vectors[second][key]) for key,v in vectors[first].items()}
        rows.append(record);print(json.dumps(dict(frame=frame,losses=record['losses']),ensure_ascii=False),flush=True)
        camera.load2device('cpu');del package,terms,current,axes,dist,vectors,value,grads;torch.cuda.empty_cache()
    result=dict(gt_geometry_read=False,heldout_read=False,optimizer_updates=0,model=str(a.model),iteration=a.iteration,frames=FRAMES,group_counts={k:int(v.sum()) for k,v in groups.items()},definitions='全TRAIN帧损失原始归一化不改；使用真实训练权重后的梯度。L2为参数子向量欧氏范数，RMS=L2/sqrt(numel)，cos为同一参数子空间点积除范数积，零梯度记null。不同参数单位/Adam预条件不同，禁止跨参数直接比较L2；deform为共享参数，不能分配到静态或动态。点组按继承gate，非GT人体语义。未据梯度单独取消photo约束。',per_frame=rows)
    (a.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','model','support','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True);main(p.parse_args())
