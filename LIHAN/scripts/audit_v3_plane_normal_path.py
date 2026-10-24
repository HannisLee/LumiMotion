"""TRAIN核对预测平面轴、Gaussian旋转及GS光栅法线的一致性和分组可见误差。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from gaussian_renderer import render
from utils.general_utils import build_rotation
from utils.no_gt_guard import install_no_gt_guard

@torch.no_grad()
def main(args):
    for key in ['source','model','narrow_support','broad_support','stages','output']:setattr(args,key,getattr(args,key).resolve())
    out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    install_no_gt_guard([args.source,args.model,args.narrow_support,args.broad_support,args.stages,out])
    torch.set_num_threads(4);model=DynamicSurfaceModel.load(args.model,args.iteration);cameras=load_train_cameras(args.source)
    data=np.load(args.broad_support);n=torch.tensor(data['normal'],device='cuda',dtype=torch.float32);n=F.normalize(n,dim=0);origin=torch.tensor(data['origin'],device='cuda',dtype=torch.float32)
    narrow=np.load(args.narrow_support)['indices'];broad=data['indices'];stages=np.load(args.stages);excluded=np.setdiff1d(stages['after_motion'],stages['after_clearance'])
    gate=model.pc.get_binary_feature().reshape(-1);sets={'original_narrow':narrow,'new_broad':np.setdiff1d(broad,narrow),'near_dynamic_excluded':excluded}
    masks={}
    for name,indices in sets.items():
        m=torch.zeros(len(gate),device='cuda',dtype=torch.bool);m[torch.tensor(indices,device='cuda')]=True;masks[name]=m
    masks['other_static']=(gate<.1)&~torch.stack(list(masks.values())).any(0)
    totals={key:np.zeros(3) for key in masks};alpha_mass=0.;max_normal_difference=0.;pixel_angle_sum=0.;pixel_count=0;rows=[]
    for camera in cameras:
        camera.load2device();geometry=model.effective(camera);axes=build_rotation(geometry.get_rotation)[:,:,2]
        sign=torch.where((axes*(camera.camera_center-geometry.get_xyz)).sum(1,keepdim=True)>=0,1.,-1.)
        world=axes*sign;zero=model.pc.get_xyz.new_zeros(3)
        package=render(camera,geometry,model.pipe,zero,0.,0.,0.,override_color=world)
        difference=float((package['render']-package['rend_normal']).abs().max());max_normal_difference=max(max_normal_difference,difference)
        cosine=(axes*n).sum(1).abs().clamp(0,1);angle=torch.rad2deg(torch.atan2(torch.cross(axes,n[None].expand_as(axes),dim=-1).norm(dim=1),cosine))
        distance=((geometry.get_xyz-origin)*n).sum(1).abs();foreground=camera.gt_alpha_mask[0].cuda()>.5;alpha=package['rend_alpha'][0]
        alpha_mass+=float(alpha[foreground].sum());static_mass=torch.zeros_like(alpha);local={}
        for name,mask in masks.items():
            m=mask.float();color=torch.stack([m,m*angle,m*distance],dim=-1)
            value=render(camera,geometry,model.pipe,zero,0.,0.,0.,override_color=color)['render']
            sums=value[:,foreground].sum(1).cpu().numpy();totals[name]+=sums;static_mass+=value[0]
            local[name]=dict(mass=float(sums[0]),angle_deg=float(sums[1]/max(sums[0],1e-8)))
        static_union=torch.stack(list(masks.values())).any(0).float()
        static_mass=render(camera,geometry,model.pipe,zero,0.,0.,0.,override_color=static_union[:,None].expand(-1,3))['render'][0]
        static_dominated=foreground&(static_mass/alpha.clamp_min(1e-8)>.9)
        normal=F.normalize(package['rend_normal'],dim=0);dot=(normal*n[:,None,None]).sum(0).abs().clamp(0,1)
        pixel_angle=torch.rad2deg(torch.acos(dot));pixel_angle_sum+=float(pixel_angle[static_dominated].sum());pixel_count+=int(static_dominated.sum())
        rows.append(dict(frame=int(camera.image_name_train_light.split('_')[-1]),normal_render_max_abs_difference=difference,groups=local));camera.load2device('cpu')
    result=dict(gt_geometry_read=False,heldout_read=False,training_updates=0,model=str(args.model),iteration=args.iteration,
        point_counts={k:int(v.sum()) for k,v in masks.items()},
        group_overlap_note='original_narrow与near_dynamic_excluded重叠67点，组独立解释不得相加；static_dominated用独立union render消除重复',max_normal_render_vs_explicit_world_axis_difference=max_normal_difference,
        normal_axis_renderer_consistent=max_normal_difference<2e-5,
        static_dominated_rendered_normal_to_predicted_plane_mean_deg=pixel_angle_sum/pixel_count,
        groups={k:dict(alpha_mass_fraction=float(v[0]/alpha_mass),visible_point_axis_to_plane_mean_deg=float(v[1]/v[0]),visible_point_plane_distance_mean=float(v[2]/v[0])) for k,v in totals.items()},
        per_frame=rows,interpretation='误差目标是父预测拟合面而非GT；仅验证轴/渲染实现与剩余预测误差分布，不作为GOAL指标')
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_frame'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','narrow-support','broad-support','stages','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True);main(p.parse_args())
