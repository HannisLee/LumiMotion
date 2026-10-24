"""用TRAIN预测前表面/尺度一致性核验近动态静态点，禁止将投影视锥存在冒充可见。"""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from gaussian_renderer import render
from utils.general_utils import build_rotation
from utils.no_gt_guard import install_no_gt_guard


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main(args):
    for key in vars(args):setattr(args,key,getattr(args,key).resolve())
    out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    broad_audit=json.loads(args.broad_audit.read_text());assert broad_audit['gt_read'] is False and broad_audit['heldout_read'] is False
    assert sha(args.support)==broad_audit['support_sha256']
    install_no_gt_guard([args.source,args.model,args.parent,args.support,args.stages,args.broad_audit,out])
    torch.set_num_threads(4);model=DynamicSurfaceModel.load(args.model,0);cameras=load_train_cameras(args.source)
    manifest=json.loads((args.source/'no_gt_manifest.json').read_text());frames=[int(c.image_name_train_light.split('_')[-1]) for c in cameras];assert frames==manifest['train_frames']
    stages=np.load(args.stages);data=np.load(args.support);extent=float(data['extent'])
    ids_np=np.setdiff1d(stages['after_motion'],stages['after_clearance']);assert len(ids_np)==1035
    ids=torch.tensor(ids_np,device='cuda');size=len(ids);count=np.zeros(size,dtype=int);early=count.copy();late=count.copy();first_dir=np.zeros((size,3));min_cos=np.ones(size);error_sum=np.zeros(size)
    stages_sum={k:0 for k in ['in_frame','alpha_interior','relative_depth','scale_depth','normal_facing','unique_visible']};rows=[]
    for camera in cameras:
        frame=int(camera.image_name_train_light.split('_')[-1]);camera.load2device();geometry=model.effective(camera)
        package=render(camera,geometry,model.pipe,model.pc.get_xyz.new_zeros(3),0.,0.,0.,override_color=torch.zeros_like(model.pc.get_xyz))
        xyz=geometry.get_xyz[ids];hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],1);clip=hom@camera.full_proj_transform
        u=((clip[:,0]/clip[:,3].clamp_min(1e-8)+1)*camera.image_width/2-.5).round().long();v=((clip[:,1]/clip[:,3].clamp_min(1e-8)+1)*camera.image_height/2-.5).round().long()
        valid=(clip[:,3]>0)&(u>=1)&(u<camera.image_width-1)&(v>=1)&(v<camera.image_height-1);stages_sum['in_frame']+=int(valid.sum())
        u=u.clamp(0,camera.image_width-1);v=v.clamp(0,camera.image_height-1);z=(hom@camera.world_view_transform)[:,2]
        interior=-F.max_pool2d(-camera.gt_alpha_mask.cuda()[None],3,1,1)[0,0]
        valid&=(interior[v,u]>.99)&(package['rend_alpha'][0,v,u]>.95);stages_sum['alpha_interior']+=int(valid.sum())
        difference=(z-package['surf_depth'][0,v,u]).abs();relative=difference/z.clamp_min(1e-6)
        valid&=relative<.01;stages_sum['relative_depth']+=int(valid.sum())
        rotation=build_rotation(geometry.get_rotation)[ids];scale=geometry.get_scaling[ids]
        z_axis=camera.world_view_transform[:3,2]
        sigma_z=(((rotation[:,:,:2]*scale[:,None,:])*z_axis[None,:,None]).sum(1).square().sum(1)).sqrt()
        absolute_limit=torch.maximum(torch.full_like(sigma_z,extent*.001),.5*sigma_z)
        valid&=difference<absolute_limit;stages_sum['scale_depth']+=int(valid.sum())
        normal=rotation[:,:,2]*model.normal_sign[ids];rendered=F.normalize(package['rend_normal'][:,v,u].T,dim=-1)
        ray=F.normalize(camera.camera_center-xyz,dim=-1)
        valid&=((normal*rendered).sum(1)>.8)&((normal*ray).sum(1)>.1);stages_sum['normal_facing']+=int(valid.sum())
        good=torch.where(valid)[0].cpu().numpy();err=relative.cpu().numpy();pixel=(v*camera.image_width+u).cpu().numpy()
        ranked=good[np.argsort(err[good],kind='stable')];_,unique=np.unique(pixel[ranked],return_index=True);selected=ranked[unique]
        stages_sum['unique_visible']+=len(selected);rays=ray.cpu().numpy();first=count[selected]==0;first_dir[selected[first]]=rays[selected[first]]
        min_cos[selected]=np.minimum(min_cos[selected],(first_dir[selected]*rays[selected]).sum(1));count[selected]+=1;early[selected]+=int(frame<=59);late[selected]+=int(frame>=81);error_sum[selected]+=err[selected]
        rows.append(dict(frame=frame,visible_unique=len(selected)));camera.load2device('cpu')
    span=np.degrees(np.arccos(np.clip(min_cos,-1,1)));reliable=(count>=10)&(early>=3)&(late>=3)&(span>=15.)
    recovered=ids_np[reliable];combined=np.union1d(data['indices'],recovered)
    np.savez_compressed(out/'support.npz',indices=combined,initial_indices=stages['after_motion'],normal=data['normal'],origin=data['origin'],extent=extent)
    np.savez_compressed(out/'visibility.npz',candidate_indices=ids_np,reliable=reliable,visible_counts=count,early_counts=early,late_counts=late,view_span_degrees=span,mean_relative_depth_error=error_sum/np.maximum(count,1))
    result=dict(gt_read=False,heldout_read=False,training_updates=0,parent_support_sha256=sha(args.support),
        source_sha256={str(args.parent/'point_cloud/iteration_35000/point_cloud.ply'):sha(args.parent/'point_cloud/iteration_35000/point_cloud.ply'),str(args.parent/'deform/iteration_35000/deform.pth'):sha(args.parent/'deform/iteration_35000/deform.pth'),str(args.support):sha(args.support),str(args.broad_audit):sha(args.broad_audit),str(args.stages):sha(args.stages)},
        train_motion_frames=frames,visibility_frames=frames,candidate_count=size,recovered_count=len(recovered),final_support=len(combined),support_sha256=sha(out/'support.npz'),
        thresholds=dict(min_visible_frames=10,min_early_frames=3,min_late_frames=3,min_view_span_degrees=15.,relative_depth_lt=.01,
            absolute_depth_lt='max(.001extent, .5*Gaussian_camera_depth_sigma)',predicted_alpha_gt=.95,train_alpha_eroded_gt=.99,normal_agreement_gt=.8,facing_gt=.1,unique_per_pixel=True),
        stages_observation_counts=stages_sum,visibility_count_quantiles=np.quantile(count,[0,.25,.5,.75,1]).tolist(),
        recovered_mean_count=float(count[reliable].mean()) if reliable.any() else None,
        rejected_reason_counts=dict(less_than_10=int((count<10).sum()),early_less_3=int((early<3).sum()),late_less_3=int((late<3).sum()),span_less_15=int((span<15).sum())),per_frame=rows,
        limitations=['只核验父预测前表面一致，不证明真实GT语义或准确几何','大footprint会影响中心周边；仍需DEV人体/鞋足独立验收','未修改13冻结支持或启动新训练'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','per_frame','train_motion_frames','visibility_frames']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','parent','support','broad-audit','stages','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
