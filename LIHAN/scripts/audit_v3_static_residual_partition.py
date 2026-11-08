"""互斥父预测支持类别的TRAIN可见残差，固定父合成/候选合成双口径。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from utils.general_utils import build_rotation
from utils.no_gt_guard import install_no_gt_guard
from gaussian_renderer import render

@torch.no_grad()
def main(a):
    for k in ['source','parent_model','model','support','broad','stages','output']:setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=True);assert not (a.output/'audit.json').exists()
    install_no_gt_guard([a.source,a.parent_model,a.model,a.support,a.broad,a.stages,a.output]);torch.set_num_threads(4)
    parent=DynamicSurfaceModel.load(a.parent_model,0);candidate=DynamicSurfaceModel.load(a.model,a.iteration)
    gate=parent.pc.get_binary_feature().reshape(-1);gate_np=gate.cpu().numpy();N=len(gate);assert N==len(candidate.pc.get_xyz)
    data=np.load(a.support);broad=np.load(a.broad)['indices'];stages=np.load(a.stages)
    n=F.normalize(torch.tensor(data['normal'],device='cuda',dtype=torch.float32),dim=0);o=torch.tensor(data['origin'],device='cuda',dtype=torch.float32);extent=float(data['extent'])
    sets={'broad_support':broad,'recovered_visible':np.setdiff1d(data['indices'],broad),'near_dynamic_rejected':np.setdiff1d(stages['after_motion'],data['indices']),
          'hull_rejected':np.setdiff1d(stages['raw_without_hull'],stages['after_hull']),'motion_rejected':np.setdiff1d(stages['after_hull'],stages['after_motion'])}
    outside=(gate<.1)&~torch.isin(torch.arange(N,device='cuda'),torch.tensor(stages['raw_without_hull'],device='cuda'))
    normal_side=(build_rotation(parent.pc.get_rotation)[:,:,2]*n).sum(1).abs()<=.7
    sets['predicted_sidewall']=torch.where(outside&normal_side)[0].cpu().numpy();sets['outside_plane_band']=torch.where(outside&~normal_side)[0].cpu().numpy()
    masks={k:torch.isin(torch.arange(N,device='cuda'),torch.tensor(v,device='cuda')) for k,v in sets.items()};stack=torch.stack(list(masks.values()))
    assert int(stack.sum(0).max())==1 and torch.equal(stack.any(0),gate<.1),'必须互斥覆盖全部父高置信静态'
    masks['nonstatic_gate']=gate>=.1
    counts=np.zeros(N,dtype=int);early=counts.copy();late=counts.copy();first_ray=np.zeros((N,3));min_cos=np.ones(N)
    totals={mode:{k:np.zeros(3) for k in masks} for mode in ['parent_fixed_composition','candidate_actual_composition']};denoms={k:0. for k in totals};rows=[]
    cameras=load_train_cameras(a.source);frames=[int(c.image_name_train_light.split('_')[-1]) for c in cameras];assert len(frames)==117 and frames==json.loads((a.source/'no_gt_manifest.json').read_text())['train_frames']
    colors=np.array([[0,180,0],[0,255,255],[255,50,0],[230,180,0],[255,0,255],[160,70,255],[80,100,255],[100,100,100]],dtype=np.float32)/255
    names=list(masks);palette=torch.tensor(colors,device='cuda');group_id=torch.zeros(N,device='cuda',dtype=torch.long)
    for j,key in enumerate(names):group_id[masks[key]]=j
    for j,camera in enumerate(cameras):
        frame=frames[j];camera.load2device();pgeo=parent.effective(camera);cgeo=candidate.effective(camera);zero=gate.new_zeros(3)
        axes=build_rotation(cgeo.get_rotation)[:,:,2];angle=torch.rad2deg(torch.atan2(torch.cross(axes,n[None].expand_as(axes),dim=-1).norm(dim=1),(axes*n).sum(1).abs()))
        distance=((cgeo.get_xyz-o)*n).sum(1).abs();fg=camera.gt_alpha_mask[0].cuda()>.5;record=dict(frame=frame,modes={});pp=None
        for mode,geometry in [('parent_fixed_composition',pgeo),('candidate_actual_composition',cgeo)]:
            values={};alpha=None
            for key,mask in masks.items():
                m=mask.float();pack=render(camera,geometry,parent.pipe,zero,0.,0.,0.,override_color=torch.stack([m,m*angle,m*distance],-1));v=pack['render'];alpha=pack['rend_alpha'][0]
                sums=v[:,fg].sum(1).cpu().numpy();totals[mode][key]+=sums;values[key]=dict(mass=float(sums[0]),axis_mean_deg=float(sums[1]/max(sums[0],1e-20)))
                if mode=='parent_fixed_composition':pp=pack
            denoms[mode]+=float(alpha[fg].sum());record['modes'][mode]=values
        # 父预测depth/alpha/尺度/normal前表面门禁；每个互斥组内同像素去重，门槛不由候选或DEV修改。
        xyz=pgeo.get_xyz;hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],1);clip=hom@camera.full_proj_transform
        u=((clip[:,0]/clip[:,3].clamp_min(1e-8)+1)*camera.image_width/2-.5).round().long();v=((clip[:,1]/clip[:,3].clamp_min(1e-8)+1)*camera.image_height/2-.5).round().long()
        valid=(gate<.1)&(clip[:,3]>0)&(u>=1)&(u<camera.image_width-1)&(v>=1)&(v<camera.image_height-1);u=u.clamp(0,camera.image_width-1);v=v.clamp(0,camera.image_height-1)
        z=(hom@camera.world_view_transform)[:,2];interior=-F.max_pool2d(-camera.gt_alpha_mask.cuda()[None],3,1,1)[0,0]
        difference=(z-pp['surf_depth'][0,v,u]).abs();relative=difference/z.clamp_min(1e-6)
        rotation=build_rotation(pgeo.get_rotation);scale=pgeo.get_scaling;z_axis=camera.world_view_transform[:3,2]
        sigma=(((rotation[:,:,:2]*scale[:,None,:])*z_axis[None,:,None]).sum(1).square().sum(1)).sqrt()
        pointnormal=rotation[:,:,2]*parent.normal_sign;rendernormal=F.normalize(pp['rend_normal'][:,v,u].T,dim=-1);ray=F.normalize(camera.camera_center-xyz,dim=-1)
        valid&=(interior[v,u]>.99)&(pp['rend_alpha'][0,v,u]>.95)&(relative<.01)&(difference<torch.maximum(torch.full_like(sigma,extent*.001),.5*sigma))&((pointnormal*rendernormal).sum(1)>.8)&((pointnormal*ray).sum(1)>.1)
        rel=relative.cpu().numpy();pixel=(v*camera.image_width+u).cpu().numpy();rays=ray.cpu().numpy();selected=[]
        for key,ids in sets.items():
            good=ids[valid[torch.tensor(ids,device='cuda')].cpu().numpy()];rank=good[np.argsort(rel[good],kind='stable')];_,unique=np.unique(pixel[rank],return_index=True);selected.extend(rank[unique].tolist())
        selected=np.array(selected,dtype=int);first=counts[selected]==0;first_ray[selected[first]]=rays[selected[first]];min_cos[selected]=np.minimum(min_cos[selected],(first_ray[selected]*rays[selected]).sum(1));counts[selected]+=1;early[selected]+=int(frame<=59);late[selected]+=int(frame>=81)
        if frame in [1,81,149]:
            pack=render(camera,cgeo,parent.pipe,zero,0.,0.,0.,override_color=palette[group_id]);rgb=(pack['render']/pack['rend_alpha'].clamp_min(1e-8)).clamp(0,1).permute(1,2,0).cpu().numpy();rgb[~fg.cpu().numpy()]=0
            font=ImageFont.truetype('/tmp/lumimotion-ppt-1017/NotoSansCJKsc-Regular.otf',16);canvas=Image.new('RGB',(800,460),'white');canvas.paste(Image.fromarray((rgb*255).astype('uint8')),(0,45));draw=ImageDraw.Draw(canvas);draw.text((8,8),'TRAIN预测互斥类别：无GT语义，无支持修改',font=font,fill='black')
            labels=['原宽支持','可见回收','近动态未回收','凸包外','运动排除','预测侧壁','面带外','非高置信静态']
            for k,label in enumerate(labels):draw.rectangle((420,55+k*42,440,75+k*42),fill=tuple((colors[k]*255).astype('uint8')));draw.text((450,52+k*42),label,font=font,fill='black')
            canvas.save(a.output/f'train_partition_{frame:04d}.png')
        rows.append(record);camera.load2device('cpu')
        if j%20==0:print('TRAIN残差分组',j+1,'/117',flush=True)
    span=np.degrees(np.arccos(np.clip(min_cos,-1,1)));reliable=(counts>=10)&(early>=3)&(late>=3)&(span>=15.)
    result=dict(gt_geometry_read=False,heldout_read=False,optimizer_updates=0,model=str(a.model),iteration=a.iteration,frames=frames,
        group_counts={k:int(v.sum()) for k,v in masks.items()},groups_mutually_exclusive=True,
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.support,a.broad,a.stages]},
        modes={mode:{k:dict(alpha_mass_fraction=float(v[0]/denoms[mode]),axis_to_predicted_plane_visible_mean_deg=float(v[1]/max(v[0],1e-20)),distance_to_plane_visible_mean=float(v[2]/max(v[0],1e-20))) for k,v in sums.items()} for mode,sums in totals.items()},
        parent_front_surface={k:dict(reliable_count=int(reliable[ids].sum()),point_count=len(ids),mean_observation_count=float(counts[ids].mean()) if len(ids) else None) for k,ids in sets.items()},
        limitations=['预测侧壁/近动态只为风险标签，不能证明真实GT鞋或侧壁语义','父固定合成使用父geometry但渲染候选轴残差，区别候选改变opacity/覆盖造成的权重变化','前表面门禁每类别内部同像素去重，不能直接与此前1035点整体去重的可见计数比较','不输出新训练支持，不自动回收点，未改变训练或评测掩膜'],per_frame=rows)
    (a.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');np.savez_compressed(a.output/'parent_visibility.npz',visible_counts=counts,reliable=reliable,view_span=span)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_frame','frames']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','parent-model','model','support','broad','stages','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True);main(p.parse_args())
