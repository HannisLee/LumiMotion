"""以邻居留出验证持续身份轨迹的局部旋转运输；零优化，无GT。"""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scripts.probe_dynamic_dl_selfcal import read_config,digest_tensor
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from utils.general_utils import build_rotation


def statistics(values):
 a=np.asarray(values).ravel();assert len(a) and np.isfinite(a).all()
 return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))


@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['model','cache','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--reference-index',type=int,default=-1)
 p.add_argument('--dense-neighbors',action='store_true')
 p.add_argument('--neighbor-mode',choices=['reference','trajectory'],default='reference')
 args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
 tracks=torch.load(args.cache,map_location='cpu',weights_only=False)
 cfg=read_config(Path(args.model)/'cfg_args')
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
 ply=Path(args.model)/'point_cloud/iteration_35000/point_cloud.ply';pc.load_ply(str(ply))
 deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
 assert deform.load_weights(args.model,35000);deform.deform.eval().requires_grad_(False)
 before={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
 sample_ids=tracks['ids'].cuda();assert torch.equal(pc.get_xyz[sample_ids].cpu(),tracks['canonical_xyz'])
 ids=torch.arange(len(pc.get_xyz),device='cuda') if args.dense_neighbors else sample_ids
 xyz=pc.get_xyz[ids]
 feature=pc.get_binary_feature()[ids];gate=feature.flatten().cpu().numpy()
 canonical=xyz.cpu().numpy().astype(np.float64);positions=tracks['positions'].numpy().astype(np.float64)
 train=np.array([i for i in range(len(positions)) if i%9!=4]);held=np.array([i for i in range(len(positions)) if i%9==4])
 supported_sample=tracks['valid'][train].sum(0).numpy()>=10
 if args.dense_neighbors:
  supported=np.zeros(len(xyz),dtype=bool);supported[tracks['ids'].numpy()]=supported_sample
 else:supported=supported_sample
 native=[];dense_positions=[];max_query_error=0.
 canonical_rotation=build_rotation(pc.get_rotation[ids])
 for i,time in enumerate(tracks['times']):
  ds=deform.step(xyz,time.cuda().reshape(1,1).expand(len(xyz),1),feature=feature)
  current_positions=xyz+ds['d_xyz']
  compared=current_positions[sample_ids] if args.dense_neighbors else current_positions
  error=float((compared-tracks['positions'][i].cuda()).abs().max())
  if args.dense_neighbors:dense_positions.append(current_positions.cpu().numpy());max_query_error=max(max_query_error,error)
  assert error<1e-4
  rotation=build_rotation(pc.rotation_activation(pc._rotation[ids]+ds['d_rotation']))
  native.append((rotation@canonical_rotation.transpose(1,2)).cpu().numpy())
 native=np.stack(native).astype(np.float64)
 if args.dense_neighbors:positions=np.stack(dense_positions).astype(np.float64)
 if args.reference_index>=0:
  assert args.reference_index in train
  canonical=positions[args.reference_index].copy()
  native=native@native[args.reference_index].swapaxes(-1,-2)[None]
 summaries={};metadata={};arrays={}
 for group,group_mask in [('dynamic',gate>.5),('low_gate',gate<=.5)]:
  selected=np.where(supported&group_mask)[0]
  metadata[group]=dict(all_tracks=int(group_mask.sum()),supported_tracks=len(selected),support_fraction=float(len(selected)/max(int(group_mask.sum()),1)))
  if len(selected)<33:
   summaries[group]=dict(status='不足33个支持身份，未拟合');continue
  pool=np.where(group_mask)[0] if args.dense_neighbors else selected
  tree=cKDTree(canonical[pool]);distance,neighbors=tree.query(canonical[selected],k=33)
  # 自身排除；距离相同时不依赖第一个返回恰好是自身。
  neighbor_ids=np.stack([pool[row][pool[row]!=center][:32] for center,row in zip(selected,neighbors)])
  if args.neighbor_mode=='trajectory':
   assert not args.dense_neighbors, '密集轨迹图未实现，不静默改变协议'
   total=np.zeros((len(selected),len(selected)));squared=np.zeros_like(total)
   for t in train:
    points=positions[t,selected];norm=(points*points).sum(-1)
    distances=np.sqrt(np.maximum(norm[:,None]+norm[None,:]-2*points@points.T,0))
    total+=distances;squared+=distances*distances
   mean=total/len(train);deviation=np.sqrt(np.maximum(squared/len(train)-mean*mean,0))
   score=mean+deviation;np.fill_diagonal(score,np.inf)
   candidates=np.argpartition(score,32,axis=1)[:,:32]
   order=np.argsort(np.take_along_axis(score,candidates,axis=1),axis=1)
   neighbor_ids=selected[np.take_along_axis(candidates,order,axis=1)]
  a=canonical[neighbor_ids]-canonical[selected,None,:]
  b=positions[:,neighbor_ids,:]-positions[:,selected,None,:]
  radius=np.sqrt(np.mean(np.sum(a*a,axis=-1),axis=-1)).clip(1e-8)
  metadata[group]['neighbor_rms_radius']=statistics(radius)
  arrays[group+'_selected']=selected;arrays[group+'_neighbors']=neighbor_ids
  for fold in [0,1]:
   fit=np.arange(fold,32,2);test=np.arange(1-fold,32,2)
   afit=a[:,fit,:];bfit=b[:,:,fit,:]
   covariance=np.einsum('nki,tnkj->tnij',afit,bfit)
   u,s,vh=np.linalg.svd(covariance)
   fix=np.broadcast_to(np.eye(3),u.shape).copy();fix[...,2,2]=np.linalg.det(vh.swapaxes(-1,-2)@u.swapaxes(-1,-2))
   rotation=vh.swapaxes(-1,-2)@fix@u.swapaxes(-1,-2)
   assert np.max(np.abs(np.linalg.det(rotation)-1))<1e-5
   reference_singular=np.linalg.svd(afit,compute_uv=False)
   rank_ok=(reference_singular[:,1]/np.maximum(reference_singular[:,0],1e-12)>.05)&(s[...,1]/np.maximum(s[...,0],1e-12)>.05)
   atest=a[:,test,:];truth=b[:,:,test,:]
   predictions={'identity':np.broadcast_to(atest,truth.shape),'gaussian':np.einsum('tnij,nkj->tnki',native[:,selected],atest),'local':np.einsum('tnij,nkj->tnki',rotation,atest)}
   errors={name:np.sqrt(np.mean(np.sum((v-truth)**2,axis=-1),axis=-1)) for name,v in predictions.items()}
   length_bound=np.sqrt(np.mean((np.linalg.norm(truth,axis=-1)-np.linalg.norm(atest,axis=-1)[None])**2,axis=-1))
   errors['length_lower_bound']=length_bound
   arrays[f'{group}_fold{fold}_rank_ok']=rank_ok
   for name,error in errors.items():arrays[f'{group}_fold{fold}_{name}']=error
   # 不以误差阈值过滤；同时报告全部身份及预注册秩门槛子集。
   for split,times in [('train',train),('held',held)]:
    for subset in ['all','rank_ok']:
     valid=np.ones_like(rank_ok[times],dtype=bool) if subset=='all' else rank_ok[times]
     key=f'{group}_fold{fold}_{split}_{subset}'
     if not valid.any():summaries[key]=dict(count=0);continue
     summaries[key]=dict(valid_fraction=float(valid.mean()),points_times=int(valid.sum()),methods={name:dict(absolute=statistics(error[times][valid]),relative=statistics((error[times]/radius[None])[valid])) for name,error in errors.items()},local_beats_identity_fraction=float((errors['local'][times][valid]<errors['identity'][times][valid]).mean()),local_beats_gaussian_fraction=float((errors['local'][times][valid]<errors['gaussian'][times][valid]).mean()))
   print(json.dumps({k:v for k,v in summaries.items() if k.startswith(f'{group}_fold{fold}_train_all')}),flush=True)
 assert before=={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
 np.savez_compressed(out/'neighbor_errors.npz',**arrays)
 inputs=[Path(__file__),Path(args.cache),ply,Path(args.model)/'cfg_args',Path(args.model)/'deform/iteration_35000/deform.pth']
 result=dict(status='PASS',scope='邻居留出预测诊断，非真实几何验收',reference_index=args.reference_index,neighbor_mode=args.neighbor_mode,dense_neighbors=args.dense_neighbors,gt_read=False,optimizer_updates=0,query_max_error=max_query_error,metadata=metadata,summary=summaries,inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(status='PASS',metadata=metadata,query_max_error=max_query_error)),flush=True)

if __name__=='__main__':main()
