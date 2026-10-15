"""固定轨迹邻域上的正则仿射运输审计；零训练、无GT。"""
import argparse,hashlib,json,shlex,sys
from pathlib import Path
import numpy as np
import torch


def stat(a):
 a=np.asarray(a).ravel()
 return dict(count=len(a),median=float(np.median(a)),mean=float(a.mean()),p90=float(np.quantile(a,.9))) if len(a) else dict(count=0,median=None,mean=None,p90=None)


def fit(a,b,ridge):
 c=np.einsum('nki,nkj->nij',a,a);d=np.einsum('nki,tnkj->tnij',a,b)
 strength=ridge*np.trace(c,axis1=-2,axis2=-1)/3
 strength=np.maximum(strength,1e-12)
 return np.linalg.solve(c[None]+strength[None,:,None,None]*np.eye(3),d+strength[None,:,None,None]*np.eye(3))


def bounded(m):
 u,s,vh=np.linalg.svd(m);values=np.clip(s,.5,2)
 values[...,-1]*=np.where(np.linalg.det(u@vh)<0,-1,1)
 result=(u*values[...,None,:])@vh
 assert np.all(np.linalg.det(result)>0)
 assert np.max(np.linalg.svd(result,compute_uv=False))<=2+1e-8
 assert np.min(np.linalg.svd(result,compute_uv=False))>=.5-1e-8
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['cache','neighbors','plane','output']:p.add_argument('--'+k,required=True)
 args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
 c=torch.load(args.cache,map_location='cpu',weights_only=False);raw=np.load(args.neighbors);plane=json.loads(Path(args.plane).read_text());assert not plane['gt_read']
 positions=c['positions'].numpy().astype(float);reference=positions[60];train=np.array([i for i in range(135) if i%9!=4 and i!=60]);held=np.array([i for i in range(135) if i%9==4])
 height=np.abs((positions-np.array(plane['plane_origin']))@np.array(plane['plane_normal']))
 threshold=.015*plane['scene_extent'];near=(height[train]<threshold).mean(0)>=.9;above=(height[train]>2*threshold).mean(0)>=.9
 summaries={};arrays={};gates={}
 # 独立已知仿射留出控制，不使用场景GT。
 rng=np.random.default_rng(1015);a=rng.normal(size=(1,32,3));truth=np.array([[1.2,.1,0],[0,.8,.1],[0,0,1.1]])
 b=np.einsum('nki,ij->nkj',a,truth)[None];recovered=fit(a[:,::2],b[:,:,::2],1e-10)
 control=float(np.abs(np.einsum('nki,tnij->tnkj',a[:,1::2],recovered)-b[:,:,1::2]).max());assert control<1e-8
 for group in ['dynamic','low_gate']:
  ids=raw[group+'_selected'];neighbors=raw[group+'_neighbors'];a=reference[neighbors]-reference[ids,None]
  b=positions[:,neighbors]-positions[:,ids,None]
  radius=np.sqrt(np.mean(np.sum(a*a,-1),-1)).clip(1e-8)
  region_masks={'all':np.ones(len(ids),bool),'near_plane':near[ids],'above_plane':above[ids],'ambiguous':~(near[ids]|above[ids])}
  arrays[group+'_ids']=ids;arrays[group+'_radius']=radius
  for ridge in [.001,.01,.1]:
   reliability=[]
   for fold in [0,1]:
    afit=a[:,fold::2];bfit=b[:,:,fold::2];atest=a[:,1-fold::2];target=b[:,:,1-fold::2]
    m=fit(afit,bfit,ridge);limit=bounded(m)
    singular=np.linalg.svd(m,compute_uv=False);det=np.linalg.det(m)
    errors={}
    for mode,mapping in [('raw',m),('bounded',limit)]:
     prediction=np.einsum('nki,tnij->tnkj',atest,mapping)
     error=np.sqrt(np.mean(np.sum((prediction-target)**2,-1),-1));errors[mode]=error
     arrays[f'{group}_{ridge}_{fold}_{mode}_error']=error
    refsv=np.linalg.svd(afit,compute_uv=False);rank=refsv[:,1]/np.maximum(refsv[:,0],1e-12)>.05
    reliable=(np.median(errors['bounded'][train]/radius[None],axis=0)<=.2)&rank
    reliability.append(reliable)
    if ridge==.01:arrays[f'{group}_primary_fold{fold}_maps']=limit.astype(np.float32)
    for split,times in [('train',train),('held',held)]:
     for region,mask in region_masks.items():
      key=f'{group}_{ridge}_fold{fold}_{split}_{region}'
      valid=np.broadcast_to(mask,(len(times),len(ids)))
      summaries[key]=dict(centers=int(mask.sum()),methods={name:dict(absolute=stat(error[times][valid]),relative=stat((error[times]/radius[None])[valid])) for name,error in errors.items()},rotation_relative=stat((raw[f'{group}_fold{fold}_local'][times]/radius[None])[valid]),raw_negative_determinant_fraction=float((det[times][valid]<=0).mean()) if valid.any() else None,raw_outside_bounds_fraction=float(((singular[times,:,0]>2)|(singular[times,:,-1]<.5))[valid].mean()) if valid.any() else None,raw_max_singular=stat(singular[times,:,0][valid]),raw_min_singular=stat(singular[times,:,-1][valid]))
   selected=reliability[0]&reliability[1]
   for region,mask in region_masks.items():
    chosen=selected&mask;gate_key=f'{group}_{ridge}_{region}'
    # 至少半数动态高于平面支持点能在训练可靠且保留时间中位误差<=0.2，才考虑接入。
    held_errors=[arrays[f'{group}_{ridge}_{fold}_bounded_error'][held][:,chosen]/radius[None,chosen] for fold in [0,1]]
    fraction=float(chosen.sum()/max(mask.sum(),1))
    gates[gate_key]=dict(centers=int(mask.sum()),reliable_centers=int(chosen.sum()),reliable_fraction=fraction,held_relative=[stat(x) for x in held_errors],passed=fraction>=.5 and all(x.size and np.median(x)<=.2 for x in held_errors))
   print(json.dumps({k:v for k,v in gates.items() if k.startswith(f'{group}_{ridge}_')}),flush=True)
 np.savez_compressed(out/'errors_and_primary_maps.npz',**arrays)
 inputs=[Path(__file__),Path(args.cache),Path(args.neighbors),Path(args.plane)]
 result=dict(status='PASS',scope='仿射运输数值审计，非真实几何验收',gt_read=False,optimizer_updates=0,primary_ridge=.01,known_affine_control_error=control,gate=gates,summary=summaries,inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
