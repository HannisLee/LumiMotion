"""冻结RAFT独立图像对应诊断；外部预训练先验，不读取场景GT，不更新主线。"""
import argparse,hashlib,json,shlex,sys
from pathlib import Path
import cv2,numpy as np,torch,torchvision
from PIL import Image
from torchvision.models.optical_flow import raft_small,Raft_Small_Weights

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--camera-roll',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4);cv2.setNumThreads(4)
 root=Path(__file__).resolve().parents[1];src=root/'output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view'
 prior=root/'output/1015-24-LH-dynamic-DL-T16图像对应/view_compatible/result.json'
 frames=json.loads((src/'transforms_train.json').read_text())['frames'];triplets=json.loads(prior.read_text())['triplets']
 selected=np.linspace(0,len(triplets)-1,12,dtype=int).tolist()
 weights=Raft_Small_Weights.C_T_V2;model=raft_small(weights=weights,progress=False).cuda().eval();pre=weights.transforms()
 hashes={};images={};masks={};sats={};flows={};alignment=[]
 def read(i):
  if i not in images:
   path=src/'images'/Path(frames[i]['file_path']).name;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
   raw=np.array(Image.open(path).convert('RGBA'));r=cv2.resize(raw,(400,400),interpolation=cv2.INTER_AREA)
   images[i]=r[:,:,:3];masks[i]=cv2.erode((r[:,:,3]>252).astype(np.uint8),np.ones((5,5),np.uint8))
   sat=(raw[:,:,:3]>=254).any(-1).reshape(400,2,400,2).any(axis=(1,3)).astype(np.uint8);sats[i]=cv2.dilate(sat,np.ones((9,9),np.uint8))
  return images[i]
 def flow(i,j):
  if (i,j) not in flows:
   source_image=read(i);target_image=read(j);H=np.eye(3)
   if a.camera_roll:
    # OpenGL相机转OpenCV；取相机相对旋转的最近SO(2)，仅消除画面旋转。
    C=np.diag([1.,-1.,-1.]);ri=np.array(frames[i]['transform_matrix'])[:3,:3]@C;rj=np.array(frames[j]['transform_matrix'])[:3,:3]@C
    U,_,Vt=np.linalg.svd((ri.T@rj)[:2,:2]);Q=U@np.diag([1.,np.linalg.det(U@Vt)])@Vt
    ci=(np.array([frames[i]['cx'],frames[i]['cy']])+.5)/2-.5;cj=(np.array([frames[j]['cx'],frames[j]['cy']])+.5)/2-.5
    H[:2,:2]=Q;H[:2,2]=ci-Q@cj
    target_image=cv2.warpPerspective(target_image,H,(400,400),flags=cv2.INTER_LINEAR)
    probe=np.array([[0.,0.,1.],[199.,199.,1.],[399.,399.,1.]])
    err=float(np.abs((probe@H.T)@np.linalg.inv(H).T-probe).max());assert err<1e-9
    alignment.append(dict(source=i,target=j,H=H.tolist(),roundtrip_max=err,angle_degrees=float(np.degrees(np.arctan2(Q[1,0],Q[0,0])))))
    if len(alignment) in [1,37]:Image.fromarray(np.concatenate([source_image,target_image],axis=1)).save(out/f'aligned_{i}_{j}.png')
   x,y=[torch.from_numpy(k.copy()).permute(2,0,1)[None].cuda() for k in [source_image,target_image]];x,y=pre(x,y)
   with torch.inference_mode():f=model(x,y,num_flow_updates=12)[-1][0].permute(1,2,0).cpu().numpy()
   if a.camera_roll:
    gy,gx=np.mgrid[:400,:400];pixels=np.stack([gx,gy],axis=-1);endpoint=pixels+f
    original=np.concatenate([endpoint,np.ones((400,400,1))],axis=-1)@np.linalg.inv(H).T
    f=(original[:,:,:2]-pixels).astype(np.float32)
    # 原图坐标有效还不够：终点必须位于实际送入网络的旋转画布内。
    outside=((endpoint<0)|(endpoint>399)).any(-1);f[outside]=1e6
   flows[i,j]=f
  return flows[i,j]
 def sample(arr,xy):return cv2.remap(arr.astype(np.float32),xy[:,0].astype(np.float32).reshape(-1,1),xy[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(len(xy),-1)
 def inside(x):return ((x>=0)&(x<=399)).all(1)
 yy,xx=np.mgrid[2:400:4,2:400:4];grid=np.c_[xx.ravel(),yy.ravel()].astype(np.float32);rows=[];saved=[]
 for ordinal,idx in enumerate(selected):
  aa,b,c=triplets[idx]['indices'];ab=flow(aa,b);ba=flow(b,aa);bc=flow(b,c);cb=flow(c,b);ac=flow(aa,c);ca=flow(c,aa)
  pa=grid;pb=pa+sample(ab,pa);pc=pa+sample(ac,pa);via=pb+sample(bc,pb)
  fb=np.linalg.norm(pb+sample(ba,pb)-pa,axis=1);fc=np.linalg.norm(pc+sample(ca,pc)-pa,axis=1);fbc=np.linalg.norm(via+sample(cb,via)-pb,axis=1);cycle=np.linalg.norm(via-pc,axis=1)
  base=(sample(masks[aa],pa)[:,0]>.99);support=base&inside(pb)&inside(pc)&inside(via)&(sample(masks[b],pb)[:,0]>.99)&(sample(masks[c],pc)[:,0]>.99)&(sample(masks[c],via)[:,0]>.99)
  unsat=(sample(sats[aa],pa)[:,0]<.01)&(sample(sats[b],pb)[:,0]<.01)&(sample(sats[c],pc)[:,0]<.01)
  row=dict(triplet_index=idx,indices=[aa,b,c],source_foreground=int(base.sum()),target_supported=int(support.sum()),unsaturated_supported=int((support&unsat).sum()),residual_medians={k:float(np.median(v[support])) if support.any() else None for k,v in [('fb',fb),('fc',fc),('fbc',fbc),('cycle',cycle)]},thresholds={})
  for th in [1.,2.,4.]:
   ok=support&unsat&(fb<th)&(fc<th)&(fbc<th)&(cycle<th)
   row['thresholds'][str(th)]=dict(before_saturation=int((support&(fb<th)&(fc<th)&(fbc<th)&(cycle<th)).sum()),count=int(ok.sum()),source_coverage=float(ok.sum()/max(base.sum(),1)),occupied_40px_cells=len(set(map(tuple,(pa[ok]//40).astype(int)))),motion_median=float(np.median(np.linalg.norm(pb[ok]-pa[ok],axis=1))) if ok.any() else None)
  ok=support&unsat&(fb<2)&(fc<2)&(fbc<2)&(cycle<2);saved.append(dict(indices=[aa,b,c],pa=pa[ok],pb=pb[ok],pc=pc[ok],fb=fb[ok],fc=fc[ok],fbc=fbc[ok],cycle=cycle[ok]))
  canvas=np.concatenate([read(aa),read(b),read(c)],axis=1).copy();ids=np.where(ok)[0];ids=ids[np.linspace(0,len(ids)-1,min(40,len(ids)),dtype=int)] if len(ids) else ids
  for k in ids:
   pts=[tuple(np.round(x+[offset,0]).astype(int)) for x,offset in [(pa[k],0),(pb[k],400),(pc[k],800)]]
   cv2.line(canvas,pts[0],pts[1],(255,70,40),1);cv2.line(canvas,pts[1],pts[2],(40,220,70),1)
  Image.fromarray(canvas).save(out/f'matches_{ordinal:02d}.png');rows.append(row);print(json.dumps(row),flush=True)
  flows.clear()
 torch.save(saved,out/'matches.pt')
 weightfile=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
 for path in [Path(__file__),prior,src/'transforms_train.json',weightfile]:hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
 result=dict(camera_roll=a.camera_roll,alignment=alignment,status='DIAGNOSTIC_ONLY',gt_read=False,optimizer_updates=0,external_pretrained_prior=True,weights=str(weights),weight_url=weights.url,torch=torch.__version__,torchvision=torchvision.__version__,selected_triplets=selected,total_available_triplets=len(triplets),rows=rows,inputs=hashes,limitation='双向及循环自洽不证明物质对应正确；未用默认目标投影筛选，未区分动静区域，不能直接用于主线训练')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
