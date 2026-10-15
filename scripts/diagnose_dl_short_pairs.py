"""全部短两帧对应诊断；外部冻结先验，离面分区不是语义真值。"""
import argparse,hashlib,json,shlex,sys
from pathlib import Path
import cv2,numpy as np,torch,torchvision
from PIL import Image
from torchvision.models.optical_flow import raft_small,Raft_Small_Weights

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--camera-roll',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4);cv2.setNumThreads(4)
 root=Path(__file__).resolve().parents[1];src=root/'output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view'
 prior=root/'output/smoke_test/1015-65-LH-dynamic-DL-T73短间隔对应/short_pair_inventory.json'
 frames=json.loads((src/'transforms_train.json').read_text())['frames'];pairs=json.loads(prior.read_text())['pairs']
 from scipy.spatial import cKDTree
 cache=root/'output/1015-07-LH-dynamic-DL-T02表面轨迹初始化/cache/tracks.pt';plane=root/'output/1015-15-LH-dynamic-DL-T05复核/seed_1015/checkpoint.pt'
 tracks=torch.load(cache,map_location='cpu',weights_only=False);pl=torch.load(plane,map_location='cpu',weights_only=False)
 uv=(tracks['pixels'].numpy()+.5)*2-.5;valid=tracks['valid'].numpy();normal=pl['plane_normal'].numpy();origin=pl['plane_origin'].numpy()
 heights=np.abs((tracks['positions'].numpy()-origin)@normal)
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
 yy,xx=np.mgrid[2:400:4,2:400:4];pa=np.c_[xx.ravel(),yy.ravel()].astype(np.float32);rows=[];observations=[]
 for ordinal,(aa,b,gap,angle) in enumerate(pairs):
  assert aa%9!=4 and b%9!=4 and 0<gap<=3 and angle<=25
  ab=flow(aa,b);ba=flow(b,aa);pb=pa+sample(ab,pa);fb=np.linalg.norm(pb+sample(ba,pb)-pa,axis=1)
  base=sample(masks[aa],pa)[:,0]>.99;support=base&inside(pb)&(sample(masks[b],pb)[:,0]>.99)
  source_unsat=sample(sats[aa],pa)[:,0]<.01;unsat=source_unsat&(sample(sats[b],pb)[:,0]<.01)
  ia=np.where(valid[aa])[0];ib=np.where(valid[b])[0];da,ka=cKDTree(uv[aa,ia]).query(pa);db,kb=cKDTree(uv[b,ib]).query(pb)
  source_off=base&(da<=4)&(heights[aa,ia[ka]]>.1);target_off=(db<=4)&(heights[b,ib[kb]]>.1)
  flags=dict(source=base,source_unsaturated=base&source_unsat,support=support,fb=support&(fb<2),final=support&(fb<2)&unsat)
  groups=dict(all=base,source_off=source_off,both_off=source_off&target_off)
  row=dict(indices=[aa,b],source_span=gap,axis_angle=angle,counts={g:{k:int((mask&f).sum()) for k,f in flags.items()} for g,mask in groups.items()});rows.append(row)
  np.savez_compressed(out/f'pair_{ordinal:02d}.npz',indices=[aa,b],pa=pa,pb=pb,fb=fb,base=base,support=support,source_unsat=source_unsat,unsat=unsat,source_off=source_off,target_off=target_off,source_ids=tracks['ids'].numpy()[ia[ka]],target_ids=tracks['ids'].numpy()[ib[kb]],source_distance=da,target_distance=db)
  canvas=np.concatenate([read(aa),read(b)],axis=1).copy()
  for k in np.where(source_off)[0]:
   color=(255,220,0)
   if flags['fb'][k]:color=(255,40,30)
   if flags['final'][k]:color=(30,255,80) if target_off[k] else (40,100,255)
   cv2.circle(canvas,tuple(np.round(pa[k]).astype(int)),1,color,-1)
   if flags['fb'][k]:cv2.circle(canvas,tuple(np.round(pb[k]+[400,0]).astype(int)),2,color,-1)
   if flags['final'][k]:observations.append(dict(indices=[aa,b],pa=pa[k].tolist(),pb=pb[k].tolist(),both_off=bool(target_off[k]),source_id=int(tracks['ids'][ia[ka[k]]]),target_nearest_id=int(tracks['ids'][ib[kb[k]]]),fb=float(fb[k])))
  Image.fromarray(canvas).save(out/f'offplane_{ordinal:02d}.png');print(json.dumps(row),flush=True);flows.clear()
 totals={g:{k:sum(r['counts'][g][k] for r in rows) for k in rows[0]['counts'][g]} for g in rows[0]['counts']}
 weightfile=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
 for path in [Path(__file__),prior,src/'transforms_train.json',weightfile,cache,plane]:hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
 result=dict(status='DIAGNOSTIC_ONLY',gt_read=False,optimizer_updates=0,external_pretrained_prior=True,camera_roll=a.camera_roll,weights=str(weights),weight_url=weights.url,torch=torch.__version__,torchvision=torchvision.__version__,rows=rows,totals=totals,alignment=alignment,inputs=hashes,limitation='离面源/目标分区均来自粗几何，不是语义真值；不按目标重投影残差筛选')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2));print(json.dumps(totals),flush=True)
if __name__=='__main__':main()
