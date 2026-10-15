"""训练图像双向SIFT和三帧循环检查；仅诊断默认身份重投影，不使用GT或预训练模型。"""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.special import expit


def stats(values):
 a=np.asarray(values,dtype=float)
 if not len(a):return dict(count=0,median=None,p90=None,mean=None)
 return dict(count=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),mean=float(a.mean()))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['source','model','cache','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--triplet-mode',choices=['temporal','view'],default='temporal')
 args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 (out/'command.txt').write_text(shlex.join(sys.argv)+'\n');cv2.setNumThreads(4);cv2.setRNGSeed(1015)
 tracks=torch.load(args.cache,map_location='cpu',weights_only=False)
 assert tracks['config']['resolution']==4
 source=Path(args.source);metadata=json.loads((source/'transforms_train.json').read_text())['frames']
 assert len(metadata)==len(tracks['frames'])
 plypath=Path(args.model)/'point_cloud/iteration_35000/point_cloud.ply';ply=PlyData.read(plypath).elements[0]
 xyz=np.stack([ply[k] for k in ['x','y','z']],axis=1);ids=tracks['ids'].numpy();assert np.array_equal(xyz[ids],tracks['canonical_xyz'].numpy())
 fea=[k.name for k in ply.properties if k.name.startswith('fea')];assert len(fea)==1
 gate=expit(2*np.asarray(ply[fea[0]])[ids])
 uv=(tracks['pixels'].numpy().astype(float)+.5)*2-.5;valid=tracks['valid'].numpy()
 train=[i for i in range(len(metadata)) if i%9!=4]
 images={};points={};descriptors={};saturation={};hashes={};key_counts={}
 sift=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.02,edgeThreshold=10)
 for i in train:
  assert Path(metadata[i]['file_path']).stem==tracks['frames'][i]['name']
  path=source/'images'/Path(metadata[i]['file_path']).name
  original=np.array(Image.open(path).convert('RGBA'));assert original.shape[:2]==(800,800)
  rgba=cv2.resize(original,(400,400),interpolation=cv2.INTER_AREA);rgb=rgba[:,:,:3];alpha=rgba[:,:,3]
  mask=cv2.erode((alpha>252).astype(np.uint8)*255,np.ones((5,5),np.uint8))
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);keys,desc=sift.detectAndCompute(gray,mask)
  points[i]=np.array([k.pt for k in keys],dtype=float).reshape(-1,2);descriptors[i]=desc
  sat=(original[:,:,:3]>=254).any(-1).reshape(400,2,400,2).any(axis=(1,3)).astype(np.uint8)
  saturation[i]=cv2.dilate(sat,np.ones((9,9),np.uint8)).astype(bool)
  images[i]=rgb;key_counts[str(i)]=len(keys);hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
 bf=cv2.BFMatcher(cv2.NORM_L2);pair_cache={}
 def match(a,b):
  key=(a,b)
  if key in pair_cache:return pair_cache[key]
  da,db=descriptors[a],descriptors[b]
  if da is None or db is None or min(len(da),len(db))<2:pair_cache[key]={};return {}
  forward=bf.knnMatch(da,db,k=2);backward=bf.knnMatch(db,da,k=2)
  fa={m.queryIdx:m.trainIdx for m,n in forward if m.distance<.75*n.distance}
  ba={m.queryIdx:m.trainIdx for m,n in backward if m.distance<.75*n.distance}
  result={a:b for a,b in fa.items() if ba.get(b)==a};pair_cache[key]=result;return result
 rows=[];triplets=[];tiles=[]
 selected_triplets=[tuple(train[j:j+3]) for j in range(len(train)-2)]
 if args.triplet_mode=='view':
  axes=np.array([metadata[i]['transform_matrix'] for i in train])[:,:3,2]
  angles=np.degrees(np.arccos(np.clip(axes@axes.T,-1,1)))
  source_frames=np.array([metadata[i]['source_frame'] for i in train]);selected_triplets=[]
  for ia in range(len(train)):
   for ib in range(ia+1,len(train)):
    if source_frames[ib]-source_frames[ia]>15:break
    if angles[ia,ib]>25:continue
    for ic in range(ib+1,len(train)):
     if source_frames[ic]-source_frames[ia]>15:break
     if angles[ia,ic]<=25 and angles[ib,ic]<=25:selected_triplets.append((train[ia],train[ib],train[ic]))
 for j,(a,b,c) in enumerate(selected_triplets):
  ab=match(a,b);bc=match(b,c);ac=match(a,c)
  cycles=[(ka,kb,bc[kb]) for ka,kb in ab.items() if kb in bc and ac.get(ka)==bc[kb]]
  # 去除同一空间关键点的多方向重复检测，保证不增加虚假观测数。
  unique={}
  for ka,kb,kc in cycles:unique.setdefault(tuple(np.round(np.r_[points[a][ka],points[b][kb],points[c][kc]],1)),(ka,kb,kc))
  cycles=list(unique.values());available=np.where(valid[a])[0]
  tree=cKDTree(uv[a,available]) if len(available) else None
  local=[]
  for ka,kb,kc in cycles:
   pa,pb,pc=points[a][ka],points[b][kb],points[c][kc]
   sat=any(saturation[t][int(np.clip(round(point[1]),0,399)),int(np.clip(round(point[0]),0,399))] for t,point in [(a,pa),(b,pb),(c,pc)])
   row=dict(a=a,b=b,c=c,ka=ka,kb=kb,kc=kc,pa=pa.tolist(),pb=pb.tolist(),pc=pc.tolist(),saturated_patch=bool(sat),associated=False)
   if tree is not None:
    distance,index=tree.query(pa)
    if distance<=4:
     identity=int(available[index]);co_visible=bool(valid[b,identity] and valid[c,identity])
     row.update(associated=True,track=identity,gaussian_id=int(ids[identity]),dynamic=bool(gate[identity]>.5),gate=float(gate[identity]),source_distance=float(distance),co_visible=co_visible,
                error_b=float(np.linalg.norm(uv[b,identity]-pb)),error_c=float(np.linalg.norm(uv[c,identity]-pc)),observed_motion=float(np.linalg.norm(pb-pa)),
                offset_corrected_error_b=float(np.linalg.norm(uv[b,identity]+pa-uv[a,identity]-pb)),predicted_b=uv[b,identity].tolist())
   rows.append(row);local.append(row)
  triplets.append(dict(indices=[a,b,c],source_frames=[int(metadata[t].get('source_frame',t)) for t in [a,b,c]],matches_ab=len(ab),cycles=len(cycles),associated=sum(r['associated'] for r in local),dynamic=sum(r.get('dynamic',False) for r in local),dynamic_cov_isible=sum(r.get('dynamic',False) and r.get('co_visible',False) for r in local)))
  if j in [0,30,60,90,117]:
   canvas=np.concatenate([images[a],images[b]],axis=1).copy();chosen=sorted(local,key=lambda r:(not r.get('dynamic',False),r['ka']))[:40]
   for row in chosen:
    x=tuple(np.round(row['pa']).astype(int));y=tuple(np.round(np.array(row['pb'])+[400,0]).astype(int));color=(255,80,40) if row.get('dynamic',False) else (40,220,80)
    cv2.line(canvas,x,y,color,1);cv2.circle(canvas,x,3,color,1);cv2.circle(canvas,y,3,color,1)
    if row.get('associated',False):
     pred=np.round(np.array(row['predicted_b'])+[400,0]).astype(int);cv2.drawMarker(canvas,tuple(pred),(40,80,255),cv2.MARKER_CROSS,6,1)
   Image.fromarray(canvas).save(out/f'matches_{j:03d}.png')
  if j%20==0:print(json.dumps(triplets[-1]),flush=True)
 summary={}
 for group in ['all_associated','dynamic','low_gate']:
  for condition in ['all','co_visible','co_visible_unsaturated']:
   selected=[r for r in rows if r['associated'] and (group=='all_associated' or r['dynamic']==(group=='dynamic')) and (condition=='all' or r['co_visible']) and (condition!='co_visible_unsaturated' or not r['saturated_patch'])]
   summary[group+'_'+condition]=dict(observations=len(selected),unique_gaussians=len({r['gaussian_id'] for r in selected}),triplets=len({(r['a'],r['b'],r['c']) for r in selected}),unique_source_target_edges=len({(r['a'],r['b'],r['ka'],r['kb']) for r in selected}),error_b=stats([r['error_b'] for r in selected]),error_c=stats([r['error_c'] for r in selected]),source_distance=stats([r['source_distance'] for r in selected]),offset_corrected_error_b=stats([r['offset_corrected_error_b'] for r in selected]),observed_motion=stats([r['observed_motion'] for r in selected]))
 inputs=[Path(__file__),Path(args.cache),plypath,source/'transforms_train.json'];hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
 result=dict(status='PASS',scope='图像匹配诊断，匹配不是GT',triplet_mode=args.triplet_mode,gt_read=False,optimizer_updates=0,opencv_version=cv2.__version__,keypoint_counts=key_counts,triplets=triplets,total_cycles=len(rows),summary=summary,inputs=hashes)
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'matches.json').write_text(json.dumps(rows,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
