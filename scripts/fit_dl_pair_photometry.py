"""固定未知光/几何的匹配观测材质配对；仅材质保留，不是未见光照泛化。"""
import argparse,hashlib,json,shlex,sys
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='output/smoke_test/1015-66-LH-dynamic-DL-T74短两帧对应');p.add_argument('--held-reference');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4);cv2.setNumThreads(4)
 base=Path(a.base);cache=Path('output/1015-07-LH-dynamic-DL-T02表面轨迹初始化/cache/tracks.pt');lp=Path('output/1015-15-LH-dynamic-DL-T05复核/seed_1015/checkpoint.pt');ep=Path('output/smoke_test/1015-49-LH-dynamic-DL-T57底项约定修正配对/supported/checkpoint.pt');src=Path('output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view')
 t=torch.load(cache,map_location='cpu',weights_only=False);l=torch.load(lp,map_location='cpu',weights_only=False);e=torch.load(ep,map_location='cpu',weights_only=False);assert hashlib.sha256(cache.read_bytes()).hexdigest()==l['cache_sha256'];meta=json.loads((src/'transforms_train.json').read_text())['frames'];lookup={int(v):i for i,v in enumerate(t['ids'])};uv=(t['pixels'].numpy()+.5)*2-.5;ambient=.10000000149011612
 shading=e['energy'].double().numpy()[:,None]*(np.maximum((t['normals'].double().numpy()*l['smooth'].double().numpy()[:,None]).sum(-1),0)+ambient)
 observations=json.loads((base/'probe/observations.json').read_text());allframes=sorted({i for r in observations if r['both_off'] for i in r['indices']});held=set(allframes[4::5]);images={};inputs=[cache,lp,ep,src/'transforms_train.json',base/'probe/observations.json',base/'probe/result.json',base/'homography/result.json',Path(__file__)]
 if a.held_reference:
  hp=Path(a.held_reference);held=set(json.loads(hp.read_text())['held_material_frames']);inputs.append(hp)
 def sample(i,xy):
  if i not in images:
   path=src/'images'/Path(meta[i]['file_path']).name;raw=np.array(Image.open(path).convert('RGBA'));rgba=cv2.resize(raw,(400,400),interpolation=cv2.INTER_AREA);c=rgba[:,:,:3].astype(np.float64)/255.;linear=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4);mask=cv2.erode((rgba[:,:,3]>252).astype(np.uint8),np.ones((5,5),np.uint8));sat=(raw[:,:,:3]>=254).any(-1).reshape(400,2,400,2).any(axis=(1,3)).astype(np.uint8);sat=cv2.dilate(sat,np.ones((9,9),np.uint8));images[i]=(linear,mask,sat);inputs.append(path)
  x,y=map(float,xy)
  if not (0<=x<=399 and 0<=y<=399):return None
  def remap(a):return cv2.remap(a.astype(np.float64),np.array([[x]],np.float32),np.array([[y]],np.float32),cv2.INTER_LINEAR).reshape(-1)
  rgb,mask,sat=images[i]
  if remap(mask)[0]<=.99 or remap(sat)[0]>=.01:return None
  return remap(rgb)
 records=[];excluded=0;candidate_count=0;probe_meta=json.loads((base/'probe/result.json').read_text())
 for j,row in enumerate(probe_meta['rows']):
  path=base/'probe'/f'pair_{j:02d}.npz';hp=base/'homography'/f'residual_{j:02d}.npz';d=np.load(path);h=np.load(hp);inputs.extend([path,hp]);aa,bb=map(int,d['indices'])
  for k in np.where(h['candidate'])[0]:
   candidate_count+=1;gid=int(d['source_ids'][k]);ti=lookup[gid];local=[]
   for i,xy in [(aa,d['pa'][k]),(bb,d['pb'][k])]:
    matched=sample(i,xy);default=sample(i,uv[i,ti])
    if matched is None or default is None:break
    if probe_meta.get('source_anchors',False) and i==aa:assert np.array_equal(matched,default)
    local.append(dict(frame=i,gid=gid,matched=matched,default=default,shading=float(shading[i,ti])))
   if len(local)!=2:excluded+=1;continue
   for rec in local:records.append(dict(**rec,nonplanar=bool(h['moving'][k])))
 assert candidate_count==sum(r['both_off'] for r in observations)
 results={};saved={};coverage={}
 for group in ['all_candidates','nonplanar']:
  bucket={}
  for r in records:
   if group=='nonplanar' and not r['nonplanar']:continue
   bucket.setdefault((r['gid'],r['frame']),[]).append(r)
  keys=sorted(bucket);ids=np.array([k[0] for k in keys]);times=np.array([k[1] for k in keys]);s=np.array([bucket[k][0]['shading'] for k in keys]);ym=np.array([np.mean([r['matched'] for r in bucket[k]],0) for k in keys]);yd=np.array([np.mean([r['default'] for r in bucket[k]],0) for k in keys]);train=np.array([i not in held for i in times]);supported=set(ids[train]);test=(~train)&np.array([i in supported for i in ids]);unique=sorted(supported)
  coverage[group]=dict(raw_endpoint_events=sum(len(v) for v in bucket.values()),unique_gid_frames=len(keys),train_groups=int(train.sum()),held_groups=int((~train).sum()),supported_held_groups=int(test.sum()),supported_held_ids=len(set(ids[test])),supported_held_frames=len(set(times[test])))
  def fit(y,scale):return {gid:(y[(ids==gid)&train]*scale[(ids==gid)&train,None]).sum(0)/np.square(scale[(ids==gid)&train]).sum() for gid in unique}
  for signal,y in [('default',yd),('matched',ym)]:
   for model,scale in [('physical',s),('constant',np.ones_like(s))]:
    material=fit(y,scale);perturbed=y.copy();perturbed[~train]=.123;check=fit(perturbed,scale);assert all(np.array_equal(material[g],check[g]) for g in unique)
    pred=np.array([material.get(g,np.zeros(3))*v for g,v in zip(ids,scale)]);metrics={}
    for label,target in [('matched_target',ym),('default_target',yd)]:
     err=np.square(pred-target).mean(-1);metrics[label]=dict(mse=float(err[test].mean()) if test.any() else None,gid_balanced_mse=float(np.mean([err[test&(ids==g)].mean() for g in set(ids[test])])) if test.any() else None)
    key=group+'/'+signal+'/'+model;results[key]=metrics;saved[key]=dict(material=material,prediction=pred,keys=keys,train=train,test=test,matched_rgb=ym,default_rgb=yd,shading=scale)
 coverage['candidate_pairs']=dict(input=candidate_count,excluded_common_default_or_matched_support=excluded,retained=candidate_count-excluded)
 torch.save(dict(groups=saved,gt_read=False,geometry_updates=0,direction_updates=0,energy_updates=0,material_fit='closed_form_train_only'),out/'checkpoint.pt')
 result=dict(gt_read=False,geometry_updates=0,direction_updates=0,energy_updates=0,ambient=ambient,held_material_frames=sorted(held),coverage=coverage,metrics=results,held_material_perturbation_check='PASS',scope='仅材质保留预测；历史灯光拟合和匹配过程看过保留图像，非未见光照泛化或GT对应评测',inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs});(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(coverage=coverage,metrics=results),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
