"""双精度固定Gaussian可见性：平面与动态观测配对灯向更新，无GT训练。"""
import argparse,json,hashlib,math
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts.audit_dl_visibility_precision import stable_visibility
from scripts.fit_dl_shadow_tracks import interpolate

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['source','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();torch.set_num_threads(4);root=Path(a.source);out=Path(a.output);out.mkdir(exist_ok=False)
 source_meta=json.loads((root/'train/result.json').read_text());assert not source_meta['gt_read']
 ck=torch.load(root/'train/checkpoint.pt',map_location='cpu',weights_only=False)
 cache_path=Path('output/1015-07-LH-dynamic-DL-T02表面轨迹初始化/cache/tracks.pt');shadow_path=Path('output/1015-15-LH-dynamic-DL-T05复核/seed_1015/checkpoint.pt');data_path=Path('output/smoke_test/1015-50-LH-dynamic-DL-T58接收平面可见性/audit/predictions.pt');energy_path=Path('output/smoke_test/1015-49-LH-dynamic-DL-T57底项约定修正配对/supported/checkpoint.pt');audit_path=energy_path.parent.parent/'audit/diagnostics.pt'
 cache=torch.load(cache_path,weights_only=False);shadow=torch.load(shadow_path,weights_only=False);data=torch.load(data_path,weights_only=False);energyfit=torch.load(energy_path,weights_only=False);audit=torch.load(audit_path,weights_only=False);ambient=json.loads(audit_path.with_name('result.json').read_text())['ambient']
 assert hashlib.sha256(cache_path.read_bytes()).hexdigest()==shadow['cache_sha256']
 rid=ck['receiver_ids'];plane_mask=data['groups']['all_unsaturated']['mask'][:,rid].cuda();plane_y=data['rgb'][:,rid].cuda();plane_albedo=ck['albedo'].cuda();plane_normals=cache['normals'][:,rid].double().cuda()
 lookup={int(v):i for i,v in enumerate(cache['ids'])};energy_ids=torch.tensor([lookup[int(v)] for v in energyfit['point_ids']]);dynamic=audit['dynamic'][energy_ids];did=energy_ids[dynamic];dyn_mask=energyfit['mask'][:,dynamic].cuda();dyn_y=cache['rgb'][:,did].double().cuda();dyn_albedo=energyfit['albedo'][dynamic].cuda();dyn_normals=cache['normals'][:,did].double().cuda()
 E=energyfit['energy'].cuda();n=shadow['plane_normal'].double().cuda();basis=shadow['basis'].double().cuda();occ=cache['ids'][shadow['occluder']];geometry=[]
 for pos,rot,scale,opacity,query in ck['geometry']:geometry.append((pos[occ].double().cuda(),(rot[occ].double()*scale[occ,None,:].double()).cuda(),opacity.double().cuda(),query.double().cuda()))
 old=F.normalize(ck['initial_directions'].double().cuda(),dim=-1);train=torch.where(shadow['train'])[0].tolist();held=torch.where(~shadow['train'])[0].tolist();times=cache['times'].double().cuda().reshape(-1)
 def predict(i,l):
  pos,tangent,opacity,query=geometry[i];projection=torch.eye(3,dtype=l.dtype,device=l.device)-l[:,None]*n[None]/(l@n)
  v,_=stable_visibility(query,pos@projection.T@basis.T,basis[None]@projection[None]@tangent,opacity)
  plane=plane_albedo*(E[i]*(v*(plane_normals[i]*l).sum(-1).clamp_min(0)+ambient))[:,None]
  dyn=dyn_albedo*(E[i]*((dyn_normals[i]*l).sum(-1).clamp_min(0)+ambient))[:,None]
  return plane,dyn
 with torch.no_grad():initial_plane,initial_dyn=map(torch.stack,zip(*(predict(i,old[i]) for i in range(135))))
 counts=[int(plane_mask[train].sum()),int(dyn_mask[train].sum())]
 scales=[float((initial_plane-plane_y)[train][plane_mask[train]].square().mean()),float((initial_dyn-dyn_y)[train][dyn_mask[train]].square().mean())]
 assert all(x>0 for x in scales)
 def components(i,l):
  plane,dyn=predict(i,l)
  a=(plane-plane_y[i])[plane_mask[i]].square().sum()/(3*counts[0]*scales[0])
  b=(dyn-dyn_y[i])[dyn_mask[i]].square().sum()/(3*counts[1]*scales[1])
  return a,b
 def metrics(plane,dyn):
  rows={}
  for split,ts in [('train',train),('held',held)]:
   for label,pred,y,mask in [('plane',plane,plane_y,plane_mask),('dynamic',dyn,dyn_y,dyn_mask),('plane_strict',plane,plane_y,data['groups']['strict_bright']['mask'][:,rid].cuda())]:
    rows[split+'_'+label]=dict(count=int(mask[ts].sum()),mse=float((pred-y)[ts][mask[ts]].square().mean()))
  return rows
 results={};saved={}
 for mode in ['plane','balanced']:
  directions=old.clone();rows=[]
  def objective(i,l):
   a,b=components(i,l);return a if mode=='plane' else .5*(a+b)
  for i in train:
   l=old[i].clone();accepted=0
   with torch.no_grad():before=float(objective(i,l))
   for step in range(5):
    variable=l.detach().requires_grad_(True);value=objective(i,variable);grad,=torch.autograd.grad(value,variable);tangent=grad-l*(grad@l);norm=tangent.norm();assert torch.isfinite(norm)
    if norm<1e-12:break
    direction=-tangent/norm
    with torch.no_grad():
     for k in range(12):
      angle=math.radians(.25)*.5**k;candidate=F.normalize(l*math.cos(angle)+direction*math.sin(angle),dim=0);score=float(objective(i,candidate))
      if math.isfinite(score) and score<float(value)-1e-14:l=candidate;accepted+=1;break
     else:break
   with torch.no_grad():
    directions[i]=l;angle=float(torch.rad2deg(torch.acos((l@old[i]).clamp(-1,1))));assert angle<1.251
    row=dict(frame=i,before=before,after=float(objective(i,l)),accepted=accepted,angle=angle,dynamic_observations=int(dyn_mask[i].sum()));rows.append(row);print(json.dumps(dict(mode=mode,**row)),flush=True)
  with torch.no_grad():
   directions[held]=interpolate(times,times[train],directions[train])[held]
   planes,dynamics=map(torch.stack,zip(*(predict(i,directions[i]) for i in range(135))))
   results[mode]=dict(metrics=metrics(planes,dynamics),accepted=sum(x['accepted'] for x in rows),max_angle=max(x['angle'] for x in rows))
   saved[mode]=dict(directions=directions.cpu(),plane=planes.cpu(),dynamic=dynamics.cpu(),rows=rows)
   # 每组立即保存，避免第二组失败丢失已完成结果。
   torch.save(saved[mode],out/(mode+'.pt'))
  print(json.dumps(dict(mode=mode,result=results[mode])),flush=True)
 result=dict(status='PASS',gt_read=False,geometry_updates=0,material_updates=0,normalization=dict(training_counts=counts,initial_training_mse=scales,weights=[.5,.5]),initial=metrics(initial_plane,initial_dyn),groups=results,missing_dynamic_frames=[i for i in train if not dyn_mask[i].any()],scope='双精度局部坐标，固定T61/T57材质和能量，平面/动态训练SSE按各区域训练数及初始MSE归一化；held只插值，未增加对应机制或几何更新',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'train/checkpoint.pt',cache_path,shadow_path,data_path,energy_path,audit_path,Path(__file__),Path('scripts/audit_dl_visibility_precision.py')]})
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
