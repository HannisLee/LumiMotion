"""固定未知灯向/几何的点观测全局能量拟合；训练albedo解析消元，无GT。"""
import argparse,json,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--energy-model',choices=['supported','nine'],required=True);p.add_argument('--output',required=True);a=p.parse_args();r=Path(a.audit);out=Path(a.output);out.mkdir(exist_ok=False);meta=json.loads((r/'result.json').read_text());assert not meta['gt_read'];assert meta.get('ambient_mode')=='energy_scaled','必须使用与主渲染底项约定一致的新审计缓存';sources=meta['source_hashes']
 for f,h in sources.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 cache_path=next(f for f in sources if f.endswith('tracks.pt'));light_path=next(f for f in sources if 'T05' in f and f.endswith('result.json'));cache=torch.load(cache_path,map_location='cpu',weights_only=False);lights=json.loads(Path(light_path).read_text());d=torch.load(r/'diagnostics.pt',map_location='cpu',weights_only=False);assert d['train']==lights['train_indices']
 use=d['mask'][d['train']].sum(0)>=10
 unsupported=[i for i in d['train'] if not bool(d['mask'][i,use].any())]
 y=d['rgb'][:,use].cuda();mask=d['mask'][:,use].cuda();dynamic=d['dynamic'][use].cuda();normals=cache['normals'][:,use].double().cuda();directions=torch.tensor(lights['directions'],dtype=torch.float64,device='cuda');cosine=(normals*directions[:,None]).sum(-1).clamp_min(0);train=d['train'];held=d['held'];t=cache['times'].double().cuda();tt=t[train];right=torch.searchsorted(tt,t).clamp(1,len(train)-1);left=right-1;w=((t-tt[left])/(tt[right]-tt[left])).clamp(0,1);ambient=meta['ambient']
 E0=d['energy'][train].cuda();baseline_energy=d['energy'].cuda();supported_indices=[i for i in train if i not in unsupported]
 node_times=t[supported_indices] if a.energy_model=='supported' else torch.linspace(float(tt[0]),float(tt[-1]),9,dtype=t.dtype,device=t.device)
 nr=torch.searchsorted(node_times,t).clamp(1,len(node_times)-1);nl=nr-1;nw=((t-node_times[nl])/(node_times[nr]-node_times[nl])).clamp(0,1)
 design=torch.zeros((len(t),len(node_times)),dtype=t.dtype,device=t.device);design.scatter_add_(1,nl[:,None],(1-nw)[:,None]);design.scatter_add_(1,nr[:,None],nw[:,None])
 node_support=(design[train]*mask[train].sum(-1)[:,None]).sum(0);assert bool((node_support>0).all())
 raw=torch.nn.Parameter(torch.zeros(len(node_times),dtype=t.dtype,device=t.device));optimizer=torch.optim.Adam([raw],lr=.02)
 def forward():
  residual=design@raw;E=baseline_energy*torch.exp(residual);E=E/E[train].mean();s=E[:,None]*(cosine+ambient);albedo=(y[train]*s[train,:,None]*mask[train,:,None]).sum(0)/(s[train].square()*mask[train]).sum(0).clamp_min(1e-12)[:,None];return albedo[None]*s[...,None],E,albedo
 def metrics(pred):
  err=(pred-y).square().sum(-1);result={}
  for name,reg in [('dynamic',dynamic),('static',~dynamic)]:
   for split,inds in [('train',train),('held',held)]:
    m=mask[inds]&reg[None];result[name+'_'+split]=float(err[inds][m].mean()/3)
  return result
 def loss(pred):
  err=(pred[train]-y[train]).square().sum(-1)/3;return .5*err[mask[train]&dynamic[None]].mean()+.5*err[mask[train]&~dynamic[None]].mean()
 before=metrics(forward()[0]);history=[]
 for step in range(200):
  optimizer.zero_grad();pred,_,_=forward();objective=loss(pred);assert torch.isfinite(objective);objective.backward();assert torch.isfinite(raw.grad).all();optimizer.step()
  if (step+1)%50==0:history.append(dict(step=step+1,objective=float(objective),metrics=metrics(forward()[0])));print(json.dumps(history[-1]),flush=True)
 with torch.no_grad():
  pred,E,albedo=forward();after=metrics(pred);state=raw.clone();old_y=y.clone();y[held]=.123456;_,E_test,alb_test=forward();isolation=max(float((E-E_test).abs().max()),float((albedo-alb_test).abs().max()));y.copy_(old_y);assert isolation==0
 torch.save(dict(energy_model=a.energy_model,node_times=node_times.cpu(),design=design.cpu(),baseline_energy=baseline_energy.cpu(),raw_energy=raw.detach().cpu(),energy=E.cpu(),albedo=albedo.cpu(),mask=mask.cpu(),point_ids=cache['ids'][use],prediction=pred.cpu(),optimizer=optimizer.state_dict()),out/'checkpoint.pt')
 restored=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);raw.data.copy_(restored['raw_energy'].cuda());reload_error=float((forward()[0]-pred).abs().max());assert reload_error==0
 result=dict(status='PASS',gt_read=False,steps=200,ambient_mode='energy_scaled',unsupported_train_indices=unsupported,protocol_status='SUPPORTED_NODES',energy_model=a.energy_model,node_count=len(node_times),node_weighted_support=node_support.cpu().tolist(),before=before,after=after,history=history,energy_min=float(E[train].min()),energy_max=float(E[train].max()),energy_rms_change=float((E[train]-E0).square().mean().sqrt()),train_mean_energy=float(E[train].mean()),held_rgb_isolation=isolation,reload_error=reload_error,source_audit_sha256=hashlib.sha256((r/'diagnostics.pt').read_bytes()).hexdigest(),scope='只拟合有训练支持的时间节点光强残差，固定未知方向和默认轨迹法线；颜色albedo训练解析消元；无几何更新、无GT、非光栅渲染结果')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
