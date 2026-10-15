"""无GT接收平面法线/光强配对；固定几何光向，训练共享材质及可选九节点光强。"""
import argparse,json,hashlib
from pathlib import Path
import torch
from scripts.audit_dl_visibility_precision import stable_visibility

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4);out=Path(a.output);out.mkdir(exist_ok=False)
 root=Path(a.reference);meta=json.loads((root/'train/result.json').read_text());assert not meta['gt_read'];ref=torch.load(root/'train/checkpoint.pt',map_location='cpu',weights_only=False)
 paths=[Path('output/1015-07-LH-dynamic-DL-T02表面轨迹初始化/cache/tracks.pt'),Path('output/1015-15-LH-dynamic-DL-T05复核/seed_1015/checkpoint.pt'),Path('output/smoke_test/1015-50-LH-dynamic-DL-T58接收平面可见性/audit/predictions.pt'),Path('output/smoke_test/1015-49-LH-dynamic-DL-T57底项约定修正配对/supported/checkpoint.pt'),Path('output/smoke_test/1015-49-LH-dynamic-DL-T57底项约定修正配对/audit/diagnostics.pt')]
 cache,state,data,efit,audit=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];assert hashlib.sha256(paths[0].read_bytes()).hexdigest()==state['cache_sha256'];ambient=json.loads(paths[-1].with_name('result.json').read_text())['ambient'];rid=ref['receiver_ids'];train=data['train'];held=data['held'];dirs=ref['initial_directions'].double().cuda();n=state['plane_normal'].double().cuda();b=state['basis'].double().cuda();occ=cache['ids'][state['occluder']];vis=[]
 with torch.no_grad():
  for i,(pos,rot,scale,op,q) in enumerate(ref['geometry']):
   pos=pos[occ].double().cuda();tan=(rot[occ].double()*scale[occ,None,:].double()).cuda();l=dirs[i];proj=torch.eye(3,dtype=l.dtype,device='cuda')-l[:,None]*n[None]/(l@n);v,_=stable_visibility(q.double().cuda(),pos@proj.T@b.T,b[None]@proj[None]@tan,op.double().cuda());vis.append(v)
 V=torch.stack(vis);y=data['rgb'][:,rid].cuda();mask=data['groups']['all_unsaturated']['mask'][:,rid].cuda();E0=efit['energy'].cuda();cos_head=(cache['normals'][:,rid].double().cuda()*dirs[:,None]).sum(-1).clamp_min(0);cos_plane=(dirs@n).clamp_min(0)[:,None].expand_as(cos_head)
 lookup={int(v):i for i,v in enumerate(cache['ids'])};eids=torch.tensor([lookup[int(v)] for v in efit['point_ids']]);dyn=audit['dynamic'][eids];did=eids[dyn];dy=cache['rgb'][:,did].double().cuda();dm=efit['mask'][:,dyn].cuda();dc=(cache['normals'][:,did].double().cuda()*dirs[:,None]).sum(-1).clamp_min(0)
 time=cache['times'].double().cuda().reshape(-1);nodes=torch.linspace(float(time[train].min()),float(time[train].max()),9,dtype=time.dtype,device='cuda');right=torch.searchsorted(nodes,time).clamp(1,8);left=right-1;w=((time-nodes[left])/(nodes[right]-nodes[left])).clamp(0,1);design=torch.zeros(135,9,dtype=time.dtype,device='cuda');design.scatter_add_(1,left[:,None],(1-w)[:,None]);design.scatter_add_(1,right[:,None],w[:,None]);assert ((design[train]*mask[train].sum(-1)[:,None]).sum(0)>0).all()
 def fit(rgb,s,m):return (rgb[train]*s[train,:,None]*m[train,:,None]).sum(0)/(s[train].square()*m[train]).sum(0).clamp_min(1e-12)[:,None]
 results={};saved={}
 for normal,cos in [('head',cos_head),('plane',cos_plane)]:
  for mode in ['fixed','nine_nodes']:
   raw=torch.nn.Parameter(torch.zeros(9,dtype=time.dtype,device='cuda'));optimizer=torch.optim.Adam([raw],lr=.02)
   def forward():
    E=E0 if mode=='fixed' else E0*torch.exp(design@raw)
    if mode!='fixed':E=E/E[train].mean()
    s=E[:,None]*(V*cos+ambient);alb=fit(y,s,mask);return alb[None]*s[...,None],E,alb
   history=[]
   for step in range(200 if mode=='nine_nodes' else 0):
    optimizer.zero_grad();pred,_,_=forward();loss=(pred-y)[train][mask[train]].square().mean();assert torch.isfinite(loss);loss.backward();assert torch.isfinite(raw.grad).all();optimizer.step()
    if step%50==49:history.append(dict(step=step+1,train_mse=float(loss)));print(json.dumps(dict(normal=normal,mode=mode,**history[-1])),flush=True)
   with torch.no_grad():
    pred,E,albedo=forward();ds=E[:,None]*(dc+ambient);dalbedo=fit(dy,ds,dm);dp=dalbedo[None]*ds[...,None];rows={}
    for split,ts in [('train',train),('held',held)]:
     for label,prediction,rgb,m in [('plane',pred,y,mask),('strict_plane',pred,y,data['groups']['strict_bright']['mask'][:,rid].cuda()),('dynamic',dp,dy,dm)]:rows[split+'_'+label]=dict(count=int(m[ts].sum()),mse=float((prediction-rgb)[ts][m[ts]].square().mean()))
    alt=y.clone();alt[held]=.456;assert torch.equal(fit(alt,E[:,None]*(V*cos+ambient),mask),albedo)
    key=normal+'/'+mode;results[key]=dict(metrics=rows,energy_min=float(E.min()),energy_max=float(E.max()),updates=200 if mode=='nine_nodes' else 0,history=history);saved[key]=dict(prediction=pred.cpu(),energy=E.cpu(),albedo=albedo.cpu(),dynamic_prediction=dp.cpu(),dynamic_albedo=dalbedo.cpu(),raw=raw.detach().cpu())
    print(json.dumps(dict(group=key,result=results[key])),flush=True)
 torch.save(dict(groups=saved,visibility=V.cpu(),receiver_ids=rid,design=design.cpu(),nodes=nodes.cpu(),gt_read=False),out/'checkpoint.pt');result=dict(status='PASS',gt_read=False,geometry_updates=0,light_direction_updates=0,statistics=results,scope='固定T05未知方向/初始粗几何/同支持；平面法线由预测几何拟合，非GT。动态材质每组train-only重新拟合，动态未进入光强目标；不是新几何或对应增量',source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[root/'train/checkpoint.pt',Path(__file__)]});(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
