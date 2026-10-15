"""固定T05接收平面上的投影遮挡RGB对照；解析材质拟合，无GT或几何更新。"""
import argparse,json,hashlib
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--energy-run',required=True);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(4)
 audit=Path(a.audit);meta=json.loads((audit/'result.json').read_text());assert not meta['gt_read'] and meta['ambient_mode']=='energy_scaled';d=torch.load(audit/'diagnostics.pt',map_location='cpu',weights_only=False);src=meta['source_hashes'];cache_path=next(f for f in src if f.endswith('tracks.pt'));light_path=next(f for f in src if 'T05' in f and f.endswith('result.json'));cache=torch.load(cache_path,map_location='cpu',weights_only=False);light=json.loads(Path(light_path).read_text());shadow=torch.load(a.shadow,map_location='cpu',weights_only=False);assert not light['gt_read'];assert shadow['cache_sha256']==light['cache_sha256']==hashlib.sha256(Path(cache_path).read_bytes()).hexdigest()
 energy_run=Path(a.energy_run);er=json.loads((energy_run/'result.json').read_text());assert er['ambient_mode']=='energy_scaled' and not er['gt_read'];e=torch.load(energy_run/'checkpoint.pt',map_location='cpu',weights_only=False);E=e['energy'].double();dirs=torch.tensor(light['directions'],dtype=torch.float64);assert torch.allclose(shadow['smooth'].double(),dirs,atol=1e-7);assert er['source_audit_sha256']==hashlib.sha256((audit/'diagnostics.pt').read_bytes()).hexdigest()
 train=d['train'];held=d['held'];y=cache['rgb'].double();cos=(cache['normals'].double()*dirs[:,None]).sum(-1).clamp_min(0);blocked=shadow['predicted']['smooth'].bool();assert blocked.shape==cache['valid'].shape;receiver=shadow['receiver'].bool();ambient=meta['ambient'];base=cache['valid']&~cache['saturated'].any(-1)&receiver[None];broader=base&(base[train].sum(0)>=10)[None];strict=d['mask']&receiver[None];results={};saved={}
 for label,mask in [('strict_bright',strict),('all_unsaturated',broader)]:
  group={};predictions={}
  for mode,v in [('no_visibility',torch.ones_like(cos)),('projected_visibility',(~blocked).double())]:
   shading=E[:,None]*(v*cos+ambient);albedo=(y[train]*shading[train,:,None]*mask[train,:,None]).sum(0)/(shading[train].square()*mask[train]).sum(0).clamp_min(1e-12)[:,None];prediction=albedo[None]*shading[...,None];predictions[mode]=prediction;err=(prediction-y).square().sum(-1);signed=(prediction-y).mean(-1);rows={}
   for split,inds in [('train',train),('held',held)]:
    for subset,extra in [('all',torch.ones_like(mask)),('blocked',blocked),('unblocked',~blocked)]:
     m=mask[inds]&extra[inds];n=int(m.sum());rows[split+'_'+subset]=dict(observations=n,points=int(m.any(0).sum()),mse=float(err[inds][m].mean()/3) if n else None,signed_mean=float(signed[inds][m].mean()) if n else None)
   # 独立验证解析拟合只使用train：不读取任何held项。
   altered=y.clone();altered[held]=.321;check=(altered[train]*shading[train,:,None]*mask[train,:,None]).sum(0)/(shading[train].square()*mask[train]).sum(0).clamp_min(1e-12)[:,None];assert torch.equal(check,albedo)
   group[mode]=rows
  results[label]=group;saved[label]=dict(mask=mask,predictions=predictions)
 torch.save(dict(groups=saved,blocked=blocked,receiver=receiver,rgb=y,train=train,held=held,ids=cache['ids']),out/'predictions.pt')
 hashes={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [cache_path,light_path,a.shadow,str(energy_run/'checkpoint.pt'),str(audit/'diagnostics.pt'),__file__]}
 result=dict(status='PASS',gt_read=False,gradient_updates=0,receiver_points=int(receiver.sum()),ambient_mode='energy_scaled',energy_frozen=True,light_frozen=True,statistics=results,source_hashes=hashes,scope='T05预测接收平面/二值膨胀投影代理，非完整光源可见性；同支持train-only解析albedo。T57光强已利用严格支持的训练RGB，非独立端到端验证；不推广动态人体')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
