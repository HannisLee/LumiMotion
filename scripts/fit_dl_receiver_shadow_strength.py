"""固定接收平面投影的全局遮挡强度训练选择；无GT、无几何更新。"""
import argparse,json,hashlib
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4);r=Path(a.audit);out=Path(a.output);out.mkdir(exist_ok=False);meta=json.loads((r/'result.json').read_text());assert not meta['gt_read'] and meta['ambient_mode']=='energy_scaled'
 for path,sha in meta['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 d=torch.load(r/'predictions.pt',map_location='cpu',weights_only=False);sources=meta['source_hashes'];cache_path=next(p for p in sources if p.endswith('tracks.pt'));light_path=next(p for p in sources if 'T05' in p and p.endswith('result.json'));energy_path=next(p for p in sources if 'T57' in p and p.endswith('checkpoint.pt'));cache=torch.load(cache_path,map_location='cpu',weights_only=False);light=json.loads(Path(light_path).read_text());energy=torch.load(energy_path,map_location='cpu',weights_only=False)['energy'].double();cos=(cache['normals'].double()*torch.tensor(light['directions'],dtype=torch.float64)[:,None]).sum(-1).clamp_min(0);y=d['rgb'];train=d['train'];held=d['held'];P=d['blocked'].double();audit_data_path=next(p for p in sources if 'T57' in p and p.endswith('diagnostics.pt'));lighting_meta=json.loads(Path(audit_data_path).with_name('result.json').read_text());assert lighting_meta['ambient_mode']=='energy_scaled';ambient=lighting_meta['ambient'];assert not (d['groups']['strict_bright']['mask']&~d['groups']['all_unsaturated']['mask']).any();results={};saved={}
 for label,g in d['groups'].items():
  m=g['mask'];grid=[];fits=[]
  for j in range(21):
   strength=j/20;s=energy[:,None]*((1-strength*P)*cos+ambient);albedo=(y[train]*s[train,:,None]*m[train,:,None]).sum(0)/(s[train].square()*m[train]).sum(0).clamp_min(1e-12)[:,None];pred=s[...,None]*albedo[None];score=float((pred[train]-y[train])[m[train]].square().mean());grid.append(dict(strength=strength,training_mse=score));fits.append((pred,albedo))
  best=min(range(len(grid)),key=lambda j:grid[j]['training_mse']);pred,albedo=fits[best];rows={}
  for split,inds in [('train',train),('held',held)]:
   for subset,extra in [('all',torch.ones_like(m)),('blocked',d['blocked']),('unblocked',~d['blocked']),('strict_subset',d['groups']['strict_bright']['mask'])]:
    selected=m[inds]&extra[inds];n=int(selected.sum());rows[split+'_'+subset]=dict(count=n,mse=float((pred[inds]-y[inds])[selected].square().mean()) if n else None)
  # 网格端点应精确复现先前硬遮挡/无遮挡解析拟合。
  for endpoint,mode in [(0,'no_visibility'),(20,'projected_visibility')]:assert torch.equal(fits[endpoint][0],g['predictions'][mode])
  results[label]=dict(strength=grid[best]['strength'],grid=grid,metrics=rows);saved[label]=dict(prediction=pred,albedo=albedo,strength=grid[best]['strength'])
 torch.save(saved,out/'checkpoint.pt');result=dict(status='PASS',gt_read=False,gradient_updates=0,analytic_fits=42,source_audit_sha256=hashlib.sha256((r/'predictions.pt').read_bytes()).hexdigest(),groups=results,scope='每种固定支持独立用train选择21点网格全局强度；非物理透射率真值、非几何改善、不可推广动态人体')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
