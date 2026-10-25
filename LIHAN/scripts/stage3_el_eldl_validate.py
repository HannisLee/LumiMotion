"""独立核验四组最终浮点指标、真实父模型继承和完整15帧。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
import torch
from LIHAN.scripts.stage3_ppt_validate import independent_ssim

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args()
 state=json.loads((a.experiment/'state.json').read_text());r=json.loads((a.evaluation/'metrics.json').read_text());model=Path(r['model']);it=r['iteration']
 source=json.loads((model/'stage1_source.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in source.items())
 parent=Path(next(p for p in source if p.endswith('.ply')))
 old=PlyData.read(str(parent))['vertex'].data;new=PlyData.read(str(model/f'point_cloud/iteration_{it}/point_cloud.ply'))['vertex'].data
 assert len(old)==len(new)
 frozen=[n for n in old.dtype.names if n in ['x','y','z','nx','ny','nz'] or n.startswith(('scale_','rot_','f_feature_','fea_'))]
 assert all(np.array_equal(old[n],new[n]) for n in frozen)
 old_deform=Path(next(p for p in source if p.endswith('.pth')))
 x=torch.load(old_deform,map_location='cpu',weights_only=False);y=torch.load(model/f'deform/iteration_{it}/deform.pth',map_location='cpu',weights_only=False)
 assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)
 records=[]
 for row in r['frames']:
  with np.load(a.evaluation/'arrays'/f"{row['frame']}.npz") as d:
   assert all(np.isfinite(d[k]).all() for k in d.files)
   psnr=float(-10*np.log10(max(np.mean((d['rgb'].astype(np.float64)-d['target'])**2),1e-12)))
   ss=independent_ssim(d['rgb'],d['target'])
  assert abs(psnr-row['psnr'])<1e-4 and abs(ss-row['ssim'])<1e-4,(row,psnr,ss)
  records.append({'frame':row['frame'],'psnr':psnr,'ssim':ss})
 assert sorted(int(x['frame']) for x in records)==list(range(10,151,10))
 result={'status':'PASS','scope':'工程与指标有效，不代表ELDL胜出','geometry_frozen':frozen,'gaussians':len(new),'deform_tensors_equal':len(x),'parent_hashes_unchanged':True,'frames':records,'numpy_scipy_metrics':{k:float(np.mean([v[k] for v in records])) for k in ['psnr','ssim']}}
 (a.evaluation/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['numpy_scipy_metrics'])
if __name__=='__main__':main()
