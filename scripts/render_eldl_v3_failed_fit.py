"""失败全图校准的固定fit代表图归档，不读取held指标或更新参数。"""
import argparse
import math
import socket
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts import calibrate_eldl_v3_raster as f

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('cache','fit','model','source','raster-cache','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--iteration',type=int,default=16000);a=p.parse_args()
 for key,value in vars(a).items():
  if isinstance(value,Path):setattr(a,key,value.resolve())
 assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==v.PYTHON.resolve()
 a.output.mkdir(parents=True,exist_ok=False)
 from utils.no_gt_guard import install_no_gt_guard
 install_no_gt_guard([a.cache.parent.parent,a.fit,a.model,a.source,a.output])
 parent=v.read(a.checkpoint.parent.parent/'endpoint_selection.json');assert parent['status']=='OPTIMIZATION_BUDGET_INSUFFICIENT'
 assert a.checkpoint.parent.name==parent['selected']
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);packed=torch.load(a.raster_cache,map_location='cpu',weights_only=False)
 pc,cameras,pipe,data,state,hashes=f.load(a)
 assert torch.equal(saved['directions'],state['directions'])
 from scene.stage2_photometric import linear_to_srgb
 rho=saved['rho'].cuda();environment=saved['environment'].cuda();q=packed['q'][1].cuda()
 def png(path,x):
  arr=x.clamp(0,1).cpu().permute(1,2,0).numpy()
  if arr.shape[2]==1:arr=np.repeat(arr,3,2)
  cv2.imwrite(str(path),(arr[...,::-1]*255+.5).astype(np.uint8))
 rows=[]
 for frame in (1,81,149):
  i=v.TRAIN.index(frame);camera=cameras[i];out=a.output/f'frame_{frame:04d}';out.mkdir()
  package=f.render(camera,pc,pipe,rho*(environment/math.pi+q[i]));pred=linear_to_srgb(package['render'].clamp_min(0));observed=camera.original_image_train_light.cuda()
  png(out/'RGB.png',torch.cat([observed,pred,(pred-observed).abs()*3],2))
  png(out/'Alpha.png',torch.cat([camera.gt_alpha_mask.cuda(),package['rend_alpha']],2))
  png(out/'Normal.png',(torch.nn.functional.normalize(package['rend_normal'],dim=0)+1)/2*package['rend_alpha'])
  env=f.render(camera,pc,pipe,rho*environment/math.pi)['render'];direct=f.render(camera,pc,pipe,rho*q[i])['render']
  png(out/'Separation.png',torch.cat([linear_to_srgb(env.clamp_min(0)),linear_to_srgb(direct.clamp_min(0)),pred],2))
  rows.append(dict(frame=frame,linear_censored_mse=float(f.objective(package['render'],camera))))
 v.write(a.output/'summary.json',dict(status='FAILED_OPTIMIZATION_BUDGET_FIT_ARCHIVE_ONLY',gt_read=False,held_rendered=False,source_checkpoint=str(a.checkpoint),frames=rows,geometry_updates=0,internal_validation_passed=False,images_sha256={str(p):v.sha(p) for p in sorted(a.output.rglob('*.png'))}))

if __name__=='__main__':main()
