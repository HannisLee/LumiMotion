"""四组Stage3统一外观评价，仅导出同视角RGB及全精度PSNR/SSIM。"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
from LIHAN.scripts.stage3_el_eldl_mixed import install_evaluation
install_evaluation()
from scripts.render_stage2 import Stage2RenderRun, save_image
from scripts.stage2_runtime import FrozenScene, supervision, file_sha256
from scene.stage2_photometric import frame_key
from utils.loss_utils import ssim

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--source',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--directions',default='');p.add_argument('--bias',type=float,default=.1);p.add_argument('--direct-multiplier',type=float,default=1.);p.add_argument('--smoke',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--samples',type=int,default=128);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 run=Stage2RenderRun(a.model,a.iteration)
 run.args.source_path=a.source
 run.args.diffuse_sample_num=a.samples
 run.args.light_t_min=a.bias
 run.args.pbr_mixed_direct_multiplier=a.direct_multiplier
 if a.directions:run.args.pbr_mixed_lights_path=a.directions
 run.context=FrozenScene(run.args,a.model,a.iteration)
 ctx=run.context
 cameras=ctx.train_cameras if a.smoke else ctx.test_cameras
 if a.smoke:cameras=[cameras[i] for i in [0,len(cameras)//2,len(cameras)-1]]
 for name in ['rgb','gt','arrays']+(['alpha','normal','separation'] if a.smoke or a.audit else []):(a.output/name).mkdir()
 rows=[];torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 with torch.no_grad():
  for cam in cameras:
   pkg=run.render(cam);target,mask=supervision(cam,ctx.background);im=pkg['render'].clamp(0,1)
   assert torch.isfinite(im).all()
   key=frame_key(cam.image_name_train_light)
   arrays=dict(rgb=im.cpu().numpy(),target=target.cpu().numpy(),mask=mask.cpu().numpy())
   np.savez_compressed(a.output/'arrays'/f'{key}.npz',**arrays)
   save_image(a.output/'rgb'/f'{key}.png',im);save_image(a.output/'gt'/f'{key}.png',target)
   if a.smoke or (a.audit and int(key) in [10,80,150]):
    save_image(a.output/'alpha'/f'{key}.png',pkg['rend_alpha'].expand(3,-1,-1))
    save_image(a.output/'normal'/f'{key}.png',(torch.nn.functional.normalize(pkg['rend_normal'],dim=0)*.5+.5)*pkg['rend_alpha'])
    from gaussian_renderer import render
    d=ctx.deformation(cam);f=ctx.pc.get_binary_feature().reshape(-1,1)
    colors=torch.cat([1-f,f,torch.zeros_like(f)],dim=1)
    sep=render(cam,ctx.pc,run.args,ctx.background,override_color=colors,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity']})['render']
    save_image(a.output/'separation'/f'{key}.png',sep)
   mse=float(np.mean((arrays['rgb'].astype(np.float64)-arrays['target'])**2))
   row=dict(frame=key,psnr=-10*math.log10(max(mse,1e-12)),ssim=float(ssim(im,target)))
   rows.append(row);print(json.dumps(row),flush=True);ctx.unload_camera(cam)
 assert all(file_sha256(Path(p))==h for p,h in ctx.source_files.items())
 report={'model':str(a.model),'iteration':a.iteration,'source':a.source,'samples':a.samples,'light_t_min':a.bias,'direct_multiplier':a.direct_multiplier,'frames':rows,'metrics':{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim']},'protocol':'400×400黑底完整浮点sRGB，逐帧计算后均值；SSIM11×11 sigma1.5；PSNR以float64独立复算','checkpoint_unchanged':True,'direction_source':a.directions or None}
 (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(report['metrics'],flush=True)
if __name__=='__main__':main()
