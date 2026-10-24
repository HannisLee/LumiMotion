"""动态EL原生Stage1：完整15测试帧只导出数值，不输出图片。"""
from pathlib import Path
import argparse,json,math
import numpy as np
import torch
from scripts.eval_albedo_gt import AlbedoRun
from gaussian_renderer import render
from utils.loss_utils import ssim
from pytorch_msssim import ssim as original_ssim
from scene.stage2_photometric import frame_key

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--source',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 run=AlbedoRun(a.model,35000,a.source);ctx=run.context;rows=[]
 with torch.no_grad():
  for cam in ctx.test_cameras:
   ctx.load_camera(cam);d=ctx.deformation(cam);pkg=render(cam,ctx.pc,run.config,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']});img=pkg['render'].clamp(0,1);target=cam.original_image_train_light.to('cuda').clamp(0,1);mask=(cam.gt_alpha_mask.to('cuda')>.5).float();mse=(img-target).square();row=dict(frame=frame_key(cam.image_name_train_light),psnr=-10*math.log10(max(float(mse.mean()),1e-12)),ssim=float(ssim(img,target)),ssim_original_impl=float(original_ssim(img[None],target[None],data_range=1.)),psnr_foreground=-10*math.log10(max(float((mse*mask).sum()/(3*mask.sum())),1e-12)))
   rows.append(row);ctx.unload_camera(cam);print(json.dumps(row),flush=True)
 assert sorted(int(x['frame']) for x in rows)==list(range(10,151,10));run.verify_unchanged()
 result=dict(model=str(a.model),iteration=35000,source=a.source,frames=rows,metrics={k:float(np.mean([x[k] for x in rows])) for k in ['psnr','ssim','ssim_original_impl','psnr_foreground']},protocol='固定15TEST、400x400、float sRGB clamp[0,1]黑底全图逐帧均值；主SSIM使用utils.loss_utils，与本次静态Stage2表一致；另保留原日志pytorch_msssim实现的全15结果',checkpoint_sha256=run.checkpoint_hashes,checkpoint_unchanged=True)
 (a.output/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['metrics'])
if __name__=='__main__':main()
