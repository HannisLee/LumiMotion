"""固定15测试视角评测展示Stage2；保存浮点RGB与四类可视化。"""
from pathlib import Path
import argparse,json,math
import numpy as np
import torch
from scripts.render_stage2 import Stage2RenderRun,save_image
from scripts.stage2_runtime import supervision,file_sha256
from scene.stage2_photometric import frame_key
from utils.loss_utils import ssim

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);run=Stage2RenderRun(a.model,a.iteration);rows=[]
 for name in ['rgb','gt','alpha','normal','albedo','arrays']:(a.output/name).mkdir()
 torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 with torch.no_grad():
  for cam in run.context.test_cameras:
   pkg=run.render(cam);target,mask=supervision(cam,run.context.background);img=pkg['render'].clamp(0,1);key=frame_key(cam.image_name_train_light)
   mse=(img-target).square();fg=float((mse*mask).sum()/(3*mask.sum()));row=dict(frame=key,psnr=-10*math.log10(max(float(mse.mean()),1e-12)),ssim=float(ssim(img,target)),psnr_foreground=-10*math.log10(max(fg,1e-12)))
   normal=torch.nn.functional.normalize(pkg['rend_normal'],dim=0);alpha=pkg['rend_alpha']
   for name,tensor in [('rgb',img),('gt',target),('alpha',alpha.expand(3,-1,-1)),('normal',(normal*.5+.5)*alpha),('albedo',pkg['base_color'])]:save_image(a.output/name/(key+'.png'),tensor)
   np.savez_compressed(a.output/'arrays'/(key+'.npz'),rgb=img.cpu().numpy(),target=target.cpu().numpy(),mask=mask.cpu().numpy(),normal=normal.cpu().numpy(),alpha=alpha.cpu().numpy())
   rows.append(row);run.context.unload_camera(cam);print(json.dumps(row),flush=True)
 assert sorted(int(r['frame']) for r in rows)==list(range(10,151,10))
 report=dict(model=str(a.model),iteration=a.iteration,frames=rows,metrics={k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim','psnr_foreground']},protocol='固定15测试，400x400，浮点sRGB clamp[0,1]，黑底全图逐帧PSNR/SSIM后取均值；前景PSNR单列；无TEST调参',config=vars(run.args))
 (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report['metrics'])
if __name__=='__main__':main()
