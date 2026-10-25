"""完整18DEV选择偏移及正方向光倍率；冻结权重，不读取TEST评分。"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
from LIHAN.scripts.stage3_el_eldl_mixed import install_evaluation
install_evaluation()
from scripts.render_stage2 import Stage2RenderRun,save_image
from scripts.stage2_runtime import FrozenScene,supervision,file_sha256
from scene.stage2_photometric import frame_key
from utils.graphics_utils import rgb_to_srgb
from utils.loss_utils import ssim

def metrics(rows):return {k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim']}
def row(image,target,key):
 return {'frame':key,'psnr':-10*math.log10(max(float((image-target).square().mean()),1e-12)),'ssim':float(ssim(image,target))}
def choose(values,base):
 eligible=[v for v in values if v['metrics']['psnr']>=base['metrics']['psnr']+.05 and v['metrics']['ssim']>=base['metrics']['ssim']+.0001]
 return max(eligible,key=lambda v:(v['metrics']['ssim'],v['metrics']['psnr'])) if eligible else base

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--source',required=True);p.add_argument('--directions',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 run=Stage2RenderRun(a.model,a.iteration);run.args.source_path=a.source;run.args.pbr_mixed_lights_path=a.directions;run.args.diffuse_sample_num=512;run.args.pbr_mixed_direct_multiplier=1.
 run.context=FrozenScene(run.args,a.model,a.iteration);ctx=run.context;cameras=ctx.test_cameras
 assert len(cameras)==18 and all(61<=int(frame_key(c.image_name_train_light))<=79 for c in cameras)
 values=[];cache={}
 with torch.no_grad():
  for bias in [.1,.01,.03,.2]:
   run.args.light_t_min=bias;torch.manual_seed(0);torch.cuda.manual_seed_all(0);rows=[];cache[bias]={}
   directory=a.output/str(bias);directory.mkdir()
   for cam in cameras:
    package=run.render(cam);target,mask=supervision(cam,ctx.background);im=package['render'].clamp(0,1);assert torch.isfinite(im).all()
    key=frame_key(cam.image_name_train_light);rows.append(row(im,target,key))
    cache[bias][key]=package['render_linear_unpremultiplied'].detach().cpu()
    if int(key) in [61,71,79]:save_image(directory/f'{key}.png',im)
    ctx.unload_camera(cam)
   value={'bias':bias,'direct_multiplier':1.,'frames':rows,'metrics':metrics(rows)};values.append(value);print(json.dumps(value,ensure_ascii=False),flush=True)
  selected_bias=choose(values[1:],values[0]);bias=selected_bias['bias'];run.args.light_t_min=bias
  # PBR在光强上严格线性：仅多渲染一次零SUN环境分量，在线性辐亮度重建倍率。
  # 测试采样Fibonacci无随机旋转；最终另以真实前向验证3帧重建一致性。
  scales=[.5,.75,1.,1.25,1.5,2.];scale_rows={m:[] for m in scales};components={}
  for cam in cameras:
   key=frame_key(cam.image_name_train_light);run.args.pbr_mixed_direct_multiplier=0.
   env=run.render(cam);target,mask=supervision(cam,ctx.background)
   zero=env['render_linear_unpremultiplied'];one=cache[bias][key].to('cuda');alpha=env['rend_alpha'];components[key]=(zero.cpu(),one.cpu(),alpha.cpu())
   for m in scales:
    im=(rgb_to_srgb((zero+m*(one-zero)).clamp_min(0))*alpha).clamp(0,1)
    scale_rows[m].append(row(im,target,key))
   ctx.unload_camera(cam)
  calibration=[{'bias':bias,'direct_multiplier':m,'frames':scale_rows[m],'metrics':metrics(scale_rows[m])} for m in scales]
  base=next(v for v in calibration if v['direct_multiplier']==1.)
  assert all(abs(base['metrics'][k]-selected_bias['metrics'][k])<1e-4 for k in ['psnr','ssim'])
  selected=choose([v for v in calibration if v['direct_multiplier']!=1.],base)
  run.args.pbr_mixed_direct_multiplier=selected['direct_multiplier'];parity={};directory=a.output/'selected';directory.mkdir()
  for cam in cameras:
   key=frame_key(cam.image_name_train_light)
   if int(key) not in [61,71,79]:continue
   pkg=run.render(cam);zero,one,alpha=(x.to('cuda') for x in components[key]);m=selected['direct_multiplier']
   reconstructed=(rgb_to_srgb((zero+m*(one-zero)).clamp_min(0))*alpha).clamp(0,1)
   actual=pkg['render'].clamp(0,1);error=float((actual-reconstructed).abs().max());assert error<1e-4,(key,error)
   parity[key]=error;save_image(directory/f'{key}.png',actual);ctx.unload_camera(cam)
 assert all(file_sha256(Path(p))==sha for p,sha in ctx.source_files.items())
 result={'status':'COMPLETED_DEV_ONLY','model':str(a.model),'iteration':a.iteration,'samples':512,'values':values,'intensity_calibration':calibration,'selected':selected,'linear_reconstruction_parity_max_abs':parity,'criterion':'先选bias，再选正SUN倍率；每步相对原值DEV PSNR至少+0.05dB且SSIM至少+0.0001，否则原值；合格中SSIM最高优先','test_read':False,'optimizer_updates':0,'gt_intensity_used':False,'checkpoint_unchanged':True}
 (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
