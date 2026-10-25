"""完整18DEV比较原生SH间接近似开关，仅诊断，不读取TEST。"""
import argparse,json,math
from pathlib import Path
import numpy as np,torch
torch.set_num_threads(4)
from LIHAN.scripts.stage3_el_eldl_mixed import install_evaluation
install_evaluation()
from scripts.render_stage2 import Stage2RenderRun,save_image
from scripts.stage2_runtime import FrozenScene,supervision
from scene.stage2_photometric import frame_key
from utils.loss_utils import ssim
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--source',required=True);p.add_argument('--directions',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
run=Stage2RenderRun(a.model,a.iteration);run.args.source_path=a.source;run.args.pbr_mixed_lights_path=a.directions;run.args.diffuse_sample_num=512;run.context=FrozenScene(run.args,a.model,a.iteration);ctx=run.context;assert len(ctx.test_cameras)==18
values=[]
with torch.no_grad():
 for disabled in [False,True]:
  run.args.wo_indirect=disabled;directory=a.output/str(disabled);directory.mkdir();rows=[]
  for cam in ctx.test_cameras:
   pkg=run.render(cam);target,mask=supervision(cam,ctx.background);im=pkg['render'].clamp(0,1);key=frame_key(cam.image_name_train_light)
   rows.append({'frame':key,'psnr':-10*math.log10(max(float((im-target).square().mean()),1e-12)),'ssim':float(ssim(im,target))})
   if int(key) in [61,71,79]:save_image(directory/f'{key}.png',im)
   ctx.unload_camera(cam)
  v={'wo_indirect':disabled,'frames':rows,'metrics':{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim']}};values.append(v);print(json.dumps(v),flush=True)
b,c=[v['metrics'] for v in values];adopt=c['psnr']>=b['psnr']+.05 and c['ssim']>=b['ssim']+.0001
(a.output/'result.json').write_text(json.dumps({'model':a.model,'iteration':a.iteration,'values':values,'adopt':adopt,'criterion':'完整18DEV，PSNR至少+0.05dB且SSIM至少+0.0001才采用；无TEST读取','test_read':False,'weights_changed':False},ensure_ascii=False,indent=2)+'\n')
