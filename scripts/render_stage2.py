"""同协议 Stage2 评测、材质可视化、方向光及 HDR 重光照。"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
from pathlib import Path
from argparse import Namespace
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio
from scripts.stage2_runtime import (load_photometric_run, FrozenScene, model_dir, checkpoint_path,
                                   supervision, file_sha256)
from scene.stage2_photometric import HDRIrradiance, frame_key, read_gt_directions, unit_vector
from gaussian_renderer.render_stage2_lambertian import render_stage2_lambertian
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def to_image(tensor):
    return (tensor.detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)


def save_image(path,tensor):
    array=to_image(tensor)
    if array.shape[-1]==1:array=array[...,0]
    Image.fromarray(array).save(path)


class Stage2RenderRun:
    def __init__(self,output,iteration):
        self.output=Path(output)
        if checkpoint_path(output,iteration).exists():
            self.context,self.material,self.light,self.args,self.state=load_photometric_run(output,iteration)
            self.mode='photometric_lambertian'
        else:
            config=self.output/'stage2_config.json'
            if not config.exists():raise ValueError('缺少 Stage2 配置：'+str(config))
            self.args=Namespace(**json.loads(config.read_text()))
            self.context=FrozenScene(self.args,output,iteration)
            from scene.light import EnvLight
            self.env=EnvLight(path=None,device='cuda',resolution=[self.args.envmap_resolution//2,self.args.envmap_resolution],
                              max_res=self.args.envmap_resolution,activation=self.args.envmap_activation)
            self.env.load_weights(str(output),iteration)
            self.mode='original_ir';self.bvh=False

    @torch.no_grad()
    def render(self,camera,direction=None,intensity=None,hdr=None,time_override=None):
        ctx=self.context;ctx.load_camera(camera)
        deformation=ctx.deformation(camera,time_override)
        if self.mode=='photometric_lambertian':
            if direction is None and hdr is None:
                direction=self.light(camera.fid if time_override is None else time_override)
            return render_stage2_lambertian(camera,ctx.pc,self.material,deformation,ctx.background,
                direction=direction,intensity=self.args.photometric_light_intensity if intensity is None else intensity,
                hdr=hdr,depth_ratio=self.args.depth_ratio)
        if direction is not None or hdr is not None:
            raise ValueError('原始 IR 对照使用原始 HDR 工具；此重光照入口针对 Lambertian。')
        from gaussian_renderer.render_ir import render_ir
        if not self.bvh:
            ctx.pc.build_bvh(d_rotation=deformation['d_rotation'],d_xyz=deformation['d_xyz'],d_scaling=deformation['d_scaling'])
            self.bvh=True
        else:ctx.pc.update_bvh(d_rotation=deformation['d_rotation'],d_xyz=deformation['d_xyz'],d_scaling=deformation['d_scaling'])
        return render_ir(camera,ctx.pc,self.args,ctx.background,**{k:deformation[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                         relight=False,training=False,env_light=self.env,opt=self.args)


def ensure_directory(path):
    path=Path(path)
    if path.exists():raise ValueError('渲染结果禁止覆盖：'+str(path))
    path.mkdir(parents=True)
    return path


@torch.no_grad()
def evaluate(run,output):
    output=ensure_directory(output);records=[]
    for camera in run.context.test_cameras:
        pkg=run.render(camera);target,mask=supervision(camera,run.context.background)
        image=pkg['render'].clamp(0,1)
        mse=float((image-target).square().mean())
        count=float(mask.sum())
        if count<=0:raise ValueError('测试前景为空。')
        fg_mse=float(((image-target).square()*mask).sum()/(count*3))
        ys,xs=torch.where(mask[0]>0)
        y0,y1=int(ys.min()),int(ys.max())+1;x0,x1=int(xs.min()),int(xs.max())+1
        cropped_image=(image*mask)[:,y0:y1,x0:x1];cropped_target=(target*mask)[:,y0:y1,x0:x1]
        rec={'frame':frame_key(camera.image_name_train_light),'time':float(camera.fid.item()),
             'psnr':-10*math.log10(max(mse,1e-12)), 'ssim':float(ssim(image,target)),
             'lpips':float(lpips(image,target,net_type='vgg')),
             'psnr_foreground':-10*math.log10(max(fg_mse,1e-12)),
             'ssim_foreground_bbox':float(ssim(cropped_image,cropped_target)),
             'lpips_foreground_bbox':float(lpips(cropped_image,cropped_target,net_type='vgg'))}
        if run.mode=='photometric_lambertian' and run.args.photometric_gt_lights_path:
            gt=read_gt_directions(run.args.photometric_gt_lights_path,run.context.center,[rec['frame']],'cuda')[0]
            direction=run.light(camera.fid)
            rec['light_angle_degrees']=float(torch.acos((gt*direction).sum().clamp(-1,1))*180/math.pi)
        records.append(rec)
        save_image(output/(rec['frame']+'_render.png'),image)
        save_image(output/(rec['frame']+'_gt.png'),target)
        save_image(output/(rec['frame']+'_mask.png'),mask)
        run.context.unload_camera(camera)
    if not records:raise ValueError('测试集合为空。')
    metrics={key:float(np.mean([r[key] for r in records])) for key in records[0] if key not in {'frame','time'}}
    report={'protocol':'各帧自身相机/形变/光照；learned 测试光照只由训练方向插值；前景SSIM/LPIPS为遮罩后包围框指标',
            'mode':run.mode,'metrics':metrics,'frames':records}
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('evaluation',metrics,flush=True)
    return report


def video_writer(path):
    return imageio.get_writer(str(path),fps=24,macro_block_size=1,codec='libx264')


@torch.no_grad()
def insights(run,output):
    output=ensure_directory(output)
    categories=['rgb','alpha','normal','albedo']
    writers={key:video_writer(output/(key+'.mp4')) for key in categories}
    for key in categories:(output/key).mkdir()
    cameras=sorted(run.context.train_cameras+run.context.test_cameras,key=lambda c:float(c.fid.item()))
    try:
        for camera in cameras:
            pkg=run.render(camera);key=frame_key(camera.image_name_train_light)
            alpha=pkg['rend_alpha'];normal=pkg.get('normal',torch.nn.functional.normalize(pkg['rend_normal'],dim=0))
            images={'rgb':pkg['render'],'alpha':alpha.expand(3,-1,-1),
                    'normal':(normal*0.5+0.5)*alpha,'albedo':pkg['base_color']}
            for category,image in images.items():
                save_image(output/category/(key+'.png'),image)
                writers[category].append_data(to_image(image))
            run.context.unload_camera(camera)
    finally:
        for writer in writers.values():writer.close()


@torch.no_grad()
def directional_relight(run,output,key='0001',frames=72,intensity=None):
    if run.mode!='photometric_lambertian':raise ValueError('方向光替换仅适用于 Lambertian。')
    if frames<4:raise ValueError('转灯视频至少4帧。')
    output=ensure_directory(output);camera=run.context.camera_by_key[key]
    # 在固定相机右/前方向形成闭合转灯轨迹，位置、时间和材质均保持不变。
    run.context.load_camera(camera)
    front=unit_vector(camera.camera_center-run.context.center)
    right=camera.world_view_transform[:3,0]
    right=unit_vector(right-(right*front).sum()*front)
    writer=video_writer(output/'rotating_directional.mp4');means=[];alpha_reference=None
    try:
        for idx in range(frames):
            angle=2*math.pi*idx/frames
            direction=unit_vector(front*math.cos(angle)+right*math.sin(angle))
            pkg=run.render(camera,direction=direction,intensity=intensity)
            if alpha_reference is None:alpha_reference=pkg['rend_alpha'].clone()
            elif not torch.equal(alpha_reference,pkg['rend_alpha']):raise AssertionError('转灯改变了 alpha。')
            save_image(output/('{:04d}.png'.format(idx)),pkg['render'])
            writer.append_data(to_image(pkg['render']));means.append(float(pkg['render'].mean()))
    finally:writer.close();run.context.unload_camera(camera)
    record={'frame_key':key,'frames':frames,'mean_brightness':means,'alpha_unchanged':True,
            'intensity':run.args.photometric_light_intensity if intensity is None else intensity}
    if max(means)-min(means)<1e-5:raise AssertionError('转灯没有产生光照响应。')
    (output/'relight.json').write_text(json.dumps(record,indent=2)+'\n')


@torch.no_grad()
def hdr_relight(run,output,path,samples=2048,yaw=0.0,exposure=1.0):
    if run.mode!='photometric_lambertian':raise ValueError('此 HDR 入口针对 Lambertian。')
    output=ensure_directory(output)
    hdr=HDRIrradiance.from_file(path,samples=samples,yaw_degrees=yaw,exposure=exposure)
    writer=video_writer(output/'hdr.mp4')
    cameras=sorted(run.context.train_cameras+run.context.test_cameras,key=lambda c:float(c.fid.item()))
    try:
        for idx,camera in enumerate(cameras):
            pkg=run.render(camera,hdr=hdr);key=frame_key(camera.image_name_train_light)
            save_image(output/(key+'.png'),pkg['render']);writer.append_data(to_image(pkg['render']))
            if idx in {0,len(cameras)//2,len(cameras)-1}:
                np.savez_compressed(output/(key+'_linear.npz'),rgb=pkg['render_linear'].cpu().numpy())
            run.context.unload_camera(camera)
    finally:writer.close()
    (output/'hdr.json').write_text(json.dumps({'path':str(Path(path).resolve()),'sha256':file_sha256(path),
        'samples':samples,'yaw_degrees':yaw,'exposure':exposure,'world_up':'Z',
        'longitude':'atan2(-y,x)','axis_reference':'原始HDR脚本轴变换 [[0,-1,0],[0,0,1],[-1,0,0]]',
        'integrator':'确定性均匀球面积分；直接漫反射，无遮挡/镜面/间接光'},ensure_ascii=False,indent=2)+'\n')


def run_cli(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path',required=True);parser.add_argument('--load_iter',type=int,required=True)
    parser.add_argument('--deform_type',default='mlp');parser.add_argument('--task',choices=['eval','insights','materials','directional','hdr','all'],default='all')
    parser.add_argument('--output_path',default='');parser.add_argument('--frame_key',default='0001')
    parser.add_argument('--relight_frames',type=int,default=72);parser.add_argument('--light_intensity',type=float,default=None)
    parser.add_argument('--hdr_filepath',default='example_envmaps/golden_bay_4k_32x16_rot330.hdr')
    parser.add_argument('--hdr_samples',type=int,default=2048);parser.add_argument('--hdr_yaw',type=float,default=0.0)
    parser.add_argument('--hdr_exposure',type=float,default=1.0)
    args=parser.parse_args(argv);model=model_dir(args.model_path,args.deform_type)
    output=ensure_directory(args.output_path or model/'renders_stage2'/('ours_'+str(args.load_iter)))
    (output/'command.json').write_text(json.dumps(vars(args),ensure_ascii=False,indent=2)+'\n')
    repository=Path(__file__).resolve().parents[1]
    provenance={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository).decode().strip(),
                'code_status':subprocess.check_output(['git','status','--short'],cwd=repository).decode(),
                'model_path':str(model.resolve()),'iteration':args.load_iter}
    (output/'code_diff.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=repository))
    (output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    run=Stage2RenderRun(model,args.load_iter)
    if args.task in {'eval','all'}:evaluate(run,output/'eval')
    if args.task in {'insights','materials','all'}:insights(run,output/'insights')
    if args.task in {'directional','all'}:directional_relight(run,output/'directional',args.frame_key,args.relight_frames,args.light_intensity)
    if args.task in {'hdr','all'}:hdr_relight(run,output/'hdr',args.hdr_filepath,args.hdr_samples,args.hdr_yaw,args.hdr_exposure)
    (output/'RENDER_COMPLETED').write_text(args.task+'\n')
    if run.mode=='photometric_lambertian':run.context.verify_frozen()
    return output


if __name__=='__main__':run_cli()
