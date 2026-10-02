"""读取真实冒烟产物，验证冻结、alpha/色彩空间、GT/learned 梯度及完整恢复。"""
import argparse
import json
from pathlib import Path
import torch
from scripts.stage2_runtime import load_photometric_run
from gaussian_renderer.render_stage2_lambertian import render_stage2_lambertian


def compare(a,b):
    if torch.is_tensor(a):
        # 原始 CUDA rasterizer 使用原子累加，跨进程不可承诺浮点逐位一致。
        if a.is_floating_point():torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        else:torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:compare(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();checks={}
    for mode,num,resume in [('GT','02','04'),('learned','03','05')]:
        base=Path('output/smoke_test/1002-'+num+'-onlyclothV4-stage2_'+mode+'/model_mlp')
        continuation=Path('output/smoke_test/1002-'+resume+'-onlyclothV4-stage2_'+mode+'_resume/model_mlp')
        ctx,material,light,config,state=load_photometric_run(base,35200)
        restored=torch.load(continuation/'stage2_photometric/iteration_35200/checkpoint.pth',map_location='cuda')
        for field in ['material','light','optimizer','camera_stack','rng_torch','rng_cuda']:
            compare(state[field],restored[field])
        camera=ctx.train_cameras[0];ctx.load_camera(camera)
        deformation=ctx.deformation(camera);direction=light(camera.fid)
        pkg=render_stage2_lambertian(camera,ctx.pc,material,deformation,ctx.background,
                                     direction=direction,intensity=config.photometric_light_intensity)
        pkg['render'].mean().backward()
        assert material.raw_albedo.grad is not None and material.raw_albedo.grad.abs().sum()>0
        if mode=='GT':assert light.raw_directions.grad is None
        else:assert light.raw_directions.grad is not None and light.raw_directions.grad.abs().sum()>0
        assert all(v.grad is None for v in vars(ctx.pc).values() if isinstance(v,torch.nn.Parameter))
        assert all(p.grad is None for p in ctx.deform.deform.parameters())
        with torch.no_grad():
            white=render_stage2_lambertian(camera,ctx.pc,material,deformation,torch.ones_like(ctx.background),
                                         direction=direction,intensity=config.photometric_light_intensity)
            torch.testing.assert_close(white['render_linear']-pkg['render_linear'],
                      (1-pkg['rend_alpha']).expand_as(pkg['render_linear']),atol=2e-6,rtol=0)
        ctx.verify_frozen();ctx.unload_camera(camera)
        raw_error=float((state['material']['raw_albedo']-restored['material']['raw_albedo']).abs().max())
        checks[mode]={'resume_numerically_equivalent':True,'resume_albedo_logit_max_abs_error':raw_error,
                      'resume_float_tolerance':{'rtol':1e-5,'atol':1e-6},
                      'rng_and_camera_stack_exact':True,'material_gradient':True,'light_gradient':mode=='learned',
                      'geometry_frozen':True,'source_hashes_unchanged':True,'linear_background_compositing':True,
                      'train_light_parameters':int(light.raw_directions.requires_grad)*len(light.times)}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(checks)


if __name__=='__main__':main()
