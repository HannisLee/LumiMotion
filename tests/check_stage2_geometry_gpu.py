"""真实checkpoint：梯度、零残差回归、评测入口与续训状态检查。"""
import argparse
import json
from pathlib import Path
import torch
from scripts.stage2_runtime import load_photometric_run
from scripts.eval_geometry_gt import GeometryRun
from gaussian_renderer.render_stage2_lambertian import render_stage2_lambertian
from tests.check_stage2_gpu import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--resume',required=True)
    p.add_argument('--iteration',type=int,default=35100);p.add_argument('--output',required=True);a=p.parse_args()
    ctx,mat,light,args,state=load_photometric_run(a.model,a.iteration)
    camera=ctx.train_cameras[1];ctx.load_camera(camera)
    def render():
        return render_stage2_lambertian(camera,ctx.pc,mat,ctx.deformation(camera),ctx.background,
                    direction=light(camera.fid),intensity=args.photometric_light_intensity)
    pkg=render();pkg['render'].mean().backward()
    geo=ctx.pc.stage2_geometry
    gradients={k:float(v.grad.abs().sum()) for k,v in geo.named_parameters()}
    assert all(v>0 for v in gradients.values())
    assert all(torch.isfinite(v.grad).all() for v in geo.parameters())
    assert all(v.grad is None for v in vars(ctx.pc).values() if isinstance(v,torch.nn.Parameter))
    evaluator=GeometryRun(a.model,a.iteration)
    import numpy as np
    actual=evaluator.render(camera,np.eye(4),np.array([1.,1.,0.,0.]))
    np.testing.assert_allclose(actual['normal'],pkg['normal'].detach().permute(1,2,0).cpu().numpy(),rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(actual['expected'],pkg['surf_depth'][0].detach().cpu().numpy(),rtol=1e-5,atol=1e-6)
    with torch.no_grad():
        geo.translation.zero_();geo.axis_angle.zero_()
        zero=render();del ctx.pc.stage2_geometry;old=render();ctx.pc.stage2_geometry=geo
        for field in ['render','normal','surf_depth','rend_alpha']:
            torch.testing.assert_close(zero[field],old[field],atol=1e-6,rtol=1e-5)
    ctx.verify_frozen()
    resumed=torch.load(Path(a.resume)/'stage2_photometric'/f'iteration_{a.iteration}'/'checkpoint.pth',map_location='cuda')
    for field in ['geometry','material','light','optimizer','camera_stack','rng_python','rng_numpy','rng_torch','rng_cuda']:
        compare(state[field],resumed[field])
    out={'geometry_rgb_gradient_l1':gradients,'zero_residual_legacy_render_equivalent':True,
         'evaluation_uses_updated_geometry':True,'resume_numerically_equivalent':True,
         'source_frozen':True,'diagnostics_before_zero':{'translation_max':float(state['geometry']['translation'].abs().max())}}
    Path(a.output).write_text(json.dumps(out,indent=2));print(out)


if __name__=='__main__':main()
