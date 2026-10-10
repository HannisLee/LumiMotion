"""独立native CUDA与旧安装包比较；source plane/真实RGB均只读，0optimizer。"""
from pathlib import Path
import sys,json,math,importlib.util,hashlib
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parents[2]
sys.path.insert(0,str(CWD));sys.path.insert(0,str(ROOT/'build'))
import torch
import torch.nn.functional as F
import diff_surfel_rasterization as legacy
spec=importlib.util.spec_from_file_location('native_audit_wrapper',ROOT/'code_snapshot/diff_surfel_rasterization/__init__.py')
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
from scripts.render_stage2 import Stage2RenderRun,save_image
import gaussian_renderer.render_ir as renderer
from scene.pbr_geometry import quaternion_normal
RUN=CWD/'output/1010-10-jumpingjacks_dir-PBR来源平面几何投影'

def settings(module,cam,bg,degree):
    return module.GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,
        tanfovx=math.tan(cam.FoVx*.5),tanfovy=math.tan(cam.FoVy*.5),bg=bg,scale_modifier=1.,
        viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,
        sh_degree=degree,campos=cam.camera_center,prefiltered=False,debug=False)

def raw(module,cam,ctx,xyz,q,scales,opacity):
    fn=module.GaussianRasterizer(settings(module,cam,ctx.background,ctx.pc.active_sh_degree))
    _,_,maps=fn(means3D=xyz,means2D=torch.zeros_like(xyz),shs=None,colors_precomp=torch.ones_like(xyz),
                opacities=opacity,scales=scales,rotations=q,cov3D_precomp=None)
    world=(maps[2:5].permute(1,2,0)@cam.world_view_transform[:3,:3].T).permute(2,0,1)
    return maps,F.normalize(world,dim=0)

def main():
    run=Stage2RenderRun(RUN/'model_mlp',55000);ctx=run.context;pc=ctx.pc
    plane=ctx.pbr_plane_projection.plane;ids=ctx.pbr_plane_projection.support
    rows=[];dest=ROOT/'inspection/images';dest.mkdir(exist_ok=False)
    original=(renderer.GaussianRasterizationSettings,renderer.GaussianRasterizer)
    with torch.no_grad():
        for frame in ['0001','0038','0075','0076','0080','0150']:
            cam=ctx.camera_by_key[frame];ctx.load_camera(cam);d=ctx.deformation(cam)
            xyz=pc.get_xyz+d['d_xyz'];q=pc.get_rotation_bias(d['d_rotation']);scale=pc.get_scaling+d['d_scaling']
            opacity=pc.get_opacity if d['d_opacity'] is None else pc.get_opacity+d['d_opacity']
            expected=(cam.camera_center-xyz[ids])@plane
            assert expected.min()>0
            row={'frame':frame,'camera_center':cam.camera_center.tolist(),
                 'camera_plane_side_min':float(expected.min()),'modes':{}}
            for name,module in [('legacy',legacy),('native',native)]:
                maps,normal=raw(module,cam,ctx,xyz,q,scale,opacity*ids[:,None])
                valid=maps[1]>.5;signed=normal[:,valid].T@plane
                row['modes'][name]={'valid_pixels':int(valid.sum()),'plane_dot_min':float(signed.min()),
                                   'negative_fraction':float((signed<0).float().mean())}
                if name=='native':assert signed.min()>.99999
                save_image(dest/f'{frame}_{name}_plane_normal.png',(normal*.5+.5)*maps[1:2])
                renderer.GaussianRasterizationSettings=module.GaussianRasterizationSettings
                renderer.GaussianRasterizer=module.GaussianRasterizer
                pkg=run.render(cam)
                assert all(torch.isfinite(pkg[k]).all() for k in ['render','rend_alpha','rend_normal','surf_normal'])
                save_image(dest/f'{frame}_{name}_rgb.png',pkg['render'])
                save_image(dest/f'{frame}_{name}_normal.png',(F.normalize(pkg['rend_normal'],dim=0)*.5+.5)*pkg['rend_alpha'])
            renderer.GaussianRasterizationSettings,renderer.GaussianRasterizer=original
            rows.append(row);ctx.unload_camera(cam)
    # 合成已知单片geometry，仅验证原生normal CUDA梯度，不读取GT。
    cam=ctx.camera_by_key['0075'];ctx.load_camera(cam)
    xyz=(-ctx.pbr_plane_projection.offset*plane)[None]
    opacity=torch.tensor([[.99]],device='cuda');scale=torch.tensor([[.5,.5]],device='cuda')
    rotation=torch.tensor([[1.,.08,.12,.03]],device='cuda',requires_grad=True)
    maps,_=raw(native,cam,ctx,xyz,F.normalize(rotation,dim=1),scale,opacity)
    fixed_mask=(maps[1]>.8).detach();assert fixed_mask.any()
    target=torch.tensor([.31,-.73,.28],device='cuda')
    def objective(q):
        _,n=raw(native,cam,ctx,xyz,F.normalize(q,dim=1),scale,opacity)
        return (n[:,fixed_mask].T@target).mean()
    value=objective(rotation);gradient=torch.autograd.grad(value,rotation)[0]
    fd=[];eps=1e-4
    with torch.no_grad():
        for k in range(4):
            plus=rotation.detach().clone();minus=plus.clone();plus[0,k]+=eps;minus[0,k]-=eps
            fd.append(float((objective(plus)-objective(minus))/(2*eps)))
    fd=torch.tensor([fd],device='cuda');assert torch.isfinite(gradient).all()
    assert torch.allclose(gradient,fd,rtol=.015,atol=.002),(gradient,fd)
    ctx.unload_camera(cam);ctx.verify_frozen()
    result={'status':'PASS_NATIVE_FACING_AND_GRADIENT_DIAGNOSTIC','rows':rows,'optimizer_steps':0,
        'gt_labels_used':False,'synthetic_gradient':gradient.tolist(),'finite_difference':fd.tolist(),
        'gradient_max_error':float((gradient-fd).abs().max()),'source_files':ctx.source_files,
        'native_extension':native._C.__file__,
        'native_extension_sha256':hashlib.sha256(Path(native._C.__file__).read_bytes()).hexdigest(),
        'environment_package_changed':False,'B05_pipeline_changed':False,'final_mae':False,
        'limitations':'同B05旧训练权重仅验证新CUDA实际法线定向/梯度/PBR功能；不据此声明新训练后画质或7°。'}
    (ROOT/'inspection/native_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
