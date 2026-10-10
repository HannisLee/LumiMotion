"""55k只读分组光栅审计，核验来源plane轴与实际world normal；不做优化、不读GT标签。"""
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parent.parent
CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import torch
import torch.nn.functional as F
import diff_surfel_rasterization as raster
from scripts.render_stage2 import Stage2RenderRun,save_image
from scene.pbr_geometry import quaternion_normal
from gaussian_renderer.render_ir import render_ir

@torch.no_grad()
def main():
    run=Stage2RenderRun(ROOT/'model_mlp',55000)
    ctx=run.context;pc=ctx.pc;plane=ctx.pbr_plane_projection.plane
    groups={'full':torch.ones(len(pc.get_xyz),dtype=torch.bool,device='cuda'),
            'projected':ctx.pbr_plane_projection.support,
            'dynamic':pc.get_binary_feature()[:,0]>=.5,
            'static_unprojected':(pc.get_binary_feature()[:,0]<.5)&~ctx.pbr_plane_projection.support}
    rows=[]
    dest=ROOT/'inspection/plane_raster_groups_55000'
    dest.mkdir(exist_ok=False)
    for frame in ['0001','0075','0150']:
        cam=ctx.camera_by_key[frame];ctx.load_camera(cam)
        d=ctx.deformation(cam)
        xyz=pc.get_xyz+d['d_xyz'];q=pc.get_rotation_bias(d['d_rotation'])
        scales=pc.get_scaling+d['d_scaling']
        opacity=pc.get_opacity if d['d_opacity'] is None else pc.get_opacity+d['d_opacity']
        settings=raster.GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,
            tanfovx=math.tan(cam.FoVx*.5),tanfovy=math.tan(cam.FoVy*.5),bg=ctx.background,
            scale_modifier=1.,viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,
            sh_degree=pc.active_sh_degree,campos=cam.camera_center,prefiltered=False,debug=False)
        fn=raster.GaussianRasterizer(raster_settings=settings)
        reference=render_ir(cam,pc,run.args,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                            opt=run.args,material_only=True)
        row={'frame':frame,'groups':{}}
        axis=quaternion_normal(q)
        assert (axis[groups['projected']]@plane).abs().min()>.99999
        for name,ids in groups.items():
            _,_,allmap=fn(means3D=xyz,means2D=torch.zeros_like(xyz),shs=None,
                         colors_precomp=torch.ones_like(xyz),opacities=opacity*ids[:,None],
                         scales=scales,rotations=q,cov3D_precomp=None)
            alpha=allmap[1:2]
            world=(allmap[2:5].permute(1,2,0)@cam.world_view_transform[:3,:3].T).permute(2,0,1)
            normal=F.normalize(world,dim=0);valid=(alpha[0]>.5)&(world.norm(dim=0)>1e-6)
            signed=normal[:,valid].T@plane
            angle=torch.acos(signed.abs().clamp(0,1))*180/math.pi
            vals={'points':int(ids.sum()),'alpha_gt_half_pixels':int(valid.sum()),
                  'absolute_plane_angle_quantiles_deg':torch.quantile(angle,angle.new_tensor([0.,.5,.9,.99,1.])).tolist(),
                  'positive_plane_normal_fraction':float((signed>0).float().mean()),
                  'mean_alpha_over_image':float(alpha.mean())}
            if name=='full':
                vals['native_buffer_max_difference']={'alpha':float((alpha-reference['rend_alpha']).abs().max()),
                                                      'world_normal':float((world-reference['rend_normal']).abs().max())}
                assert vals['native_buffer_max_difference']['world_normal']<1e-6
            if name=='projected':
                vals['axis_absolute_dot_min']=float((axis[ids]@plane).abs().min())
                assert float(angle.max())<.1
            row['groups'][name]=vals
            save_image(dest/f'{frame}_{name}_normal.png',(normal*.5+.5)*alpha)
            save_image(dest/f'{frame}_{name}_alpha.png',alpha.expand(3,-1,-1))
        rows.append(row);ctx.unload_camera(cam)
    ctx.verify_frozen()
    result={'status':'PASS_READ_ONLY_RASTER_NORMAL_AUDIT','iteration':55000,'rows':rows,
            'installed_module':raster.__file__,'installed_extension':raster._C.__file__,
            'extension_sha256':hashlib.sha256(Path(raster._C.__file__).read_bytes()).hexdigest(),
            'source_files':ctx.source_files,'gt_geometry_normal_material_inputs':False,'optimizer_steps':0,
            'final_mae':False,'limitations':'分组单独渲染并不等于联合渲染中该组的可见贡献，alpha区域不能用于改变评测mask。'}
    (ROOT/'inspection/plane_raster_normals.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
