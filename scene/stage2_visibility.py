"""从只读Stage1预测计算的直接光可见度；不使用GT网格或GT深度。

固定参考几何、四点近似面积光、无可见度反向传播。用于小范围几何微调，
大位移时需要重新评估这一近似。HDR及替换方向光不复用捕获光照的阴影。
"""
import math
import torch
from torch.nn import functional as F
from utils.general_utils import build_scaling_rotation
from scene.stage2_photometric import frame_key


class ReferenceVisibility:
    def __init__(self, context, args):
        self.context=context;self.args=args;self.cache={}
        if args.geometry_shadow_bias<=0 or args.geometry_shadow_size<0:
            raise ValueError('阴影起点偏移必须为正，面积光尺寸非负。')

    @torch.no_grad()
    def get(self,camera,material,light):
        key=frame_key(camera.image_name_train_light)
        if key in self.cache:return self.cache[key].to(device='cuda',dtype=torch.float32)
        from gaussian_renderer.render_stage2_lambertian import render_stage2_lambertian
        ctx=self.context;pc=ctx.pc;d=ctx.deformation(camera)
        geometry=getattr(pc,'stage2_geometry',None)
        if geometry is not None:del pc.stage2_geometry
        try:
            package=render_stage2_lambertian(camera,pc,material,d,ctx.background,direction=light(camera.fid))
        finally:
            if geometry is not None:pc.stage2_geometry=geometry
        xyz=pc.get_xyz+d['d_xyz'];scales=pc.get_scaling+d['d_scaling']
        rotations=pc.get_rotation_bias(d['d_rotation']);opacity=pc.get_opacity
        transform=build_scaling_rotation(torch.cat((scales,torch.full_like(scales[:,:1],1e-6)),-1),rotations)
        # 原boundings在opacity<alpha_min时sqrt(log)为NaN；这里只为新阴影分支置零半径。
        radius=(2*(opacity/pc.alpha_min).log().clamp_min(0)).sqrt()[:,None]
        vertices=radius*(pc.unit_icosahedron_vertices[None]@transform.transpose(-1,-2))+xyz[:,None]
        faces=pc.unit_icosahedron_faces[None]+torch.arange(len(xyz),device=xyz.device)[:,None,None]*12
        ids=torch.arange(len(xyz),device=xyz.device)[:,None].expand(-1,faces.shape[1])
        pc.gaussian_tracer.build_bvh(vertices.reshape(-1,3),faces.reshape(-1,3),ids.reshape(-1))
        h,w=camera.image_height,camera.image_width
        y,x=torch.meshgrid(torch.arange(h,device=xyz.device,dtype=xyz.dtype)+.5,
                           torch.arange(w,device=xyz.device,dtype=xyz.dtype)+.5,indexing='ij')
        rays=torch.stack(((x-w/2)*2*math.tan(camera.FoVx/2)/w,
                         (y-h/2)*2*math.tan(camera.FoVy/2)/h,torch.ones_like(x)),-1)
        c2w=camera.world_view_transform.T.inverse()
        positions=((rays@c2w[:3,:3].T)*package['surf_depth'].permute(1,2,0)+c2w[:3,3]).reshape(-1,3)
        valid=package['rend_alpha'].reshape(-1)>.5
        visibility=torch.ones(h*w,device=xyz.device)
        if valid.any():
            center=light.position(camera.fid)
            axis=F.normalize(center-ctx.center,dim=0)
            up=torch.tensor([0.,0.,1.],device=xyz.device)
            if (axis*up).sum().abs()>.99:up=torch.tensor([0.,1.,0.],device=xyz.device)
            right=F.normalize(torch.cross(axis,up,dim=0),dim=0);up=torch.cross(right,axis,dim=0)
            offsets=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],device=xyz.device)
            lamps=center+(offsets[:,:1]*right+offsets[:,1:]*up)*self.args.geometry_shadow_size*.25
            if hasattr(light,'sample_positions'):lamps=light.sample_positions(camera.fid)
            points=positions[valid,None,:]
            directions=F.normalize(lamps[None]-points,dim=-1)
            traced=pc.trace(points+directions*self.args.geometry_shadow_bias,directions,
                features=torch.zeros(len(xyz),4,device=xyz.device),xyz=xyz,scales=scales,
                rotation=rotations,opacity=opacity,use_zeros=True)
            visibility[valid]=(1-traced['alpha']).mean(-1)
        result=visibility.reshape(1,h,w).clamp(0,1)
        if not torch.isfinite(result).all():raise FloatingPointError('参考阴影非有限。')
        self.cache[key]=result.to(device='cpu',dtype=torch.float16)
        return self.cache[key].to(device='cuda',dtype=torch.float32)
