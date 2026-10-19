"""动态A/B共用当前surfel、全场景光源可见度和线性域物理着色。"""
import math
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scene import GaussianModel, DeformModel
from scene.stage2_photometric import DirectionalLightTable, srgb_to_linear, linear_to_srgb
from scripts.probe_dynamic_dl_selfcal import read_config
from utils.general_utils import build_rotation, build_scaling_rotation


class EffectiveSurface:
    """同一份当前形变同时用于光栅、法线和BVH，不丢弃XYZ输入导数。"""
    def __init__(self, pc, values):
        self.pc, self.values = pc, values

    def __getattr__(self, key):
        return getattr(self.pc, key)

    @property
    def get_xyz(self):return self.pc.get_xyz+self.values['d_xyz']

    @property
    def get_rotation(self):return self.pc.get_rotation_bias(self.values['d_rotation'])

    def get_rotation_bias(self,bias=None):
        if bias is None or (isinstance(bias,(int,float)) and bias==0):return self.get_rotation
        return F.normalize(self.get_rotation+bias,dim=-1)

    @property
    def get_scaling(self):return self.pc.get_scaling+self.values['d_scaling']

    @property
    def get_opacity(self):
        value=self.values['d_opacity']
        return self.pc.get_opacity+(0. if value is None else value)


def load_train_cameras(source,frame_ids=None):
    """先移除TRAIN核验的固定背景透射项，再AREA；保留物体环境照明。"""
    import cv2
    import numpy as np
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from LIHAN.scripts.dynamic_v3_radiometry import black_observation
    radiometry_path=Path(source).parent/'input_audit_v1/radiometry_v1.json'
    radiometry=json.loads(radiometry_path.read_text())
    assert radiometry['status']=='PASS_TRAIN_FIXED_BACKGROUND' and not radiometry['gt_read']
    assert hashlib.sha256((Path(source)/'no_gt_manifest.json').read_bytes()).hexdigest()==radiometry['input_manifest_sha256']
    infos=readCamerasFromTransforms(str(source),'transforms_train.json','images',False,no_bg=False)
    config=SimpleNamespace(resolution=2,data_device='cpu',load2gpu_on_the_fly=True)
    cameras=[]
    for i,info in enumerate(infos):
        if frame_ids is not None and int(info.image_name_train_light.split('_')[-1]) not in frame_ids:continue
        camera=loadCam(config,i,info,1.)
        assert hasattr(info.image_train_light,'array'),'Stage2只接受登记的16位输入'
        rgb=info.image_train_light.array[...,:3]
        alpha=info.image_train_light.array[...,3]
        linear=black_observation(rgb,alpha,radiometry['background_linear'])
        resized=cv2.resize(linear,(camera.image_width,camera.image_height),interpolation=cv2.INTER_AREA)
        camera.observed_linear=torch.from_numpy(resized.astype(np.float32)).permute(2,0,1)
        cameras.append(camera)
    return cameras


class FullDynamicVisibility:
    """每个调用重建完整当前几何；固定V近似梯度，不按点数/迭代误用跨时刻缓存。"""
    def __init__(self,bias):
        assert math.isfinite(bias) and bias>0
        self.bias=bias
        self.build_count=0

    @torch.no_grad()
    def build(self,geometry):
        xyz,scale,rotation,opacity=geometry.get_xyz,geometry.get_scaling,geometry.get_rotation,geometry.get_opacity
        assert torch.isfinite(xyz).all() and torch.isfinite(rotation).all()
        assert torch.isfinite(scale).all() and (scale>0).all()
        assert torch.isfinite(opacity).all() and (opacity>=0).all() and (opacity<=1).all()
        scale3=torch.cat([scale,torch.full_like(scale[:,:1],1e-6)],-1)
        transform=build_scaling_rotation(scale3,rotation)
        radius=(2*(opacity.clamp_min(1e-12)/geometry.alpha_min).log().clamp_min(0)).sqrt()[:,None]
        vertices=radius*(geometry.unit_icosahedron_vertices[None]@transform.transpose(-1,-2))+xyz[:,None]
        faces=geometry.unit_icosahedron_faces[None]+torch.arange(len(xyz),device=xyz.device)[:,None,None]*12
        ids=torch.arange(len(xyz),device=xyz.device)[:,None].expand(-1,faces.shape[1])
        assert torch.isfinite(vertices).all()
        geometry.gaussian_tracer.build_bvh(vertices.reshape(-1,3),faces.reshape(-1,3),ids.reshape(-1))
        self.geometry=geometry
        self.build_count+=1

    @torch.no_grad()
    def trace(self,points,directions):
        geometry=self.geometry
        if directions.ndim==2:
            directions=directions[None].expand(len(points),-1,-1)
        directions=F.normalize(directions,dim=-1).contiguous()
        origins=(points[:,None]+directions*self.bias).contiguous()
        result=geometry.trace(origins,directions,features=torch.zeros(len(geometry.get_xyz),4,device=points.device),
            xyz=geometry.get_xyz,scales=geometry.get_scaling,rotation=geometry.get_rotation,
            opacity=geometry.get_opacity,use_zeros=True)
        visibility=(1-result['alpha']).clamp(0,1)
        assert torch.isfinite(visibility).all()
        return visibility

    def __call__(self,geometry,directions):
        self.build(geometry)
        return self.trace(geometry.get_xyz,directions.detach())


def sh2_basis(n):
    """法线上的9项实SH；不接受视线或Gaussian ID。"""
    x,y,z=n.unbind(-1)
    return torch.stack([torch.ones_like(x)*.28209479177387814,
        -.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x,
        1.0925484305920792*x*y,-1.0925484305920792*y*z,
        .31539156525252005*(3*z*z-1),-1.0925484305920792*x*z,
        .5462742152960396*(x*x-y*y)],-1)


def quaternion_product(a,b):
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    return torch.stack([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],-1)


def native_normal_operator(canonical_raw_quaternion,rotation_delta,sign):
    """固定当前加性形变头，按真实原生surfel规则运输待估法线。

    保持canonical四元数模长及切向自旋，以最小旋转把当前物理轴转到候选轴。
    每个观测实际法线为Q(normalize(q_candidate+d_rotation_t))的第三列乘固定sign。
    这是原生GS局部模型的精确候选前向；不假称XYZ Jacobian已与它一致。
    """
    raw=canonical_raw_quaternion.detach()
    delta=rotation_delta.detach()
    sign=sign.detach().reshape(-1,1)
    canonical=build_rotation(raw)[:,:,2]*sign
    def operator(candidate):
        cross=torch.cross(canonical,candidate,dim=-1)
        cosine=(canonical*candidate).sum(-1,keepdim=True)
        # 与当前轴反向的180度候选切向自旋不唯一，交给可靠性门禁拒绝。
        change=F.normalize(torch.cat([1+cosine,cross],-1),dim=-1)
        quaternion=quaternion_product(change,raw)
        full=F.normalize(quaternion[None]+delta,dim=-1)
        world=build_rotation(full.reshape(-1,4))[:,:,2].reshape(*full.shape[:-1],3)
        return world*sign[None]
    return operator


class SharedRadiance(torch.nn.Module):
    def __init__(self,count,times,directions,initial_intensity,environment_scale,device='cuda'):
        super().__init__()
        assert .1<initial_intensity<20 and environment_scale>0
        self.raw_rho=torch.nn.Parameter(torch.zeros(count,3,device=device))
        self.register_buffer('gauge_support',torch.ones(count,dtype=torch.bool,device=device))
        self.raw_intensity=torch.nn.Parameter(torch.tensor(math.log((initial_intensity-.1)/(20-initial_intensity)),device=device))
        self.environment_coefficients=torch.nn.Parameter(torch.zeros(9,3,device=device))
        self.register_buffer('environment_scale',torch.tensor(environment_scale,device=device))
        self.lights=DirectionalLightTable(times.to(device),directions.to(device),learned=False)

    @property
    def rho(self):
        assert self.gauge_support.any()
        center=self.raw_rho[self.gauge_support].median().detach()
        return .03+.94*(self.raw_rho-center).sigmoid()

    @property
    def intensity(self):return .1+19.9*self.raw_intensity.sigmoid()

    def environment(self,normals):
        # 每通道9系数，tanh有界容量；softplus保证所有world法线辐照度非负。
        field=sh2_basis(normals)@(.5*self.environment_coefficients.tanh())
        return self.environment_scale*F.softplus(field+math.log(math.expm1(1.)))


class DynamicSurfaceModel:
    def __init__(self,parent,iteration,lighting,stats,path='A',allow_smoke_parent=False):
        assert path in ('A','B')
        assert iteration==35000 or allow_smoke_parent
        self.parent,self.iteration,self.path=Path(parent),iteration,path
        self.cfg=read_config(self.parent/'cfg_args')
        self.pc=GaussianModel(self.cfg.sh_degree,no_binary_separation=self.cfg.no_binary_separation,fea_dim=self.cfg.hyper_dim)
        self.pc.load_ply(str(self.parent/f'point_cloud/iteration_{iteration}/point_cloud.ply'))
        self.pc.active_sh_degree=min(self.cfg.sh_degree,iteration//1000)
        self.deform=DeformModel(deform_type=self.cfg.deform_type,is_blender=self.cfg.is_blender,
            hyper_dim=self.cfg.hyper_dim,pred_color=self.cfg.pred_color)
        assert self.deform.load_weights(str(self.parent),iteration)
        self.deform.deform.eval()
        for value in vars(self.pc).values():
            if isinstance(value,torch.Tensor):value.requires_grad_(False)
        self.deform.deform.requires_grad_(False)
        self.pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
        self.radiance=SharedRadiance(len(self.pc.get_xyz),lighting['times'],lighting['directions'],
            lighting['intensity'],math.pi*min(stats['mean_linear_luminance'],stats['q_base'])/.5)
        self.normal_sign=torch.ones(len(self.pc.get_xyz),1,device='cuda')
        self.normal_sign_valid=torch.zeros(len(self.pc.get_xyz),dtype=torch.bool,device='cuda')
        with torch.no_grad():
            extent=(torch.quantile(self.pc.get_xyz,.99,dim=0)-torch.quantile(self.pc.get_xyz,.01,dim=0)).norm()
        self.visibility=FullDynamicVisibility(float(extent)*.003)

    def checkpoint(self,folder,step,optimizer=None,config=None,ps_reference=None):
        """真实更新后保存完整推理状态；重载不依赖祖先模型或训练图像。"""
        import random
        import numpy as np
        folder=Path(folder)
        ply=folder/f'point_cloud/iteration_{step}/point_cloud.ply'
        deform=folder/f'deform/iteration_{step}/deform.pth'
        state_dir=folder/f'dynamic_v3/iteration_{step}'
        assert not ply.exists() and not deform.exists() and not state_dir.exists(),'禁止覆盖检查点'
        self.pc.save_ply(str(ply));self.deform.save_weights(str(folder),step)
        state_dir.mkdir(parents=True,exist_ok=False)
        state=dict(format_version=1,path=self.path,parent_iteration=self.iteration,completed_updates=step,
            cumulative_actual_updates=self.iteration-1+step,radiance=self.radiance.state_dict(),
            normal_sign=self.normal_sign,normal_sign_valid=self.normal_sign_valid,
            bias=self.visibility.bias,config=config,ps_reference=ps_reference,
            optimizer=None if optimizer is None else optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state())
        torch.save(state,state_dir/'state.pt')
        receipt=dict(path=self.path,step=step,completed_updates=step,cumulative_actual_updates=self.iteration-1+step,
            parent_iteration=self.iteration,normal_sign_query_independent=True,
            checkpoint_sha256={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ply,deform,state_dir/'state.pt')},gt_read=False)
        (state_dir/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        return receipt

    @classmethod
    def load(cls,folder,step):
        """只读取当前目录的原生PLY/deform、配置与独立物理状态。"""
        folder=Path(folder)
        receipt=json.loads((folder/f'dynamic_v3/iteration_{step}/receipt.json').read_text())
        for relative,expected in receipt['checkpoint_sha256'].items():
            assert hashlib.sha256((folder/relative).read_bytes()).hexdigest()==expected,relative
        state=torch.load(folder/f'dynamic_v3/iteration_{step}/state.pt',map_location='cpu',weights_only=False)
        assert state['format_version']==1 and state['completed_updates']==step
        obj=cls.__new__(cls)
        obj.parent=folder;obj.iteration=state['parent_iteration'];obj.path=state['path']
        obj.cfg=read_config(folder/'cfg_args')
        obj.pc=GaussianModel(obj.cfg.sh_degree,no_binary_separation=obj.cfg.no_binary_separation,fea_dim=obj.cfg.hyper_dim)
        obj.pc.load_ply(str(folder/f'point_cloud/iteration_{step}/point_cloud.ply'))
        obj.pc.active_sh_degree=obj.cfg.sh_degree
        obj.deform=DeformModel(deform_type=obj.cfg.deform_type,is_blender=obj.cfg.is_blender,
            hyper_dim=obj.cfg.hyper_dim,pred_color=obj.cfg.pred_color)
        assert obj.deform.load_weights(str(folder),step)
        obj.deform.deform.eval().requires_grad_(False)
        obj.pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
        lights=state['radiance']
        obj.radiance=SharedRadiance(len(obj.pc.get_xyz),lights['lights.times'],lights['lights.raw_directions'],
            math.pi,float(lights['environment_scale']))
        obj.radiance.load_state_dict(lights)
        obj.radiance.requires_grad_(False)
        obj.normal_sign=state['normal_sign'].cuda();obj.normal_sign_valid=state['normal_sign_valid'].cuda()
        obj.visibility=FullDynamicVisibility(state['bias'])
        obj.config=state['config'];obj.loaded_step=step
        return obj

    def effective(self,camera):
        values=self.deform.step(self.pc.get_xyz,camera.fid.reshape(1,1).expand(len(self.pc.get_xyz),1),
            iteration=35000,feature=self.pc.get_binary_feature(),camera_center=camera.camera_center,is_training=False)
        return EffectiveSurface(self.pc,{k:values[k] for k in ('d_xyz','d_rotation','d_scaling','d_opacity','d_color')})

    @torch.no_grad()
    def initialize_orientation(self,cameras):
        """相机可见时确定canonical轴符号，之后着色不按查询相机翻转。"""
        from gaussian_renderer import render
        score=torch.zeros_like(self.normal_sign[:,0]);count=torch.zeros_like(score)
        background=self.pc.get_xyz.new_zeros(3)
        for camera in cameras:
            camera.load2device()
            geometry=self.effective(camera)
            package=render(camera,geometry,self.pipe,background,0.,0.,0.,override_color=torch.zeros_like(geometry.get_xyz))
            points=torch.cat([geometry.get_xyz,torch.ones_like(geometry.get_xyz[:,:1])],-1)
            clip=points@camera.full_proj_transform
            h,w=camera.image_height,camera.image_width
            projected=clip[:,:2]/clip[:,3:].clamp_min(1e-8)
            u=((projected[:,0]+1)*w/2-.5).round().long()
            v=((projected[:,1]+1)*h/2-.5).round().long()
            inside=(clip[:,3]>0)&(u>=1)&(u<w-1)&(v>=1)&(v<h-1)
            u,v=u.clamp(0,w-1),v.clamp(0,h-1)
            depth=(points@camera.world_view_transform)[:,2]
            error=(depth-package['surf_depth'][0,v,u]).abs()/depth.clamp_min(1e-6)
            alpha=camera.gt_alpha_mask.cuda()[0]
            interior=-F.max_pool2d(-alpha[None,None],3,stride=1,padding=1)[0,0]
            valid=inside&(error<.01)&(interior[v,u]>.99)&(package['rend_alpha'][0,v,u]>.95)
            normal=build_rotation(geometry.get_rotation)[:,:,2]
            cosine=(normal*F.normalize(camera.camera_center-geometry.get_xyz,dim=-1)).sum(-1)
            valid &= cosine.abs()>.1
            score+=cosine*valid;count+=valid
            camera.load2device('cpu')
        self.normal_sign=torch.where(score[:,None]>=0,1.,-1.)
        self.normal_sign_valid=count>0
        self.radiance.gauge_support.copy_(self.normal_sign_valid)
        assert self.normal_sign_valid.any(),'无法从TRAIN可见几何确定有向法线'
        return dict(visible_surfaces=int(self.normal_sign_valid.sum()),total_surfaces=len(score),
                    flipped_surfaces=int((self.normal_sign<0).sum()),
                    unobserved_surfaces=int((count==0).sum()),sign_is_query_camera_independent=True)

    def components(self,camera,geometry=None,direct_weight=.5):
        assert self.normal_sign_valid.any(),'物理着色前必须完成TRAIN固定符号初始化'
        geometry=self.effective(camera) if geometry is None else geometry
        normal=build_rotation(geometry.get_rotation)[:,:,2]*self.normal_sign
        direction=self.radiance.lights(camera.fid)[None]
        visibility=self.visibility(geometry,direction)
        rho=self.radiance.rho
        direct=rho/math.pi*self.radiance.intensity*visibility*(normal@direction.T).clamp_min(0)
        if self.path=='A':
            environment=rho/math.pi*self.radiance.environment(normal)
            color=environment+direct_weight*direct
        else:
            environment=torch.zeros_like(direct)
            color=direct
        return dict(geometry=geometry,normal=normal,visibility=visibility,environment=environment,direct=direct,color=color)

    def render(self,camera,direct_weight=.5,components=False,protect_opacity=False):
        from gaussian_renderer import render
        values=self.components(camera,direct_weight=direct_weight)
        geometry=values['geometry'];zero=self.pc.get_xyz.new_zeros(3)
        package=render(camera,geometry,self.pipe,zero,0.,0.,0.,override_color=values['color'])
        package['render_linear']=package['render']
        package['render']=linear_to_srgb(package['render_linear'].clamp_min(0))
        package['surface_components']=values
        if protect_opacity:
            protected=render(camera,geometry,self.pipe,zero,0.,0.,0.,override_color=torch.zeros_like(geometry.get_xyz),detach_opacity=True)
            package['geometry_for_normal']=protected
        if components:
            env=render(camera,geometry,self.pipe,zero,0.,0.,0.,override_color=values['environment'])
            dl=render(camera,geometry,self.pipe,zero,0.,0.,0.,override_color=values['direct'])
            expected=env['render']+direct_weight*dl['render'] if self.path=='A' else dl['render']
            assert torch.equal(package['rend_alpha'],env['rend_alpha']) and torch.equal(package['rend_alpha'],dl['rend_alpha'])
            package['component_sum_max_error']=(expected-package['render_linear']).abs().max()
            package['environment_linear']=env['render'];package['direct_linear']=dl['render']
        return package
