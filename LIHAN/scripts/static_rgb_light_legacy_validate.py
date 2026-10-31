"""历史Stage1同协议DEV验证；可启用原生逐帧形变，额外记录模型内部统计。"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scene.deform_model import DeformModel
from utils.general_utils import build_rotation
from LIHAN.scripts import static_rgb_light_extract as base
from LIHAN.scripts.static_rgb_light_prepare import write, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True)
    p.add_argument('--deform',action='store_true')
    a=p.parse_args()
    original_load,original_render=base.load,base.render
    stats=[]; positions=[]; directions=[]
    def load(model,iteration):
        g=original_load(model,iteration)
        n=build_rotation(g.get_rotation)[:,:,2]
        eig=torch.linalg.eigvalsh(n.T@n/len(n)).cpu().tolist()
        g.internal_normal_distribution=dict(second_moment_eigenvalues=eig,
            condition=eig[-1]/max(eig[0],1e-12),
            abs_z_quantiles=torch.quantile(n[:,2].abs(),torch.tensor([0.,.5,.9,.99,1.],device='cuda')).cpu().tolist())
        if a.deform:
            g.legacy_deform=DeformModel(deform_type='mlp',is_blender=True,hyper_dim=1,pred_color=True)
            assert g.legacy_deform.load_weights(str(model),iteration), '缺少形变checkpoint'
            g.legacy_deform.deform.eval()
        return g
    def render(c,g,pipe,bg,*unused,**kwargs):
        d=dict(d_xyz=0.,d_rotation=0.,d_scaling=0.,d_opacity=None,d_color=None)
        if a.deform:
            d=g.legacy_deform.step(g.get_xyz,c.fid.unsqueeze(0).expand(len(g.get_xyz),-1),
                 iteration=a.iteration,feature=g.get_binary_feature(),camera_center=c.camera_center)
            positions.append((g.get_xyz+d['d_xyz']).cpu().numpy())
            directions.append(build_rotation(g.get_rotation_bias(d['d_rotation']))[:,:,2].cpu().numpy())
        r=original_render(c,g,pipe,bg,d['d_xyz'],d['d_rotation'],d['d_scaling'],
             d_opacity=d['d_opacity'],d_color=d['d_color'])
        normal=F.normalize(r['rend_normal'],dim=0)
        mask=(c.gt_alpha_mask[0].cuda()>.99)&(r['rend_alpha'][0]>.95)
        nn=normal[:,mask]
        eig=torch.linalg.eigvalsh(nn@nn.T/max(nn.shape[1],1)).cpu().tolist()
        stats.append(dict(frame=int(c.image_name_train_light.split('_')[-1]),
             normal_second_moment_eigenvalues=eig,normal_condition=eig[-1]/max(eig[0],1e-12),
             displacement_max=float(d['d_xyz'].norm(dim=1).max()) if torch.is_tensor(d['d_xyz']) else 0.,
             displacement_median=float(d['d_xyz'].norm(dim=1).median()) if torch.is_tensor(d['d_xyz']) else 0.,
             displacement_fraction_over_001=float((d['d_xyz'].norm(dim=1)>.001).float().mean()) if torch.is_tensor(d['d_xyz']) else 0.,
             displacement_fraction_over_01=float((d['d_xyz'].norm(dim=1)>.01).float().mean()) if torch.is_tensor(d['d_xyz']) else 0.,
             canonical_normal_distribution=g.internal_normal_distribution))
        return r
    base.load,base.render=load,render
    result=base.validate(a.source,a.model,a.iteration,a.output)
    stability=None
    if positions:
        x=np.asarray(positions); n=np.asarray(directions)
        spread=np.sqrt(np.square(x-x.mean(0)).sum(-1).mean(0))
        angle=np.degrees(np.arccos(np.clip(np.abs((n*n[:1]).sum(-1)),0,1)))
        stability=dict(scope='同一Gaussian在DEV18帧的模型几何变化；非GT误差',
            position_rms_quantiles=np.quantile(spread,[0,.5,.9,.99,1]).tolist(),
            position_rms_fraction_over_001=float((spread>.001).mean()),
            normal_axis_angle_vs_first_quantiles=np.quantile(angle,[0,.5,.9,.99,1]).tolist())
    write(a.output/'legacy_internal.json',dict(gt_used=False,
        render_mode='原生逐帧形变与颜色' if a.deform else '静态canonical PLY',
        checkpoint_sha256=sha(a.model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply'),
        deform_sha256=sha(a.model/f'deform/iteration_{a.iteration}/deform.pth'),
        temporal_stability=stability,per_frame=stats))


if __name__=='__main__':main()
