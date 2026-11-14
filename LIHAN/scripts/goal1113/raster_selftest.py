"""仅合成高斯，验证物理公式的光栅线性合成。"""
import math
import torch
from . import physics as P


@torch.no_grad()
def run():
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getProjectionMatrix
    device='cuda'; dtype=torch.float64
    means=torch.tensor([[0.,0.,2.],[.15,.05,2.1]],device=device)
    normals=torch.tensor([[[0.,0.,-1.],[0.,0.,-1.]]],device=device,dtype=dtype)
    view=torch.nn.functional.normalize(-means.to(dtype),dim=-1)[None]
    rho=torch.tensor([[.3,.4,.5],[.6,.4,.2]],device=device,dtype=dtype)
    light=torch.tensor([[0.,0.,-1.]],device=device,dtype=dtype)
    env=torch.full((128,3),.3,device=device,dtype=dtype)
    direct,static=P.components(normals,view,rho,env,light,1.)
    proj=getProjectionMatrix(znear=.01,zfar=100.,fovX=math.pi/2,fovY=math.pi/2).T.cuda()
    settings=GaussianRasterizationSettings(image_height=32,image_width=32,tanfovx=1.,tanfovy=1.,
        bg=torch.zeros(3,device=device),scale_modifier=1.,viewmatrix=torch.eye(4,device=device),
        projmatrix=proj,sh_degree=0,campos=torch.zeros(3,device=device),prefiltered=False,debug=False)
    raster=GaussianRasterizer(settings)
    def draw(colors):
        out,_,_=raster(means3D=means,means2D=torch.zeros_like(means),shs=None,colors_precomp=colors.float().contiguous(),
             opacities=torch.ones((2,1),device=device)*.8,scales=torch.ones((2,2),device=device)*.2,
             rotations=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]],device=device),cov3D_precomp=None)
        return out
    d=draw(direct[0]); s=draw(static[0]); both=draw(direct[0]+static[0])
    if d.abs().sum()<=0 or s.abs().sum()<=0: raise AssertionError('合成高斯未投影')
    error=(both-d-s).abs().max().item()
    if error>2e-5: raise AssertionError(f'光栅线性误差 {error}')
    return {'synthetic_raster_additivity_max_abs':error,'indirect':0}
