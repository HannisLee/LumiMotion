"""独立多观测PS数学：完整逆转置归一化，观测生成detach伪法线。"""
import math

import torch
import torch.nn.functional as F


def transport_normal(normal, inverse_transpose):
    """列向量约定；非刚性分母依赖待估法线，不能忽略。"""
    mapped = torch.einsum('tnij,nj->tni', inverse_transpose, normal)
    return F.normalize(mapped, dim=-1)


def effective_light(inverse_transpose, directions):
    if directions.ndim == 2:
        directions = directions[:, None, :].expand(-1, inverse_transpose.shape[1], -1)
    return torch.einsum('tnij,tni->tnj', inverse_transpose, directions)


def local_response_design(normal, transport, directions, visibility, observation_valid, intensity):
    """实际运输在单位法线切向的两列导数，加共享rho的幅度列。

    法线仅有两自由度，第三列表示未知光度尺度。刚性、无遮挡截断时，
    该矩阵与传统三列灯向矩阵相差正交基变换。非线性运输不能沿用LS初值矩阵。
    """
    normal=normal.detach()
    reference=F.one_hot(normal.abs().argmin(-1),3).to(normal.dtype)
    first=F.normalize(torch.cross(normal,reference,dim=-1),dim=-1)
    second=torch.cross(normal,first,dim=-1)
    def response(candidate):
        return (transport(F.normalize(candidate,dim=-1))*directions).sum(-1).clamp_min(0)
    with torch.enable_grad():
        cosine,derivative_a=torch.autograd.functional.jvp(response,normal,first,create_graph=False)
        _,derivative_b=torch.autograd.functional.jvp(response,normal,second,create_graph=False)
    scale=visibility*observation_valid*(float(intensity)/math.pi)
    return torch.stack([derivative_a,derivative_b,cosine],-1).detach()*scale[...,None]


def estimate_normals(observed, directions, inverse_transpose, visibility, valid, saturated,
                     intensity, steps=200, minimum_observations=6, minimum_singular_ratio=.05,
                     normal_operator=None):
    """只从真实观测求解；不接受当前预测法线作为初始化或目标。

    observed为T×N×3线性观测（A传入已detach的EL残差，B传原始RGB）。
    inverse_transpose为固定当前形变的T×N×3×3法线运输；可见度也固定。
    不除以visibility；近零直射、饱和与负残差不进入PS，完整RGB损失另行保留。
    """
    observed, directions, inverse_transpose, visibility = [x.detach() for x in
        (observed, directions, inverse_transpose, visibility)]
    assert observed.ndim == 3 and observed.shape[-1] == 3
    t, count, _ = observed.shape
    assert inverse_transpose.shape == (t,count,3,3)
    assert valid.shape == visibility.shape == (t,count)
    assert saturated.shape == observed.shape
    assert float(intensity) > 0
    transport = (lambda n: transport_normal(n,inverse_transpose)) if normal_operator is None else normal_operator
    if directions.ndim == 2:
        directions = directions[:,None,:].expand(-1,count,-1)
    assert directions.shape == observed.shape
    directions=F.normalize(directions,dim=-1)
    finite = torch.isfinite(observed).all(-1) & torch.isfinite(inverse_transpose).all((-1,-2))
    finite &= torch.isfinite(visibility) & torch.isfinite(directions).all(-1)
    channel_valid = valid[...,None] & finite[...,None] & (~saturated) & (observed>=0)
    channel_valid &= visibility[...,None] > .2
    mask = channel_valid.float()
    # 各通道观测先合并为亮度用于独立LS初始化，不引用当前几何法线。
    colors = observed.nan_to_num(0.).clamp_min(0)
    lum = (colors*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    observation_valid = channel_valid.any(-1)
    design = effective_light(inverse_transpose.nan_to_num(0.), directions)
    design=design*visibility.nan_to_num(0.)[...,None]*(float(intensity)/math.pi)
    design=design*observation_valid[...,None]
    gram=torch.einsum('tni,tnj->nij',design,design)
    rhs=torch.einsum('tni,tn->ni',design,lum)
    ridge=gram.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-8)*1e-7
    solution=torch.linalg.solve(gram+ridge[:,None,None]*torch.eye(3,device=gram.device,dtype=gram.dtype),rhs)
    raw=torch.nn.Parameter(F.normalize(solution,dim=-1))
    initial=raw.detach().clone()
    with torch.no_grad():
        shading=visibility*(transport(initial)*directions).sum(-1).clamp_min(0)*(float(intensity)/math.pi)
        rho=(colors*shading[...,None]*mask).sum(0)/(shading.square()[...,None]*mask).sum(0).clamp_min(1e-8)
        rho=rho.clamp(.030001,.969999)
    material_raw=torch.nn.Parameter(torch.logit((rho-.03)/.94))
    optimizer=torch.optim.Adam([raw,material_raw],lr=.03)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        normal=F.normalize(raw,dim=-1)
        world=transport(normal)
        cosine=(world*directions).sum(-1).clamp_min(0)
        rho=.03+.94*material_raw.sigmoid()
        prediction=rho[None]*visibility[...,None]*cosine[...,None]*(float(intensity)/math.pi)
        difference=(prediction-colors)*mask
        robust=(difference.square()+.02**2).sqrt()-.02
        per_track=(robust*mask).sum((0,2))/mask.sum((0,2)).clamp_min(1)
        loss=per_track.mean()
        if not torch.isfinite(loss):
            raise ValueError('独立PS求解非有限，禁止回退当前预测法线')
        loss.backward()
        optimizer.step()
        if step in (0,steps-1):history.append(float(loss.detach()))
    with torch.no_grad():
        normal=F.normalize(raw,dim=-1)
        world=transport(normal)
        cosine=(world*directions).sum(-1).clamp_min(0)
        rho=.03+.94*material_raw.sigmoid()
        prediction=rho[None]*visibility[...,None]*cosine[...,None]*(float(intensity)/math.pi)
        relative_error=(((prediction-colors).abs()*mask).sum((0,2))/
                        (colors.abs()*mask).sum((0,2)).clamp_min(.01))
        actual_design=local_response_design(normal,transport,directions,visibility,observation_valid,intensity)
        # 每个RGB通道有各自未知rho；不能把来自不同通道的不足观测拼成假秩。
        # 保守要求至少一个完整通道自身满足帧数与三列秩，其他通道仍参与稳健拟合。
        informative=channel_valid&(cosine[...,None]>1e-6)
        per_channel=actual_design[:,:,None,:]*informative[...,None]
        channel_singular=torch.linalg.svdvals(per_channel.permute(1,2,0,3))
        channel_ratio=channel_singular[:,:,-1]/channel_singular[:,:,0].clamp_min(1e-8)
        channel_number=informative.sum(0)
        eligible=channel_number>=minimum_observations
        score=channel_ratio*eligible
        best=score.argmax(-1)
        row=torch.arange(count,device=normal.device)
        singular=channel_singular[row,best]
        ratio=score[row,best]
        initial_singular=torch.linalg.svdvals(design.permute(1,0,2))
        number=channel_number[row,best]
        reliable=(number>=minimum_observations)&(ratio>=minimum_singular_ratio)
        reliable &= (normal.norm(dim=-1)>.9)&torch.isfinite(normal).all(-1)&(relative_error<=.5)
        confidence=torch.exp(-(relative_error/.25)**2)*ratio.clamp(max=.25)/.25
        confidence=confidence*reliable
        return dict(normal=normal.detach(),albedo=rho.detach(),confidence=confidence.detach(),
                    reliable=reliable.detach(),valid_observations=number.detach(),
                    effective_singular_values=singular.detach(),effective_singular_ratio=ratio.detach(),
                    channel_singular_ratios=channel_ratio.detach(),channel_valid_observations=channel_number.detach(),
                    selected_confidence_channel=best.detach(),
                    initial_linear_singular_ratio=(initial_singular[:,-1]/initial_singular[:,0].clamp_min(1e-8)).detach(),
                    confidence_design='逐RGB通道实际运输两切向导数及未知rho幅度；至少单通道独立满足帧数/秩；背光不计帧',
                    relative_photometric_error=relative_error.detach(),initial_normal=initial,
                    loss_history=history,targets_stopped_gradient=True,
                    transport='normalized_inverse_transpose' if normal_operator is None else 'explicit_native_surface_operator',
                    normal_source='多观测图像与固定运输/光照/可见度求解；无当前法线输入')
