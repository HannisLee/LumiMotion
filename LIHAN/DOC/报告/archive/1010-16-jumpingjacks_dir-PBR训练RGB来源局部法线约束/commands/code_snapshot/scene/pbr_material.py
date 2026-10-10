"""原生PBR材质可选初始化修复，只读取Stage1预测参数。"""
import math
import torch
from utils.sh_utils import RGB2SH, SH2RGB


def validate_albedo_initialization(args):
    if not (getattr(args, 'pbr_albedo_interior_init', False) or
            getattr(args, 'pbr_albedo_project_after_step', False)):
        return
    if args.render_mode != 'original_ir' or args.pbr_light_mode != 'gt_directional':
        raise ValueError('PBR材质内部区间初始化只支持原生方向光Stage2。')
    margin = args.pbr_albedo_init_margin
    if not math.isfinite(margin) or not 0 < margin < .47:
        raise ValueError('PBR材质初始化margin必须有限且在(0,.47)内。')


@torch.no_grad()
def initialize_native_albedo(pc, margin=.001):
    """仅投影真实_albedo_dc初值，恢复硬clamp外的梯度；不改SH着色来源。"""
    if not math.isfinite(margin) or not 0 < margin < .47:
        raise ValueError('PBR材质初始化margin必须有限且在(0,.47)内。')
    dc = pc._albedo_dc
    if dc.ndim != 3 or dc.shape[1:] != (1,3) or not dc.shape[0] or not torch.isfinite(dc).all():
        raise ValueError('PBR来源材质DC形状或有限性无效。')
    sh_source = getattr(pc, '_albedo_dc_stage1', None)
    if sh_source is not None and dc.data_ptr() == sh_source.data_ptr():
        raise ValueError('PBR材质DC与Stage1 SH来源共享存储，拒绝污染SH来源。')
    rgb = SH2RGB(dc.detach())
    initial_appearance = rgb.clamp(.03,.97)
    projected = rgb.clamp(.03+margin,.97-margin)
    dc.copy_(RGB2SH(projected))
    actual = SH2RGB(dc)
    return dict(method='source_prediction_native_DC_interior_projection',
                margin=margin, interval=[.03+margin,.97-margin],
                points=len(dc), source_clamped_channels_fraction=float(((rgb<=.03)|(rgb>=.97)).float().mean()),
                initialized_channels_fraction=float((projected!=rgb).float().mean()),
                after_clamped_channels_fraction=float(((actual<=.03)|(actual>=.97)).float().mean()),
                max_initial_appearance_difference_srgb=float((actual.clamp(.03,.97)-initial_appearance).abs().max()),
                gt_material_inputs=False, gt_normal_inputs=False, gt_geometry_inputs=False,
                stage1_SH_source_modified=False)


@torch.no_grad()
def project_native_albedo(pc, margin=.001):
    """原生Adam更新后保持真实DC在可导区间，避免下一批次再次进入硬clamp死区。"""
    if not math.isfinite(margin) or not 0 < margin < .47:
        raise ValueError('PBR材质投影margin必须有限且在(0,.47)内。')
    dc = pc._albedo_dc
    if dc.ndim != 3 or dc.shape[1:] != (1,3) or not dc.shape[0] or not torch.isfinite(dc).all():
        raise ValueError('PBR材质投影DC形状或有限性无效。')
    sh_source = getattr(pc, '_albedo_dc_stage1', None)
    if sh_source is not None and dc.data_ptr() == sh_source.data_ptr():
        raise ValueError('PBR材质DC与Stage1 SH来源共享存储，拒绝污染SH来源。')
    pc._albedo_dc.copy_(RGB2SH(SH2RGB(pc._albedo_dc).clamp(.03+margin,.97-margin)))
