"""Stage2可选来源输出gate：优化PLY feature，始终冻结来源MLP。"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from scene.pbr_deformation import _source_assets, _sha

METADATA = 'pbr_feature_gate.json'
METHOD = 'stage1_fixed_MLP_native_output_feature_gate'
PROBABILITY_METHOD = 'stage1_fixed_MLP_native_probability_output_feature_gate'


def native_probability(feature):
    """与GaussianModel.get_binary_feature(eval=True)逐操作相同。"""
    u = torch.ones_like(feature)*.5
    return torch.sigmoid(2*(feature+torch.log(u)-torch.log(1-u)))


@torch.no_grad()
def encode_probability(probability, source, source_probability):
    """有限PLY logit；未改变的概率保留来源raw feature逐bit原值。"""
    if (probability.shape != source.shape or probability.dtype != source.dtype or
            not torch.isfinite(probability).all() or
            (probability < 0).any() or (probability > 1).any()):
        raise ValueError('gate概率必须与来源同shape/dtype且在[0,1]内。')
    raw = .5*(probability.log()-torch.log1p(-probability))
    raw = torch.where(probability == 0, -80., raw)
    raw = torch.where(probability == 1, 80., raw)
    return torch.where(probability == source_probability, source, raw)


def validate_feature_gate(args):
    if not getattr(args, 'pbr_optimize_feature_gate', False):
        return
    if (args.render_mode != 'original_ir' or args.pbr_light_mode != 'gt_directional' or
            not getattr(args, 'pbr_optimize_geometry', False) or
            not getattr(args, 'pbr_deformation_anchor', False) or
            getattr(args, 'no_binary_separation', False) or getattr(args, 'hyper_dim', 1) != 1 or
            getattr(args, 'deform_type', 'mlp') != 'mlp'):
        raise ValueError('输出gate微调需要方向光原生PBR几何、来源XYZ锚定及单维MLP feature。')
    for key in ['pbr_feature_gate_lr', 'pbr_feature_gate_max_logit_delta']:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError('gate学习率与logit信赖域须有限且为正。')
    weight = args.lambda_pbr_feature_gate_anchor
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('来源gate锚定权重须有限且非负。')
    mode = getattr(args, 'pbr_feature_gate_parameterization', 'raw_logit')
    if mode not in ['raw_logit', 'probability']:
        raise ValueError('未知gate参数化方式。')
    if mode == 'probability':
        maximum = args.pbr_feature_gate_max_probability_delta
        if not math.isfinite(maximum) or not 0 < maximum <= 1:
            raise ValueError('gate概率信赖域须在(0,1]内。')


def gate_anchor(feature, source, maximum):
    return ((feature-source)/maximum).square().mean()


@torch.no_grad()
def project_gate(feature, source, maximum):
    feature.copy_(source+(feature-source).clamp(-maximum, maximum))


def _source_gate(model, iteration, pc, deform):
    _, files, snapshots = _source_assets(model, iteration)
    source = PlyData.read(model/f'point_cloud/iteration_{iteration}/point_cloud.ply')['vertex'].data
    feature = np.asarray(source['fea_0']).copy()[:, None]
    if feature.dtype != np.float32 or tuple(feature.shape) != tuple(pc.feature.shape) or not np.isfinite(feature).all():
        raise ValueError('来源gate shape/dtype/有限性无效。')
    weights = torch.load(model/f'deform/iteration_{iteration}/deform.pth', map_location='cpu')
    current = deform.deform.state_dict()
    if set(weights) != set(current) or any(not torch.equal(weights[k], current[k].detach().cpu()) for k in weights):
        raise ValueError('输出gate只允许完全冻结的原来源MLP。')
    return torch.as_tensor(feature, device=pc.feature.device), files, snapshots


class NativePBRFeatureGate:
    def __init__(self, model, pc, deform, args, source_iteration):
        validate_feature_gate(args)
        model = Path(model)
        self.source, files, snapshots = _source_gate(model, source_iteration, pc, deform)
        if not torch.equal(pc.feature.detach(), self.source):
            raise ValueError('gate只允许从原35k feature值初始化。')
        self.maximum = args.pbr_feature_gate_max_logit_delta
        self.parameterization = getattr(args, 'pbr_feature_gate_parameterization', 'raw_logit')
        self.probability = None
        if self.parameterization == 'probability':
            self.source_probability = native_probability(self.source)
            self.maximum = args.pbr_feature_gate_max_probability_delta
            self.probability = torch.nn.Parameter(self.source_probability.clone())
        deform.deform.requires_grad_(False)
        record = dict(format_version=1, method=METHOD, source_iteration=source_iteration,
            points=len(self.source), source_files=files, source_snapshots=snapshots,
            max_logit_delta=self.maximum, learning_rate=args.pbr_feature_gate_lr,
            anchor_weight=args.lambda_pbr_feature_gate_anchor, source_MLP_frozen=True,
            gt_material_inputs=False, gt_normal_inputs=False, gt_geometry_inputs=False,
            gt_light_photometry_inputs=False)
        if self.probability is not None:
            record.update(format_version=2, method=PROBABILITY_METHOD,
                parameterization='probability', max_probability_delta=self.maximum,
                max_logit_delta=None, finite_endpoint_logit=80.,
                forward='native_saved_PLY_sigmoid', gradient='probability_coordinate_identity',
                optimizer_state_saved=False)
        asset = model/METADATA
        if asset.exists():
            raise FileExistsError('gate来源记录禁止覆盖。')
        asset.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        self.metadata_sha = _sha(asset)

    def loss(self, pc):
        if self.probability is not None:
            return gate_anchor(self.probability, self.source_probability, self.maximum)
        return gate_anchor(pc.feature, self.source, self.maximum)

    def optimizer_parameters(self, pc):
        return [self.probability] if self.probability is not None else [pc.feature]

    def output(self, pc):
        native = pc.get_binary_feature()
        if self.probability is None:
            return native
        # 前向逐bit使用真实保存logit的原生值；梯度以概率坐标传递。
        return native.detach()+(self.probability-self.probability.detach())

    @torch.no_grad()
    def project(self, pc):
        if self.probability is not None:
            if not torch.isfinite(self.probability).all():
                raise ValueError('优化后的gate概率非有限。')
            lower = (self.source_probability-self.maximum).clamp(0, 1)
            upper = (self.source_probability+self.maximum).clamp(0, 1)
            self.probability.copy_(torch.maximum(lower, torch.minimum(upper, self.probability)))
            pc.feature.copy_(encode_probability(self.probability, self.source, self.source_probability))
            # 同步float32逆sigmoid量化，下一步与实际PLY前向完全一致。
            self.probability.copy_(pc.get_binary_feature())
            return
        project_gate(pc.feature, self.source, self.maximum)


def validate_feature_gate_model(model, pc, deform, args, iteration):
    """所有冻结渲染入口检查相同来源、实际PLY信赖域与checkpoint绑定。"""
    if not getattr(args, 'pbr_optimize_feature_gate', False):
        return None
    validate_feature_gate(args)
    model = Path(model)
    asset = model/METADATA
    record = json.loads(asset.read_text())
    probability_mode = getattr(args, 'pbr_feature_gate_parameterization', 'raw_logit') == 'probability'
    if (record.get('format_version') != (2 if probability_mode else 1) or
            record.get('method') != (PROBABILITY_METHOD if probability_mode else METHOD) or
            record.get('source_MLP_frozen') is not True or
            any(record.get(k) is not False for k in ['gt_material_inputs', 'gt_normal_inputs',
                'gt_geometry_inputs', 'gt_light_photometry_inputs'])):
        raise ValueError('gate来源协议无效。')
    source_iteration = record.get('source_iteration')
    source, files, snapshots = _source_gate(model, source_iteration, pc, deform)
    if (record.get('points') != len(source) or record.get('source_files') != files or
            record.get('source_snapshots') != snapshots or source_iteration != args.load_iter or
            iteration < source_iteration or
            (not probability_mode and record.get('max_logit_delta') != args.pbr_feature_gate_max_logit_delta) or
            record.get('learning_rate') != args.pbr_feature_gate_lr or
            record.get('anchor_weight') != args.lambda_pbr_feature_gate_anchor):
        raise ValueError('gate配置/来源不匹配。')
    if not torch.isfinite(pc.feature).all():
        raise ValueError('checkpoint feature非有限。')
    if probability_mode:
        if (record.get('parameterization') != 'probability' or
                record.get('max_probability_delta') != args.pbr_feature_gate_max_probability_delta or
                record.get('max_logit_delta') is not None or record.get('finite_endpoint_logit') != 80. or
                record.get('forward') != 'native_saved_PLY_sigmoid' or
                record.get('gradient') != 'probability_coordinate_identity' or
                record.get('optimizer_state_saved') is not False or
                (native_probability(pc.feature.detach())-native_probability(source)).abs().max() >
                args.pbr_feature_gate_max_probability_delta+1e-6):
            raise ValueError('checkpoint概率gate协议或信赖域无效。')
    elif (pc.feature.detach()-source).abs().max() > args.pbr_feature_gate_max_logit_delta+1e-5:
        raise ValueError('checkpoint feature超出来源logit信赖域。')
    if iteration > source_iteration:
        marker = json.loads((model/f'pbr_directional/iteration_{iteration}/lighting.json').read_text())
        if marker.get('iteration') != iteration or marker.get('feature_gate_metadata_sha256') != _sha(asset):
            raise ValueError('checkpoint没有绑定同一来源gate。')
    elif not torch.equal(pc.feature.detach(), source):
        raise ValueError('来源iteration gate已改变。')
    return record
