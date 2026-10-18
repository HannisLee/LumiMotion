"""在已验证纯几何初始化后只注入TRAIN预适配DC，保留同一物理训练实现。"""
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from scene.eldl_v3_geometry_init import GeometryWarmstartTrainer, KEYS, read, sha, require
from scene.eldl_v3_trainer import ELDLTrainer

ENTRY = 'scripts/train_eldl_v3_material_warmstart.py'


def validate_material(args):
    source = Path(args.source_path).resolve()
    control = source.parent
    repo = control.parent.parent
    smoke = args.smoke_schedule
    index = ([3, 20] if smoke else [1000, 16000]).index(args.geometry_step)
    names = ['1019-20-ELDL-V3-短SH固定几何材质预适配', '1019-21-ELDL-V3-长SH固定几何材质预适配'] if smoke else ['1019-14-LH-static-jumpingjacks-ELDL-V3-P01短SH固定几何材质预适配4050', '1019-15-LH-static-jumpingjacks-ELDL-V3-P02长SH固定几何材质预适配4050']
    experiment = repo / ('output/smoke_test' if smoke else 'output') / names[index]
    fit = experiment / 'material_fit'
    manifest_path = fit / 'material_manifest.json'
    review_path = control / ('material_preadapt_smoke_pair_v2' if smoke else 'material_preadapt_pair_v1') / 'receipt.json'
    require(Path(args.material_manifest).resolve() == manifest_path and Path(args.material_pair_review).resolve() == review_path, '只允许本轮登记双父材质及配对验收。')
    review, manifest = read(review_path), read(manifest_path)
    audit_source = repo / 'scripts/audit_eldl_v3_material_preadapt.py'
    require(review['input_sha256'].get(str(audit_source)) == sha(audit_source) == 'ddceeef47915cdcb7addf76cdc5fe1ffdf2e5648fece99d5cf75b3bc3f830f04', '配对收据不是最终已审计门禁。')
    require(review['smoke'] == manifest['smoke'] == smoke and review['gt_read'] is False and review['external_test_read'] is False, '配对验收身份/GT边界不符。')
    expected_status = 'PASS_MATERIAL_PREADAPT_SMOKE_ENGINEERING' if smoke else 'PASS_MATERIAL_PREADAPT_PAIR_BUDGET'
    require(review['status'] == expected_status and review['formal_use_permitted'] == review['pair_eligible'] == (not smoke), '双父预算未共同通过。')
    require(review['allowed_geometry_steps'] == ([3, 20] if smoke else [1000, 16000]) and len(review['entries']) == 2 and review['common_camera_order'], '配对端点/共同序列不完整。')
    pair_files = []
    parent = read(args.geometry_parent_manifest)
    for j, step in enumerate(review['allowed_geometry_steps']):
        other_fit = repo / ('output/smoke_test' if smoke else 'output') / names[j] / 'material_fit'
        other_paths = [other_fit / 'material_manifest.json', other_fit / 'configuration.json', other_fit / f'dc_step_{20 if smoke else 4050}.pt']
        other, other_config = read(other_paths[0]), read(other_paths[1])
        entry = review['entries'][j]
        require(entry['manifest'] == str(other_paths[0]) and entry['material'] == str(other_paths[2]) and entry['parent_geometry_step'] == other['parent_geometry_step'] == step, '配对另一端不是固定端点。')
        require(entry['manifest_sha256'] == sha(other_paths[0]) and entry['material_sha256'] == other['material_sha256'] == sha(other_paths[2]) and other['configuration_sha256'] == sha(other_paths[1]), '配对另一端资产变化。')
        require(other['smoke'] == smoke and other['budget_eligible'] == entry['budget_eligible'] == (not smoke) and other['source'] == other_config['source'] == str(source), '配对另一端预算/源变化。')
        geometry = other['geometry_binding']
        require(geometry == other_config['geometry_binding'] and geometry['parent_manifest_sha256'] == args.geometry_initialization_binding['parent_manifest_sha256'] and geometry['parent_manifest'] == args.geometry_initialization_binding['parent_manifest'] and geometry['geometry_sha256'] == entry['geometry_sha256'] == parent['endpoints'][str(step)]['geometry_sha256'], '配对另一端不属于同一SH父轨迹。')
        require(other_config['input_sha256'].get(str(source / 'estimated_eldl_train.json')) == sha(source / 'estimated_eldl_train.json'), '配对另一端固定光不同。')
        pair_files.extend(other_paths)
    endpoint = review['entries'][index]
    require(endpoint['manifest'] == str(manifest_path) and endpoint['manifest_sha256'] == sha(manifest_path) and endpoint['parent_geometry_step'] == args.geometry_step, '材质manifest未被当前配对收据绑定。')
    steps = 20 if smoke else 4050
    material = fit / f'dc_step_{steps}.pt'
    config_path = fit / 'configuration.json'
    config = read(config_path)
    require(manifest['material'] == endpoint['material'] == str(material) and manifest['material_sha256'] == endpoint['material_sha256'] == sha(material), '必须使用登记终点DC。')
    require(manifest['configuration_sha256'] == sha(config_path) and manifest['geometry_binding'] == config['geometry_binding'] == args.geometry_initialization_binding, '材质与当前SH父几何/点序不一致。')
    require(manifest['parent_geometry_step'] == args.geometry_step and manifest['source'] == config['source'] == str(source), '父步数或原始源不符。')
    require(manifest['completed_optimizer_steps'] == steps and manifest['only_tensor'] == config['only_trainable_tensor'] == '_albedo_dc' and manifest['gt_supervision'] is False and manifest['external_test_read'] is False, '终点预算或训练边界不符。')
    require(manifest['budget_eligible'] == (not smoke) and manifest['formal_use_permitted'] is False and manifest['pair_eligible'] is False, '单父资格与配对资格不符。')
    require(manifest['non_dc_bitexact'] and manifest['all_geometry_render_maps_bitexact'] and manifest['no_optimizer_rng_transfer'], '冻结/不继承状态声明不足。')
    require(config['input_sha256'].get(str(source / 'estimated_eldl_train.json')) == sha(source / 'estimated_eldl_train.json'), '预适配固定光与本次不同。')
    files = list(dict.fromkeys([review_path, manifest_path, material, config_path, audit_source, *pair_files]))
    for path in files[1:]:
        require(review['input_sha256'].get(str(path)) == sha(path), '配对验收未覆盖当前材质资产。')
    return dict(material=str(material), material_sha256=sha(material), material_manifest=str(manifest_path), material_manifest_sha256=sha(manifest_path), pair_review=str(review_path), pair_review_sha256=sha(review_path), geometry_sha256=args.geometry_initialization_binding['geometry_sha256'], parent_geometry_step=args.geometry_step, points=manifest['points'], source=str(source), lighting_sha256=sha(source / 'estimated_eldl_train.json'), preadapt_updates=steps, train_frames=config['train_frames'], gt_supervision=False, external_test_read=False, only_tensor='_albedo_dc', optimizer_loaded=False, rng_loaded=False, approved_paths=[str(p) for p in files], input_sha256={str(p): sha(p) for p in files})


class MaterialWarmstartTrainer(GeometryWarmstartTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.args
        binding = config.material_initialization_binding
        pc = self.gaussians
        model = Path(config.model_path)
        original = {key: value.detach().clone() for key, value in vars(pc).items() if isinstance(value, torch.Tensor) and key != '_albedo_dc'}
        material = torch.load(binding['material'], map_location='cpu', weights_only=True)
        require(set(material) == {'_albedo_dc'}, '不得继承其他材质/几何/优化器。')
        dc = material['_albedo_dc']
        require(dc.dtype == torch.float32 and dc.shape == pc._albedo_dc.shape == (binding['points'], 1, 3) and torch.isfinite(dc).all(), '材质shape/点数/有限性不符。')
        c0 = .28209479177387814
        require(float(dc.min()) >= (.031 - .5) / c0 - 1e-6 and float(dc.max()) <= (.969 - .5) / c0 + 1e-6, '材质超登记边界。')
        with torch.no_grad():
            pc._albedo_dc.copy_(dc.cuda())
        require(all(torch.equal(value, getattr(pc, key)) for key, value in original.items()), '材质注入改变非DC状态。')
        require(not pc.optimizer.state and not self.light_optimizer.state and self.viewpoint_stack is None, '必须新Adam和空相机栈。')
        require(not pc.xyz_gradient_accum.any() and not pc.denom.any() and not pc.max_radii2D.any(), '增密统计不是零。')
        require(all(not p.requires_grad for p in self.deform.deform.parameters()), 'MLP必须冻结。')
        # 保留父构造阶段的共同灰色零步，新的实际训练零步另存，失败现场也可追溯。
        before = model / 'pre_material_conversion'
        before.mkdir()
        for relative in ('point_cloud/iteration_0', 'deform/iteration_0', 'static_dl_lighting/iteration_0', 'optimizer_0.pt'):
            src = model / relative
            dst = before / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        torch.save({'_albedo_dc': dc.clone()}, model / 'effective_initial_material.pt')
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        current = self.iteration
        self.iteration = 0
        ELDLTrainer._save_physical(self)
        self.iteration = current
        zero_path = model / 'static_dl_lighting/iteration_0/lighting.json'
        zero = read(zero_path)
        zero.update(actual_entrypoint=ENTRY, relative_physical_step=0, completed_optimizer_calls=0, geometry_parent_manifest_sha256=config.geometry_initialization_binding['parent_manifest_sha256'], geometry_initialization_step=config.geometry_step, material_manifest_sha256=binding['material_manifest_sha256'], next_optimizer_step=1, initial_lrs={g['name']: float(g['lr']) for g in pc.optimizer.param_groups})
        zero_path.write_text(json.dumps(zero, ensure_ascii=False, indent=2) + '\n')
        root = Path(__file__).resolve().parents[1]
        sources = {str(root / relative): sha(root / relative) for relative in (ENTRY, 'scene/eldl_v3_material_init.py')}
        record = dict(status='PASS_TRAIN_MATERIAL_CONVERSION', actual_entrypoint=ENTRY, material_binding=binding, loaded_tensor_keys=['_albedo_dc'], points=len(pc.get_xyz), geometry_bitexact=True, other_attributes_bitexact=True, train_material_loaded=True, mlp_loaded=False, optimizer_loaded=False, rng_loaded=False, empty_adam=True, densification_statistics_zero=True, viewpoint_stack_empty=True, reseeded_after_conversion=True, seed=0, relative_physical_step=0, gt_supervision=False, external_test_read=False, effective_initial_material_sha256=sha(model / 'effective_initial_material.pt'), source_sha256=sources, geometry_initialization_interpretation='geometry_initialization.json记录注入材质前的纯几何构造；实际训练零步为该构造加本文件登记DC')
        (model / 'material_initialization.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        for name in ('eldl_v3_training.json', 'unknown_light_training.json', 'static_dl_training.json'):
            path = model / name
            value = read(path)
            value.update(actual_entrypoint=ENTRY, material_initialization=record, weights_loaded_from_parent=True, initialization='同P00四几何及独立135TRAIN预适配DC；新Adam/RNG，未加载SH颜色、MLP或GT权重', train_material_loaded=True)
            if 'new_sources' in value:
                value['new_sources'].update(sources)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def _save_physical(self):
        super()._save_physical()
        path = Path(self.args.model_path) / f'static_dl_lighting/iteration_{self.iteration}/lighting.json'
        value = read(path)
        value.update(actual_entrypoint=ENTRY, material_manifest_sha256=self.args.material_initialization_binding['material_manifest_sha256'])
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
