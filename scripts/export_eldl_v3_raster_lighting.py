"""汇合全图校准、冻结验证、实际目检及方向反证，导出探索性固定TRAIN灯光。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'calibration', 'validation', 'profile', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    for name, value in vars(a).items():
        setattr(a, name, value.resolve())
    require(not a.output.exists(), '禁止覆盖已有资产或失败结果')
    source = read(a.source / 'no_gt_manifest.json')
    require(set(x.name for x in a.source.iterdir()) ==
            {'images', 'points3d.ply', 'transforms_train.json', 'no_gt_manifest.json'}, '只接受原四项TRAIN输入')
    require(source['gt_read'] is False and source['init_weights_loaded'] is False and
            source['dataset'] == 'LH-static-jumpingjacks-ELDL-V3', '必须从头无GT源')
    frames = [i for i in range(1, 151) if i % 10]
    require(source['train_frames'] == frames, '必须恰好135 TRAIN')
    selected = read(a.calibration / 'endpoint_selection.json')
    require(selected['status'] == 'FIT_BUDGET_PASS' and not selected['held_used'] and
            selected['selected_by'] == 'terminal_fit_mse_only', '终点必须预算充分且仅按fit选择')
    require(selected['selected'] == 'init_3' and selected['epoch'] == 30 and
            selected['cumulative_epochs'] == 90, '当前只接纳预登记v2终点')
    checkpoint = a.calibration / selected['selected'] / 'checkpoint_epoch_30.pt'
    validation = read(a.validation / 'summary.json')
    manual = read(a.validation / 'manual_review.json')
    receipt = read(a.validation / 'validation_receipt.json')
    profile = read(a.profile / 'summary.json')
    require(validation['status'] == 'PASS_NUMERIC_PENDING_DIRECTION_PROFILE_AND_REVIEW' and
            all(validation['checks'].values()), '冻结数值验证未通过')
    require(not any(validation[k] for k in ('gt_read', 'external_test_read', 'held_used_to_reselect')) and
            validation['selected_init'] == 3, '验证越界或重新选择')
    require(manual['status'] == 'PASS_MANUAL_ENGINEERING_REVIEW_WITH_LIMITATIONS' and
            manual['actual_review'] and manual['no_gt_or_external_test_read'] and
            not manual['held_used_to_reselect'], '需要实际四类目检')
    require(manual['numeric_summary_sha256'] == sha(a.validation / 'summary.json') and
            receipt['numeric_summary_sha256'] == sha(a.validation / 'summary.json') and
            receipt['manual_review_sha256'] == sha(a.validation / 'manual_review.json') and
            receipt['selected_checkpoint_sha256'] == sha(checkpoint) and receipt['exit_code'] == 0,
            '验收收据绑定失败')
    require(profile['eligible_for_exploratory_geometry'] and
            profile['status'] in ('FINITE_LOCAL_SUPPORT_NO_UNIQUENESS_CLAIM', 'UNRESOLVED_NO_REFUTATION') and
            profile['direction_identifiability'] == 'unresolved' and
            profile['selected_checkpoint_sha256'] == sha(checkpoint), '方向反证不准入或终点变化')
    require(len(manual['actually_viewed_contacts_sha256']) >= 9 and
            set(manual['four_categories_observations']) >= {'RGB', 'Alpha', 'Normal', 'Separation'},
            '缺少实际目检证据')
    bindings = {}
    for record in (validation['input_sha256'], profile['input_sha256'],
                   manual['actually_viewed_contacts_sha256'], manual['actually_viewed_individual_RGB_sha256']):
        for path, expected in record.items():
            require(sha(path) == expected, '历史验收输入变化：' + path)
            bindings[path] = expected
    require(bindings.get(str(a.source / 'no_gt_manifest.json')) == sha(a.source / 'no_gt_manifest.json') and
            bindings.get(str(checkpoint)) == sha(checkpoint), '验证未绑定当前源/终点')
    for relative, expected in source['files'].items():
        path = (a.source / relative).resolve()
        require(path.is_relative_to(a.source) and sha(path) == expected, 'TRAIN源SHA变化')
    for path in (a.calibration / 'endpoint_selection.json', a.validation / 'summary.json',
                 a.validation / 'manual_review.json', a.validation / 'validation_receipt.json',
                 a.profile / 'summary.json', Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(not ck['constant'] and ck['epoch'] == 30 and ck['cumulative_epoch'] == 90, '错误终点')
    directions, environment = ck['directions'], ck['environment']
    require(directions.shape == (135, 3) and torch.isfinite(directions).all() and
            (directions.norm(dim=-1) - 1).abs().max() < 1e-5, '方向不合法')
    require(environment.shape == (3,) and torch.isfinite(environment).all() and
            ((environment / math.pi > .01) & (environment / math.pi < 30)).all(), 'EL不合法')
    meta_path = a.calibration.parent / 'cache_v3_geometry_bias/manifest.json'
    require(str(meta_path) in bindings, '缺失bias来源')
    bias = read(meta_path)['shadow_bias']
    records = {f'{frame:04d}': {'light_dir_world': direction}
               for frame, direction in zip(frames, directions.tolist())}
    provenance = dict(input_sha256=bindings, no_weights_transferred=True,
                      geometry_source='I00只见117fit；只导出灯参数，正式几何从原随机100k点重新训练',
                      old_sparse_protocol_status='FAILED_PRESERVED',
                      scope='仅光度迁移、偏移稳定性和有限局部方向反证通过；不证明物理唯一性或GT准确性',
                      limitations=profile['limitation'])
    asset = dict(source_split='TRAIN', gt_read=False, engineering_only=False,
                 internal_validation_passed=True, exploratory=True, direction_identifiability='unresolved',
                 validation_protocol='ELDL_V3_FULL_RASTER_TRAIN_CALIBRATION_V2',
                 validation_scope='photometric_transfer_and_bias', environment_mode='shared_rgb_constant',
                 ambient_rgb=environment.tolist(), direct_intensity=math.pi, frames=records,
                 shadow_bias=bias, provenance=provenance,
                 held_direction_rule='fit时间nlerp；18内部开发保留帧在正式135TRAIN中成为已见数据')
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(a.source / 'images', a.output / 'images')
    for name in ('points3d.ply', 'transforms_train.json'):
        shutil.copy2(a.source / name, a.output / name)
    for name, value in [('estimated_eldl_train.json', asset),
                        ('estimated_directions_train.json', dict(frames=records, gt_read=False,
                         source_split='TRAIN', provenance=provenance))]:
        (a.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    source.update(lighting_provenance=provenance, lighting_frozen=True)
    source['files'] = {str(x.relative_to(a.output)): sha(x) for x in sorted(a.output.rglob('*')) if x.is_file()}
    (a.output / 'no_gt_manifest.json').write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS_EXPLORATORY_EXPORT', source=str(a.output),
                         ambient_rgb=asset['ambient_rgb'], shadow_bias=bias,
                         lighting_sha256=sha(a.output / 'estimated_eldl_train.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
