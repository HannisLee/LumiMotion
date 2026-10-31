"""第三轮冻结后唯一一次独立方向评测；不估计、不选参、不搜索坐标或符号。"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import static_rgb_light_evaluate as base

ROOT = base.ROOT
RUN = ROOT / 'output/1031-06-LH-static-ELDL-V3-第三轮z偏置校正方向光初始化'
PREREG = ROOT / 'LIHAN/DOC/讨论/1031-07-未知方向光第三轮z偏置诊断与预注册方案.md'
PREREG_SHA = 'dbfe6cbbac3076b89680ce7b6d77d9f449e8fb07637cda6773165be24c417426'
OBS = ROOT / 'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz'
OBS_SHA = 'eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47'
CODE = ROOT / 'LIHAN/scripts/static_rgb_light_fit_r3.py'
REVIEW = ROOT / 'output/smoke_test/1031-03-SYNTH-第三轮独立数值审查/receipt.json'
SOURCE = ROOT / 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3'


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def self_test():
    """只用人工构造向量，验证点积、截断、角度单位。"""
    prediction = np.tile([1., 0., 0.], (3, 1))
    target = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
    actual = base.evaluate({'manual': prediction}, target)['manual']
    np.testing.assert_allclose(actual, [0., 90., 180.], atol=1e-12, rtol=0)
    return dict(status='PASS', expected_degrees=[0, 90, 180], actual_degrees=actual.tolist(),
                gt_read=False, coordinate_or_sign_search=False)


def validate():
    """所有验证都只涉及允许输入、模型、拟合产物与代码。"""
    manifest_path = RUN / 'freeze_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require = base.require
    require(manifest['status'] == 'R3_FROZEN_READY', '本轮尚未正式冻结')
    require(manifest['gt_read'] is False and manifest['final_method'] == 'E4', '冻结状态或最终方法错误')
    require(bool(manifest['frozen_utc']), '缺少冻结时间')
    require(manifest['coordinate_system'] == 'world', '必须为世界坐标')
    require(manifest['direction_convention'] == 'surface_to_light', '必须为surface_to_light')
    require(manifest['preregistration_sha256'] == PREREG_SHA == base.sha(PREREG), '预注册文件改变')
    require(manifest['observations_sha256'] == OBS_SHA == base.sha(OBS), '输入观测改变')
    require(manifest['code_sha256'] == base.sha(CODE), '正式拟合代码改变')
    files = manifest['files']
    require(isinstance(files, dict) and bool(files), '缺少完整冻结files')
    checked = {}
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(ROOT), 'files必须为仓库相对路径')
        require(path.name != 'lights.json' and not any(t in name.lower() for t in
                ('gt_light_directions', 'gt_normal', 'gt_depth', 'gt_albedo', 'gt_material', 'gt_shadow')),
                f'禁止读取GT冻结输入：{name}')
        require(base.sha(path) == expected, f'冻结哈希不符：{name}')
        checked[name] = expected
    mandatory = [PREREG, OBS, CODE, REVIEW,
                 RUN / 'visual_inspection.json', RUN / 'fitting/predictions.npz',
                 RUN / 'fitting/method_config.json', RUN / 'fitting/support.npz',
                 RUN / 'fitting/support_manifest.json', RUN / 'fitting/ablation_stability_table.json']
    for path in mandatory:
        require(str(path.relative_to(ROOT)) in files, f'必须冻结文件：{path}')
    eval_manifest = json.loads((RUN / 'eval_code_manifest.json').read_text())
    require(eval_manifest['status'] == 'EVALUATION_CODE_FROZEN' and eval_manifest['gt_read'] is False,
            '独立评测代码尚未冻结')
    require(eval_manifest['freeze_manifest_sha256'] == base.sha(manifest_path), '评测代码锁定的主冻结改变')
    expected_eval_files = {str(Path(__file__).relative_to(ROOT)), str(Path(base.__file__).relative_to(ROOT))}
    require(set(eval_manifest['files']) == expected_eval_files, '评测代码依赖清单不完整')
    for name, expected in eval_manifest['files'].items():
        require(base.sha(ROOT / name) == expected, f'独立评测代码改变：{name}')
    fit_files = {str(p.relative_to(ROOT)) for p in (RUN / 'fitting').rglob('*') if p.is_file()}
    require(fit_files.issubset(files), '有拟合产物未纳入冻结')
    require(any(p.endswith('iteration_35000/point_cloud.ply') for p in files), '缺少模型点云冻结')
    require(any(p.endswith('iteration_35000/deform.pth') for p in files), '缺少模型形变冻结')
    review = json.loads(REVIEW.read_text())
    require(review['status'] == 'PASS' and review['gt_read'] is False, '独立数值审查未通过')
    require(review['reviewed_code_sha256'] == manifest['code_sha256'], '数值审查代码与正式代码不符')
    inspection = json.loads((RUN / 'visual_inspection.json').read_text())
    require(inspection.get('status') in {'PASS', 'PASS_FOR_FREEZE_WITH_DOCUMENTED_LIMITATIONS'} and
            inspection.get('gt_read') is False and len(inspection.get('items', [])) == 4, '真实无GT目检未通过')
    config = json.loads((RUN / 'fitting/method_config.json').read_text())
    require(config['gt_read'] is False and config['final_method'] == 'E4', '配置冻结声明错误')
    require(config['selected'] == manifest['selected'], '最终选择不一致')
    require(config['fit_gaussians'] == 2683 and config['heldout_gaussians'] == 648, '支持域改变')
    require([v['seed'] for v in config['bootstrap_stability']] == [103108, 103109], 'bootstrap种子或数量改变')
    support = json.loads((RUN / 'fitting/support_manifest.json').read_text())
    require(support['support_sha256'] == base.sha(RUN / 'fitting/support.npz'), '支持点哈希不符')
    with np.load(RUN / 'fitting/support.npz', allow_pickle=False) as saved:
        require(tuple(len(saved[k]) for k in ('body_fit', 'body_val', 'split_A', 'split_B')) ==
                (2683, 648, 1450, 1233), '支持点数量不符')
    pred_path = RUN / 'fitting/predictions.npz'
    require(base.sha(pred_path) == manifest['prediction_sha256'], '预测文件改变')
    with np.load(pred_path, allow_pickle=False) as saved:
        require(set(saved.files) == {'frames', *base.METHODS}, '预测key必须恰为frames及E0-E4')
        require(np.array_equal(saved['frames'], base.FRAMES), '预测帧必须为全部1..150且顺序一致')
        predictions = {k: np.asarray(saved[k], dtype=np.float64) for k in base.METHODS}
    max_norm_error = {}
    for method, value in predictions.items():
        require(value.shape == (150, 3) and np.isfinite(value).all(), f'{method}方向形状或数值错误')
        norms = np.linalg.norm(value, axis=1)
        max_norm_error[method] = float(max(abs(norms - 1)))
        require(max_norm_error[method] < 1e-10, f'{method}必须预先保存单位向量')
        predictions[method] = value / norms[:, None]
    if manifest['selected'] == 'E3_fallback':
        require(np.array_equal(predictions['E4'], predictions['E3']), '回退E4不是E3副本')
    return manifest, predictions, dict(status='PASS', checked_at=now(), gt_read=False,
        freeze_manifest_sha256=base.sha(manifest_path), checked_file_count=len(checked),
        eval_code_manifest_sha256=base.sha(RUN / 'eval_code_manifest.json'),
        checked_files=checked, prediction_sha256=manifest['prediction_sha256'],
        max_direction_norm_error=max_norm_error, final_method='E4', selected=manifest['selected'],
        independent_review=review, visual_inspection=inspection)


def read_gt_once():
    """仅一次打开lights.json；哈希使用同一内存字节，不重新打开文件。"""
    payload = (SOURCE / 'lights.json').read_bytes()
    gt_hash = hashlib.sha256(payload).hexdigest()
    records = json.loads(payload)
    base.require(set(records) == {f'{i:04d}' for i in base.FRAMES}, 'GT必须恰为0001..0150')
    directions = []
    for frame in base.FRAMES:
        item = records[f'{frame:04d}']
        base.require(item.get('source_type') == 'SUN', f'{frame}主灯不是SUN')
        if 'direction_convention' in item:
            base.require(item['direction_convention'] ==
                'Unit vector from surface toward sun; max(dot(normal, light_dir_world), 0)', 'GT方向约定错误')
        direction = np.asarray(item['light_dir_world'], dtype=np.float64)
        base.require(direction.shape == (3,) and np.isfinite(direction).all(), 'GT方向无效')
        norm = np.linalg.norm(direction)
        base.require(norm > 1e-12, 'GT方向为零')
        directions.append(direction / norm)
    return np.stack(directions), gt_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-gt-evaluation', action='store_true', help='仅收到明确开始第三轮GT指令后使用')
    parser.add_argument('--self-test-only', action='store_true', help='仅执行人工角度自检，不访问任何实验或GT')
    args = parser.parse_args()
    test = self_test()
    if args.self_test_only:
        print(json.dumps(test, ensure_ascii=False, indent=2)); return
    frozen, predictions, integrity = validate()
    if not args.allow_gt_evaluation:
        print(json.dumps(dict(integrity=integrity, self_test=test), ensure_ascii=False, indent=2)); return
    output = RUN / 'evaluation'
    base.require(not output.exists(), '禁止覆盖已有评测输出或失败结果')
    receipt = RUN / 'gt_evaluation_receipt.json'
    dump(receipt, dict(status='GT_EVALUATION_STARTED', started_at=now(), integrity=integrity,
        self_test=test, evaluation_code_sha256=base.sha(__file__), base_code_sha256=base.sha(base.__file__),
        output=str(output), failure_policy='任何失败保留收据及产物，禁止二次打开GT或重新评测'))
    output.mkdir()
    dump(output / 'integrity_validation.json', integrity)
    dump(output / 'self_test.json', test)
    started = now()
    gt, gt_hash = read_gt_once()
    finished = now()
    errors = base.evaluate(predictions, gt)
    metrics = base.write_outputs(output, predictions, gt, errors, 'E4')
    final_error = errors['E4']
    worst = np.argsort(-final_error, kind='stable')[:12]
    delta = predictions['E4'] - gt
    dump(output / 'failure_analysis.json', dict(method='E4', selected=frozen['selected'],
        worst_frames=[dict(frame=int(base.FRAMES[i]), angular_error_deg=float(final_error[i]),
            prediction=predictions['E4'][i].tolist(), gt=gt[i].tolist(), delta=delta[i].tolist()) for i in worst],
        mean_prediction_minus_gt=delta.mean(axis=0).tolist(), positive_z_error_count=int((delta[:, 2] > 0).sum()),
        interpretation='仅描述冻结方向与GT方向的差异；这些差异不能独立归因于法线、深度、材质或阴影。未读取其他GT属性，未反向修改方法。'))
    result = dict(status='PASS' if metrics['E4']['mean_deg'] <= 5 else 'FAILED', final_method='E4',
        selected=frozen['selected'], target_mean_deg=5., methods=metrics, evaluated_at=now(),
        gt_read_started_at=started, gt_read_finished_at=finished, gt_file_open_count=1, gt_sha256=gt_hash,
        freeze_manifest_sha256=integrity['freeze_manifest_sha256'], prediction_sha256=frozen['prediction_sha256'],
        evaluation_code_sha256=base.sha(__file__), base_code_sha256=base.sha(base.__file__),
        coordinate_system='world', direction_convention='surface_to_light', sign_or_coordinate_search=False,
        gt_used_for_estimation=False, gt_feedback_parameter_tuning=False, stage2_training=False,
        final_rule='仅冻结E4 Mean Angular Error <=5度判PASS；E0-E3仅描述性消融')
    dump(output / 'results.json', result)
    dump(output / 'evaluation_complete.json', dict(status='COMPLETE', completed_at=now(), result=result['status'],
         receipt_sha256=base.sha(receipt), artifacts={p.name: base.sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
