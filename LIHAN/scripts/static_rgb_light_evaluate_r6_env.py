"""T01(1108-01) R6-ENV 冻结后唯一一次独立方向评测；不估计、不选参、不搜索坐标/符号/时间偏移。
同次评测 R6、E4(读冻结predictions)、E3；E4指标必须复现冻结 1031-12 评测(差<=1e-9度)。PASS判据见规划第10.2节。
评测代码口径沿用 static_rgb_light_evaluate_r5b.py（decompose、read_gt_once）。冻结前不得运行 --allow-gt-evaluation。
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from LIHAN.scripts import static_rgb_light_evaluate as base

ROOT = base.ROOT
E4_RUN = ROOT / 'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角'
PLAN = ROOT / 'LIHAN/DOC/规划/1108-01-T01环境光建模与RGB损失新Stage2规划.md'
OBS = ROOT / 'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz'
OBS_SHA = 'eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47'
SOURCE = ROOT / 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3'
METHODS = ('R6', 'E4', 'E3')
FRAMES = base.FRAMES


def now(): return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    with Path(path).open('x') as h:
        json.dump(value, h, ensure_ascii=False, indent=2, allow_nan=False); h.write('\n')


def decompose(pred, gt):
    """仰角差(pred-GT,度)与包裹方位差(pred-GT,度)；仅描述，不可与三维夹角相加。"""
    el = np.degrees(np.arcsin(np.clip(pred[:, 2], -1, 1)) - np.arcsin(np.clip(gt[:, 2], -1, 1)))
    az = np.degrees(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(gt[:, 1], gt[:, 0])))))
    return el, az


def pass_rule(full_converged, integrity_ok, mean_r6, mean_e4):
    """规划10.2：收敛 + 完整性 + Mean_R6 严格小于同次计算的 Mean_E4(全精度)。返回 (status, reasons)。"""
    reasons = []
    if not full_converged: reasons.append('未收敛')
    if not integrity_ok: reasons.append('完整性失败')
    if not (mean_r6 < mean_e4): reasons.append('Mean未严格优于E4')
    return ('PASS' if not reasons else 'FAILED'), reasons


def self_test():
    """人工向量：夹角、分解、PASS规则逻辑；不访问实验或GT。"""
    prediction = np.tile([1., 0., 0.], (3, 1)); target = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
    actual = base.evaluate({'manual': prediction}, target)['manual']; np.testing.assert_allclose(actual, [0., 90., 180.], atol=1e-12, rtol=0)
    el, az = decompose(np.array([[np.sqrt(.5), 0, np.sqrt(.5)]]), np.array([[0., np.sqrt(.5), np.sqrt(.5)]])); np.testing.assert_allclose([el[0], az[0]], [0., -90.], atol=1e-9)
    assert pass_rule(True, True, 4.0, 4.1)[0] == 'PASS' and pass_rule(True, True, 4.1, 4.1)[0] == 'FAILED' and pass_rule(False, True, 1., 4.)[1] == ['未收敛'] and pass_rule(True, False, 1., 4.)[1] == ['完整性失败']
    return dict(status='PASS', expected_degrees=[0, 90, 180], actual_degrees=actual.tolist(), decomposition_check=[float(el[0]), float(az[0])], pass_rule_logic='PASS', gt_read=False, coordinate_or_sign_search=False)


def validate(run):
    manifest_path = run / 'freeze_manifest.json'; manifest = json.loads(manifest_path.read_text()); require = base.require
    require(manifest['status'] == 'R6_FROZEN_READY' and manifest['gt_read'] is False and manifest['final_method'] == 'R6', '冻结状态错误')
    require(bool(manifest['frozen_utc']) and manifest['coordinate_system'] == 'world' and manifest['direction_convention'] == 'surface_to_light', '坐标/约定/冻结时间错误')
    require(manifest['plan_sha256'] == base.sha(PLAN) and manifest['observations_sha256'] == OBS_SHA == base.sha(OBS), '规划或输入改变')
    code = ROOT / 'LIHAN/scripts/static_rgb_light_fit_r6_env.py'; require(manifest['code_sha256'] == base.sha(code), '拟合代码改变')
    files = manifest['files']; require(isinstance(files, dict) and files, '缺少冻结files')
    for name, expected in files.items():
        path = (ROOT / name).resolve(); require(not Path(name).is_absolute() and path.is_relative_to(ROOT), 'files必须为仓库相对路径')
        require(path.name != 'lights.json' and not any(t in name.lower() for t in ('gt_light_directions', 'gt_normal', 'gt_depth', 'gt_albedo', 'gt_material', 'gt_shadow')), f'禁止GT冻结输入：{name}')
        require(base.sha(path) == expected, f'冻结哈希不符：{name}')
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
    for p in [PLAN, OBS, code, ROOT / 'LIHAN/scripts/static_rgb_light_fit_r3.py', ROOT / 'LIHAN/scripts/static_rgb_light_fit_r5b.py', run / 'visual_inspection.json', run / 'fitting/predictions.npz', run / 'fitting/env.npz',
              run / 'fitting/method_config.json', run / 'fitting/support.npz', run / 'fitting/r3_reproduction.json', run / 'fitting/e4_regression.json', E4_RUN / 'fitting/predictions.npz']:
        require(rel(p) in files, f'必须冻结文件：{p}')
    em = json.loads((run / 'eval_code_manifest.json').read_text()); require(em['status'] == 'EVALUATION_CODE_FROZEN' and em['gt_read'] is False and em['freeze_manifest_sha256'] == base.sha(manifest_path), '评测代码未冻结或与主冻结不符')
    require(set(em['files']) == {rel(__file__), rel(base.__file__)}, '评测代码依赖清单不完整')
    for name, expected in em['files'].items(): require(base.sha(ROOT / name) == expected, f'评测代码改变：{name}')
    fit_files = {rel(p) for p in (run / 'fitting').rglob('*') if p.is_file()}; require(fit_files.issubset(files), '有拟合产物未纳入冻结')
    require(any(p.endswith('iteration_35000/point_cloud.ply') for p in files) and any(p.endswith('iteration_35000/deform.pth') for p in files), '缺少Stage1模型冻结')
    st = [json.loads((ROOT / k).read_text()) for k in files if k.endswith('numeric_tests.json')]; require(len(st) == 1 and st[0]['status'] == 'PASS' and st[0]['gt_read'] is False, '合成自检(S1-S10)未通过')
    vi = json.loads((run / 'visual_inspection.json').read_text()); require(vi.get('gt_read') is False and len(vi.get('items', [])) == 5, '目检记录不完整')
    cfg = json.loads((run / 'fitting/method_config.json').read_text()); require(cfg['gt_read'] is False and cfg['final_method'] == 'R6', '配置声明错误')
    reg = json.loads((run / 'fitting/e4_regression.json').read_text()); require(reg['ok'] is True and reg['max_angle_diff_deg'] <= 1e-6, 'E4回归未通过')
    r3r = json.loads((run / 'fitting/r3_reproduction.json').read_text()); require(max(r3r['max_abs_diff'].values()) <= 1e-6, 'E1-E3未复现')
    sup = json.loads((run / 'fitting/support_manifest.json').read_text()); require(sup['equals_frozen'] is True and sup['linearization_qc']['luminance_max_abs_err'] <= 1e-5, '支撑集/线性化一致性失败')
    pred_path = run / 'fitting/predictions.npz'; require(base.sha(pred_path) == manifest['prediction_sha256'], '预测文件改变')
    with np.load(pred_path, allow_pickle=False) as s:
        require(set(s.files) == {'frames', 'E0', 'E1', 'E2', 'E3', 'R6'} and np.array_equal(s['frames'], FRAMES), '预测key/帧错误'); preds = {k: np.asarray(s[k], dtype=np.float64) for k in ('E3', 'R6')}
    with np.load(E4_RUN / 'fitting/predictions.npz', allow_pickle=False) as s: preds['E4'] = np.asarray(s['E4'], dtype=np.float64)
    for m, v in preds.items():
        require(v.shape == (150, 3) and np.isfinite(v).all(), f'{m}形状/有限性错误'); nrm = np.linalg.norm(v, axis=1); require(float(abs(nrm - 1).max()) < 1e-10, f'{m}非单位向量'); preds[m] = v / nrm[:, None]
    ok = dict(full_converged=bool(cfg['full_converged']), integrity_ok=True)
    return manifest, preds, ok, dict(status='PASS', checked_at=now(), gt_read=False, freeze_manifest_sha256=base.sha(manifest_path), checked_file_count=len(files), prediction_sha256=manifest['prediction_sha256'],
                                      e4_regression=reg, full_certificate=cfg['full_certificate'], self_test_receipt=st[0], visual_inspection=vi)


def read_gt_once():
    payload = (SOURCE / 'lights.json').read_bytes(); gt_hash = hashlib.sha256(payload).hexdigest(); records = json.loads(payload)
    base.require(set(records) == {f'{i:04d}' for i in FRAMES}, 'GT必须恰为0001..0150'); out = []
    for f in FRAMES:
        item = records[f'{f:04d}']; base.require(item.get('source_type') == 'SUN', f'{f}主灯不是SUN')
        if 'direction_convention' in item: base.require(item['direction_convention'] == 'Unit vector from surface toward sun; max(dot(normal, light_dir_world), 0)', 'GT方向约定错误')
        v = np.asarray(item['light_dir_world'], dtype=np.float64); base.require(v.shape == (3,) and np.isfinite(v).all() and np.linalg.norm(v) > 1e-12, 'GT方向无效'); out.append(v / np.linalg.norm(v))
    return np.stack(out), gt_hash


def metrics_for(pred, gt):
    err = base.evaluate({'m': pred}, gt)['m']; el, az = decompose(pred, gt); pe = np.degrees(np.arcsin(np.clip(pred[:, 2], -1, 1))); ge = np.degrees(np.arcsin(np.clip(gt[:, 2], -1, 1)))
    return err, dict(mean_deg=float(err.mean()), median_deg=float(np.median(err)), p90_deg=float(np.percentile(err, 90)), max_deg=float(err.max()), within_5deg_count=int((err <= 5).sum()), frame_count=len(err),
                     azimuth_mae_deg=float(abs(az).mean()), azimuth_p50_deg=float(np.median(abs(az))), azimuth_p90_deg=float(np.percentile(abs(az), 90)), azimuth_signed_mean_deg=float(az.mean()),
                     elevation_mae_deg=float(abs(el).mean()), elevation_p50_deg=float(np.median(abs(el))), elevation_p90_deg=float(np.percentile(abs(el), 90)), elevation_signed_mean_deg=float(el.mean()),
                     pred_elevation_range_deg=[float(pe.min()), float(pe.max())], pred_gt_elevation_pearson=float(np.corrcoef(pe, ge)[0, 1]) if pe.std() > 0 else None), el, az


def write_outputs(output, preds, gt, errs, els, azs):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    with (output / 'per_frame.csv').open('x', newline='') as h:
        w = csv.writer(h); w.writerow(['frame', 'method', 'pred_x', 'pred_y', 'pred_z', 'gt_x', 'gt_y', 'gt_z', 'angular_error_deg', 'within_5deg'])
        for m in METHODS:
            for i, f in enumerate(FRAMES): w.writerow([int(f), m, *preds[m][i], *gt[i], float(errs[m][i]), bool(errs[m][i] <= 5)])
    with (output / 'elevation_azimuth_decomposition.csv').open('x', newline='') as h:
        w = csv.writer(h); w.writerow(['frame', 'method', 'elev_err_deg_pred_minus_gt', 'azimuth_err_deg_pred_minus_gt'])
        for m in METHODS:
            for i, f in enumerate(FRAMES): w.writerow([int(f), m, float(els[m][i]), float(azs[m][i])])
    (output / 'figures').mkdir()
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in METHODS: ax.plot(FRAMES, errs[m], label=m)
    ax.axhline(5, c='k', ls='--'); ax.set(xlabel='Frame', ylabel='Angular error (deg)'); ax.legend(); fig.tight_layout(); fig.savefig(output / 'figures/angular_error_curve.png', dpi=160); plt.close(fig)
    ge = np.degrees(np.arcsin(gt[:, 2])); fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True); axes[0].plot(FRAMES, ge, c='k', label='GT')
    for m in METHODS: axes[0].plot(FRAMES, np.degrees(np.arcsin(preds[m][:, 2])), label=m); axes[1].plot(FRAMES, azs[m], label=f'{m} az err')
    for v in (40, 50): axes[0].axhline(v, c='gray', ls=':')
    axes[0].set_ylabel('Elevation (deg)'); axes[0].legend(); axes[1].axhline(0, c='gray', lw=.8); axes[1].set_ylabel('Azimuth err (deg)'); axes[1].legend(); axes[1].set_xlabel('Frame')
    fig.tight_layout(); fig.savefig(output / 'figures/elevation_azimuth_posthoc.png', dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path); ap.add_argument('--allow-gt-evaluation', action='store_true'); ap.add_argument('--self-test-only', action='store_true'); a = ap.parse_args()
    test = self_test()
    if a.self_test_only: print(json.dumps(test, ensure_ascii=False, indent=2)); return
    base.require(a.run is not None, '需要--run'); run = (a.run if a.run.is_absolute() else ROOT / a.run).resolve()
    frozen, preds, ok, integrity = validate(run)
    if not a.allow_gt_evaluation: print(json.dumps(dict(integrity=integrity, self_test=test), ensure_ascii=False, indent=2)); return
    output = run / 'evaluation'; base.require(not output.exists(), '禁止覆盖已有评测输出或失败结果'); receipt = run / 'gt_evaluation_receipt.json'
    dump(receipt, dict(status='GT_EVALUATION_STARTED', started_at=now(), integrity=integrity, self_test=test, evaluation_code_sha256=base.sha(__file__), base_code_sha256=base.sha(base.__file__), output=str(output),
                       failure_policy='任何失败保留收据及产物，禁止二次打开GT或重新评测'))
    output.mkdir(); dump(output / 'integrity_validation.json', integrity); dump(output / 'self_test.json', test)
    started = now(); gt, gt_hash = read_gt_once(); finished = now()
    errs, mets, els, azs = {}, {}, {}, {}
    for m in METHODS: errs[m], mets[m], els[m], azs[m] = metrics_for(preds[m], gt)
    # E4 必须复现冻结评测
    frozen_e4 = json.loads((E4_RUN / 'evaluation/results.json').read_text())['methods']['E4']; dev = {k: abs(mets['E4'][k] - frozen_e4[k]) for k in ('mean_deg', 'median_deg', 'p90_deg', 'within_5deg_count')}
    base.require(max(dev.values()) <= 1e-9, f'E4未复现冻结评测：{dev}')
    write_outputs(output, preds, gt, errs, els, azs)
    keys = ['mean_deg', 'median_deg', 'p90_deg', 'within_5deg_count', 'azimuth_mae_deg', 'elevation_mae_deg', 'elevation_signed_mean_deg']
    comparison = {k: dict(R6=mets['R6'][k], E4=mets['E4'][k], R6_minus_E4=mets['R6'][k] - mets['E4'][k]) for k in keys}; comparison['pred_elevation_range_deg'] = dict(R6=mets['R6']['pred_elevation_range_deg'], E4=mets['E4']['pred_elevation_range_deg'])
    status, reasons = pass_rule(ok['full_converged'], ok['integrity_ok'], float(errs['R6'].mean()), float(errs['E4'].mean()))
    worst = np.argsort(-errs['R6'], kind='stable')[:12]
    result = dict(status=status, fail_reasons=reasons, final_method='R6', criterion='full收敛 且 完整性通过 且 Mean_R6 < Mean_E4(全精度,同次计算)', methods=mets, comparison_R6_vs_E4=comparison, e4_reproduction_deviation=dev,
                  worst_frames_R6=[dict(frame=int(FRAMES[i]), angular_error_deg=float(errs['R6'][i]), elev_err_deg=float(els['R6'][i]), azimuth_err_deg=float(azs['R6'][i])) for i in worst],
                  evaluated_at=now(), gt_read_started_at=started, gt_read_finished_at=finished, gt_file_open_count=1, gt_sha256=gt_hash, freeze_manifest_sha256=integrity['freeze_manifest_sha256'], prediction_sha256=frozen['prediction_sha256'],
                  evaluation_code_sha256=base.sha(__file__), base_code_sha256=base.sha(base.__file__), coordinate_system='world', direction_convention='surface_to_light', sign_or_coordinate_search=False,
                  gt_used_for_estimation=False, gt_feedback_parameter_tuning=False, stage3_run=False, note='PASS不表示恢复了物理环境光或albedo；方位差与三维夹角不可相加')
    dump(output / 'results.json', result)
    dump(output / 'evaluation_complete.json', dict(status='COMPLETE', completed_at=now(), result=status, receipt_sha256=base.sha(receipt), artifacts={str(p.relative_to(output)): base.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(status=status, reasons=reasons, comparison=comparison), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
