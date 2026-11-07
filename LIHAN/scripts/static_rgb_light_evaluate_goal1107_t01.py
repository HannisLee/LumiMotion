"""T01 冻结后一次 GT 批次；复用 r5b 数学路径，不回传拟合或选档。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import static_rgb_light_evaluate_r5b as old

base = old.base
ROOT = base.ROOT
RUN = ROOT / 'output/1107-01-LH-static-ELDL-V3-T01仰角正则消融'
FREEZE_SHA = '597ab3e60a90a1a52c0c9a03332fd36db33bcd75b1b604bd4ae88462e5f66f7d'
BASELINE = old.RUN / 'fitting/predictions.npz'


def validate():
    """仅核验冻结的无 GT 证据；任何不一致均在 GT 读取前停止。"""
    require = base.require
    require(base.sha(RUN / 'freeze_manifest.json') == FREEZE_SHA, '冻结清单改变')
    m = json.loads((RUN / 'freeze_manifest.json').read_text())
    require(m['status'] == 'PASS_NO_GT_FIT_FREEZE_REVIEW', '冻结未通过')
    for entry in m['files']:
        path = RUN / entry['path']
        require(path.resolve().is_relative_to(RUN.resolve()), '冻结路径越界')
        require(base.sha(path) == entry['sha256'], f'冻结文件改变：{path}')
    for key in ('plan', 'implementation'):
        require(base.sha(ROOT / m[key]['path']) == m[key]['sha256'], f'{key}改变')
    launch = json.loads((RUN / 'fitting/launch.json').read_text())
    for path, expected in launch['inputs'].items():
        require(Path(path).name != 'lights.json', '白名单不应含 GT')
        require(base.sha(ROOT / path) == expected, f'输入改变：{path}')
    fit = RUN / 'fitting'
    rows = json.loads((fit / 'candidate_table.json').read_text())['candidates']
    require(len(rows) == 7 and all(len(r['certificates']) == 7 for r in rows), '流程不完整')
    require(all(c['converged'] for r in rows for c in r['certificates'].values()), '证书异常')
    require(json.loads((fit / 'fit_complete.json').read_text())['gt_read'] is False, '拟合GT声明异常')
    config = json.loads((fit / 'method_config.json').read_text())
    require(config['gt_read'] is False, '配置GT声明异常')
    repro = json.loads((fit / 'baseline_reproduction.json').read_text())
    require(repro['status'] == 'PASS' and max(repro['differences'].values()) == 0, '原档复现异常')
    # 独立按预注册规则重算无 GT 选择及敏感性；不调用优化代码。
    def pick(delta=.02, tau=.02, el_weight=1.):
        score = lambda r: r['split_mean_deg'] + .5*r['split_p90_deg'] + el_weight*r['split_el_mae_deg'] + r['boot_mean_deg'] + r['hilo_mean_deg']
        pool = [r for r in rows[1:] if r['all_gates'] and score(r) < score(rows[0])-delta]
        if not pool:
            return 'E4_fallback_no_strict_noGT_improvement', None
        best = min(map(score, pool))
        selected = min((r for r in pool if score(r) <= best+tau), key=lambda r: (r['body_heldout_mae'], r['name']))
        return selected['name'], score(selected)-score(rows[0])
    selected, _ = pick()
    require(selected == m['selection']['selected'] == 'C06', '主选择不符')
    sensitivity = json.loads((fit / 'sensitivity.json').read_text())['rows']
    require(len(sensitivity) == 12, '敏感性缺行')
    for row in sensitivity:
        args = {row['axis']: row['value']}
        chosen, delta = pick(**args)
        require(chosen == row['selected'] and abs(delta-row['score_difference']) < 1e-12, '敏感性不符')
    predictions = {}
    with np.load(fit / 'predictions.npz', allow_pickle=False) as saved:
        require(np.array_equal(saved['frames'], base.FRAMES), '帧号异常')
        predictions['主预测C06'] = saved['E4'].copy()
    with np.load(BASELINE, allow_pickle=False) as saved:
        require(np.array_equal(saved['frames'], base.FRAMES), '原档帧号异常')
        predictions['历史E4'] = saved['E4'].copy()
    for row in rows:
        predictions[row['name']] = np.load(fit / f"runs/{row['name']}_full/direction.npy", allow_pickle=False)
    require(np.array_equal(predictions['主预测C06'], predictions['C06']), '主预测未绑定C06')
    require(np.array_equal(predictions['历史E4'], predictions['C00']), 'C00未精确复现原档')
    for name, value in predictions.items():
        require(value.shape == (150, 3) and np.isfinite(value).all(), f'{name}无效')
        norms = np.linalg.norm(value, axis=1)
        require(np.max(abs(norms-1)) < 1e-10, f'{name}非单位方向')
        predictions[name] = value/norms[:, None]
    return predictions, dict(status='PASS', checked_at=old.now(), freeze_manifest_sha256=FREEZE_SHA,
        checked_frozen_files=len(m['files']), checked_inputs=len(launch['inputs']),
        selected=selected, sensitivity_rows_verified=12, gt_read=False,
        baseline_prediction_sha256=base.sha(BASELINE), prediction_sha256=base.sha(fit / 'predictions.npz'))


def describe(value):
    return dict(mean=float(value.mean()), std=float(value.std()), min=float(value.min()), max=float(value.max()),
        quantiles=dict(zip(('P0','P5','P25','P50','P75','P95','P100'), map(float, np.percentile(value, [0,5,25,50,75,95,100])))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-gt-evaluation', action='store_true')
    args = parser.parse_args()
    test = old.self_test()
    predictions, integrity = validate()
    if not args.allow_gt_evaluation:
        print(json.dumps(dict(self_test=test, integrity=integrity), ensure_ascii=False, indent=2))
        return
    out = RUN / 'evaluation'
    base.require(not out.exists(), '评测目录已存在，禁止覆盖')
    code_paths = [Path(__file__).resolve(), Path(old.__file__).resolve(), Path(base.__file__).resolve()]
    code = {str(p.relative_to(ROOT)): base.sha(p) for p in code_paths}
    old.dump(RUN / 'eval_code_manifest.json', dict(status='EVALUATION_CODE_FROZEN', gt_read=False,
        freeze_manifest_sha256=FREEZE_SHA, files=code, self_test=test, frozen_at=old.now()))
    receipt = RUN / 'gt_evaluation_receipt.json'
    old.dump(receipt, dict(status='GT_EVALUATION_STARTED', started_at=old.now(), integrity=integrity,
        evaluation_code=code, output=str(out), failure_policy='保留所有收据与产物，禁止二次GT读取',
        gt_open_count_planned=1, main_prediction='主预测C06', control='历史E4',
        diagnostic_candidates=[f'C{i:02d}' for i in range(7)]))
    out.mkdir()
    (out / '1107-01-T01冻结README原文.md').write_bytes((RUN / 'README.md').read_bytes())
    old.dump(out / 'integrity_validation.json', integrity)
    old.dump(out / 'self_test.json', test)
    started = old.now()
    # 唯一 GT 打开操作；复用 r5b 原函数，方向和哈希来自同一内存字节。
    gt, gt_hash = old.read_gt_once()
    old.dump(out / 'gt_read_receipt.json', dict(status='GT_READ_COMPLETE', started_at=started,
        finished_at=old.now(), gt_file_open_count=1, gt_sha256=gt_hash))
    np.savez_compressed(out / 'evaluation_cache.npz', frames=base.FRAMES, gt=gt, **predictions)
    errors = base.evaluate(predictions, gt)
    result = {}
    with (out / 'per_frame.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['方法','帧','预测x','预测y','预测z','GT_x','GT_y','GT_z','三维误差度','预测方位度','GT方位度','预测仰角度','GT仰角度','方位误差度','仰角误差度'])
        for name, pred in predictions.items():
            el, az = old.decompose(pred, gt)
            az = (az+180)%360-180
            pred_el = np.degrees(np.arcsin(np.clip(pred[:,2], -1, 1)))
            gt_el = np.degrees(np.arcsin(np.clip(gt[:,2], -1, 1)))
            pred_az = np.degrees(np.arctan2(pred[:,1],pred[:,0]))
            gt_az = np.degrees(np.arctan2(gt[:,1],gt[:,0]))
            value = errors[name]
            result[name] = dict(mean_deg=float(value.mean()), p50_deg=float(np.percentile(value,50)),
                p90_deg=float(np.percentile(value,90)), within_5deg_count=int((value<=5).sum()), frame_count=len(value),
                decomposition={axis: dict(bias_deg=float(err.mean()), mae_deg=float(abs(err).mean()),
                    p50_abs_deg=float(np.percentile(abs(err),50)), p90_abs_deg=float(np.percentile(abs(err),90)))
                    for axis,err in [('azimuth',az),('elevation',el)]}, elevation_deg=describe(pred_el))
            for i, frame in enumerate(base.FRAMES):
                writer.writerow([name,int(frame),*pred[i],*gt[i],float(value[i]),float(pred_az[i]),float(gt_az[i]),float(pred_el[i]),float(gt_el[i]),float(az[i]),float(el[i])])
    status = 'PASS' if result['主预测C06']['mean_deg'] < result['历史E4']['mean_deg'] else 'FAILED'
    old.dump(out / 'results.json', dict(status=status, final_rule='150帧三维Mean严格低于历史E4全精度Mean',
        methods=result, gt_elevation_deg=describe(gt_el), selected='C06', evaluated_at=old.now(),
        gt_sha256=gt_hash, gt_file_open_count=1, freeze_manifest_sha256=FREEZE_SHA,
        evaluation_code=code, coordinate_system='world', direction_convention='surface_to_light',
        gt_feedback_parameter_tuning=False, gt_reselection=False,
        candidate_metrics_usage='全部候选仅事后诊断，不改变冻结主选择'))
    old.dump(out / 'evaluation_complete.json', dict(status='COMPLETE', result=status, completed_at=old.now(),
        receipt_sha256=base.sha(receipt), files={p.name:base.sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status=status, main=result['主预测C06'], control=result['历史E4']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
