"""1109-01 T01 冻结后一次性独立 GT 评测（Academic-analysis 运行）。

  python -m LIHAN.scripts.goal1109.evaluate_t01_uniform_ambient --run output/1109-NN-... --self-test-only      # 只做合成自检，不读 GT
  python -m LIHAN.scripts.goal1109.evaluate_t01_uniform_ambient --run output/1109-NN-...                      # 只做完整性核验，不读 GT
  python -m LIHAN.scripts.goal1109.evaluate_t01_uniform_ambient --run output/1109-NN-... --allow-gt-evaluation # 整个评测只打开一次 lights.json

规划 8.2–8.6。坐标约定：世界系、surface_to_light、150 帧一一对应；不搜索符号、轴或时间偏移。GT 只在 read_gt_once() 中打开一次。
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from LIHAN.scripts import static_rgb_light_evaluate as base
from LIHAN.scripts.goal1109 import ua_core as C

E4_MEAN_FROZEN = 4.247987100361879
FRAMES = np.arange(1, 151)
SOURCE = C.SOURCE
GT_OPEN_COUNT = [0]
E4_EVAL = C.BASE.parent / 'evaluation/results.json'
C01_EVAL = C.C01_RUN_DIR / 'evaluation/results.json'
R6_EVAL = C.R6_RUN_DIR / 'evaluation/results.json'
C01_PRED = C.C01_RUN_DIR / 'fitting/predictions.npz'
R6_PRED = C.R6_RUN_DIR / 'fitting/predictions.npz'
SELF_TESTS = [f'S{i}' for i in range(1, 14)] + ['S14a', 'S14b', 'S15']
FORBIDDEN_NAME_TOKENS = ('gt_light_directions', 'gt_normal', 'gt_depth', 'gt_albedo', 'gt_material', 'gt_shadow')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(cond, msg):
    if not cond:
        raise ValueError(msg)


def dump(p, v, mode='x'):
    with Path(p).open(mode) as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


# ------------------------------------------------------------------ 指标与规则（纯函数，自检覆盖）
def decompose(pred, gt):
    """仰角差（pred-GT，度）与包裹到 [-180,180) 的方位差（pred-GT，度）。仅描述，不可与三维夹角相加。"""
    el = np.degrees(np.arcsin(np.clip(pred[:, 2], -1, 1)) - np.arcsin(np.clip(gt[:, 2], -1, 1)))
    az = np.degrees(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(gt[:, 1], gt[:, 0])))))
    return el, az


def metrics_for(pred, gt):
    err = base.evaluate({'m': pred}, gt)['m']
    el, az = decompose(pred, gt)
    pe = np.degrees(np.arcsin(np.clip(pred[:, 2], -1, 1)))
    ge = np.degrees(np.arcsin(np.clip(gt[:, 2], -1, 1)))
    m = dict(mean_deg=float(err.mean()), p50_deg=float(np.percentile(err, 50)), p90_deg=float(np.percentile(err, 90)), max_deg=float(err.max()),
             within_5deg_count=int((err <= 5).sum()), frame_count=len(err),
             azimuth=dict(mae_deg=float(abs(az).mean()), p50_abs_deg=float(np.percentile(abs(az), 50)), p90_abs_deg=float(np.percentile(abs(az), 90)), signed_mean_deg=float(az.mean())),
             elevation=dict(mae_deg=float(abs(el).mean()), p50_abs_deg=float(np.percentile(abs(el), 50)), p90_abs_deg=float(np.percentile(abs(el), 90)), signed_mean_deg=float(el.mean())),
             pred_elevation_range_deg=[float(pe.min()), float(pe.max())], pred_elevation_std_deg=float(pe.std()),
             pred_gt_elevation_pearson=float(np.corrcoef(pe, ge)[0, 1]) if pe.std() > 0 and ge.std() > 0 else None)
    return err, el, az, m


def pass_rule(integrity_ok, a0_valid, a0_converged, mean_a0, mean_e4):
    """规划 8.3：三条同时满足才 PASS，其余 FAILED 并写明原因。"""
    reasons = []
    if not integrity_ok:
        reasons.append('完整性失败')
    if not a0_valid:
        reasons.append('无有效预测')
    elif not a0_converged:
        reasons.append('未收敛')
    if a0_valid and not (mean_a0 < mean_e4):
        reasons.append('Mean 未严格优于 E4')
    return ('PASS' if not reasons else 'FAILED'), reasons


def attribution(mean_a1, mean_a0, tau):
    """规划 8.4（原样使用的结论文字）。"""
    d = mean_a1 - mean_a0
    if abs(d) <= tau:
        txt = 'A1 与 A0 接近：在本方法的抽样敏感度内，环境光项对光向精度的贡献不可分辨，即贡献有限'
        kind = 'close'
    elif d > tau:
        txt, kind = '环境光项对光向精度有正贡献', 'positive'
    else:
        txt, kind = '环境光项对光向精度有负贡献', 'negative'
    return dict(delta_mean_A1_minus_A0=d, tau=tau, kind=kind, conclusion=txt)


def ambiguity_answer(R_a, R_b, a0_hilo, c01_hilo, a0_ab, c01_ab):
    """规划 8.6 问题一的预注册判定规则。"""
    c1, c2, c3 = R_a > R_b, a0_hilo <= c01_hilo, a0_ab <= c01_ab
    r1, r2, r3 = R_a < R_b, a0_hilo > c01_hilo, a0_ab > c01_ab
    items = dict(R_a_gt_R_b=bool(c1), hilo_le_C01=bool(c2), AB_le_C01=bool(c3))
    if c1 and c2 and c3:
        return dict(answer='减少', items=items)
    if r1 and r2 and r3:
        return dict(answer='增加', items=items)
    return dict(answer='混合', items=items)


def mean_angle(a, b):
    return float(base.evaluate({'m': a / np.linalg.norm(a, axis=1, keepdims=True)}, b / np.linalg.norm(b, axis=1, keepdims=True))['m'].mean())


def frozen_ref_mean(path, key):
    return json.loads(Path(path).read_text())['methods'][key]


# ------------------------------------------------------------------ 完整性核验（不读 GT）
def integrity(run):
    man_path = run / 'freeze_manifest.json'
    m = json.loads(man_path.read_text())
    require(m['status'] == 'UA_FROZEN_READY' and m['gt_read'] is False, '尚未冻结或冻结声明错误')
    require(m['plan_sha256'] == C.PLAN_SHA == sha(C.PLAN), '规划文件改变')
    files = m['files']
    checked = {}
    for name, want in files.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'files 必须为仓库相对路径')
        require(p.name != 'lights.json' and not any(t in name.lower() for t in FORBIDDEN_NAME_TOKENS), f'禁止 GT 冻结输入：{name}')
        require(sha(p) == want, f'冻结哈希不符：{name}')
        checked[name] = want
    rel = lambda p: str(Path(p))
    must = [C.PLAN, C.SCRIPT_DIR / 'fit_t01_uniform_ambient.py', C.SCRIPT_DIR / 'ua_core.py', r3_obs(), run / 'fitting/A0/predictions.npz', run / 'fitting/A1/predictions.npz',
            run / 'fitting/A0/ambient.npz', run / 'visual_inspection.json']
    for p in must:
        require(rel(p) in files, f'冻结清单缺少 {p}')
    res = {}
    # 目标回归 I-R
    for arm in ('E4', 'C01'):
        r = json.loads((run / f'regress/{arm}/regress.json').read_text())
        require(r['ok'] is True and r['gt_read'] is False, f'目标回归 {arm} 未通过')
        res[f'regress_{arm}'] = dict(rel_diff=r['rel_diff'], tolerance=r['tolerance'])
    # 自检 S1–S15
    st = json.loads(Path(m['selftest_json']).read_text())
    require(st['status'] == 'PASS' and st['gt_read'] is False, '自检未通过')
    for t in SELF_TESTS:
        require(st['tests'][t]['status'] == 'PASS', f'自检 {t} 未通过')
    res['self_test'] = dict(path=m['selftest_json'], tests=SELF_TESTS)
    # 视觉检查
    vi = json.loads((run / 'visual_inspection.json').read_text())
    require(vi.get('gt_read') is False and len(vi.get('items', [])) >= 4, 'visual_inspection 不完整')
    # 支撑 / bootstrap / RGB 一致性 / 初值白名单
    with np.load(C.BASE / 'support.npz', allow_pickle=False) as old:
        for arm in ('A0', 'A1'):
            with np.load(run / f'fitting/{arm}/support.npz', allow_pickle=False) as new:
                for k in old.files:
                    require(np.array_equal(old[k], new[k]), f'{arm} 支撑 {k} 与 1031-12 不一致')
            for s in C.BOOT:
                require(np.array_equal(np.load(run / f'fitting/{arm}/bootstrap_{s}_sample.npy'), np.load(C.BASE / f'bootstrap_{s}_sample.npy')), f'{arm} bootstrap {s} 不一致')
            launch = json.loads((run / f'fitting/{arm}/launch.json').read_text())
            require(launch['gt_read'] is False and launch['kappa_consistency_max_abs'] <= 1e-5, f'{arm} RGB 一致性失败')
            for k, want in launch['input_and_code_sha256'].items():
                if Path(k).name in ('ua_core.py', 'ua_diag.py', 'ua_figs.py', 'ua_selftest.py', 'fit_t01_uniform_ambient.py') and k not in files:
                    continue
                if Path(k).is_file():
                    require(sha(k) == want, f'{arm} 启动哈希与当前不符：{k}')
    # 白名单初值哈希（EXPECTED）
    for k, want in C.g1108.EXPECTED.items():
        require(sha(C.BASE / k) == want, f'白名单哈希不符 {k}')
    # 预测有效性
    status = {}
    for arm in ('A0', 'A1'):
        fc = json.loads((run / f'fitting/{arm}/fit_complete.json').read_text())
        status[arm] = fc['status']
        with np.load(run / f'fitting/{arm}/predictions.npz', allow_pickle=False) as p:
            require(np.array_equal(p['frames'], FRAMES), f'{arm} frames 错误')
            for k in p.files:
                if k.startswith(arm + '_') or k == 'reference_E4':
                    v = p[k]
                    require(v.shape == (150, 3) and np.isfinite(v).all() and np.abs(np.linalg.norm(v, axis=1) - 1).max() <= 1e-10, f'{arm}/{k} 非单位向量')
    with np.load(C.BASE / 'predictions.npz', allow_pickle=False) as p:
        e4 = p['E4'].copy()
    for arm in ('A0', 'A1'):
        with np.load(run / f'fitting/{arm}/predictions.npz', allow_pickle=False) as p:
            require(np.array_equal(p['reference_E4'], e4), f'{arm} reference_E4 与冻结 E4 不一致')
    res['fit_status'] = status
    res['manifest_sha256'] = sha(man_path)
    res['checked_file_count'] = len(checked)
    return m, dict(status='PASS', gt_read=False, checked_utc=now(), **res), checked


def r3_obs():
    from LIHAN.scripts import static_rgb_light_fit_r3 as r3
    return r3.OBS


# ------------------------------------------------------------------ GT 读取（全程只一次）
def read_gt_once():
    GT_OPEN_COUNT[0] += 1
    require(GT_OPEN_COUNT[0] == 1, 'GT 已打开过，禁止二次读取')
    payload = (SOURCE / 'lights.json').read_bytes()
    gt_hash = hashlib.sha256(payload).hexdigest()
    records = json.loads(payload)
    require(set(records) == {f'{i:04d}' for i in FRAMES}, 'GT 必须恰为 0001..0150')
    out = []
    for f in FRAMES:
        item = records[f'{f:04d}']
        require(item.get('source_type') == 'SUN', f'{f} 主灯不是 SUN')
        if 'direction_convention' in item:
            require(item['direction_convention'] == 'Unit vector from surface toward sun; max(dot(normal, light_dir_world), 0)', 'GT 方向约定错误')
        v = np.asarray(item['light_dir_world'], dtype=np.float64)
        require(v.shape == (3,) and np.isfinite(v).all() and np.linalg.norm(v) > 1e-12, 'GT 方向无效')
        out.append(v / np.linalg.norm(v))
    return np.stack(out), gt_hash


# ------------------------------------------------------------------ 自检（人工向量；不读 GT）
def self_test():
    target = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    pred = np.tile([1., 0., 0.], (4, 1))
    err, el, az, m = metrics_for(pred, target)
    np.testing.assert_allclose(err, [0, 90, 180, 90], atol=1e-12)
    e2, a2 = decompose(np.array([[np.sqrt(.5), 0, np.sqrt(.5)]]), np.array([[0., np.sqrt(.5), np.sqrt(.5)]]))
    np.testing.assert_allclose([e2[0], a2[0]], [0., -90.], atol=1e-9)
    assert pass_rule(True, True, True, 4.0, 4.1) == ('PASS', [])
    assert pass_rule(True, True, True, 4.1, 4.1)[0] == 'FAILED'        # 严格小于
    assert pass_rule(True, True, False, 3.0, 4.1)[1] == ['未收敛']
    assert pass_rule(False, True, True, 3.0, 4.1)[1] == ['完整性失败']
    assert pass_rule(True, False, False, np.nan, 4.1)[1] == ['无有效预测']
    assert attribution(4.0, 4.1, .2)['kind'] == 'close' and attribution(5.0, 4.1, .2)['kind'] == 'positive' and attribution(3.0, 4.1, .2)['kind'] == 'negative'
    assert attribution(4.3, 4.1, .2)['kind'] == 'close' or attribution(4.3, 4.1, .2)['kind'] == 'positive'
    assert ambiguity_answer(.5, .4, 1, 2, 1, 2)['answer'] == '减少' and ambiguity_answer(.3, .4, 3, 2, 3, 2)['answer'] == '增加' and ambiguity_answer(.5, .4, 3, 2, 1, 2)['answer'] == '混合'
    return dict(status='PASS', gt_read=False, coordinate_or_sign_search=False)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--allow-gt-evaluation', action='store_true')
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    C.check_root()
    test = self_test()
    if a.self_test_only:
        print(json.dumps(test, ensure_ascii=False, indent=2))
        return
    run = a.run
    man, integ, checked = integrity(run)
    if not a.allow_gt_evaluation:
        Path(run / 'evaluation_integrity_only.json').write_text(json.dumps(integ, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(integ, ensure_ascii=False, indent=2))
        return
    out = run / 'evaluation'
    require(not out.exists(), '禁止二次评测或覆盖')
    receipt = run / 'gt_evaluation_receipt.json'
    dump(receipt, dict(status='GT_EVALUATION_STARTED', utc=now(), gt_file_open_count_planned=1, freeze_manifest_sha256=integ['manifest_sha256'], code_sha256=sha(__file__),
                       gt_used_for_decisions=False, failure_policy='任何失败均保留本收据，禁止自动重开 GT 评测'))
    out.mkdir()
    dump(out / 'integrity_validation.json', integ)
    dump(out / 'self_test.json', test)
    preds = {}
    flows = {}
    for arm in ('A0', 'A1'):
        with np.load(run / f'fitting/{arm}/predictions.npz', allow_pickle=False) as p:
            for k in p.files:
                if k.startswith(arm + '_'):
                    preds[f'{arm}/{k[len(arm) + 1:]}'] = p[k].astype(np.float64)
        flows[arm] = json.loads((run / f'fitting/{arm}/fit_complete.json').read_text())
    with np.load(C.BASE / 'predictions.npz', allow_pickle=False) as p:
        preds['E4'] = p['E4'].astype(np.float64)
    with np.load(C01_PRED, allow_pickle=False) as p:
        preds['C01/full'] = p['C01_full'].astype(np.float64)
        c01_aux = {k: p[f'C01_{k}'].astype(np.float64) for k in ('hi', 'lo', 'A', 'B')}
    with np.load(R6_PRED, allow_pickle=False) as p:
        preds['R6'] = p['R6'].astype(np.float64)
    for k in preds:
        v = preds[k]
        require(v.shape == (150, 3) and np.isfinite(v).all() and np.abs(np.linalg.norm(v, axis=1) - 1).max() <= 1e-10, f'{k} 预测无效')
        preds[k] = v / np.linalg.norm(v, axis=1, keepdims=True)
    gt, gt_hash = read_gt_once()                       # ← 全程唯一一次 GT 打开
    errs, els, azs, mets = {}, {}, {}, {}
    for k, v in preds.items():
        errs[k], els[k], azs[k], mets[k] = metrics_for(v, gt)
    # 参照重算与各自冻结评测一致（<=1e-9 度）
    consistency = {}
    for name, path, key in [('E4', E4_EVAL, 'E4'), ('C01/full', C01_EVAL, 'C01/full'), ('R6', R6_EVAL, 'R6')]:
        fr = frozen_ref_mean(path, key)
        consistency[name] = {kk: abs(mets[name][kk] - fr[kk]) for kk in ('mean_deg', 'p90_deg', 'within_5deg_count') if kk in fr}
        fr_p50 = fr.get('median_deg', fr.get('p50_deg'))
        if fr_p50 is not None:
            consistency[name]['p50_deg'] = abs(mets[name]['p50_deg'] - fr_p50)
        require('mean_deg' in consistency[name], f'冻结评测 {name} 缺少 mean_deg')
    ref_ok = all(v <= 1e-9 for d in consistency.values() for v in d.values()) and abs(mets['E4']['mean_deg'] - E4_MEAN_FROZEN) <= 1e-9
    a0_valid = 'A0/full' in preds
    a0_conv = bool(json.loads((run / 'fitting/A0/runs/full/certificate.json').read_text())['converged']) if a0_valid else False
    mean_a0 = mets['A0/full']['mean_deg'] if a0_valid else float('nan')
    status, reasons = pass_rule(bool(ref_ok), a0_valid, a0_conv, mean_a0, mets['E4']['mean_deg'])
    if not ref_ok:
        reasons.append('参照重算与冻结评测不一致')
        status = 'FAILED'
    tau = man['tau_deg']
    attr = attribution(mets['A1/full']['mean_deg'], mean_a0, tau) if (a0_valid and 'A1/full' in preds) else dict(kind='不可判定')
    if attr.get('kind') != '不可判定':
        attr.update(elev_signed_bias_diff_A1_minus_A0=mets['A1/full']['elevation']['signed_mean_deg'] - mets['A0/full']['elevation']['signed_mean_deg'],
                    azimuth_mae_diff_A1_minus_A0=mets['A1/full']['azimuth']['mae_deg'] - mets['A0/full']['azimuth']['mae_deg'])
    cross = {}
    for lab, (p, q) in {'C01-E4': ('C01/full', 'E4'), 'A0-C01': ('A0/full', 'C01/full'), 'A1-A0': ('A1/full', 'A0/full'), 'A0-E4': ('A0/full', 'E4')}.items():
        if p in mets and q in mets:
            cross[lab] = dict(mean_deg=mets[p]['mean_deg'] - mets[q]['mean_deg'], p50_deg=mets[p]['p50_deg'] - mets[q]['p50_deg'], p90_deg=mets[p]['p90_deg'] - mets[q]['p90_deg'],
                              within_5deg_count=mets[p]['within_5deg_count'] - mets[q]['within_5deg_count'],
                              elevation_signed_bias_deg=mets[p]['elevation']['signed_mean_deg'] - mets[q]['elevation']['signed_mean_deg'],
                              azimuth_mae_deg=mets[p]['azimuth']['mae_deg'] - mets[q]['azimuth']['mae_deg'])
    amb_ans = None
    if a0_valid:
        try:
            k45 = json.loads((run / 'fitting/A0/diagnostics/K_A4_A5_fisher.json').read_text())
            R_a = k45['modes']['u_e0']['R']
            R_b = k45['refs']['C01']['modes']['u_e0']['R']
            with np.load(run / 'fitting/A0/predictions.npz', allow_pickle=False) as p:
                a0hl, a0ab = mean_angle(p['A0_hi'], p['A0_lo']), mean_angle(p['A0_A'], p['A0_B'])
            amb_ans = ambiguity_answer(R_a, R_b, a0hl, mean_angle(c01_aux['hi'], c01_aux['lo']), a0ab, mean_angle(c01_aux['A'], c01_aux['B']))
            amb_ans.update(R_a_A0_u_e0=R_a, R_b_C01_u_e0=R_b, A0_hilo_mean_deg=a0hl, C01_hilo_mean_deg=mean_angle(c01_aux['hi'], c01_aux['lo']), A0_AB_mean_deg=a0ab,
                           C01_AB_mean_deg=mean_angle(c01_aux['A'], c01_aux['B']), r_a=k45.get('r_a'), corr_amb_rhobar=k45.get('corr_amb_rhobar'))
        except Exception as exc:                        # 诊断缺失不阻塞主结论
            amb_ans = dict(answer='不可判定', reason=repr(exc))
    result = dict(status=status, fail_reasons=reasons, criterion='完整性通过 且 A0 full 收敛 且 Mean_A0 < Mean_E4（全精度严格）', methods=mets, gt_sha256=gt_hash, gt_file_open_count=GT_OPEN_COUNT[0],
                  reference_consistency=consistency, reference_consistency_ok=bool(ref_ok), E4_mean_frozen=E4_MEAN_FROZEN, tau_deg=tau,
                  A1_attribution=attr, cross_attribution=cross, ambiguity_answer=amb_ans, a0_full_converged=a0_conv,
                  note_cross='A0 相对 E4 同时改动三处（偏置形态+上界、RGB、去正则）；A0-E4 只作总效果，不得单独归因于环境项',
                  freeze_manifest_sha256=integ['manifest_sha256'], evaluated_utc=now(), coordinate_system='world', direction_convention='surface_to_light',
                  sign_search_performed=False, gt_used_for_estimation=False, gt_feedback_parameter_tuning=False)
    dump(out / 'results.json', result)
    with (out / 'per_frame.csv').open('x', newline='') as h:
        w = csv.writer(h)
        w.writerow(['frame', 'method', 'pred_x', 'pred_y', 'pred_z', 'gt_x', 'gt_y', 'gt_z', 'angle_deg', 'azimuth_pred_minus_gt_deg', 'elevation_pred_minus_gt_deg'])
        for k in preds:
            for i in range(150):
                w.writerow([i + 1, k, *preds[k][i], *gt[i], float(errs[k][i]), float(azs[k][i]), float(els[k][i])])
    figures(out, run, preds, errs, els, azs, gt, a0_valid)
    dump(out / 'evaluation_complete.json', dict(status='COMPLETE', result=status, utc=now(), gt_file_open_count=GT_OPEN_COUNT[0],
                                                 artifacts={p.name: sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    dump(receipt, dict(status='GT_EVALUATION_COMPLETE', utc=now(), gt_file_open_count=GT_OPEN_COUNT[0], gt_sha256=gt_hash, freeze_manifest_sha256=integ['manifest_sha256'],
                       code_sha256=sha(__file__), gt_used_for_decisions=False), mode='w')
    print(json.dumps(dict(status=status, reasons=reasons, A0=mets.get('A0/full', {}).get('mean_deg'), E4=mets['E4']['mean_deg'], A1=mets.get('A1/full', {}).get('mean_deg')), ensure_ascii=False))
    if not ref_ok:
        raise SystemExit('参照重算与冻结评测不一致（结果已写入，状态 FAILED）')


def figures(out, run, preds, errs, els, azs, gt, a0_valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out / 'figures').mkdir()
    show = [k for k in ('A0/full', 'A1/full', 'E4', 'C01/full') if k in preds]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(FRAMES, np.rad2deg(np.unwrap(np.arctan2(gt[:, 1], gt[:, 0]))), 'k', label='GT')
    ax[1].plot(FRAMES, np.degrees(np.arcsin(gt[:, 2])), 'k', label='GT')
    for k in show:
        ax[0].plot(FRAMES, np.rad2deg(np.unwrap(np.arctan2(preds[k][:, 1], preds[k][:, 0]))), label=k)
        ax[1].plot(FRAMES, np.degrees(np.arcsin(preds[k][:, 2])), label=k)
    for v in (40, 50):
        ax[1].axhline(v, c='gray', ls=':')
    ax[0].set_title('azimuth (unwrapped)')
    ax[1].set_title('elevation')
    for a_ in ax:
        a_.legend()
        a_.grid(alpha=.2)
        a_.set_xlabel('frame')
    fig.tight_layout()
    fig.savefig(out / 'figures/trajectories_vs_gt.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k in show:
        ax.plot(FRAMES, errs[k], label=k)
    ax.axhline(5, c='k', ls='--')
    ax.set(xlabel='frame', ylabel='angular error (deg)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'figures/angular_error_curve.png', dpi=160)
    plt.close(fig)
    if a0_valid:
        z = np.load(run / 'fitting/A0/ambient.npz')
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(FRAMES, els['A0/full'], label='A0 elevation error (pred-GT)')
        ax[0].axhline(0, c='k', lw=.6)
        ax[0].legend()
        ax[0].set_xlabel('frame')
        if 'profile_grid' in z.files:
            ax[1].plot(z['profile_grid'], z['profile_F'])
            ax[1].axvline(float(z['amb_full']), c='r', ls='--')
            ax[1].set_xlabel('amb')
            ax[1].set_title('profile F(amb) (descriptive only)')
        fig.tight_layout()
        fig.savefig(out / 'figures/elevation_error_vs_amb_profile.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
