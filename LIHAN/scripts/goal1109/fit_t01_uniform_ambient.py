"""1109-01 T01 全局环境光单标量 Stage 2 拟合入口（A0 / A1 / 目标回归 / 合成自检）。不读取 GT，不 import 任何评测模块，不打开 lights.json。

  python -u -m LIHAN.scripts.goal1109.fit_t01_uniform_ambient --arm {A0,A1} --output <ROOT>/fitting/<ARM> [--smoke]
  python -u -m LIHAN.scripts.goal1109.fit_t01_uniform_ambient --self-test --output output/smoke_test/1109-NN-SYNTH-T01全局环境光自检
  python -u -m LIHAN.scripts.goal1109.fit_t01_uniform_ambient --arm {E4,C01} --regress-only --output <ROOT>/regress/<ARM>

规划：LIHAN/DOC/规划/1109-01-T01全局环境光单标量Stage2规划.md（SHA256 eb12c0e4...）。变量命名：仰角内部参数 nu，环境光标量 amb。
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts import static_rgb_light_fit_r5b as r5b
from LIHAN.scripts import static_rgb_light_fit_r6_env as r6
from LIHAN.scripts.static_rgb_light_fit_r3 import sha, write, sub, ols_score
from LIHAN.scripts.static_rgb_light_fit_r5b import orbit_sign, a_from_z, elev_deg
from LIHAN.scripts.goal1109 import ua_core as C
from LIHAN.scripts.goal1109 import ua_diag as D
from LIHAN.scripts.goal1109 import ua_figs as FG

CODE_SHA = {   # 被 import 的既有代码（冻结值：1108-05 / 1108-03 freeze_manifest 登记）
    'LIHAN/scripts/static_rgb_light_fit_r3.py': '0359461d5c4436698d4eee4d417a564ca273edf0091bfa87153f1ef0eb834517',
    'LIHAN/scripts/static_rgb_light_fit_r5b.py': 'ea6a90084d61d88ee87dbcd5230793de65f56e8b1f562b2125a68d90dd6d12e2',
    'LIHAN/scripts/static_rgb_light_fit_goal1108_02_t01.py': '81e7b7d1cf505583c9eda1ba281aec3b02a4eddbdcef6d71a5b66c591b8e8cb1',
    'LIHAN/scripts/static_rgb_light_fit_r6_env.py': 'bf7bc70c2adf5d155c140f9668b11575a74eed20ced672859d689302439c5d87',
}
GOAL1109_FILES = ['ua_core.py', 'ua_diag.py', 'ua_figs.py', 'ua_selftest.py', 'fit_t01_uniform_ambient.py']


def jsonable(v):
    """递归转换 numpy 标量/数组；NaN/Inf -> None（write 使用 allow_nan=False）。"""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if torch.is_tensor(v):
        return jsonable(v.detach().cpu().numpy())
    return v


def W(path, obj):
    write(path, jsonable(obj))


# ------------------------------------------------------------------ 输入校验
def registered_hashes():
    """从 1108-05 / 1108-03 的 freeze_manifest 取登记哈希（按文件名匹配），与当前代码比对。"""
    reg = {}
    m5 = json.loads((C.C01_RUN_DIR / 'freeze_manifest.json').read_text())
    for sec in ('sha256', 'input_and_code_sha256'):
        for k, v in m5.get(sec, {}).items():
            reg[Path(k).name] = v
    m3 = json.loads((C.R6_RUN_DIR / 'freeze_manifest.json').read_text())
    for k, v in m3['files'].items():
        if 'frozen_code' not in k:
            reg.setdefault(Path(k).name, v)
    return reg


def verify_inputs(include_model=True):
    C.check_root()
    out = {}
    if sha(C.PLAN) != C.PLAN_SHA:
        raise RuntimeError('规划文件哈希与冻结值不符')
    out[str(C.PLAN)] = C.PLAN_SHA
    reg = registered_hashes()
    for path, want in CODE_SHA.items():
        got = sha(path)
        if got != want:
            raise RuntimeError(f'既有代码哈希不符：{path}')
        if Path(path).name in reg and reg[Path(path).name] != got:
            raise RuntimeError(f'既有代码与冻结清单登记值不符：{path}')
        out[path] = got
    if sha(r3.OBS) != r3.OBS_HASH:
        raise RuntimeError('observations.npz 哈希不符')
    out[str(r3.OBS)] = r3.OBS_HASH
    for k, want in C.g1108.EXPECTED.items():
        p = C.BASE / k
        if sha(p) != want:
            raise RuntimeError(f'1031-12 冻结输入哈希不符：{p}')
        out[str(p)] = want
    for f in GOAL1109_FILES:
        p = C.SCRIPT_DIR / f
        if p.is_file():
            out[str(p)] = sha(p)
    if include_model:
        for p in [C.MODEL / 'point_cloud/iteration_35000/point_cloud.ply', C.MODEL / 'deform/iteration_35000/deform.pth']:
            out[str(p)] = sha(p)    # 只记录哈希，不加载 PLY 的 f_dc/f_rest
    return out


def real_bundle():
    obs = C.load_obs_rgb()
    support, supports, picks = C.build_support(obs['n'], obs['intensity'], obs['w'], obs['xyz'], obs['gaussian_ids'], obs['frames'])
    initial, reference, historical = C.load_initial(supports)
    c01 = np.load(C.C01_RUN_DIR / 'fitting/runs/full/direction.npy')
    return dict(obs=obs, support=support, supports=supports, picks=picks, initial=initial, reference=reference, historical=historical, c01=c01)


# ------------------------------------------------------------------ 目标回归（E4 / C01 配置切换验证）
def regress(arm, out):
    assert arm in ('E4', 'C01')
    hashes = verify_inputs(include_model=False)
    out.mkdir(parents=True, exist_ok=False)
    W(out / 'launch.json', dict(argv=sys.argv, pid=os.getpid(), arm=arm, regress_only=True, input_and_code_sha256=hashes, gt_read=False))
    obs = C.load_obs_rgb()
    support, supports, _ = C.build_support(obs['n'], obs['intensity'], obs['w'], obs['xyz'], obs['gaussian_ids'], obs['frames'])
    fi = support['body_fit']
    n, I, w = sub(obs['n'], fi), obs['intensity'][:, fi], obs['w'][:, fi]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ctx = C.make_ctx(n, I, w, device)
    e3 = np.load(C.BASE / 'runs/E3/direction.npy')
    s = orbit_sign(e3)
    if arm == 'E4':
        l = np.load(C.E4_RUN / 'direction.npy')
        phi = np.unwrap(np.arctan2(l[:, 1], l[:, 0]))
        nu = np.arctanh(np.clip((np.arcsin(l[:, 2]) - C.E_C) / C.E_MAX, -.999999999, .999999999))
        frozen = json.loads((C.E4_RUN / 'config.json').read_text())['final_loss']
        tol = 1e-9
        with torch.no_grad():
            val = float(C.objective_e4(torch.as_tensor(phi, dtype=torch.float64, device=device), torch.as_tensor(nu, dtype=torch.float64, device=device), ctx, s)[0])
        src = str(C.E4_RUN / 'direction.npy')
        cfg = dict(model='E4: luminance + b_g(KKT) + 1e-5(rho^2+b^2) + lam_a=1, lam_e=10, lam_rev=0.1')
    else:
        pz = np.load(C.C01_RUN_DIR / 'fitting/runs/full/parameters.npz')
        phi, nu = pz['phi'], pz['a']
        frozen = json.loads((C.C01_RUN_DIR / 'fitting/runs/full/config.json').read_text())['final_loss']
        tol = 1e-10
        with torch.no_grad():
            val = float(C.objective_c01(torch.as_tensor(phi, dtype=torch.float64, device=device), torch.as_tensor(nu, dtype=torch.float64, device=device), ctx, s)[0])
        src = str(C.C01_RUN_DIR / 'fitting/runs/full/parameters.npz')
        cfg = dict(model='C01: luminance + b_g(KKT) + 0.1 rev，无系数/平滑正则')
    rel = abs(val - frozen) / abs(frozen)
    ok = bool(np.isfinite(rel) and rel <= tol)
    W(out / 'regress.json', dict(arm=arm, ok=ok, recomputed_final_loss=val, frozen_final_loss=frozen, rel_diff=rel, tolerance=tol, source=src, orbit_sign=s,
                                 device=device, config=cfg, refuse_to_fit=True, gt_read=False, utc=C.utc()))
    print(json.dumps(dict(arm=arm, ok=ok, recomputed=val, frozen=frozen, rel_diff=rel)), flush=True)
    if not ok:
        raise SystemExit(2)


# ------------------------------------------------------------------ 拟合臂
def audit_json(arm, final_pen, probe):
    items = []
    for name, src, old in [('L_coef', 'static_rgb_light_fit_r5b.py:68', 1e-5), ('a1', 'static_rgb_light_fit_r5b.py:43', 'lambda_a=1'), ('a2', 'static_rgb_light_fit_r5b.py:43', 'lambda_a=1'),
                           ('e1', 'static_rgb_light_fit_r5b.py:43', 'lambda_e=10'), ('e2', 'static_rgb_light_fit_r5b.py:43', 'lambda_e=10')]:
        items.append(dict(name=name, source=src, old_weight=old, new_weight=0.0, raw_value_at_full_endpoint=None if name == 'L_coef' else final_pen.get(name),
                          weighted_contribution=0.0, gradient_contribution=0.0, evidence='objective_ua 的 total 不含该项：torch.equal(total, data+0.1*rev) 与总梯度逐位等于参考梯度（合成断言）'))
    return dict(gt_read=False, arm=arm, objective='L_data + 0.1 R_rev', items=items,
                retained=dict(rev=0.1, orbit_init_and_sign='保留', total_orbit_check='|sum dphi|>=pi', elevation_band_deg=[40, 50], rho_denominator_1e_6='保留并披露：岭收缩',
                              E3_init='历史 E3（含 1e-5 系数项与 E2 三帧平均）固定读取', preconditioner='identity'),
                absent=['epsilon/SH 环境项', 'b_g', '可学习光强', 'rho/amb 正则', 'ρ 色度或均值锚定', 'Huber/IRLS/geo_penalty/balance_factor'],
                physical_constraints=['amb>=0', 'rho in [0,1]', '|l_t|=1'], parameter_inventory=dict(requires_grad=['phi(150)', 'nu(150)'], optimizer_groups=['phi lr=0.01', 'nu lr=0.11']),
                synthetic_probe=probe)


def run_arm(B, out, arm, smoke=False, flows=None, hashes=None, steps=C.STEPS, quiet=False):
    """B：数据包（真实或合成）。out：fitting/<ARM>。返回 dict(status, ...)。"""
    assert arm in ('A0', 'A1')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    obs = B['obs']
    n, y, rgb, I, w = obs['n'], obs['y'], obs['rgb'], obs['intensity'], obs['w']
    fi, vi = B['support']['body_fit'], B['support']['body_val']
    supports, initial = B['supports'], B['initial']
    reference, c01 = B['reference'], B['c01']
    W(out / 'launch.json', dict(argv=sys.argv, pid=os.getpid(), arm=arm, smoke=smoke, input_and_code_sha256=hashes or {}, gt_read=False, started_utc=C.utc(),
                                 kappa_consistency_max_abs=obs.get('kappa_consistency_max_abs')))
    np.savez_compressed(out / 'support.npz', **B['support'])
    for s_, pick in B['picks'].items():
        np.save(out / f'bootstrap_{s_}_sample.npy', pick)
    probe = C.regularization_audit_probe()
    flows = flows or (['full'] if smoke else C.ARM_FLOWS[arm])
    ends, infos, params, status = {}, {}, {}, {}
    t_start = time.time()
    for flow in flows:
        source = flow if flow in supports else 'full'
        idx, init = supports[source], initial[source]
        if flow == 'hi':
            nu0 = np.arctanh(.9)
        elif flow == 'lo':
            nu0 = -np.arctanh(.9)
        else:
            nu0 = a_from_z(init[:, 2].mean())
        mode = {'ainit_lo': 'lo', 'ainit_hi': 'hi'}.get(flow)
        t0 = time.time()
        try:
            end, info = C.fit_traj_ua(sub(n, idx), y[:, idx], w[:, idx], init, nu0, out, flow, arm=arm,
                                      amb_init_mode=mode, steps=steps, quiet=quiet)
        except (FloatingPointError, RuntimeError) as exc:
            if '非有限loss' not in str(exc):
                raise
            status[flow] = dict(valid=False, converged=False, status='NO_FINITE_ENDPOINT', reason=str(exc))
            W(out / 'runs' / flow / 'failure.json', status[flow])
            W(out / 'diagnostics_partial.json', status)
            continue
        pz = np.load(out / 'runs' / flow / 'parameters.npz')
        amb, rho = float(pz['amb']), pz['rho']
        valid = C.endpoint_valid(end, rho, amb)
        ends[flow], infos[flow], params[flow] = end, info, dict(amb=amb, rho=rho)
        status[flow] = dict(valid=bool(valid), converged=bool(info['converged']), status='CONVERGED' if info['converged'] else 'NOT_CONVERGED', seconds=time.time() - t0,
                            certificate_path=info['certificate'].get('path'), final_loss=info['final_loss'], amb=amb, band=C.g1108.band(end))
        if not valid:
            status[flow]['status'] = 'INVALID_ENDPOINT'
        W(out / 'diagnostics_partial.json', status)
    full_valid = bool(status.get('full', {}).get('valid', False))
    if smoke:
        res = dict(arm=arm, certificate=infos['full']['certificate'] if 'full' in infos else None, converged=bool(infos['full']['converged']) if 'full' in infos else False,
                   final_loss=infos['full']['final_loss'] if 'full' in infos else None, amb=params['full']['amb'] if 'full' in params else None, seconds=time.time() - t_start,
                   cuda_max_memory_allocated_MiB=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None), flows=status, gt_read=False)
        W(out / 'smoke_result.json', res)
        print('UA_SMOKE', json.dumps(jsonable(dict(converged=res['converged'], seconds=res['seconds'], amb=res['amb']))), flush=True)
        return dict(status='SMOKE_DONE', smoke=res)

    selected = C.choose(full_valid, arm)
    invalid_aux = [k for k, v in status.items() if k != 'full' and not v.get('valid', False)]
    selected['diagnostic_incomplete'] = invalid_aux
    valid_ends = {k: v for k, v in ends.items() if status[k]['valid']}
    selected['repeat_status'] = C.repeat_status(valid_ends.get('full'), valid_ends.get('repeat_full'))
    diag_dir = out / 'diagnostics'
    diag_dir.mkdir()
    errors = []
    results = {}
    if full_valid:
        results = run_diagnostics(B, out, arm, valid_ends, params, infos, status, diag_dir, errors)
    else:
        W(diag_dir / 'N_A.json', dict(status='N/A', reason='full 无有效终点', gt_read=False))
    # 预测
    pred = dict(frames=obs['frames'], reference_E4=reference, **B['historical'])
    for k, v in valid_ends.items():
        pred[f'{arm}_{k}'] = v
    np.savez_compressed(out / 'predictions.npz', **pred)
    amb_arrays = {f'amb_{k}': np.float64(params[k]['amb']) for k in params}
    amb_arrays.update(results.get('ambient_arrays', {}))
    np.savez_compressed(out / 'ambient.npz', **amb_arrays)
    W(out / 'regularization_audit.json', audit_json(arm, infos['full']['penalties'] if 'full' in infos else {}, probe))
    W(out / 'method_config.json', dict(
        arm=arm, gt_read=False, coordinate_system='world', direction_convention='surface_to_light', directional_intensity=1, learnable_intensity=False,
        model='y_hat_tgc = rho_gc*(amb + max(0,n.l_t)), V=1, amb>=0 (A1: amb=0), rho in [0,1]', loss='linear RGB, per-channel equal weight 1/3, weights w_tg',
        lam_a=0, lam_e=0, lam_rev=C.LAM_REV, l_coef=0, elevation_band_deg=[40, 50], plan=str(C.PLAN), plan_sha256=C.PLAN_SHA, flows=flows, seed=C.SEED,
        search=dict(grid_points=C.GRID_N + 1, a_max0=C.A_MAX0, expansions_max=C.EXPAND_MAX, golden_tol=C.GOLD_TOL), flags=results.get('flags', []),
        full_converged=bool(status.get('full', {}).get('converged', False)), input_and_code_sha256=hashes or {}))
    W(out / 'selection.json', selected)
    final = C.final_status(full_valid, invalid_aux + errors)
    W(out / 'fit_complete.json', dict(status=final, arm=arm, selected=selected, diagnostic_errors=errors, flags=results.get('flags', []), flows=status, gt_read=False,
                                      visual_inspection='待 exp-run 实际打开全部图后写 visual_inspection.json', finished_utc=C.utc()))
    print('UA_FIT_COMPLETE', arm, final, flush=True)
    return dict(status=final, flows=status, results=results)


def safe(name, fn, diag_dir, errors, store):
    try:
        r = fn()
    except Exception as exc:       # 诊断失败不丢拟合结果；记录并标记 INCOMPLETE_DIAGNOSTICS
        r = dict(status='ERROR', error=repr(exc), traceback=traceback.format_exc(), gt_read=False)
        errors.append(name)
    W(diag_dir / f'{name}.json', r)
    store[name] = r
    return r


def run_diagnostics(B, out, arm, ends, params, infos, status, diag_dir, errors):
    obs = B['obs']
    n, y, rgb, I, w = obs['n'], obs['y'], obs['rgb'], obs['intensity'], obs['w']
    fi, vi = B['support']['body_fit'], B['support']['body_val']
    dc = D.make_dctx(sub(n, fi), y[:, fi], rgb[:, fi], I[:, fi], w[:, fi])
    dcv = D.make_dctx(sub(n, vi), y[:, vi], rgb[:, vi], I[:, vi], w[:, vi])
    full = ends['full']
    amb = params['full']['amb']
    rho = params['full']['rho']
    ref = B['reference']
    c01 = B['c01']
    store, flags = {}, []
    arrays = {}
    A0 = arm == 'A0'

    def collect(r):
        if isinstance(r, dict):
            for f in r.get('flags', []) if isinstance(r.get('flags', []), list) else []:
                flags.append(f)
    # ---- 7.1 通用轴 ----
    def general():
        g = dict(gt_read=False, flows={}, differences={}, flags=[])
        for k, v in status.items():
            g['flows'][k] = {kk: v.get(kk) for kk in ('valid', 'converged', 'status', 'certificate_path', 'final_loss', 'amb', 'seconds', 'band')}
        if not status['full']['converged']:
            g['flags'].append('NOT_CONVERGED')
        for k in ['A', 'B', 'boot103108', 'boot103109', 'hi', 'lo', 'ainit_lo', 'ainit_hi', 'repeat_full']:
            if k in ends:
                g['differences'][f'{k}_vs_full'] = C.g1108.diff(ends[k], full)
                if g['differences'][f'{k}_vs_full']['large_disagreement'] and k != 'repeat_full':
                    g['flags'].append(f'LARGE_DISAGREEMENT:{k}')
        for a_, b_ in [('A', 'B'), ('hi', 'lo')]:
            if a_ in ends and b_ in ends:
                g['differences'][f'{a_}_vs_{b_}'] = C.g1108.diff(ends[a_], ends[b_])
        g['repeat_status'] = C.repeat_status(full, ends.get('repeat_full'))
        if g['repeat_status'] == 'NONDETERMINISTIC':
            g['flags'].append('NONDETERMINISTIC')
        g['repeat_max_deg'] = float(C.chord_deg(full, ends['repeat_full']).max()) if 'repeat_full' in ends else None
        # E4 口径留出光度
        mine = ols_score(sub(n, vi), I[:, vi], w[:, vi], full)
        e4 = ols_score(sub(n, vi), I[:, vi], w[:, vi], ref)
        g['body_heldout_clipped_mae'] = dict(this=mine, frozen_E4=e4, ratio=mine / e4,
                                              note='r3.ols_score 内部含无约束 (rho,b) OLS，仅作 E4 口径留出诊断，不进入拟合与输出')
        if mine > 1.05 * e4:
            g['flags'].append('PHOTO_DEGRADED')
        g['band'] = C.g1108.band(full)
        if g['band']['nominal_fraction'] > .2:
            g['flags'].append('NOMINAL_BAND_ACTIVE')
        if g['band']['physical_boundary_near']:
            g['flags'].append('PHYSICAL_BOUNDARY_NEAR')
        q = D.insample_quality(dc, full, amb, rho)
        g['channel_rms_ratio'] = q['channel_rms_ratio_max_over_min']
        if q['channel_rms_ratio_max_over_min'] > 2:
            g['flags'].append('CHANNEL_HETEROSCEDASTIC')
        return g
    gen = safe('general_7_1', general, diag_dir, errors, store)
    collect(gen)
    # ---- RGB 质量 ----
    def rgbq():
        out_ = dict(gt_read=False, in_sample_body_fit=D.insample_quality(dc, full, amb, rho), heldout_body_val=D.heldout_quality(dcv, full, amb))
        out_['references_PSNR_Y_body_fit'] = {'E4_frozen_direction(kkt rho,b)': D.ref_luminance_quality(dc, ref, I[:, fi], w[:, fi]),
                                              'C01_frozen_direction(kkt rho,b)': D.ref_luminance_quality(dc, c01, I[:, fi], w[:, fi])}
        out_['definitions'] = '见规划 7.3；SSIM_pts 为 Gaussian 点集全局 SSIM，非图像 SSIM'
        return out_
    safe('rgb_quality', rgbq, diag_dir, errors, store)
    W(out / 'rgb_quality.json', store['rgb_quality'])
    # ---- K 项 ----
    refs = dict(E4=ref, C01=c01)
    if A0:
        def k1():
            r_, g_, F_ = D.ka1_profile(dc, full, amb)
            arrays['profile_grid'], arrays['profile_F'] = g_, F_
            return r_
        r1 = safe('K_A1_profile', k1, diag_dir, errors, store)
        collect(r1)
        r2 = safe('K_A2_als', lambda: D.ka2_als(dc, full, amb), diag_dir, errors, store)
        collect(r2)
        amb_by = {k: params[k]['amb'] for k in params if status.get(k, {}).get('valid')}
        r3_ = safe('K_A3_stability', lambda: D.ka3_stability(amb_by, ends), diag_dir, errors, store)
        collect(r3_)
    r45 = safe('K_A4_A5_fisher', lambda: D.ka4_ka5(dc, full, amb, rho, arm, refs, I[:, fi], w[:, fi]), diag_dir, errors, store)
    collect(r45)
    if A0:
        def empirical():
            ks = [k for k in ['full', 'A', 'B', 'boot103108', 'boot103109', 'hi', 'lo'] if k in params and status[k]['valid']]
            av = np.array([params[k]['amb'] for k in ks])
            res = dict(flows=ks, amb=av.tolist(), gt_read=False, note='跨流程经验相关，只作描述（支撑不同）')
            for c in range(3):
                rb = np.array([params[k]['rho'][(params[k]['rho'][:, c] > 1e-12) & (params[k]['rho'][:, c] < 1 - 1e-12), c].mean() for k in ks])
                res[f'rho_bar_ch{c}'] = rb.tolist()
                res[f'pearson_ch{c}'] = float(np.corrcoef(av, rb)[0, 1]) if len(ks) >= 3 and av.std() > 0 and rb.std() > 0 else None
            return res
        safe('K_A4_empirical_corr', empirical, diag_dir, errors, store)
    r5p = safe('K_A5p_elevation_profile', lambda: D.ka5p_profile(dc, full, amb, arm, refs, I[:, fi], w[:, fi]), diag_dir, errors, store)
    r6_ = safe('K_A6_uniformity', lambda: D.ka6_uniformity(dc, full, amb, rho, coverage_only=not A0), diag_dir, errors, store)
    collect(r6_)
    ac = None
    if A0:
        r7 = safe('K_A7_chroma', lambda: D.ka7_chroma(dc, full, amb), diag_dir, errors, store)
        collect(r7)
        ac = r7.get('amb_c')
    r8 = safe('K_A8_energy_rho', lambda: D.ka8_energy(dc, full, amb, rho, compute_noub=A0), diag_dir, errors, store)
    collect(r8)
    flags = sorted(set(flags))
    W(out / 'flags.json', dict(flags=flags, gt_read=False, note='旗标只报告，不改输出、不重跑'))
    # ---- 图 ----
    def make_figs():
        FD = figure_data(B, out, arm, dc, ends, params, store, arrays, full, amb, rho, ref, c01, status)
        return FG.figures(out, arm, FD)
    checks = safe('figure_checks', make_figs, diag_dir, errors, store)
    amb_arr = dict(arrays)
    if A0 and ac is not None:
        amb_arr['amb_c'] = np.array(ac)
    if A0 and isinstance(r6_, dict) and 'bins' in r6_:
        amb_arr['amb_bin'] = np.array([np.nan if b['amb'] is None else b['amb'] for b in r6_['bins']])
    if isinstance(r8, dict) and r8.get('amb_noUB') is not None:
        amb_arr['amb_noUB'] = np.float64(r8['amb_noUB'])
    return dict(flags=flags, ambient_arrays=amb_arr, store=store)


def figure_data(B, out, arm, dc, ends, params, store, arrays, full, amb, rho, ref, c01, status):
    T, G, _ = dc['yt'].shape
    pred = D.predict_ua(dc, full, amb, rho)
    res = pred - dc['yt'].numpy()
    wn = dc['w_np']
    res_mean = np.zeros((T, 3))
    q10, q90 = np.zeros((T, 3)), np.zeros((T, 3))
    for t in range(T):
        m = wn[t] > 0
        if m.any():
            res_mean[t] = (wn[t, m, None] * res[t, m]).sum(0) / wn[t, m].sum()
            q10[t], q90[t] = np.quantile(res[t, m], [.1, .9], axis=0)
    zbar = (wn * dc['n'][..., 2]).sum(0) / np.maximum(wn.sum(0), 1e-300)
    order = np.argsort(zbar)
    heat = np.where(wn > 0, res.mean(-1), np.nan)[:, order].T
    hist = {}
    rel = {}
    for k in ends:
        hp = out / 'runs' / k / 'history.json'
        if hp.exists():
            h = json.loads(hp.read_text())
            traj = [e['amb'] for e in h.get('adam', []) if 'amb' in e] + [e['amb'] for e in h.get('optimization', []) if 'amb' in e]
            hist[k] = dict(final_losses=h['final_losses'], amb_traj=traj if traj else [0.0])
        if k != 'full':
            rel[k] = C.chord_deg(ends[k], full)
    prof = None
    if arm == 'A0' and 'profile_grid' in arrays:
        als = [r['amb_end'] for r in store.get('K_A2_als', {}).get('starts', [])]
        prof = (arrays['profile_grid'], arrays['profile_F'], amb, als)
    bs = None
    k6 = store.get('K_A6_uniformity', {})
    if arm == 'A0' and isinstance(k6, dict) and 'g' in k6.get('backlit', {}):
        x = D.shade_t(dc, full).numpy()
        tt, gg = np.nonzero((x <= 0) & (wn > 0))
        rng = np.random.default_rng(C.SEED)
        pick = rng.choice(len(tt), min(len(tt), 20000), replace=False)
        tt, gg = tt[pick], gg[pick]
        c = rng.integers(0, 3, len(tt))
        xs = rho[gg, c] * dc['n'][tt, gg, 2]
        ys = dc['yt'].numpy()[tt, gg, c] - rho[gg, c] * amb
        bs = (xs, ys, (k6['backlit']['delta'], k6['backlit']['g'], float(rho.mean())))
    a_bin = []
    if isinstance(k6, dict) and k6.get('bins'):
        a_bin = [(f"[{b['lo']:.2f},{b['hi']:.2f})", b['amb']) for b in k6['bins']]
    k8 = store.get('K_A8_energy_rho', {})
    eta = k8['eta_lin'] if isinstance(k8, dict) and 'eta_lin' in k8 else {'all': [0, 0, 0], 'lit': [0, 0, 0], 'back': [0, 0, 0]}
    return dict(frames=B['obs']['frames'], res_mean=res_mean, res_q10=q10, res_q90=q90, res_heat=heat, support_count=(wn > 0).sum(1), dirs=ends, refs=dict(E4=ref, C01=c01),
                rho=rho, eta=eta, a_bin=a_bin, amb=amb, hist=hist, rel_angle=rel, profile=prof, back_scatter=bs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=['A0', 'A1', 'E4', 'C01'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--regress-only', action='store_true')
    a = ap.parse_args()
    C.check_root()
    if a.self_test:
        from LIHAN.scripts.goal1109 import ua_selftest
        ua_selftest.run_all(a.output)
        return
    if a.arm in ('E4', 'C01'):
        if not a.regress_only:
            ap.error(f'--arm {a.arm} 只允许与 --regress-only 一起使用（拒绝拟合）')
        regress(a.arm, a.output)
        return
    if a.arm is None or a.regress_only:
        ap.error('需要 --arm A0|A1（拟合）或 --arm E4|C01 --regress-only')
    hashes = verify_inputs()
    B = real_bundle()
    run_arm(B, a.output, a.arm, smoke=a.smoke, hashes=hashes)


if __name__ == '__main__':
    main()
