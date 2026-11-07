"""T01(1107-03) I6：环境光泄漏监测 M1-M6（无 GT；E1 vs E2 同口径）。

输入：
  --e1/--e2            Stage3 模型目录(含 point_cloud/、envmap/)；--iteration 55000
  --b00                B00 Stage1 模型目录(M5 逐 Gaussian 法线对比基线)
  --e1-fit/--e1-dev    pipeline1107_evaluate.py --split fit/dev --dump-leak 的输出目录(E2 同理)
  --e1-test/--e2-test  pipeline1107_evaluate.py --split test 的输出目录(M6 分量能量，读 metrics.json)
输出：<output>/leakage.json 与若干 png。

分层变量(规划第6节)：环境可见度 V 分三档 [0,.5)/[.5,.8)/[.8,1]；SUN 直射项 s=cos*V_sun 分 s<0.05(阴影)/s>=0.05(受光)。
各模型使用自己的 V 和 s 分层(几何微调后两个模型的 V 不同)。旗标阈值为预注册，只用于如实报告，不用于选择。
M5 的“逐像素渲染法线相对 B00”需要 B00 的法线渲染，当前评测入口不提供 -> 记为 N/A 并写明原因；
逐 Gaussian 旋转法线相对 B00 的角度变化完整实现。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

C0 = 0.28209479177387814
CLAMP_LOW, CLAMP_HIGH = 0.03, 0.97
LUMA = np.array([0.2126, 0.7152, 0.0722])
V_EDGES = (0.5, 0.8)
S_THRESHOLD = 0.05
THRESHOLDS = dict(m1_low_v_relative=0.10, m1_lit_relative=0.03, m2_abs_r=0.05, m3_mean=-0.01, m3_ratio=2.0,
                  m4_env_ratio=1.2, m5_p90_deg=2.0)


# ---------------------------------------------------------------- 基础统计
def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, np.power((np.maximum(x, 0.04045) + 0.055) / 1.055, 2.4))


def describe(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return dict(n=0, mean=None, median=None, p5=None, p95=None)
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    return dict(n=int(x.size), mean=float(x.mean()), median=float(p50), p5=float(p5), p95=float(p95))


def strata(v, s):
    """返回 {层名: 布尔掩码}。s 缺失(长度 0)时不产生 s 层。"""
    out = {'all': np.ones(len(v), bool), 'V<0.5': v < V_EDGES[0],
           '0.5<=V<0.8': (v >= V_EDGES[0]) & (v < V_EDGES[1]), 'V>=0.8': v >= V_EDGES[1]}
    if len(s) == len(v):
        out['shadow(s<0.05)'] = s < S_THRESHOLD
        out['lit(s>=0.05)'] = s >= S_THRESHOLD
    return out


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def relative(a, b):
    """(b-a)/|a|；a 为 0 或缺失时 None。"""
    if a is None or b is None or abs(a) < 1e-12:
        return None
    return float((b - a) / abs(a))


# ---------------------------------------------------------------- 逐像素池
def load_pool(*run_dirs):
    """把若干 evaluate 输出目录的 leak/*.npz 拼成一个池；返回 dict of arrays。"""
    cols = dict(albedo=[], resid=[], v=[], s=[], env=[], sun=[], ind=[], normal=[])
    frames = 0
    for run_dir in run_dirs:
        for path in sorted((Path(run_dir) / 'leak').glob('*.npz')):
            with np.load(path) as d:
                n = int(d['fg'].sum())
                cols['albedo'].append(d['albedo_lin'].astype(np.float32))
                cols['resid'].append(d['resid_lin'].astype(np.float32).mean(1))
                cols['v'].append(d['env_vis'].astype(np.float32))
                cols['s'].append(d['s'].astype(np.float32) if d['s'].size == n else np.zeros(0, np.float32))
                cols['env'].append(d['env'].astype(np.float32))
                cols['sun'].append(d['sun'].astype(np.float32) if d['sun'].shape[0] == n else np.zeros((0, 3), np.float32))
                cols['ind'].append(d['ind'].astype(np.float32))
                cols['normal'].append(d['normal'].astype(np.float32))
            frames += 1
    if not frames:
        raise ValueError('没有找到 leak/*.npz：评测时须加 --dump-leak。')
    pool = {k: np.concatenate(v, 0) for k, v in cols.items()}
    pool['lum'] = pool['albedo'].astype(np.float64) @ LUMA
    pool['frames'] = frames
    return pool


# ---------------------------------------------------------------- Gaussian 级
def read_ply(path):
    from plyfile import PlyData
    v = PlyData.read(str(path)).elements[0]
    names = [p.name for p in v.properties]
    dc = np.stack([np.asarray(v[f'albedo_dc_{i}']) for i in range(3)], 1)
    rot = np.stack([np.asarray(v[f'rot_{i}']) for i in range(4)], 1)
    out = dict(albedo_dc=dc.astype(np.float64), rot=rot.astype(np.float64), n=len(dc))
    if 'x' in names:
        out['xyz'] = np.stack([np.asarray(v[k]) for k in 'xyz'], 1).astype(np.float64)
    return out


def quaternion_normal(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), 1)


def gaussian_albedo(dc):
    """与 GaussianModel.get_albedo 一致：clamp(dc*C0+0.5, 0.03, 0.97)；返回 (线性albedo, 贴边比例, 原始值)。"""
    raw = dc * C0 + 0.5
    lin = srgb_to_linear(np.clip(raw, CLAMP_LOW, CLAMP_HIGH))
    clamped = dict(low=float((raw <= CLAMP_LOW).mean()), high=float((raw >= CLAMP_HIGH).mean()))
    return lin, clamped, raw


def angle_degrees(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def gaussian_stats(ply_path, b00_path=None):
    g = read_ply(ply_path)
    lin, clamped, _ = gaussian_albedo(g['albedo_dc'])
    out = dict(n=g['n'], clamp_fraction=clamped, albedo_linear={
        c: describe(lin[:, i]) for i, c in enumerate('RGB')}, luminance=describe(lin @ LUMA))
    out['_lin'] = lin
    if b00_path is not None:
        b = read_ply(b00_path)
        if b['n'] != g['n']:
            raise ValueError('Gaussian 数量与 B00 不一致(Stage3 不应致密化)。')
        angle = angle_degrees(quaternion_normal(g['rot']), quaternion_normal(b['rot']))
        out['normal_change_vs_b00_deg'] = dict(mean=float(angle.mean()), p50=float(np.percentile(angle, 50)),
                                               p90=float(np.percentile(angle, 90)), p99=float(np.percentile(angle, 99)),
                                               fraction_ge_59=float((angle >= 59).mean()))
        if 'xyz' in g and 'xyz' in b:
            disp = np.linalg.norm(g['xyz'] - b['xyz'], axis=1)
            out['displacement_vs_b00'] = dict(mean=float(disp.mean()), p90=float(np.percentile(disp, 90)),
                                              max=float(disp.max()))
        out['_angle'] = angle
    return out


# ---------------------------------------------------------------- 环境光(M4)
def load_env(model, iteration):
    import torch
    path = Path(model) / f'envmap/iteration_{iteration}/envmap.pth'
    state = torch.load(path, map_location='cpu', weights_only=False)
    base = state['state_dict']['base'].numpy().astype(np.float64)
    activation = state['activation']
    value = np.exp(base) if activation == 'exp' else (1 / (1 + np.exp(-base)) if activation == 'sigmoid' else base)
    mixed = Path(model) / f'envmap/iteration_{iteration}/mixed.json'
    intensity = json.loads(mixed.read_text())['effective_intensity'] if mixed.exists() else None
    return value, intensity


def env_summary(value):
    h = value.shape[0]
    theta = (np.arange(h) + 0.5) / h * np.pi
    w = np.sin(theta)[:, None, None] * np.ones_like(value)
    return dict(plain_mean=float(value.mean()), solid_angle_mean=float((value * w).sum() / w.sum()),
                top_half_mean=float(value[: h // 2].mean()), bottom_half_mean=float(value[h // 2:].mean()),
                shape=list(value.shape))


# ---------------------------------------------------------------- M1-M6
def m1(pool1, pool2, g1, g2):
    res = dict(pixel={}, gaussian={})
    for name, pool in (('E1', pool1), ('E2', pool2)):
        masks = strata(pool['v'], pool['s'])
        res['pixel'][name] = {layer: dict(luminance=describe(pool['lum'][m]),
                                          channels={c: describe(pool['albedo'][m, i]) for i, c in enumerate('RGB')})
                              for layer, m in masks.items()}
    res['gaussian']['E1'] = {k: v for k, v in g1.items() if not k.startswith('_')}
    res['gaussian']['E2'] = {k: v for k, v in g2.items() if not k.startswith('_')}
    diff = g2['_lin'] - g1['_lin']
    res['gaussian']['E2_minus_E1_linear_albedo'] = {c: describe(diff[:, i]) for i, c in enumerate('RGB')}
    res['gaussian']['E2_minus_E1_luminance'] = describe(diff @ LUMA)
    res['gaussian']['mean_abs_diff'] = float(np.abs(diff).mean())
    low = relative(res['pixel']['E1']['V<0.5']['luminance']['median'], res['pixel']['E2']['V<0.5']['luminance']['median'])
    lit_l = res['pixel']['E1'].get('lit(s>=0.05)', {}).get('luminance', {}).get('median')
    lit_r = res['pixel']['E2'].get('lit(s>=0.05)', {}).get('luminance', {}).get('median')
    lit = relative(lit_l, lit_r)
    res['low_v_relative_change'], res['lit_relative_change'] = low, lit
    res['flag'] = None if low is None or lit is None else bool(
        low > THRESHOLDS['m1_low_v_relative'] and abs(lit) < THRESHOLDS['m1_lit_relative'])
    res['rule'] = '低可见度档E2 albedo亮度中位数比E1高>10%(相对)，且受光档差异<3%'
    return res


def m2(pool1, pool2):
    res = {}
    for name, pool in (('E1', pool1), ('E2', pool2)):
        res[name] = dict(r_lum_s=pearson(pool['lum'], pool['s']) if len(pool['s']) == len(pool['lum']) else None,
                         r_lum_v=pearson(pool['lum'], pool['v']))
    deltas = [abs(res['E2'][k]) - abs(res['E1'][k]) for k in ('r_lum_s', 'r_lum_v')
              if res['E1'][k] is not None and res['E2'][k] is not None]
    res['delta_abs_r'] = deltas
    res['flag'] = None if not deltas else bool(max(deltas) > THRESHOLDS['m2_abs_r'])
    res['rule'] = 'E2的|r|比E1大>0.05(albedo亮度与s、与V任一)'
    return res


def m3(pool1, pool2):
    res = {}
    for name, pool in (('E1', pool1), ('E2', pool2)):
        masks = strata(pool['v'], pool['s'])
        res[name] = {}
        for layer, m in masks.items():
            r = pool['resid'][m]
            d = describe(r)
            d['mae'] = float(np.abs(r).mean()) if r.size else None
            res[name][layer] = d
    flagged = []
    for layer in ('shadow(s<0.05)', 'V<0.5'):
        a, b = res['E1'].get(layer, {}).get('mean'), res['E2'].get(layer, {}).get('mean')
        if a is None or b is None:
            continue
        flagged.append(b < THRESHOLDS['m3_mean'] or abs(b) >= THRESHOLDS['m3_ratio'] * max(abs(a), 1e-12))
    res['flag'] = None if not flagged else bool(any(flagged))
    res['rule'] = 'E2在阴影或低V档平均残差(渲染-目标，线性域)<-0.01，或|mean|达E1的2倍以上'
    return res


def m4(env1, env2, sun1, sun2):
    s1, s2 = env_summary(env1), env_summary(env2)
    ratio = s2['solid_angle_mean'] / s1['solid_angle_mean']
    return dict(E1=s1, E2=s2, sun_intensity=dict(E1=sun1, E2=sun2), env_mean_ratio_E2_over_E1=float(ratio),
                flag=bool(ratio > THRESHOLDS['m4_env_ratio']),
                rule='E2/E1环境光(立体角加权)均值比>1.2')


def m5(g1, g2, pool1=None, pool2=None):
    res = dict(E1=g1.get('normal_change_vs_b00_deg'), E2=g2.get('normal_change_vs_b00_deg'),
               displacement=dict(E1=g1.get('displacement_vs_b00'), E2=g2.get('displacement_vs_b00')))
    # 描述量：同像素位置不可配对，这里只给 E1/E2 各自渲染法线的 z 分量分布按V分层，作为现象描述
    res['pixel_vs_b00'] = 'N/A：评测入口不渲染B00法线；旗标按逐Gaussian p90变化量判断(见flag_basis)'
    if g1.get('_angle') is not None and g2.get('_angle') is not None:
        d = res['E2']['p90'] - res['E1']['p90']
        res['p90_difference_deg'] = d
        res['flag'] = bool(d > THRESHOLDS['m5_p90_deg'])
        res['flag_basis'] = '逐Gaussian全体p90变化量E2-E1>2度(无逐Gaussian可见度，未按V分层；规划按V分层的像素版为N/A)'
    else:
        res['flag'] = None
    return res


def m6(test1, test2):
    """正确性检查：E2间接分量恒为0、E1>0；读取evaluate的TEST metrics.json逐帧能量。"""
    out = {}
    for name, path in (('E1', test1), ('E2', test2)):
        frames = json.loads(Path(path).read_text())['frames']
        out[name] = {f['frame']: dict(env=f.get('energy_env'), sun=f.get('energy_sun'), indirect=f.get('energy_indirect'),
                                       indirect_max=f.get('energy_indirect_max')) for f in frames}
    e1_positive = all((v['indirect_max'] or 0) > 0 for v in out['E1'].values())
    e2_zero = all((v['indirect_max'] if v['indirect_max'] is not None else 1) == 0 for v in out['E2'].values())
    out['check_E1_indirect_positive'] = bool(e1_positive)
    out['check_E2_indirect_zero'] = bool(e2_zero)
    out['engineering_error'] = bool(not (e1_positive and e2_zero))
    out['rule'] = '正确性检查：E2间接分量必须恒为0、E1必须>0，否则判工程错误'
    return out


# ---------------------------------------------------------------- 图
def make_figures(output, pool1, pool2, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, layer in zip(axes, ('all', 'shadow(s<0.05)', 'V<0.5')):
        for name, pool, color in (('E1', pool1, 'tab:blue'), ('E2', pool2, 'tab:red')):
            m = strata(pool['v'], pool['s']).get(layer)
            if m is not None and m.any():
                ax.hist(pool['resid'][m], bins=100, range=(-0.3, 0.3), alpha=.5, label=name, color=color, density=True)
        ax.set_title('residual (render-target, linear) ' + layer)
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / 'm3_residual_hist.png', dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    layers = ['V<0.5', '0.5<=V<0.8', 'V>=0.8']
    x = np.arange(len(layers))
    for k, name in enumerate(('E1', 'E2')):
        ax.bar(x + k * 0.35, [res['m1']['pixel'][name][l]['luminance']['median'] or 0 for l in layers], 0.35, label=name)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(layers)
    ax.set_ylabel('median albedo luminance (linear)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / 'm1_albedo_by_visibility.png', dpi=120)
    plt.close(fig)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run(a):
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    ply = lambda m: Path(m) / f'point_cloud/iteration_{a.iteration}/point_cloud.ply'
    b00_ply = Path(a.b00) / f'point_cloud/iteration_{a.b00_iteration}/point_cloud.ply'
    pool1 = load_pool(a.e1_fit, a.e1_dev)
    pool2 = load_pool(a.e2_fit, a.e2_dev)
    g1, g2 = gaussian_stats(ply(a.e1), b00_ply), gaussian_stats(ply(a.e2), b00_ply)
    env1, sun1 = load_env(a.e1, a.iteration)
    env2, sun2 = load_env(a.e2, a.iteration)
    res = dict(thresholds=THRESHOLDS, frames=dict(E1=pool1['frames'], E2=pool2['frames']),
               pixels=dict(E1=len(pool1['v']), E2=len(pool2['v'])), strata_note='FIT117+DEV18 前景像素池；各模型用自己的V与s分层')
    res['m1'] = m1(pool1, pool2, g1, g2)
    res['m2'] = m2(pool1, pool2)
    res['m3'] = m3(pool1, pool2)
    res['m4'] = m4(env1, env2, sun1, sun2)
    res['m5'] = m5(g1, g2)
    res['m6'] = m6(Path(a.e1_test) / 'metrics.json', Path(a.e2_test) / 'metrics.json')
    flags = {k: res[k].get('flag') for k in ('m1', 'm2', 'm3', 'm4', 'm5')}
    res['flags'] = flags
    res['flags_triggered'] = int(sum(1 for v in flags.values() if v))
    res['flags_not_evaluable'] = [k for k, v in flags.items() if v is None]
    res['report_rule'] = '触发几个旗标如实写几个；渲染指标支持且旗标>=2时，结论必须加“伴随泄漏迹象”限定；旗标不用于选择'
    (output / 'leakage.json').write_text(json.dumps(clean(res), ensure_ascii=False, indent=2) + '\n')
    make_figures(output, pool1, pool2, res)
    print(json.dumps(dict(flags=flags, triggered=res['flags_triggered'], m6_engineering_error=res['m6']['engineering_error']),
                     ensure_ascii=False))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('e1', 'e2', 'b00'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('e1-fit', 'e1-dev', 'e2-fit', 'e2-dev', 'e1-test', 'e2-test'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--iteration', type=int, default=55000)
    p.add_argument('--b00-iteration', type=int, default=35000)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
