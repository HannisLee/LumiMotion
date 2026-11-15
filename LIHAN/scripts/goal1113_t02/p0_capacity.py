"""T02 前置诊断 P0：表达容量预检（无训练、无GT、仅CPU）。
依据 LIHAN/DOC/规划/1114-04-T02候选方案草案.md 第3节。只读复用 goal1113 的 physics/inputs 常量与公式；
不读 lights.json/GT/评测目录；不改任何 T01 产物。运行：
  CUDA_VISIBLE_DEVICES="" python LIHAN/scripts/goal1113_t02/p0_capacity.py <输出目录>
"""
import sys, json, math, hashlib, platform, time
from pathlib import Path
ROOT = Path.cwd().resolve()   # LIHAN 是符号链接，不能用 __file__ 解析仓库根；须在仓库根目录运行
assert (ROOT/'scene').is_dir() and (ROOT/'LIHAN/scripts/goal1113').is_dir()
sys.path.insert(0, str(ROOT))
import numpy as np, torch, scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, identity
from LIHAN.scripts.goal1113 import physics as P, inputs as I

T01 = ROOT/'output/1113-01-LH-static-ELDL-V3-T01-BguidedEnvMap'
YW = np.array(P.Y)
OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=False)
LOG = open(OUT/'p0.log', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s+'\n'); LOG.flush()
def sha(p):
    assert 'lights.json' not in str(p).lower()
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def stop(status, obj):
    obj['status'] = status; dump('p0_result.json', obj); log('STOP', status); sys.exit(2)

# ---------------- 输入与哈希 ----------------
OBS = ROOT/I.OBS
if sha(OBS) != I.EXPECTED[I.OBS]: stop('OBS_HASH_MISMATCH', {})
assert not (ROOT/I.RUNTIME/'rgba').exists()
wl = [OBS, ROOT/I.RUNTIME/'rgb_only_manifest.json', ROOT/I.RUNTIME/'transforms_all.json',
      T01/'inputs/prior.npz', T01/'inputs/config.json', T01/'fitting/A/reconstruction.npz', T01/'fitting/A/loss.json',
      T01/'fitting/A/diagnostic.json', T01/'fitting/B/diagnostic.json', T01/'fitting/B/checkpoint_2000.pt',
      ROOT/'LIHAN/scripts/goal1113/physics.py', ROOT/'LIHAN/scripts/goal1113/inputs.py', ROOT/'LIHAN/scripts/goal1113_t02/p0_capacity.py',
      ROOT/'LIHAN/DOC/规划/1114-04-T02候选方案草案.md']
hashes = {str(p.relative_to(ROOT)): sha(p) for p in wl}
for p in (I.RUNTIME/'rgb_only_manifest.json', I.RUNTIME/'transforms_all.json'):
    assert hashes[str(p)] == I.EXPECTED[p], p
env_info = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                highs_via='scipy.optimize.linprog(method=highs-ds)', cuda_visible=str(__import__('os').environ.get('CUDA_VISIBLE_DEVICES')))
log('环境', env_info); log('哈希', json.dumps(hashes, indent=1))

KEYS = ('frames','gaussian_ids','xyz','frame_xyz','normals','intensity','weights','rgb')
with np.load(OBS, allow_pickle=False) as z: d = {k: z[k] for k in KEYS}
n_all = d['normals'].astype(np.float64); w_all = d['weights'].astype(np.float64)
rgb = d['rgb'].astype(np.float64); y = d['intensity'].astype(np.float64)
assert np.array_equal(d['frames'], np.arange(1, 151)) and np.isfinite(w_all).all() and (w_all >= 0).all() and w_all.sum() > 0
valid = w_all > 0
rgb = np.where(valid[..., None], rgb, 0.)
lin = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
inten = np.where(valid, y, 0.)
assert np.max(np.abs((lin@YW)[valid]-inten[valid])) <= 1e-5
# 支持集：与 goal1113.inputs.load_observations 同一确定性规则（复制，不读 BASE）
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    q20 = np.nanquantile(y, .2, axis=0); q95 = np.nanquantile(y, .95, axis=0)
eligible = (np.isfinite(y).sum(0) >= 8) & (q95-q20 > .015)
xyz = d['xyz']; mean_n = n_all.mean(0) if n_all.ndim == 3 else n_all
body = (xyz[:, 2] >= .05) & (abs(mean_n[:, 2]) <= .85)
cells = np.floor(xyz*20).astype(np.int64)
hval = cells[:, 0]*73856093 ^ cells[:, 1]*19349663 ^ cells[:, 2]*83492791
held = hval % 5 == 0
fi = np.flatnonzero(eligible & body & ~held)
splitA = fi[(hval[fi]//5) % 2 == 0]; splitB = fi[(hval[fi]//5) % 2 != 0]
if (len(fi), len(np.flatnonzero(eligible & body & held)), len(splitA), len(splitB)) != (2683, 648, 1450, 1233):
    stop('SUPPORT_COUNT_MISMATCH', {})
prior = np.load(T01/'inputs/prior.npz', allow_pickle=False)
cfg = json.load(open(T01/'inputs/config.json'))
K, s, c0, E0 = cfg['K'], cfg['scale'], cfg['c0'], cfg['e0']
rho_y = prior['rho_y'][fi]; b = prior['b'][fi]; ldir = prior['direction']
assert abs(max(1., float(rho_y.max()))-K) < 1e-12
# 相机：经原有接口，白名单目录 source_all
from LIHAN.scripts.static_rgb_light_extract import cameras
cams = cameras(ROOT/I.RUNTIME, 'all')
assert [int(c.image_name_train_light.split('_')[-1]) for c in cams] == list(range(1, 151))
centers = np.stack([c.camera_center.detach().cpu().numpy() for c in cams]).astype(np.float64)
fxyz = d['frame_xyz']
view = centers[:, None, :]-fxyz; ln = np.linalg.norm(view, axis=-1, keepdims=True)
assert np.isfinite(ln).all() and not (ln <= 1e-8).any(); view /= ln
nt = n_all if n_all.ndim == 3 else np.broadcast_to(n_all[None], (150, *n_all.shape))
N = torch.from_numpy(np.ascontiguousarray(nt[:, fi])); V = torch.from_numpy(np.ascontiguousarray(view[:, fi]))
W = torch.from_numpy(np.ascontiguousarray(w_all[:, fi])); TG = lin[:, fi]
assert bool((W.sum(0) > 0).all())
Q = W/W.sum(0, keepdim=True); G = len(fi)
nl_dir = torch.einsum('tgc,tc->tg', N, torch.from_numpy(ldir)).clamp_min(0)   # max(n·l,0)，方向固定为 E4
log('G=', G, 'K=', K, 's=', s, 'E0=', E0)

# ---------------- 通用 BRDF 基（与 physics.brdf_basis 同公式，网格可变） ----------------
def grid(nth, nph):
    i = torch.arange(nth, dtype=torch.float64); j = torch.arange(nph, dtype=torch.float64)
    th = (i+.5)*math.pi/nth; ph = (j+.5)*2*math.pi/nph
    t, p = torch.meshgrid(th, ph, indexing='ij')
    om = torch.stack((t.sin()*p.cos(), t.sin()*p.sin(), t.cos()), -1).reshape(-1, 3)
    so = ((2*math.pi/nph)*(torch.cos(i*math.pi/nth)-torch.cos((i+1)*math.pi/nth)))[:, None].expand(nth, nph).reshape(-1)
    return om, so
def basis(n, v, nth, nph):
    om, so = grid(nth, nph); EPS = P.EPS; aa = P.ALPHA**2
    nv = (n*v).sum(-1).clamp(0, 1); nl = torch.einsum('...c,jc->...j', n, om).clamp(0, 1)
    h0 = v[..., None, :]+om; hn = torch.linalg.vector_norm(h0, dim=-1, keepdim=True); h = h0/hn.clamp_min(EPS)
    nh = (n[..., None, :]*h).sum(-1).clamp(0, 1); vh = (v[..., None, :]*h).sum(-1).clamp(0, 1)
    fr = .04+.96*(1-vh).pow(5); ggx = aa/(math.pi*((nh.square()*(aa-1)+1).square()))
    g1 = lambda c: 2*c/(c+torch.sqrt(aa+(1-aa)*c.square())).clamp_min(EPS)
    spec = ggx*g1(nv)[..., None]*g1(nl)*fr/(4*nv[..., None].clamp_min(EPS)*nl.clamp_min(EPS))
    mask = (nv[..., None] > 1e-6) & (nl > 0) & (hn[..., 0] > EPS); f = nl*so
    return torch.where(mask, (1-fr)*f/math.pi, 0.), torch.where(mask, spec*f, 0.)
# 与 physics.brdf_basis 逐元素核验（8x16）
dd, ss = basis(N[:, :64], V[:, :64], 8, 16); d0, s0 = P.brdf_basis(N[:, :64], V[:, :64])
basis_err = float(max((dd-d0).abs().max(), (ss-s0).abs().max())); log('basis vs physics 最大误差', basis_err)
if basis_err > 1e-12: stop('BASIS_MISMATCH', dict(basis_err=basis_err))

CH = 32
def build_M(nth, nph, rho):
    """rho: G×3。返回 M[g,j,c]=Σ_t q_tg(rho_gc dif_tgj+spec_tgj)。"""
    J = nth*nph; M = torch.zeros(G, J, 3, dtype=torch.float64)
    for a in range(0, G, CH):
        sl = slice(a, min(a+CH, G)); df, sp = basis(N[:, sl], V[:, sl], nth, nph); q = Q[:, sl]
        qd = torch.einsum('tg,tgj->gj', q, df); qs = torch.einsum('tg,tgj->gj', q, sp)
        M[sl] = qd[..., None]*rho[sl][:, None, :]+qs[..., None]
    return M
def energy_and_sbar(nth, nph, rho, L):
    """L: J×3 非负辐亮度。返回 E（RGB能量占比，与 fit.diagnostic 同口径）与 Sbar。"""
    sums = 0.; den = 0.; sb = []
    for a in range(0, G, CH):
        sl = slice(a, min(a+CH, G)); df, sp = basis(N[:, sl], V[:, sl], nth, nph)
        S = torch.einsum('tgj,jc->tgc', df, L)*rho[sl][None]+torch.einsum('tgj,jc->tgc', sp, L)
        D = K*rho[sl][None]*nl_dir[:, sl][..., None]; w = W[:, sl, None]
        sums += float((w*S).sum()); den += float((w*(D+S)).sum())
        sb.append((W[:, sl]*(S@torch.tensor(YW))).sum(0)/W[:, sl].sum(0))
    return sums/den, torch.cat(sb).numpy()
def stats(res):
    r = np.abs(res)/s; qn = np.quantile(r, [.5, .9, .95])
    return dict(mean=float(r.mean()), P50=float(qn[0]), P90=float(qn[1]), P95=float(qn[2]), max=float(r.max()),
                signed_mean=float((res/s).mean()))
bt = torch.from_numpy(b)
rho_gray = torch.from_numpy(np.repeat((rho_y/K)[:, None], 3, 1))

# ---------------- 复现核验：T01 A/B 终点 ----------------
M8 = build_M(8, 16, rho_gray)
ck = torch.load(T01/'fitting/B/checkpoint_2000.pt', map_location='cpu', weights_only=False)['params']
envB = ck['env'].double(); rho_ck = ck['rho'].double()
rho_diff = float((rho_ck-rho_gray).abs().max())
envA = torch.full((128, 3), c0, dtype=torch.float64)
dB = json.load(open(T01/'fitting/B/diagnostic.json')); dA = json.load(open(T01/'fitting/A/diagnostic.json'))
check = dict(rho_ckpt_vs_prior_max_abs=rho_diff)
for name, env, ref in (('A', envA, dA), ('B', envB, dB)):
    sbar_m = np.einsum('gjc,jc->g', M8.numpy(), env.numpy()*YW)    # Y(S)=Σ_j Σ_c Yc·M_gjc·env_jc（灰 rho 时 M 各通道相同）
    sbar_m = (M8[:, :, 0].numpy() @ (env.numpy()@YW))
    st = stats(sbar_m-b)
    E, sb2 = energy_and_sbar(8, 16, rho_gray, env)
    check[name] = dict(b_mean=st['mean'], b_mean_T01=ref['b_mean'], diff=abs(st['mean']-ref['b_mean']),
                       p90=st['P90'], p90_T01=ref['b_p90'], signed=st['signed_mean'], signed_T01=ref['b_signed_mean'],
                       E=E, E_T01=ref['energy'], sbar_chunk_vs_M=float(np.abs(sb2-sbar_m).max()))
log('复现核验', json.dumps(check, indent=1)); dump('p0_validation.json', check)
if not (rho_diff < 1e-10 and check['A']['diff'] < 1e-6 and check['B']['diff'] < 1e-6 and
        abs(check['B']['E']-dB['energy']) < 1e-6 and check['B']['sbar_chunk_vs_M'] < 1e-9):
    stop('VALIDATION_FAILED_IMPLEMENTATION_ERROR', dict(validation=check))

# ---------------- LP ----------------
def solve_lp(A, bvec, label):
    """min (1/G)Σ|A x - b|/s , x≥0；A: G×m。HiGHS 对偶单纯形。"""
    m = A.shape[1]; As = csr_matrix(A); Id = identity(G, format='csr')
    Aub = vstack([hstack([As, -Id]), hstack([-As, -Id])]).tocsr()
    bub = np.r_[bvec, -bvec]; c = np.r_[np.zeros(m), np.full(G, 1/(G*s))]
    t0 = time.time()
    r = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(0, None)]*(m+G), method='highs-ds')
    log(label, 'LP status', r.status, r.message, 'obj', r.fun, 'sec', round(time.time()-t0, 1))
    if r.status != 0: stop('LP_NOT_OPTIMAL_'+label, dict(message=r.message))
    x = r.x[:m]; res = A@x-bvec
    assert abs(float(np.abs(res).mean()/s)-r.fun) < 1e-9
    return x, res, r

def energy_gray(nth, nph, rho, ysol):
    return energy_and_sbar(nth, nph, rho, torch.from_numpy(np.repeat(ysol[:, None], 3, 1)))

results = dict(inputs_hashes=hashes, env=env_info, validation=check, K=K, s=s, E0=E0, G=G)

# (主) T01 模型：灰 rho，8x16，变量 y_j（灰环境）
x, res, r = solve_lp(M8[:, :, 0].numpy(), b, 'main')
E_main, sb_chk = energy_gray(8, 16, rho_gray, x)
assert np.abs(sb_chk-(M8[:, :, 0].numpy()@x)).max() < 1e-9
results['main'] = dict(stats(res), E=E_main, n_vars=128, lp_obj=float(r.fun), n_zero_texels=int((x <= 1e-12).sum()),
                       env_Y_max=float(x.max()), simplex_iter=int(r.nit))
np.savez_compressed(OUT/'main_solution.npz', y=x, residual=res)
F = results['main']
# 自检
selfcheck = dict(F_mean=F['mean'], T01_B_b_mean=dB['b_mean'], ok=bool(F['mean'] <= dB['b_mean']+1e-12))
results['selfcheck'] = selfcheck
log('自检', selfcheck)
if not selfcheck['ok']: stop('SELFCHECK_FAILED_IMPLEMENTATION_ERROR', results)

# (a) D1 色度 rho0；变量 L_jc（384）
wsum = W.sum(0).numpy()
chi = np.einsum('tg,tgc->gc', w_all[:, fi], TG)/np.maximum(np.einsum('tg,tg->g', w_all[:, fi], inten[:, fi]), 1e-12)[:, None]
den0 = np.einsum('tg,tg->g', w_all[:, fi], inten[:, fi]) <= 1e-12
chi[den0] = 1.
r0 = (rho_y/K)[:, None]*chi; mx = r0.max(1)
scale = 1/np.maximum(1., mx); rho_chroma = r0*scale[:, None]
n_scaled = int((mx > 1).sum())
Mc = build_M(8, 16, torch.from_numpy(rho_chroma))
Aa = (Mc.numpy()*YW[None, None, :]).reshape(G, 128*3)
xa, resa, ra = solve_lp(Aa, b, 'a_D1')
La = xa.reshape(128, 3)
Ea, _ = energy_and_sbar(8, 16, torch.from_numpy(rho_chroma), torch.from_numpy(La))
results['a_D1_chroma'] = dict(stats(resa), E=Ea, n_vars=384, scaled_fraction=n_scaled/G, n_scaled=n_scaled, n_den0=int(den0.sum()),
                              chi_mean=chi.mean(0).tolist(), n_zero_vars=int((xa <= 1e-12).sum()),
                              env_mean_rgb=La.mean(0).tolist(),
                              rho_Y_loss_max=float(np.abs((rho_chroma@YW)-(rho_y/K)).max()),
                              note='scaled_fraction>0.05 将触发 D1 的 RHO_INIT_CONFLICT')
log('(a)', results['a_D1_chroma'])

# (b) 16x32，灰 rho
M16 = build_M(16, 32, rho_gray)
xb, resb, rb = solve_lp(M16[:, :, 0].numpy(), b, 'b_16x32')
Eb, _ = energy_gray(16, 32, rho_gray, xb)
results['b_16x32'] = dict(stats(resb), E=Eb, n_vars=512, n_zero_texels=int((xb <= 1e-12).sum()),
                          note='与8x16求积不嵌套，仅作分辨率瓶颈检验')
log('(b)', results['b_16x32'])

# (c) 尾部高斯
rA = np.load(T01/'fitting/A/reconstruction.npz')['b_residual']; assert len(rA) == G
ra_abs = np.abs(rA); thr = np.quantile(ra_abs, .9); tail = ra_abs >= thr; rest = ~tail
wt = w_all[:, fi]; nz = nt[:, fi, 2]
eff = wt.sum(0)**2/np.maximum((wt**2).sum(0), 1e-30)
mz = (wt*nz).sum(0)/wt.sum(0); maz = (wt*np.abs(nz)).sum(0)/wt.sum(0)
qs = lambda a: dict(zip(['P5', 'P25', 'P50', 'P75', 'P95'], np.quantile(a, [.05, .25, .5, .75, .95]).tolist()))
results['c_tail'] = dict(
    threshold_abs_residual=float(thr), n_tail=int(tail.sum()),
    weight_sum_tail=float(wt[:, tail].sum()), weight_share_tail=float(wt[:, tail].sum()/wt.sum()),
    weight_sum_per_gauss_tail_mean=float(wt[:, tail].sum(0).mean()), weight_sum_per_gauss_rest_mean=float(wt[:, rest].sum(0).mean()),
    eff_frames_tail=dict(mean=float(eff[tail].mean()), median=float(np.median(eff[tail]))),
    eff_frames_rest=dict(mean=float(eff[rest].mean()), median=float(np.median(eff[rest]))),
    wmean_nz_tail=dict(mean=float(mz[tail].mean()), **qs(mz[tail])), wmean_nz_rest=dict(mean=float(mz[rest].mean()), **qs(mz[rest])),
    wmean_abs_nz_tail=dict(mean=float(maz[tail].mean()), **qs(maz[tail])), wmean_abs_nz_rest=dict(mean=float(maz[rest].mean()), **qs(maz[rest])),
    frac_abs_nz_gt_0p5_tail=float((maz[tail] > .5).mean()), frac_abs_nz_gt_0p5_rest=float((maz[rest] > .5).mean()),
    LP_main_residual_share_of_tail=float(np.abs(res)[tail].sum()/np.abs(res).sum()),
    A_residual_share_of_tail=float(ra_abs[tail].sum()/ra_abs.sum()),
    rhoY_tail_mean=float(rho_y[tail].mean()), rhoY_rest_mean=float(rho_y[rest].mean()),
    b_over_s_tail_mean=float((b[tail]/s).mean()), b_over_s_rest_mean=float((b[rest]/s).mean()))
log('(c)', json.dumps(results['c_tail'], indent=1))

# 情形 R/U（机械）
caseR = bool(F['mean'] <= .10 and F['P90'] <= .25)
results['case'] = dict(rule='F_mean<=0.10 且 F_P90<=0.25', F_mean=F['mean'], F_P90=F['P90'], case='R' if caseR else 'U')
results['status'] = 'COMPLETE'
log('情形', results['case'])
dump('p0_result.json', results)
