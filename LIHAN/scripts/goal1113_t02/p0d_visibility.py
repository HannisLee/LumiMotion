"""T02 前置诊断 P0(d)：用冻结 Stage1 几何追踪的环境光可见性 V 计算 b 下限（无训练、无GT、无渲染）。
依据 LIHAN/DOC/规划/1114-04-T02候选方案草案.md 第3节与D3②。必须经 bootstrap（载入仓库CUDA光栅器/追踪器）在仓库根运行：
  CUDA_VISIBLE_DEVICES=<gpu> python LIHAN/scripts/bootstrap_1011.py <仓库根> LIHAN.scripts.goal1113_t02.p0d_visibility <输出目录>
阶段1：逐帧建BVH并对 body_fit 的每个 texel 中心方向追踪单条射线（V不求梯度，逐帧落盘，可断点续跑）。
阶段2：以 M^V 求解与 P0 主问题同形的 LP（灰ρ 128变量；色度ρ0 384变量），V=1 时必须逐元素复现 P0。
不读 lights.json/GT/评测目录；不改 T01/P0 任何产物。
"""
import sys, json, math, hashlib, platform, time, os
from pathlib import Path
ROOT = Path.cwd().resolve()
assert (ROOT/'scene').is_dir() and (ROOT/'LIHAN/scripts/goal1113').is_dir()
sys.path.insert(0, str(ROOT))
import numpy as np, torch, scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, identity
from scipy.stats import spearmanr
from LIHAN.scripts.goal1113 import physics as P, inputs as I

T01 = ROOT/'output/1113-01-LH-static-ELDL-V3-T01-BguidedEnvMap'
P0 = ROOT/'output/1114-01-LH-static-ELDL-V3-T02-P0capacity'
YW = np.array(P.Y)
OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
(OUT/'partial').mkdir(exist_ok=True)
LOG = open(OUT/'p0d.log', 'a')
GEOMETRY_SHADOW_BIAS = 0.1          # 沿用 arguments/stage2.py 默认值，不调参
ITER = 35000
TRACE_BUDGET_S = 7200.0             # 约2 GPU小时上限，超过则按结构冲突停止
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
    obj['status'] = status; dump('p0d_result.json', obj); log('STOP', status); sys.exit(2)

# ---------------- 输入与哈希（仅T01白名单） ----------------
OBS = ROOT/I.OBS
if sha(OBS) != I.EXPECTED[I.OBS]: stop('OBS_HASH_MISMATCH', {})
assert not (ROOT/I.RUNTIME/'rgba').exists()
wl = [OBS, ROOT/I.RUNTIME/'rgb_only_manifest.json', ROOT/I.RUNTIME/'transforms_all.json',
      ROOT/I.MODEL/f'point_cloud/iteration_{ITER}/point_cloud.ply', ROOT/I.MODEL/f'deform/iteration_{ITER}/deform.pth',
      T01/'inputs/prior.npz', T01/'inputs/config.json', T01/'fitting/A/reconstruction.npz',
      T01/'fitting/A/diagnostic.json', T01/'fitting/B/diagnostic.json', T01/'fitting/B/checkpoint_2000.pt',
      P0/'p0_result.json', ROOT/'LIHAN/scripts/goal1113/physics.py', ROOT/'LIHAN/scripts/goal1113/inputs.py',
      ROOT/'LIHAN/scripts/goal1113_t02/p0_capacity.py', ROOT/'LIHAN/scripts/goal1113_t02/p0d_visibility.py',
      ROOT/'LIHAN/DOC/规划/1114-04-T02候选方案草案.md']
hashes = {str(p.relative_to(ROOT)): sha(p) for p in wl}
for p in (I.RUNTIME/'rgb_only_manifest.json', I.RUNTIME/'transforms_all.json'):
    assert hashes[str(p)] == I.EXPECTED[p], p
env_info = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                highs_via='scipy.optimize.linprog(method=highs-ds)', cuda_visible=str(os.environ.get('CUDA_VISIBLE_DEVICES')))
log('=== 启动', time.strftime('%F %T'), env_info); log('哈希', json.dumps(hashes, indent=1))

KEYS = ('frames','gaussian_ids','xyz','frame_xyz','normals','intensity','weights','rgb')
with np.load(OBS, allow_pickle=False) as z: d = {k: z[k] for k in KEYS}
n_all = d['normals'].astype(np.float64); w_all = d['weights'].astype(np.float64)
rgb = d['rgb'].astype(np.float64); y = d['intensity'].astype(np.float64)
assert np.array_equal(d['frames'], np.arange(1, 151)) and np.isfinite(w_all).all() and (w_all >= 0).all() and w_all.sum() > 0
valid = w_all > 0
rgb = np.where(valid[..., None], rgb, 0.)
lin = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
inten = np.where(valid, y, 0.)
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
if len(fi) != 2683: stop('SUPPORT_COUNT_MISMATCH', {})
G = len(fi)
prior = np.load(T01/'inputs/prior.npz', allow_pickle=False)
cfg = json.load(open(T01/'inputs/config.json'))
K, s, c0, E0 = cfg['K'], cfg['scale'], cfg['c0'], cfg['e0']
rho_y = prior['rho_y'][fi]; b = prior['b'][fi]; ldir = prior['direction']
gid_fit = d['gaussian_ids'][fi].astype(np.int64)
frame_xyz = d['frame_xyz'].astype(np.float32)          # (150, Nobs, 3)

# ---------------- 8x16 方向网格（与 physics.brdf_basis 同一 texel 中心） ----------------
def grid(nth, nph):
    i = torch.arange(nth, dtype=torch.float64); j = torch.arange(nph, dtype=torch.float64)
    th = (i+.5)*math.pi/nth; ph = (j+.5)*2*math.pi/nph
    t, p = torch.meshgrid(th, ph, indexing='ij')
    om = torch.stack((t.sin()*p.cos(), t.sin()*p.sin(), t.cos()), -1).reshape(-1, 3)
    so = ((2*math.pi/nph)*(torch.cos(i*math.pi/nth)-torch.cos((i+1)*math.pi/nth)))[:, None].expand(nth, nph).reshape(-1)
    return om, so
OM, SO = grid(8, 16)

# ======================= 阶段1：V 追踪 =======================
VFILE = OUT/'V_fit.npz'
trace_report = {}
if not VFILE.exists():
    from LIHAN.scripts.static_rgb_light_extract import cameras, load
    from scene.deform_model import DeformModel
    from utils.general_utils import build_rotation
    cams = cameras(ROOT/I.RUNTIME, 'all')
    assert [int(c.image_name_train_light.split('_')[-1]) for c in cams] == list(range(1, 151))
    g = load(ROOT/I.MODEL, ITER)
    deformation = DeformModel(deform_type='mlp', is_blender=True, hyper_dim=1, pred_color=True)
    assert deformation.load_weights(str(ROOT/I.MODEL), ITER), '缺少冻结形变权重'
    deformation.deform.eval()
    log('Stage1 高斯数', g.get_xyz.shape[0], 'alpha_min', g.alpha_min, 'tmin', g.gaussian_tracer.transmittance_min)
    dirs = OM.float().cuda()                                         # (128,3)
    gid_t = torch.from_numpy(gid_fit).cuda()
    opacity_all = g.get_opacity.detach()

    def frame_state(t):
        c = cams[t]; c.load2device()
        with torch.no_grad():
            dd = deformation.step(g.get_xyz, c.fid.unsqueeze(0).expand(len(g.get_xyz), -1), iteration=ITER,
                                  feature=g.get_binary_feature(), camera_center=c.camera_center)
            xyz_t = g.get_xyz+dd['d_xyz']; sc = g.get_scaling+dd['d_scaling']; rot = g.get_rotation_bias(dd['d_rotation'])
        return dd, xyz_t, sc, rot

    @torch.no_grad()
    def trace_points(xyz_t, sc, rot, opacity, points, dirs_, chunk):
        """points (P,3)、dirs_ (B,3) -> 透射率 (P,B)；OOM 时调用方缩小 chunk 重试（射线数不变）。"""
        out = []
        for a in range(0, len(points), chunk):
            pt = points[a:a+chunk]
            rays_d = dirs_[None].expand(len(pt), -1, -1).contiguous()
            pk = g.trace((pt[:, None]+rays_d*GEOMETRY_SHADOW_BIAS).contiguous(), rays_d,
                         features=torch.zeros(len(g.get_xyz), 4, device='cuda'), xyz=xyz_t, scales=sc, rotation=rot,
                         opacity=opacity, use_zeros=True)
            v = (1-pk['alpha']).clamp(0, 1)
            if not torch.isfinite(v).all(): raise ValueError('透射率有非有限值')
            out.append(v.cpu())
        return torch.cat(out)

    # ---- 先决检查：frame_xyz 与追踪侧形变一致 ----
    dd, xyz_t, sc, rot = frame_state(0)
    err0 = float((xyz_t[gid_t].cpu()-torch.from_numpy(frame_xyz[0][fi])).abs().max())
    log('frame0 形变 xyz 与 observations.frame_xyz 最大差', err0)
    if err0 > 1e-4: stop('FRAME_XYZ_MISMATCH', dict(max_abs_err_frame0=err0))
    trace_report['frame_xyz_err_frame0'] = err0

    # ---- 小规模正确性检查：2 帧各 2048 条射线，对比独立朴素实现 ----
    def naive_vis(xyz_t, sc, rot, opacity, ro, rd):
        """朴素 float64 实现：逐高斯 ray-surfel 相交，alpha=min(.99,o*exp(-.5|p|^2))，alpha<alpha_min 跳过，
        T=Π(1-alpha)，T<transmittance_min 视为被完全遮挡(trace 在此情形返回 alpha=1 即 V=0)。仅 d>0（射线前向）。"""
        R = build_rotation(rot).double(); e = 1e-7; scd = sc.double()
        sinv = 1.0/(scd+e*(scd == 0).double())
        nrm = R[:, :, 2]; ru = R[:, :, 0]*sinv[:, 0:1]; rv = R[:, :, 1]*sinv[:, 1:2]
        mu = xyz_t.double(); op = opacity.double()[:, 0]
        res = torch.empty(len(ro), dtype=torch.float64)
        for i in range(len(ro)):
            o_g = ((ro[i].double()[None]-mu)*nrm).sum(-1); d_g = (nrm*rd[i].double()[None]).sum(-1)
            dd_ = -o_g*d_g/(d_g*d_g).clamp_min(1e-6)
            pos = ro[i].double()[None]+dd_[:, None]*rd[i].double()[None]-mu
            pg = torch.stack(((ru*pos).sum(-1), (rv*pos).sum(-1)), -1)
            al = (op*torch.exp(-.5*(pg*pg).sum(-1))).clamp_max(.99)
            keep = (al >= g.alpha_min) & (dd_ > 1.2e-7)
            T = torch.prod(1-al[keep])
            res[i] = 0. if float(T) < g.gaussian_tracer.transmittance_min else float(T)
        return res.cpu()
    rng = np.random.default_rng(20261114)
    naive = dict(frames=[], n_rays=0, abs_diff=[], mism=[])
    allgood = True
    for t in (0, 74):
        dd, xyz_t, sc, rot = frame_state(t)
        g.build_bvh(dd['d_rotation'], dd['d_xyz'], dd['d_scaling'])
        pts = torch.from_numpy(frame_xyz[t][fi]).cuda()
        Vt = trace_points(xyz_t, sc, rot, opacity_all, pts, dirs, 512)
        sel_g = rng.integers(0, G, 2048); sel_j = rng.integers(0, 128, 2048)
        ro = pts[torch.from_numpy(sel_g).cuda()]+dirs[torch.from_numpy(sel_j).cuda()]*GEOMETRY_SHADOW_BIAS
        rd = dirs[torch.from_numpy(sel_j).cuda()]
        vn = naive_vis(xyz_t, sc, rot, opacity_all, ro, rd)
        vt = Vt[torch.from_numpy(sel_g), torch.from_numpy(sel_j)].double()
        diff = (vn-vt).abs()
        naive['frames'].append(t); naive['n_rays'] += 2048
        naive['abs_diff'].append(dict(frame=t, max=float(diff.max()), frac_le_1e3=float((diff <= 1e-3).float().mean()),
                                      mean=float(diff.mean()), n_naive_lt1=int((vn < 1).sum()), n_trace_lt1=int((vt < 1).sum())))
        # 全零 opacity 单位透射率（trace 对已知简单场景）
        Vz = trace_points(xyz_t, sc, rot, torch.zeros_like(opacity_all), pts[:256], dirs, 256)
        zerr = float((Vz-1).abs().max()); naive['abs_diff'][-1]['zero_opacity_max_err'] = zerr
        log('朴素实现对比', naive['abs_diff'][-1])
        if zerr > 1e-7: allgood = False
        if naive['abs_diff'][-1]['frac_le_1e3'] < .99: allgood = False
    naive['pass'] = allgood
    trace_report['naive_check'] = naive
    dump('trace_correctness.json', trace_report)
    if not allgood: stop('TRACE_CORRECTNESS_CHECK_FAILED', dict(trace_report=trace_report))

    # ---- 全量追踪（逐帧落盘，断点续跑） ----
    tfile = OUT/'partial/trace_time.json'
    spent = json.load(open(tfile))['seconds'] if tfile.exists() else 0.
    chunk = 1024; frame_err_max = err0
    for t in range(150):
        pf = OUT/f'partial/V_t{t:03d}.npy'
        if pf.exists(): continue
        t0 = time.time()
        while True:
            try:
                dd, xyz_t, sc, rot = frame_state(t)
                e = float((xyz_t[gid_t].cpu()-torch.from_numpy(frame_xyz[t][fi])).abs().max()); frame_err_max = max(frame_err_max, e)
                if e > 1e-4: stop('FRAME_XYZ_MISMATCH', dict(frame=t, max_abs_err=e))
                g.build_bvh(dd['d_rotation'], dd['d_xyz'], dd['d_scaling'])
                pts = torch.from_numpy(frame_xyz[t][fi]).cuda()
                Vt = trace_points(xyz_t, sc, rot, opacity_all, pts, dirs, chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); chunk //= 2; log('OOM，chunk 缩小为', chunk)
                if chunk < 16: stop('OOM_CHUNK_TOO_SMALL', dict(frame=t))
        np.save(pf, Vt.numpy().astype(np.float32))
        spent += time.time()-t0; json.dump(dict(seconds=spent), open(tfile, 'w'))
        if t % 10 == 0 or t == 149: log(f'trace frame {t+1}/150 chunk={chunk} 本帧 {time.time()-t0:.1f}s 累计 {spent:.0f}s')
        if spent > TRACE_BUDGET_S:
            stop('TRACE_BUDGET_EXCEEDED_STRUCTURE_CONFLICT', dict(frames_done=t+1, seconds=spent))
    V = np.stack([np.load(OUT/f'partial/V_t{t:03d}.npy') for t in range(150)])
    assert V.shape == (150, G, 128)
    trace_report.update(trace_seconds=spent, frame_xyz_err_max_all=frame_err_max)
    np.savez_compressed(VFILE, V=V, frame_ids=np.arange(1, 151), body_fit=fi, texel_dirs=OM.numpy())
    dump('trace_report.json', trace_report)
    del g, deformation
    torch.cuda.empty_cache()
else:
    log('V_fit.npz 已存在，跳过追踪')
    if (OUT/'trace_report.json').exists(): trace_report = json.load(open(OUT/'trace_report.json'))

# ======================= 阶段2：M^V 与 LP =======================
vhash = sha(VFILE); log('V_fit.npz SHA256', vhash)
with np.load(VFILE) as z:
    Vn = z['V']; assert np.array_equal(z['body_fit'], fi)
finite = bool(np.isfinite(Vn).all()); inrange = bool((Vn >= 0).all() and (Vn <= 1).all())
log('V 有限', finite, '范围[0,1]', inrange, 'shape', Vn.shape)
if not (finite and inrange): stop('V_INVALID', dict(finite=finite, inrange=inrange, V_sha256=vhash))
Vt_ = torch.from_numpy(Vn)                                              # float32，(150,G,128)

# 相机/视线/权重（与P0同）
from LIHAN.scripts.static_rgb_light_extract import cameras
cams = cameras(ROOT/I.RUNTIME, 'all')
centers = np.stack([c.camera_center.detach().cpu().numpy() for c in cams]).astype(np.float64)
view = centers[:, None, :]-d['frame_xyz'].astype(np.float64); ln = np.linalg.norm(view, axis=-1, keepdims=True); view /= ln
nt = n_all if n_all.ndim == 3 else np.broadcast_to(n_all[None], (150, *n_all.shape))
N = torch.from_numpy(np.ascontiguousarray(nt[:, fi])); VW = torch.from_numpy(np.ascontiguousarray(view[:, fi]))
W = torch.from_numpy(np.ascontiguousarray(w_all[:, fi])); TG = lin[:, fi]
Q = W/W.sum(0, keepdim=True)
nl_dir = torch.einsum('tgc,tc->tg', N, torch.from_numpy(ldir)).clamp_min(0)

def basis(n, v, nth=8, nph=16):
    om, so = grid(nth, nph); EPS = P.EPS; aa = P.ALPHA**2
    nv = (n*v).sum(-1).clamp(0, 1); nl = torch.einsum('...c,jc->...j', n, om).clamp(0, 1)
    h0 = v[..., None, :]+om; hn = torch.linalg.vector_norm(h0, dim=-1, keepdim=True); h = h0/hn.clamp_min(EPS)
    nh = (n[..., None, :]*h).sum(-1).clamp(0, 1); vh = (v[..., None, :]*h).sum(-1).clamp(0, 1)
    fr = .04+.96*(1-vh).pow(5); ggx = aa/(math.pi*((nh.square()*(aa-1)+1).square()))
    g1 = lambda c: 2*c/(c+torch.sqrt(aa+(1-aa)*c.square())).clamp_min(EPS)
    spec = ggx*g1(nv)[..., None]*g1(nl)*fr/(4*nv[..., None].clamp_min(EPS)*nl.clamp_min(EPS))
    mask = (nv[..., None] > 1e-6) & (nl > 0) & (hn[..., 0] > EPS); f = nl*so
    return torch.where(mask, (1-fr)*f/math.pi, 0.), torch.where(mask, spec*f, 0.)
CH = 32
def build_M(rho, use_v, phys=False):
    """M[g,j,c]=Σ_t q_tg V_tgj (rho_gc dif_tgj+spec_tgj)。use_v=False 即 V=1（P0 的 M）。phys=True 用 physics.brdf_basis。"""
    M = torch.zeros(G, 128, 3, dtype=torch.float64)
    for a in range(0, G, CH):
        sl = slice(a, min(a+CH, G))
        df, sp = P.brdf_basis(N[:, sl], VW[:, sl]) if phys else basis(N[:, sl], VW[:, sl])
        q = Q[:, sl]
        if use_v: q = q[..., None]*Vt_[:, sl].double()
        else: q = q[..., None].expand(-1, -1, 128)
        qd = (q*df).sum(0); qs = (q*sp).sum(0)
        M[sl] = qd[..., None]*rho[sl][:, None, :]+qs[..., None]
    return M
def energy(rho, L, use_v):
    sums = 0.; den = 0.
    for a in range(0, G, CH):
        sl = slice(a, min(a+CH, G)); df, sp = basis(N[:, sl], VW[:, sl])
        if use_v: vv = Vt_[:, sl].double(); df = df*vv; sp = sp*vv
        S = torch.einsum('tgj,jc->tgc', df, L)*rho[sl][None]+torch.einsum('tgj,jc->tgc', sp, L)
        D = K*rho[sl][None]*nl_dir[:, sl][..., None]; w = W[:, sl, None]
        sums += float((w*S).sum()); den += float((w*(D+S)).sum())
    return sums/den
def stats(res):
    r = np.abs(res)/s; qn = np.quantile(r, [.5, .9, .95])
    return dict(mean=float(r.mean()), P50=float(qn[0]), P90=float(qn[1]), P95=float(qn[2]), max=float(r.max()),
                signed_mean=float((res/s).mean()))
def solve_lp(A, bvec, label):
    m = A.shape[1]; As = csr_matrix(A); Id = identity(G, format='csr')
    Aub = vstack([hstack([As, -Id]), hstack([-As, -Id])]).tocsr()
    bub = np.r_[bvec, -bvec]; c = np.r_[np.zeros(m), np.full(G, 1/(G*s))]
    t0 = time.time()
    r = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(0, None)]*(m+G), method='highs-ds')
    log(label, 'LP status', r.status, r.message, 'obj', r.fun, 'sec', round(time.time()-t0, 1))
    if r.status != 0: stop('LP_NOT_OPTIMAL_'+label, dict(message=r.message, V_sha256=vhash))
    x = r.x[:m]; res = A@x-bvec
    assert abs(float(np.abs(res).mean()/s)-r.fun) < 1e-9
    return x, res, r

rho_gray = torch.from_numpy(np.repeat((rho_y/K)[:, None], 3, 1))
p0 = json.load(open(P0/'p0_result.json'))
results = dict(inputs_hashes=hashes, V_sha256=vhash, env=env_info, K=K, s=s, E0=E0, G=G, trace=trace_report,
               P0_main=p0['main'], P0_a=p0['a_D1_chroma'])

# ---- V=1 复现核验 ----
M1 = build_M(rho_gray, False)                        # 本脚本基，V=1
M1p = build_M(rho_gray, False, phys=True)            # physics.brdf_basis
basis_err = float((M1-M1p).abs().max())
# 与 P0 的 M8 同一构造：用 V=全1 张量走 use_v 路径也应一致
Vt_save = Vt_; Vt_ = torch.ones_like(Vt_save)
M1v = build_M(rho_gray, True); one_err = float((M1v-M1).abs().max()); Vt_ = Vt_save
x1, res1, r1 = solve_lp(M1[:, :, 0].numpy(), b, 'V1_main')
st1 = stats(res1)
repro = dict(M_basis_vs_physics_max=basis_err, M_V1path_vs_M_max=one_err, F_V1=st1['mean'], F_P0=p0['main']['mean'],
             diff_mean=abs(st1['mean']-p0['main']['mean']), diff_P90=abs(st1['P90']-p0['main']['P90']),
             diff_P50=abs(st1['P50']-p0['main']['P50']), diff_signed=abs(st1['signed_mean']-p0['main']['signed_mean']))
# T01 B 终点环境复现
ck = torch.load(T01/'fitting/B/checkpoint_2000.pt', map_location='cpu', weights_only=False)['params']
envB = ck['env'].double(); dB = json.load(open(T01/'fitting/B/diagnostic.json'))
stB = stats(M1[:, :, 0].numpy() @ (envB.numpy()@YW)-b)
repro['B_env_b_mean'] = stB['mean']; repro['B_env_b_mean_T01'] = dB['b_mean']; repro['B_diff'] = abs(stB['mean']-dB['b_mean'])
log('V=1 复现核验', json.dumps(repro, indent=1)); results['V1_reproduction'] = repro
if not (basis_err < 1e-12 and one_err < 1e-12 and repro['diff_mean'] < 1e-9 and repro['diff_P90'] < 1e-9 and repro['B_diff'] < 1e-6):
    stop('V1_REPRODUCTION_FAILED_IMPLEMENTATION_ERROR', results)

# ---- 主：灰 ρ，M^V，128 变量 ----
MV = build_M(rho_gray, True)
x, res, r = solve_lp(MV[:, :, 0].numpy(), b, 'V_main')
E_V = energy(rho_gray, torch.from_numpy(np.repeat(x[:, None], 3, 1)), True)
results['F_V_main'] = dict(stats(res), E=E_V, n_vars=128, n_zero_texels=int((x <= 1e-12).sum()), env_Y_max=float(x.max()),
                           lp_obj=float(r.fun), simplex_iter=int(r.nit))
np.savez_compressed(OUT/'main_solution_V.npz', y=x, residual=res)
# 参考：T01 B 终点环境在 V 下的 b 残差（仅信息）
results['T01_Benv_under_V'] = stats(MV[:, :, 0].numpy() @ (envB.numpy()@YW)-b)

# ---- (a) 色度 ρ0 变体，384 变量 ----
wsum_y = np.einsum('tg,tg->g', w_all[:, fi], inten[:, fi])
chi = np.einsum('tg,tgc->gc', w_all[:, fi], TG)/np.maximum(wsum_y, 1e-12)[:, None]
chi[wsum_y <= 1e-12] = 1.
r0 = (rho_y/K)[:, None]*chi; mx = r0.max(1); rho_chroma = r0/np.maximum(1., mx)[:, None]
Mc = build_M(torch.from_numpy(rho_chroma), True)
Aa = (Mc.numpy()*YW[None, None, :]).reshape(G, 384)
xa, resa, ra = solve_lp(Aa, b, 'V_a_chroma')
Ea = energy(torch.from_numpy(rho_chroma), torch.from_numpy(xa.reshape(128, 3)), True)
results['F_V_a_chroma'] = dict(stats(resa), E=Ea, n_vars=384, n_scaled=int((mx > 1).sum()))

# ---- V 统计与诊断 ----
Vf = Vn.astype(np.float64)
vq = np.quantile(Vf.reshape(-1)[::7], [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]).tolist()
Vg = Vf.mean((0, 2))                                               # 逐高斯 简单均值（时间×texel）
Qn = Q.numpy().astype(np.float64)
Vt_w = np.einsum('tg,tgj->gj', Qn, Vf)                              # q 加权时间平均 (G,128)
Vbar_plain = Vt_w@(SO.numpy()/SO.sum().item())                      # 立体角加权，无余弦
cosw = np.zeros(G); cosden = np.zeros(G)
for a in range(0, G, CH):
    sl = slice(a, min(a+CH, G))
    nl = torch.einsum('tgc,jc->tgj', N[:, sl], OM).clamp_min(0).numpy()*SO.numpy()[None, None]
    cosw[sl] = np.einsum('tg,tgj,tgj->g', Qn[:, sl], nl, Vf[:, sl]); cosden[sl] = np.einsum('tg,tgj->g', Qn[:, sl], nl)
Vbar_cos = cosw/np.maximum(cosden, 1e-30)
n_allzero = int((Vf.max((0, 2)) == 0).sum())
results['V_stats'] = dict(mean=float(Vf.mean()), quantiles_P0_1_5_10_25_50_75_90_95_99_100=vq,
                          sampled_every7th_note='分位数基于每7个元素抽样，均值为全量',
                          frac_exact0=float((Vf == 0).mean()), frac_exact1=float((Vf == 1).mean()),
                          frac_lt0p5=float((Vf < .5).mean()), n_gauss_all_zero_V=n_allzero,
                          per_gauss_mean_quantiles=dict(zip(['P5', 'P25', 'P50', 'P75', 'P95'], np.quantile(Vg, [.05, .25, .5, .75, .95]).tolist())),
                          q_weighted_cos_Vbar=dict(mean=float(Vbar_cos.mean()), **dict(zip(['P5', 'P50', 'P95'], np.quantile(Vbar_cos, [.05, .5, .95]).tolist()))),
                          finite=finite, in_range=inrange)
bs = b/rho_y
sp = {k: dict(zip(['rho', 'p'], map(float, spearmanr(bs, v)))) for k, v in
      (('cos_weighted_primary', Vbar_cos), ('solid_angle_weighted', Vbar_plain), ('plain_mean_t_j', Vg))}
rA = np.load(T01/'fitting/A/reconstruction.npz')['b_residual']; ra_abs = np.abs(rA)
tail = ra_abs >= np.quantile(ra_abs, .9); rest = ~tail
results['spearman_b_over_rhoY_vs_Vbar'] = sp
results['tail_visibility'] = dict(n_tail=int(tail.sum()), Vbar_cos_tail=float(Vbar_cos[tail].mean()), Vbar_cos_rest=float(Vbar_cos[rest].mean()),
                                  Vbar_plain_tail=float(Vbar_plain[tail].mean()), Vbar_plain_rest=float(Vbar_plain[rest].mean()),
                                  Vbar_cos_tail_median=float(np.median(Vbar_cos[tail])), Vbar_cos_rest_median=float(np.median(Vbar_cos[rest])))

# ---- 机械判定 ----
F0 = p0['main']; FV = results['F_V_main']
red = (F0['mean']-FV['mean'])/F0['mean']; red90 = (F0['P90']-FV['P90'])/F0['P90']
if FV['mean'] <= .10 and FV['P90'] <= .25: case = 'V-R'
elif red >= .05: case = 'V-P'
else: case = 'V-U'
results['comparison'] = dict(F_P0_mean=F0['mean'], F_P0_P90=F0['P90'], F_V_mean=FV['mean'], F_V_P90=FV['P90'],
                             rel_reduction_mean=float(red), rel_reduction_P90=float(red90),
                             rule='V-R: mean<=0.10且P90<=0.25; V-P: 非V-R且mean相对降幅>=5%; V-U: 降幅<5%(含变差); 5%仅为报告分档')
results['case'] = case
results['status'] = 'COMPLETE'
log('情形', case, '相对降幅 mean', red, 'P90', red90)
dump('p0d_result.json', results)
log('=== 完成', time.strftime('%F %T'))
