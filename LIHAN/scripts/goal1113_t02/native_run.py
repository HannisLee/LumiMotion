"""T02-native：沿用 T01 A/B/C 与冻结代码，仅四处差异（见 DOC/修改/1114-09）。
1) 环境图 16(纬)×32(经)=512 texel；2) 删除 L_static（b 不进任何 loss，静态L1仅无梯度记录）；
3) 删除防护判定（占比/b残差仅记录，不触发停止）；4) 其余步数/学习率/α0/K/种子/R_rev 与 T01 相同。
必须在仓库根目录用 python -m LIHAN.scripts.goal1113_t02.native_run 运行。"""
import argparse, json, math, os, shlex, socket, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from LIHAN.scripts.goal1113 import fit as F, inputs as I, physics as P, optimizer as O

H, W = 16, 32
NT = H * W


def grid(device, dtype=torch.float64):
    i = torch.arange(H, device=device, dtype=dtype); j = torch.arange(W, device=device, dtype=dtype)
    theta = (i + .5) * math.pi / H; phi = (j + .5) * 2 * math.pi / W
    t, p = torch.meshgrid(theta, phi, indexing='ij')
    omega = torch.stack((t.sin()*p.cos(), t.sin()*p.sin(), t.cos()), -1).reshape(-1, 3)
    solid = ((2*math.pi/W)*(torch.cos(i*math.pi/H)-torch.cos((i+1)*math.pi/H)))[:, None].expand(H, W).reshape(-1)
    return omega, solid


def objective_backward(data, params, K, scale, sign, chunk=128, backward=True):
    """仅 RGB + 0.1*R_rev 有梯度；静态L1在 no_grad 下记录，不进入 total。"""
    n, v, target, w, b = (data[k] for k in ('n','v','target','w','b'))
    T, G = w.shape; total_w = w.sum()
    if total_w <= 0 or (w.sum(0) <= 0).any(): raise ValueError('零权重')
    totals = [0., 0.]
    for s in range(0, G, chunk):
        sl = slice(s, min(s+chunk, G))
        l = P.direction(params['phi'], params['a'])
        basis = (data['basis'][0][:,sl], data['basis'][1][:,sl]) if 'basis' in data else None
        direct, static = P.components(n[:,sl], v[:,sl], params['rho'][sl], params['env'], l, K, basis)
        ww = w[:,sl]
        rgb = (ww[...,None]*(direct+static-target[:,sl]).square()).sum()/(3*scale**2*total_w)
        if not torch.isfinite(rgb): raise FloatingPointError('非有限损失')
        with torch.no_grad():
            sbar = (ww*(static @ static.new_tensor(P.Y))).sum(0)/ww.sum(0)
            totals[1] += float((sbar-b[sl]).abs().sum()/(G*scale))
        if backward and rgb.requires_grad: rgb.backward()
        totals[0] += rgb.item()
    rev = P.reverse_penalty(params['phi'], sign)
    if backward and rev.requires_grad: (.1*rev).backward()
    return dict(rgb=totals[0], static=totals[1], reverse=rev.item(), total=totals[0]+.1*rev.item())


def setup(device):
    """复制 fit.setup；仅环境图尺寸改为 512，其余逻辑不变。"""
    from LIHAN.scripts.static_rgb_light_extract import cameras
    hashes = I.whitelist(); src = I.load_observations(); cams = cameras(I.RUNTIME, 'all')
    if [int(c.image_name_train_light.split('_')[-1]) for c in cams] != list(range(1,151)): raise ValueError('相机帧号不符')
    centers = np.stack([c.camera_center.detach().cpu().numpy() for c in cams]).astype(np.float64)
    view = centers[:,None,:]-src['raw']['frame_xyz']; length = np.linalg.norm(view,axis=-1,keepdims=True)
    if not np.isfinite(length).all() or (length<=1e-8).any(): raise ValueError('视线退化')
    view /= length; src['view'] = view
    fi = src['support']['body_fit']; w = src['w'][:,fi]
    if (w.sum(0)<=0).any(): raise ValueError('fit 高斯无有效观测')
    dev = torch.device(device)
    t = lambda x: torch.as_tensor(x,dtype=torch.float64,device=dev)
    data = dict(n=t(src['n'][:,fi]),v=t(view[:,fi]),target=t(src['target'][:,fi]),w=t(w),b=t(src['b'][fi]))
    rho_y = src['rho_y'][fi]; K = max(1., float(rho_y.max()))
    l = src['direction']; phi = np.unwrap(np.arctan2(l[:,1],l[:,0])); span = phi[-1]-phi[0]
    if abs(span) < math.pi: raise ValueError('初始绕行不足π')
    sign = float(np.sign(span))
    z = (np.arcsin(np.clip(l[:,2],-1,1))-math.pi/4)/(math.pi/36)
    if (abs(z)>=1).any(): raise ValueError('初始仰角不在可表示范围')
    params = dict(rho=t(np.repeat((rho_y/K)[:,None],3,axis=1)).clone(),env=t(np.ones((NT,3))).clone(),
                  phi=t(phi).clone(),a=t(np.arctanh(z)).clone())
    with torch.no_grad():
        initial = P.direction(params['phi'],params['a'])
        dynamic = K*params['rho'][None]*torch.einsum('tgc,tc->tg',data['n'],initial).clamp_min(0)[...,None]
        expected = t(rho_y)[None,:,None]*torch.einsum('tgc,tc->tg',data['n'],t(l)).clamp_min(0)[...,None]
        diff = float((dynamic-expected).abs().max())
        if diff > 1e-10: raise ValueError('E4 动态代数不符')
        bases = [P.brdf_basis(data['n'][:,j:j+128],data['v'][:,j:j+128]) for j in range(0,len(fi),128)]
        data['basis'] = (torch.cat([x[0] for x in bases],1),torch.cat([x[1] for x in bases],1))
        env = t(np.ones((NT,3))); vals = []
        for j in range(0,len(fi),128):
            sl = slice(j,min(j+128,len(fi))); basis = (data['basis'][0][:,sl],data['basis'][1][:,sl])
            _,s = P.components(data['n'][:,sl],data['v'][:,sl],params['rho'][sl],env,initial,K,basis)
            vals.append((data['w'][:,sl]*(s@s.new_tensor(P.Y))).sum(0)/data['w'][:,sl].sum(0))
        u = torch.cat(vals); b = data['b']; c0 = max(float((u*b).sum()/(u.square().sum()+1e-12)),1e-6)
        params['env'].fill_(c0)
    scale = max(float((data['w']*t(src['intensity'][:,fi])).sum()/data['w'].sum()),1e-3)
    e0 = float((w*src['b'][fi][None]).sum()/(w*(rho_y[None]*np.maximum(np.einsum('tgc,tc->tg',src['n'][:,fi],l),0)+src['b'][fi][None])).sum())
    return src, data, params, dict(K=K,c0=c0,scale=scale,e0=e0,sign=sign,equivalence_error=diff,hashes=hashes,
                                   env_shape=[H,W],loss='RGB+0.1*R_rev;no L_static;no guard')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume'); a = ap.parse_args(); out = Path(a.output)
    P.grid = grid                                  # 同一进程内 BRDF 基/求积改为 16×32
    P.guard = lambda energy, e0, mean, p90: (True, 'INFO_ONLY')   # 防护仅记录不判停
    O.objective_backward = objective_backward
    F.setup = setup
    started = datetime.now(timezone.utc).isoformat(); code = 1
    try:
        F.run(out, a.device, a.resume); code = 0
    finally:
        r = dict(schema_version=1, mode='fit', command=shlex.join(sys.orig_argv), cwd=os.getcwd(), hostname=socket.gethostname(),
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(), started_utc=started,
                 finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=code, log_path='run.log')
        out.mkdir(parents=True, exist_ok=True)
        path = out/'fit_execution.json'
        if path.exists(): path.rename(out/f'fit_execution_previous_{os.getpid()}.json')
        path.write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__': main()
