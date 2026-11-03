"""T02 固定预算优化器；尚未接入完整入口及自检，禁止真实数据运行。"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from LIHAN.scripts import static_rgb_light_fit_r5b as r5
from LIHAN.scripts.static_rgb_light_shbg_t01_core import photometric
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters, trajectory


def fit(n, y, w, H, anchor, lambda_bg, dest, run='full', control=False):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    nt, yt, wt, ht = map(tensor, (n, np.nan_to_num(y), np.where(np.isfinite(y) & np.isfinite(w) & (w > 0), w, 0), H))
    phi0, a0 = anchor_parameters(anchor); phi0, a0 = tensor(phi0), tensor(a0)
    sign = r5.orbit_sign(anchor)
    start = math.atanh(.5) * (1 if run == 'hi' else -1 if run == 'lo' else 0)
    U = torch.nn.Parameter(torch.full((5,), start, dtype=torch.float64, device=device))
    V = torch.nn.Parameter(torch.full((5,), start, dtype=torch.float64, device=device))
    c = None if control else torch.nn.Parameter(tensor(math.log(9) * (1 if run == 'hi' else -1 if run == 'lo' else 0)))
    params = [U, V] + ([] if control else [c]); history = []; last_lbfgs = None
    def state(): return [z.detach().clone() for z in params]
    def restore(values):
        with torch.no_grad():
            for p, value in zip(params, values): p.copy_(value)
    def calc():
        phi, a, e, direction = trajectory(phi0, a0, U, V)
        gamma = torch.zeros((), dtype=torch.float64, device=device) if control else torch.sigmoid(c)
        x = torch.relu(torch.einsum('tgc,tc->tg', nt, direction))
        data, rho, b = photometric(x, yt, wt, ht, gamma)
        reg = r5.penalties(phi, e, sign)
        return data + reg['a1'] + reg['a2'] + 10*(reg['e1']+reg['e2']) + .1*reg['rev'] + lambda_bg*gamma.square(), direction, gamma, phi, a, rho, b
    def checked(backward=False):
        result = calc(); loss = result[0]
        if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in params): raise FloatingPointError('非有限目标或参数')
        if backward:
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params): raise FloatingPointError('非有限梯度')
        return result
    def snapshot():
        with torch.no_grad():
            result = checked(); return float(result[0]), result[1].cpu().numpy(), float(result[2])
    def record(stage, step, loss):
        history.append(dict(stage=stage, step=step, loss=float(loss)))
        if step == 1 or step % 50 == 0: print(json.dumps(history[-1]), flush=True)
    def lbfgs(count):
        nonlocal last_lbfgs
        opt = torch.optim.LBFGS(params, lr=1, history_size=50, line_search_fn='strong_wolfe', max_iter=20, tolerance_grad=1e-12, tolerance_change=1e-15)
        def closure():
            opt.zero_grad(); return checked(backward=True)[0]
        trace = []; metrics = None
        for k in range(count):
            opt.step(closure); value = snapshot(); trace.append(value); record('LBFGS', k+1, value[0]); last_lbfgs = opt.state_dict()
            if len(trace) > 5:
                old = trace[-6]; rel = abs(value[0]-old[0])/max(abs(old[0]), 1e-30)
                angle = float(r5.ang_deg(value[1], old[1]).max()); dg = abs(value[2]-old[2])
                metrics = (rel, angle, dg)
                if rel < 1e-9 and angle < .005 and (control or dg < 1e-5): break
        return metrics, len(trace)
    def probe():
        saved = state(); before = snapshot(); opt = torch.optim.Adam(params, lr=.001)
        try:
            for _ in range(500):
                opt.zero_grad(); checked(backward=True); opt.step()
            after = snapshot()
            return float(r5.ang_deg(before[1], after[1]).max()), (before[0]-after[0])/max(abs(before[0]), 1e-30), abs(before[2]-after[2])
        finally: restore(saved)
    def certificate(window):
        p = probe()
        ok = window is not None and window[0] < 1e-9 and window[1] < .005 and (control or window[2] < 1e-5) and p[0] < .1 and p[1] < 1e-4 and (control or p[2] < 1e-3)
        return dict(window=window, probe=p, converged=bool(ok))
    try:
        initial = snapshot()
        if abs(float(torch.diff(trajectory(phi0, a0, U, V)[0]).sum().detach())) < np.pi: raise ValueError('初始环绕不足 180 度')
        opt = torch.optim.Adam(params, lr=.03)
        for k in range(1, 1001):
            opt.zero_grad(); loss=checked(backward=True)[0]; record('Adam', k, loss.detach()); opt.step()
            if k in (600, 850):
                for group in opt.param_groups: group['lr'] *= .3
        window, used = lbfgs(200); cert = certificate(window)
        if not cert['converged'] and used < 400:
            window, more = lbfgs(400-used); used += more; cert = certificate(window)
        cert['outer_iterations'] = used
        if not cert['converged']:
            before = cert; opt = torch.optim.Adam(params, lr=.003); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000, eta_min=0)
            for k in range(6000):
                opt.zero_grad(); loss=checked(backward=True)[0]; record('补充Adam', k+1, loss.detach()); opt.step(); sched.step()
            window, used = lbfgs(60); cert = certificate(window); cert['pre_supplement'] = before; cert['supplement_outer_iterations'] = used
        with torch.no_grad():
            loss, direction, gamma, phi, a, rho, b = checked()
            saturated = bool((torch.abs(torch.tanh(U)) > .98).any() or (torch.abs(torch.tanh(V)) > .98).any() or (not control and not (.01 <= float(gamma) <= .99)))
            cert.update(initial_loss=initial[0], initial_max_anchor_deg=float(r5.ang_deg(initial[1], anchor).max()), initial_gamma=initial[2], orbit_sign=sign, loss=float(loss), reason='BOUNDARY_SATURATION' if saturated else None if cert['converged'] else 'NOT_CONVERGED')
            cert['converged'] = bool(cert['converged'] and not saturated)
            np.save(dest/'direction.npy', direction.cpu().numpy())
            np.savez_compressed(dest/'parameters.npz', U=U.cpu().numpy(), V=V.cpu().numpy(), u=(np.pi/180*torch.tanh(U)).cpu().numpy(), v=(.1*torch.tanh(V)).cpu().numpy(), phi=phi.cpu().numpy(), a=a.cpu().numpy(), gamma=float(gamma), rho=rho.cpu().numpy(), b=b.cpu().numpy())
    except Exception as exc:
        cert = dict(converged=False, reason='OPTIMIZER_EXCEPTION', exception=repr(exc))
    torch.save(dict(parameters=[p.detach().cpu() for p in params], adam_state=opt.state_dict() if 'opt' in locals() else None, lbfgs_state=last_lbfgs), dest/'checkpoint.pt')
    (dest/'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2))
    (dest/'certificate.json').write_text(json.dumps(cert, ensure_ascii=False, indent=2))
    return cert
