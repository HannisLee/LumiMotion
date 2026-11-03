"""T01 固定预算轨迹优化器；禁止用于未通过完整自检的实验。"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from LIHAN.scripts import static_rgb_light_fit_r5b as r5
from LIHAN.scripts.static_rgb_light_shbg_t01_core import photometric


def fit(n, y, w, H, anchor, gamma_initial, elevation_initial, lambda_bg, dest,
        budgets=(1000, 200, 400, 500, 6000, 60), force_tc=False):
    """固定 T-A/T-B/T-C 与 C1/C2/C3；异常不触发额外 fallback。"""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    nt, yt, wt, ht = map(tensor, (n, np.nan_to_num(y), np.where(np.isfinite(y), w, 0), H))
    phi0 = np.unwrap(np.arctan2(anchor[:, 1], anchor[:, 0])); sign = r5.orbit_sign(anchor)
    if abs(np.diff(phi0).sum()) < np.pi:
        raise ValueError('初始环绕量不足 180 度')
    ph = torch.nn.Parameter(tensor(phi0).clone())
    av = torch.nn.Parameter(tensor(np.broadcast_to(elevation_initial, len(phi0)).copy()))
    c = torch.nn.Parameter(tensor(math.log(gamma_initial/(1-gamma_initial))).clone())
    def direction(p, a):
        return r5.compose(torch.stack([torch.cos(p), torch.sin(p)], 1), r5.elev(a))
    def objective(p, a, cc):
        e = r5.elev(a); l = direction(p, a)
        x = torch.relu(torch.einsum('tgc,tc->tg', nt, l))
        gamma = torch.sigmoid(cc)
        data, rho, b = photometric(x, yt, wt, ht, gamma)
        reg = r5.penalties(p, e, sign)
        return data + reg['a1'] + reg['a2'] + 10*(reg['e1']+reg['e2']) + .1*reg['rev'] + lambda_bg*gamma.square(), rho, b
    def snap():
        return [p.detach().clone() for p in (ph, av, c)]
    def restore(s):
        with torch.no_grad():
            for p, v in zip((ph, av, c), s): p.copy_(v)
    def measure(p, a, cc):
        with torch.no_grad():
            return direction(p, a).cpu().numpy(), float(torch.sigmoid(cc))
    T = len(phi0); D1 = torch.diff(torch.eye(T, dtype=torch.float64, device=device), dim=0); D2 = torch.diff(D1, dim=0)
    def precond(lam, weight):
        L2 = lam/(T*1e-4*weight)
        M = torch.eye(T, dtype=torch.float64, device=device) + L2*D1.T@D1 + L2**2*D2.T@D2
        ev, U = torch.linalg.eigh(M)
        return U@torch.diag(ev.rsqrt())@U.T
    Pp, Pa = precond(1, .5), precond(10, 1.)
    hist = []; last_lbfgs = {}
    def checked(loss, params=None):
        if not torch.isfinite(loss): raise FloatingPointError('非有限目标')
        p,a,cc=(ph,av,c) if params is None else params
        if not all(torch.isfinite(z).all() for z in (p,a,cc)): raise FloatingPointError('非有限参数')
        g=torch.sigmoid(cc)
        if not (g>0 and g<1): raise FloatingPointError('背景幅度数值到达边界')
    def record(stage, step, value, cc=None):
        cc=c if cc is None else cc
        item = dict(stage=stage, step=step, loss=float(value), gamma=float(torch.sigmoid(cc).detach()))
        hist.append(item)
        if step==1 or step%50==0: print(json.dumps(item), flush=True)
    def lbfgs(count):
        base = snap(); qp = torch.nn.Parameter(torch.zeros_like(ph)); qa = torch.nn.Parameter(torch.zeros_like(av)); qc = torch.nn.Parameter(torch.zeros_like(c))
        opt = torch.optim.LBFGS([qp,qa,qc], lr=1, history_size=50, line_search_fn='strong_wolfe', max_iter=20, tolerance_grad=1e-12, tolerance_change=1e-15)
        def eff(): return base[0]+Pp@qp, base[1]+Pa@qa, base[2]+qc
        def closure():
            opt.zero_grad(); params=eff(); value = objective(*params)[0]; checked(value,params); value.backward(); return value
        trace=[]; dirs=[]; gs=[]; metrics=None
        for k in range(count):
            opt.step(closure); p,a,cc=eff(); value=objective(p,a,cc)[0]; checked(value,(p,a,cc))
            restore([p.detach(),a.detach(),cc.detach()])
            last_lbfgs.update(state=opt.state_dict(),base=[z.detach().cpu() for z in base],q=[z.detach().cpu() for z in (qp,qa,qc)])
            dr,g=measure(p,a,cc); trace.append(float(value.detach())); dirs.append(dr); gs.append(g)
            record('LBFGS',k+1,value.detach(),cc)
            if len(trace)>5:
                rel=abs(trace[-6]-trace[-1])/max(abs(trace[-6]),1e-30)
                mx=float(r5.ang_deg(dirs[-6],dirs[-1]).max()); dg=abs(gs[-6]-gs[-1])
                metrics=(rel,mx,dg)
                if rel<1e-9 and mx<.005 and dg<1e-5: break
        last_lbfgs.update(state=opt.state_dict(),base=[z.detach().cpu() for z in base],q=[z.detach().cpu() for z in (qp,qa,qc)])
        restore([z.detach() for z in eff()]); return metrics, len(trace)
    def probe():
        state=snap(); d0,g0=measure(*state); loss0=float(objective(*state)[0].detach()); opt=torch.optim.Adam([{'params':[ph],'lr':1e-4},{'params':[av],'lr':1e-3},{'params':[c],'lr':1e-3}])
        try:
            for _ in range(budgets[3]):
                opt.zero_grad(); loss=objective(ph,av,c)[0]; checked(loss); loss.backward(); opt.step()
            d1,g1=measure(ph,av,c); loss1=float(objective(ph,av,c)[0].detach()); return float(r5.ang_deg(d0,d1).max()), (loss0-loss1)/max(abs(loss0),1e-30), abs(g1-g0)
        finally: restore(state)
    def certificate(m):
        p=probe(); good=m is not None and m[0]<1e-9 and m[1]<.005 and m[2]<1e-5 and p[0]<.1 and p[1]<1e-4 and p[2]<1e-3
        return dict(window=m, probe=p, converged=bool(good))
    try:
        opt=torch.optim.Adam([{'params':[ph],'lr':.01},{'params':[av],'lr':.11},{'params':[c],'lr':.03}])
        for k in range(1,budgets[0]+1):
            opt.zero_grad(); loss=objective(ph,av,c)[0]; checked(loss); record('TA',k,loss.detach()); loss.backward(); opt.step()
            if k in (600,850):
                for group in opt.param_groups: group['lr']*=.3
        m,used=lbfgs(budgets[1]); cert=certificate(m)
        while not cert['converged'] and used<budgets[2]:
            limit=(budgets[1]-used) if used<budgets[1] else (budgets[2]-used)
            m,extra=lbfgs(limit); used+=extra; cert=certificate(m)
        cert['outer_iterations']=used
        if not cert['converged'] or force_tc:
            before=cert; base=snap(); qp=torch.nn.Parameter(torch.zeros_like(ph)); qa=torch.nn.Parameter(torch.zeros_like(av)); qc=torch.nn.Parameter(torch.zeros_like(c))
            opt=torch.optim.Adam([qp,qa,qc],lr=.003); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,budgets[4],eta_min=0)
            for k in range(budgets[4]):
                opt.zero_grad(); params=(base[0]+Pp@qp,base[1]+Pa@qa,base[2]+qc); loss=objective(*params)[0]; checked(loss,params); record('TC',k+1,loss.detach(),base[2]+qc); loss.backward(); opt.step(); sched.step(); restore([base[0]+Pp@qp,base[1]+Pa@qa,base[2]+qc])
            restore([base[0]+Pp@qp,base[1]+Pa@qa,base[2]+qc]); m,used_tc=lbfgs(budgets[5]); cert=certificate(m); cert['pre_TC']=before; cert['TC_outer_iterations']=used_tc
        with torch.no_grad():
            loss,rho,b=objective(ph,av,c); dr,g=measure(ph,av,c); checked(loss)
            np.save(dest/'direction.npy',dr); np.savez_compressed(dest/'parameters.npz',phi=ph.cpu().numpy(),a=av.cpu().numpy(),c=c.cpu().numpy(),gamma=g,rho=rho.cpu().numpy(),b=b.cpu().numpy())
            torch.save({'phi':ph.detach().cpu(),'a':av.detach().cpu(),'c':c.detach().cpu(),'adam_state':opt.state_dict(),'lbfgs_state':last_lbfgs},dest/'checkpoint.pt')
        cert['loss']=float(loss); cert['reason']=None if cert['converged'] else 'NOT_CONVERGED'
    except Exception as exc:
        cert=dict(converged=False,reason='OPTIMIZER_EXCEPTION',exception=repr(exc)); torch.save({'phi':ph.detach().cpu(),'a':av.detach().cpu(),'c':c.detach().cpu(),'adam_state':opt.state_dict() if 'opt' in locals() else None,'lbfgs_state':last_lbfgs},dest/'checkpoint.pt')
    (dest/'history.json').write_text(json.dumps(hist,indent=2)); (dest/'certificate.json').write_text(json.dumps(cert,indent=2)); return cert
