"""固定真实对应的光向/能量局部可辨识性与残差分解；只诊断，不训练或读GT。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch


def subspace(A):
    u,s,v=np.linalg.svd(A,full_matrices=False)
    cutoff=max(float(s[0])*1e-8,1e-12) if len(s) else 1e-12
    rank=int((s>cutoff).sum())
    return u[:,:rank],s,rank


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);fit=Path(a.fit);torch.set_num_threads(4)
    c=torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False);p=torch.load(fit/'pairs.pt',map_location='cpu',weights_only=False)
    assert c['protocol']=='block_selfcal_paired_v1' and c['geometry_updates']==0
    p={k:(v.double() if v.is_floating_point() else v) for k,v in p.items()}
    B=c['time_basis'].double();baseD=c['base_direction'].double();baseE=c['base_energy'].double();train=c['train_frames']
    def predictions(x):
        D=torch.nn.functional.normalize(baseD+B@x[:24].reshape(8,3),dim=-1)
        E=baseE*(B@x[24:]).exp();return D,E/E[train].mean()
    def residual(x,is_train):
        D,E=predictions(x);m=p['train'] if is_train else ~p['train'];ia=p['a'][m];ib=p['b'][m]
        sa=E[ia]*((p['na'][m]*D[ia]).sum(-1).clamp_min(0)+.1);sb=E[ib]*((p['nb'][m]*D[ib]).sum(-1).clamp_min(0)+.1)
        ya=p['ya'][m];yb=p['yb'][m];rho=(sa[:,None]*ya+sb[:,None]*yb)/(sa.square()+sb.square())[:,None]
        return torch.stack([sa[:,None]*rho-ya,sb[:,None]*rho-yb],dim=1).flatten()/np.sqrt(int(m.sum())*6)
    results={}
    for name,x in [('initial',torch.zeros(32,dtype=torch.double)),('final',torch.cat([c['direction_controls'].flatten(),c['log_energy_controls']]).double())]:
        r=residual(x,True).detach().numpy();rh=residual(x,False).detach().numpy()
        J=torch.autograd.functional.jacobian(lambda z:residual(z,True),x,vectorize=True).numpy()
        Jh=torch.autograd.functional.jacobian(lambda z:residual(z,False),x,vectorize=True).numpy()
        ud,sd,rd=subspace(J[:,:24]);ue,se,re=subspace(J[:,24:]);uj,sj,rj=subspace(J)
        conditional=J[:,:24]-ue@(ue.T@J[:,:24]);uc,sc,rc=subspace(conditional)
        canonical=np.linalg.svd(ud.T@ue,compute_uv=False)
        diagnostic={}
        for mode,sl in [('direction',slice(0,24)),('energy',slice(24,32)),('joint',slice(0,32))]:
            M=J[:,sl];MH=Jh[:,sl];delta=np.linalg.lstsq(M,-r,rcond=1e-8)[0]
            diagnostic[mode]=dict(linear_train_mse=float(np.mean((r+M@delta)**2)*len(r)),linear_held_mse=float(np.mean((rh+MH@delta)**2)*len(rh)),parameter_norm=float(np.linalg.norm(delta)),scope='无约束局部线性诊断，不作为真实前向或可接受更新')
        D,E=predictions(x);parts={}
        for split,is_train in [('train',True),('held',False)]:
            m=p['train'] if is_train else ~p['train'];ca=(p['na'][m]*D[p['a'][m]]).sum(-1).numpy();cb=(p['nb'][m]*D[p['b'][m]]).sum(-1).numpy()
            rr=residual(x,is_train).detach().numpy().reshape(-1,6);sse=(rr**2).sum(-1)
            labels={'both_nonpositive':(ca<=0)&(cb<=0),'one_nonpositive':(ca<=0)^(cb<=0),'both_positive':(ca>0)&(cb>0)}
            parts[split]={k:dict(count=int(v.sum()),mse_share=float(sse[v].sum()/sse.sum())) for k,v in labels.items()}
            order=np.argsort(-sse);parts[split]['top_10_share']=float(sse[order[:10]].sum()/sse.sum())
            parts[split]['edges']={str(int(e)):dict(count=int((p['edge'][m].numpy()==e).sum()),mse_share=float(sse[p['edge'][m].numpy()==e].sum()/sse.sum())) for e in np.unique(p['edge'][m].numpy())}
        results[name]=dict(train_mse=float(r@r),held_mse=float(rh@rh),rank_direction=rd,rank_energy=re,rank_joint=rj,rank_direction_after_energy=rc,singular_direction=sd.tolist(),singular_energy=se.tolist(),singular_joint=sj.tolist(),singular_direction_after_energy=sc.tolist(),canonical_correlations=canonical.tolist(),direction_jacobian_energy_explainable_fraction=float(np.linalg.norm(ue@(ue.T@J[:,:24]))**2/np.linalg.norm(J[:,:24])**2),linear_diagnostic=diagnostic,residual_parts=parts)
        np.savez(out/f'{name}.npz',residual=r,held_residual=rh,jacobian=J,held_jacobian=Jh,conditional_direction=conditional)
    sources=[fit/'checkpoint.pt',fit/'pairs.pt',Path(__file__)]
    result=dict(gt_read=False,updates=0,train_count=int(p['train'].sum()),held_count=int((~p['train']).sum()),states=results,inputs={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},scope='固定109/69对应及粗法线的双精度局部分析；不证明全局可辨识性，不使用线性解训练')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if not kk.startswith('singular') and kk not in ['residual_parts','canonical_correlations']} for k,v in results.items()},indent=2))
if __name__=='__main__':main()
