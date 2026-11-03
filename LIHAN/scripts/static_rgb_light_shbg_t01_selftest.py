"""T01 合成数据自检（持续完善中；不是正式自检入口）。"""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from LIHAN.scripts import static_rgb_light_shbg_t01_core_selftest as coretest
from LIHAN.scripts.static_rgb_light_shbg_t01_core import photometric
from LIHAN.scripts.static_rgb_light_shbg_t01_optimizer import fit
from LIHAN.scripts.static_rgb_light_shbg_t01_scoring import select
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import camera_fields, raw_camera_fields
from LIHAN.scripts.static_rgb_light_fit_r3 import kkt
from LIHAN.scripts.static_rgb_light_shbg_t01_templates import template
from LIHAN.scripts.static_rgb_light_shbg_t01_choose import ols_mask, ols_fixed


def main():
    torch.set_num_threads(2)
    coretest.main(); rng=np.random.default_rng(19)
    # 与枚举活跃集的非负最小二乘对照（实现保留旧 KKT 的 1e-6 稳定项）。
    x=rng.uniform(.2,1,(40,3)); w=rng.uniform(.5,1,(40,3))
    y=np.column_stack([.8*x[:,0]+.3, -.4*x[:,1]+.7, .6*x[:,2]-.2])
    r,b=kkt(torch.tensor(x),torch.tensor(y),torch.tensor(w)); r=r.numpy(); b=b.numpy()
    assert r[0]>0 and b[0]>0 and r[1]==0 and b[1]>0 and r[2]>0 and b[2]==0
    for j in range(3):
        A=np.column_stack((x[:,j],np.ones(len(x))))*np.sqrt(w[:,j,None]); z=y[:,j]*np.sqrt(w[:,j])
        candidates=[np.zeros(2),np.array([max(0,A[:,0]@z/(A[:,0]@A[:,0])),0]),np.array([0,max(0,A[:,1]@z/(A[:,1]@A[:,1]))])]
        both=np.linalg.lstsq(A,z,rcond=None)[0]
        if (both>=0).all(): candidates.append(both)
        reference=min(candidates,key=lambda c:np.linalg.norm(A@c-z))
        np.testing.assert_allclose([r[j],b[j]],reference,atol=1e-6,rtol=0)
    # 零背景时光度目标对方向变量的梯度与原目标一致。
    u=torch.tensor(rng.normal(size=(40,3)),dtype=torch.float64,requires_grad=True)
    xt=torch.sigmoid(u); yt=torch.tensor(y); wt=torch.tensor(w); ht=torch.ones_like(xt)
    new=photometric(xt,yt,wt,ht,torch.tensor(0.))[0]
    rho,bias=kkt(xt,yt,wt); old=(wt*.5*(xt*rho+bias-yt).square()).sum()/wt.sum()+1e-5*(rho.square().mean()+bias.square().mean())
    ga=torch.autograd.grad(new,u,retain_graph=True)[0]; gb=torch.autograd.grad(old,u)[0]
    assert abs((new-old).item())<=1e-10 and (ga-gb).abs().max().item()<=1e-7
    # 评分点只由 E3/E4 锁定；候选退化时拒绝，不能丢弃难点。
    angles=np.linspace(0,np.pi,150); nn=np.tile(np.stack([np.cos(angles),np.sin(angles),np.zeros(150)],axis=1)[:,None,:],(1,2,1))
    reference=np.tile([1.,0.,0.],(150,1)); degenerate=np.tile([0.,0.,1.],(150,1))
    values=.2+.5*np.maximum(nn[:,:,0],0); weights=np.ones_like(values)
    common=ols_mask(nn,values,weights,[reference]); assert common.all()
    assert np.isfinite(ols_fixed(nn,values,weights,reference,common)) and np.isinf(ols_fixed(nn,values,weights,degenerate,common))
    # 背景确实影响目标，并可从独立随机时变模板中恢复共享幅度。
    x=torch.tensor(rng.uniform(0,1,(120,12))); H=torch.tensor(rng.uniform(0,.2,(120,12)))
    w=torch.ones_like(x); y=.2+.7*x+.55*H; c=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([c],lr=.1)
    for _ in range(300):
        opt.zero_grad(); loss=photometric(x,y,w,H,c.sigmoid())[0]; loss.backward(); opt.step()
    assert abs(c.sigmoid().item()-.55)<.01
    assert photometric(x,y,w,H,c.sigmoid())[0] < photometric(x,y,w,H,torch.tensor(0.))[0]
    # 纯合成相机：未知字段值不解析；帧顺序独立检查。
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp); frames=[dict(file_path=f'r_{i:04d}',source_frame=i,transform_matrix=np.eye(4).tolist(),unused={'secret':[1,2,3]}) for i in range(1,151)]
        (p/'camera.json').write_text(json.dumps({'unused':{'other':True},'frames':frames}))
        ids,centers=camera_fields(p/'camera.json'); assert np.array_equal(ids,np.arange(1,151)) and centers.shape==(150,3)
        raw={'frames':{f'{i:04d}':{'frame':i,'extrinsics':{'camera_to_world':np.eye(4).tolist(),'other':[1,2,3]},'ignored':{'x':1}} for i in range(1,151)},'ignored':[1,2]}
        (p/'raw.json').write_text(json.dumps(raw)); rawids,matrices=raw_camera_fields(p/'raw.json')
        assert np.array_equal(ids,rawids) and np.array_equal(centers,matrices[:,:3,3])
        T=12; G=6; phi=np.linspace(0,2*np.pi,T); l=np.stack([np.cos(phi)/np.sqrt(2),np.sin(phi)/np.sqrt(2),np.full(T,1/np.sqrt(2))],1)
        n=rng.normal(size=(T,G,3)); n/=np.linalg.norm(n,axis=-1,keepdims=True)
        x=np.maximum(np.einsum('tgc,tc->tg',n,l),0); H=rng.uniform(0,.1,(T,G)); y=.2+.7*x+.4*H
        result=fit(n,y,np.ones_like(y),H,l,.5,0,1e-4,p/'tc',budgets=(2,6,8,2,3,6),force_tc=True)
        assert 'pre_TC' in result and (p/'tc'/'checkpoint.pt').exists()
        bad=n.copy(); bad[0,0,0]=np.nan
        failed=fit(bad,y,np.ones_like(y),H,l,.5,0,1e-4,p/'failure',budgets=(2,6,8,2,3,6))
        assert not failed['converged'] and failed['reason']=='OPTIMIZER_EXCEPTION' and (p/'failure'/'checkpoint.pt').exists()
    base=dict(all_gates=True,J=1.,G4b_mae=.1,G4a_mae=.1,L=1,lambda_bg=.001,name='C02')
    other=dict(base,name='C01',lambda_bg=.0001,J=1.019)
    chosen,status=select([other,base]); assert chosen['name']=='C02' and status=='SELECTED_SH'
    assert select([dict(base,all_gates=False)])[1]=='FALLBACK_E4'
    print('预注册数值组件合成检查通过；无真实数据和 GT。')

if __name__=='__main__': main()
