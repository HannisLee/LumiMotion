"""T02 合成自检；所有已知值均由随机合成产生，不是数据集 GT。"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from LIHAN.scripts import static_rgb_light_shbg_t01_core_selftest as old_core
from LIHAN.scripts import static_rgb_light_shbg_t01_extract_selftest as extract_test
from LIHAN.scripts import static_rgb_light_shbg_t02_core_selftest as core_test
from LIHAN.scripts.static_rgb_light_fit_r3 import kkt
from LIHAN.scripts.static_rgb_light_shbg_t01_core import photometric, sh_luminance, validate_alignment
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import camera_fields, raw_camera_fields, digest
from LIHAN.scripts.static_rgb_light_shbg_t01_scoring import select
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters, design, project, final_eta
from LIHAN.scripts.static_rgb_light_shbg_t02_templates import template
from LIHAN.scripts.static_rgb_light_shbg_t02_choose import bg_statistics, heldout_score
from LIHAN.scripts.static_rgb_light_shbg_t02_freeze import freeze


def main():
    torch.set_num_threads(2); rng=np.random.default_rng(20261103)
    print('所有已知数值均为合成数据；不读取真实数据和 GT。')
    old_core.main(); extract_test.main(); core_test.main()
    frames=np.arange(1,151); ids=np.arange(4)
    for bad_ids,bad_frames in ((ids[::-1],frames),(ids,frames[::-1])):
        try: validate_alignment(bad_ids,ids,bad_frames)
        except ValueError: pass
        else: raise AssertionError('ID/帧错位未拒绝')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp); camera=path/'camera.json'
        f=[dict(file_path=f'r_{i:04d}',source_frame=int(i),transform_matrix=np.eye(4).tolist(),ignored={'token':[1,2]}) for i in frames]
        camera.write_text(json.dumps(dict(frames=f,ignored={'nested':[1,2]})))
        fr,centers=camera_fields(camera); assert np.array_equal(fr,frames) and centers.shape==(150,3)
        raw={'frames':{f'{i:04d}':{'frame':int(i),'extrinsics':{'camera_to_world':np.eye(4).tolist(),'ignored':[1]},'ignored':True} for i in frames}}
        camera.write_text(json.dumps(raw)); fr,matrices=raw_camera_fields(camera); assert np.array_equal(fr,frames) and matrices.shape==(150,4,4)
        camera.write_text('old'); old=digest(camera); camera.write_text('new'); assert digest(camera)!=old
        from LIHAN.scripts import static_rgb_light_shbg_t02 as entry
        receipt={'status':'PASS','code_sha256':{'synthetic':old},'plan_sha256':entry.PLAN_HASH}
        (path/'receipt.json').write_text(json.dumps(receipt))
        with patch.object(entry,'code_hashes',return_value={'synthetic':digest(camera)}):
            try: entry.check_selftest(path)
            except ValueError: pass
            else: raise AssertionError('代码哈希变化未拒绝')
        try: freeze(path)
        except FileNotFoundError: pass
        else: raise AssertionError('冻结未拒绝缺失文件')
    coeff=np.zeros((1,3,9)); coeff[:,:,3]=.1
    xyz=np.broadcast_to(np.array([2.,0.,0.]),(150,1,3)).copy(); centers=np.broadcast_to(np.array([1.,0.,0.]),(150,3)).copy()
    lum,raw=sh_luminance(coeff,xyz,centers,1); srgb=.5-.1*.4886025119029199; linear=((srgb+.055)/1.055)**2.4
    np.testing.assert_allclose(lum,linear,atol=1e-14); np.testing.assert_allclose(raw,srgb)
    theta=np.arange(150)*2*np.pi/150; anchor=np.stack((np.cos(theta)/np.sqrt(2),np.sin(theta)/np.sqrt(2),np.full(150,1/np.sqrt(2))),axis=1)
    phi,a=anchor_parameters(anchor); n=rng.normal(size=(150,3,3)); n/=np.linalg.norm(n,axis=2,keepdims=True)
    d=design(n,phi,a); w=rng.uniform(.5,1,(150,3)); valid=np.ones_like(w,bool)
    h=rng.normal(size=(150,3)); rec=project(h,d,w,valid)
    # 人工正交信号经过统一缩放后保持方向。
    h2=np.empty_like(h)
    for g in range(3): h2[:,g]=d[:,g]@rng.normal(size=12)+rec['q'][:,g]
    rec2=project(h2,d,w,valid)
    for g in range(3):
        alpha=np.dot(rec['q'][:,g],rec2['q'][:,g])/np.dot(rec['q'][:,g],rec['q'][:,g])
        assert np.max(np.abs(rec2['q'][:,g]-alpha*rec['q'][:,g]))<1e-8 and alpha>0
    # 全零列、秩亏与缺失帧。
    singular=np.zeros((150,1,12)); singular[:,0,0]=1; singular[:,0,1]=1
    mask=np.ones((150,1),bool); mask[::4]=False
    rr=project(rng.normal(size=(150,1)),singular,np.ones((150,1)),mask)
    assert rr['rank'][0]==1 and rr['counts'][0]==112 and rr['error'][0]<1e-8
    # 高度相关的列可用差分解释 q，逐列相关远小于门槛。
    z=rng.normal(size=150); z-=z.mean(); z/=np.linalg.norm(z)
    q=rng.normal(size=150); q-=z*np.dot(z,q); q/=np.linalg.norm(q)
    basis=np.zeros((150,1,12)); basis[:,0,0]=z; basis[:,0,1]=z+.01*q
    correlations=np.abs(basis[:,0,:2].T@q)/np.linalg.norm(basis[:,0,:2],axis=0)
    eta=final_eta(q[:,None],basis,np.ones((150,1)),np.ones((150,1),bool))[0]
    assert correlations.max()<.2 and eta>.999999
    # 最终方向跨越明暗边界：以最终直射为合成 h，保留初始投影 q。
    tangent=np.stack((-np.sin(phi),np.cos(phi),np.zeros(150)),axis=1)
    shift=rng.uniform(-.012,.002,size=150); normals=tangent+shift[:,None]*anchor
    normals/=np.linalg.norm(normals,axis=1,keepdims=True); normals=normals[:,None,:]
    delta=np.deg2rad(1.); final_phi=phi+delta
    final_l=np.stack((np.cos(np.pi/4)*np.cos(final_phi),np.cos(np.pi/4)*np.sin(final_phi),np.full(150,np.sin(np.pi/4))),axis=1)
    initial_dot=np.einsum('tgc,tc->tg',normals,anchor); final_dot=np.einsum('tgc,tc->tg',normals,final_l)
    assert np.any((initial_dot<=0)&(final_dot>0))
    d0=design(normals,phi,a); df=design(normals,final_phi,a)
    first=project(np.maximum(final_dot,0),d0,np.ones((150,1)),np.ones((150,1),bool)); before=first['q'].copy()
    assert final_eta(before,df,np.ones((150,1)),first['cal'])[0]>.2 and np.array_equal(before,first['q'])
    y=.2+.5*np.maximum(np.einsum('tgc,tc->tg',n,anchor),0)+.3*rec['H']; weights=np.ones_like(y)
    class OddOnly:
        def __init__(self,arr): self.arr=arr
        def __getitem__(self,key):
            if key!=slice(None,None,2): raise AssertionError('校准读取偶数帧')
            return self.arr[key]
    hold=template(h,n,OddOnly(y),OddOnly(weights),np.arange(3),anchor,True)
    changed=y.copy(); changed[1::2]+=100; weights2=weights.copy(); weights2[1::2]*=7
    hold2=template(h,n,changed,weights2,np.arange(3),anchor,True)
    np.testing.assert_array_equal(hold['q'],hold2['q']); np.testing.assert_array_equal(hold['beta'],hold2['beta'])
    def calibrate_gamma(yy,ww,H):
        xt=torch.tensor(np.maximum(np.einsum('tgc,tc->tg',n,anchor),0)[::2]); yt=torch.tensor(yy[::2]); wt=torch.tensor(ww[::2]); ht=torch.tensor(H[::2]); c=torch.tensor(0.,dtype=torch.float64,requires_grad=True); opt=torch.optim.Adam([c],lr=.03)
        for _ in range(100): opt.zero_grad(); loss=photometric(xt,yt,wt,ht,c.sigmoid())[0]; loss.backward(); opt.step()
        rho,b=kkt(xt,yt-c.sigmoid()*ht,wt)
        return c.detach().numpy(),rho.detach().numpy(),b.detach().numpy()
    for first,second in zip(calibrate_gamma(y,weights,hold['H']),calibrate_gamma(changed,weights2,hold2['H'])): np.testing.assert_array_equal(first,second)
    x=torch.tensor(rng.uniform(.1,1,(150,3)),requires_grad=True); yt=torch.tensor(y); wt=torch.ones_like(x); H=torch.tensor(hold['H']); zero=torch.tensor(0.)
    new=photometric(x,yt,wt,H,zero)[0]; rho,b=kkt(x,yt,wt); old=(wt*.5*(x*rho+b-yt).square()).sum()/wt.sum()+1e-5*(rho.square().mean()+b.square().mean())
    grad1=torch.autograd.grad(new,x,retain_graph=True)[0]; grad2=torch.autograd.grad(old,x)[0]
    assert abs(float(new-old))<1e-10 and (grad1-grad2).abs().max()<1e-7
    c=torch.tensor(0.,dtype=torch.float64,requires_grad=True); loss=photometric(x,yt,wt,H,c.sigmoid())[0]; loss.backward(); assert torch.isfinite(c.grad) and abs(c.grad)>1e-10
    xx=torch.tensor(rng.uniform(.2,1,(100,3))); ww=torch.ones_like(xx); hh=torch.tensor(rng.uniform(0,.1,(100,3))); known=.2+.7*xx+.55*hh
    c=torch.tensor(0.,dtype=torch.float64,requires_grad=True); opt=torch.optim.Adam([c],lr=.1)
    for _ in range(400): opt.zero_grad(); z=photometric(xx,known,ww,hh,c.sigmoid())[0]; z.backward(); opt.step()
    assert abs(float(c.sigmoid())-.55)<.01
    negative=torch.full_like(xx,-.5); rho,b=kkt(xx,negative,ww); assert (rho==0).all() and (b==0).all()
    from LIHAN.scripts.static_rgb_light_shbg_t02_choose import calibration_mask
    mask23=weights.copy(); mask23[46:150:2,0]=0
    mask24=weights.copy(); mask24[48:150:2,0]=0
    assert not calibration_mask(n,OddOnly(y),OddOnly(mask23),anchor)[0]
    assert calibration_mask(n,OddOnly(y),OddOnly(mask24),anchor)[0]
    missing=weights.copy(); missing[1,:]=0
    assert np.isinf(heldout_score(n,y,missing,anchor,np.ones(3,bool),hold['H'],.3)[0])
    s1,count1=heldout_score(n,y,weights,anchor,np.ones(3,bool),hold['H'],.3)
    s2,count2=heldout_score(n,changed,weights,anchor,np.ones(3,bool),hold['H'],.3)
    assert s1!=s2 and count1==count2
    rms,ratio=bg_statistics(hold['q'],.3,y,weights); assert rms>0 and ratio>0
    row=dict(all_gates=True,J=1.,G4b_mae=.1,G4a_mae=.1,L=1,lambda_bg=.001,name='C02')
    picked,status=select([dict(row,name='C01',lambda_bg=.0001,J=1.019),row]); assert picked['name']=='C02' and status=='SELECTED_SH'
    assert select([dict(row,all_gates=False)])[1]=='FALLBACK_E4'
    print('T02 合成数值与基础隔离检查通过。')
    from LIHAN.scripts import static_rgb_light_shbg_t02_optimizer_selftest, static_rgb_light_shbg_t02_integration_selftest
    static_rgb_light_shbg_t02_optimizer_selftest.main()
    static_rgb_light_shbg_t02_integration_selftest.main()
    print('T02 全部合成自检完成。')

if __name__=='__main__': main()
