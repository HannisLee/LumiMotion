"""仅合成数据：执行真实优化预算，检查探针恢复和补充预算。"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters, trajectory
from LIHAN.scripts.static_rgb_light_shbg_t02_optimizer import boundary_saturated, fit


def main():
    torch.set_num_threads(2); rng=np.random.default_rng(720)
    theta=np.arange(150)*2*np.pi/150; anchor=np.stack((np.cos(theta)/np.sqrt(2),np.sin(theta)/np.sqrt(2),np.full(150,1/np.sqrt(2))),axis=1)
    n=rng.normal(size=(150,3,3)); n/=np.linalg.norm(n,axis=2,keepdims=True)
    x=np.maximum(np.einsum('tgc,tc->tg',n,anchor),0); y=.2+.7*x; w=np.ones_like(y); H=np.zeros_like(y)
    assert not boundary_saturated(torch.zeros(5),torch.zeros(5),.5)
    assert boundary_saturated(torch.full((5,),3.),torch.zeros(5),.5)
    assert boundary_saturated(torch.zeros(5),torch.zeros(5),.009)
    assert not boundary_saturated(torch.zeros(5),torch.zeros(5),0.,True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        def interrupt_step(*unused,**kwargs):
            assert (root/'interrupted/checkpoint.pt').is_file() and (root/'interrupted/history.json').is_file()
            raise RuntimeError('合成中断')
        with patch.object(torch.optim.Adam,'step',interrupt_step):
            interrupted=fit(n,y,w,H,anchor,0.,root/'interrupted',control=True)
        assert interrupted['reason']=='OPTIMIZER_EXCEPTION' and (root/'interrupted/checkpoint.pt').is_file()
        cert=fit(n,y,w,H,anchor,0.,root/'control',control=True)
        assert cert['converged'] and cert['initial_gamma']==0 and cert['reason'] is None
        ckpt=torch.load(root/'control/checkpoint.pt',map_location='cpu',weights_only=False)
        assert len(ckpt['parameters'])==2
        phi,a=anchor_parameters(anchor)
        with np.load(root/'control/parameters.npz',allow_pickle=False) as p:
            _,_,_,l=trajectory(torch.tensor(phi),torch.tensor(a),torch.tensor(p['U']),torch.tensor(p['V']))
            np.testing.assert_allclose(l.numpy(),np.load(root/'control/direction.npy'),rtol=0,atol=1e-14)
        # 固定角度探针强制证书失败，从而执行完整 400 + 6000 + 60 预算。
        with patch('LIHAN.scripts.static_rgb_light_shbg_t02_optimizer.r5.ang_deg',side_effect=lambda a,b:np.full(150,.2)):
            failed=fit(n,y,w,H,anchor,0.,root/'supplement',control=True)
        hist=json.loads((root/'supplement/history.json').read_text())
        assert sum(item['stage']=='Adam' for item in hist)==1000
        assert sum(item['stage']=='补充Adam' for item in hist)==6000
        assert failed['pre_supplement']['outer_iterations']==400 and failed['supplement_outer_iterations']==60 and failed['reason']=='NOT_CONVERGED'
        with np.load(root/'supplement/parameters.npz',allow_pickle=False) as p:
            _,_,_,l=trajectory(torch.tensor(phi),torch.tensor(a),torch.tensor(p['U']),torch.tensor(p['V']))
            np.testing.assert_allclose(l.numpy(),np.load(root/'supplement/direction.npy'),rtol=0,atol=1e-14)
    print('PASS：合成优化器预算、探针恢复、控制变量和边界拒绝。')

if __name__=='__main__': main()
