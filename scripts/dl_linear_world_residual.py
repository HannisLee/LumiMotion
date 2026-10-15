"""共享偏移加线性时间残差的明确前向，归一化时间范围[0,1]。"""
import torch
PROTOCOL='correspondence_world_linear_v1'

def linear_residual(coefficients,time):
    assert coefficients.ndim==3 and coefficients.shape[1:]==(2,3)
    t=torch.as_tensor(time,device=coefficients.device,dtype=coefficients.dtype).reshape(())
    assert 0. <= float(t) <= 1.
    return coefficients[:,0]+(t-.5)*coefficients[:,1]

def residual_for_state(state,time):
    r=state['world_residual'].cuda()
    if state['protocol']==PROTOCOL:
        assert state['time_basis']==dict(center=.5,minimum=0.,maximum=1.,kind='constant_plus_linear')
        return linear_residual(r,time)
    assert state['protocol']=='correspondence_world_shared_v1'
    return r

def validate_linear(state):
    r=state['world_residual']
    assert r.shape==(len(state['active_ids']),2,3) and torch.isfinite(r).all()
    assert state['time_basis']==dict(center=.5,minimum=0.,maximum=1.,kind='constant_plus_linear')
    assert max(float(linear_residual(r,t).norm(dim=-1).max()) for t in [0.,1.])<=.00100002
