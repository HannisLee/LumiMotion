"""固定晚期统一步长乘子；从名义lr计算，不修改Adam状态或非活跃组。"""
import math

ACTIVE_GROUPS=('xyz','albedo_dc','opacity','scaling','rotation')


def multiplier(step,start=16000,end=20000):
    if not 0 < start < end:raise ValueError('需正且递增的固定退火边界。')
    return 10.**(-min(1.,max(0.,(step-start)/(end-start))))


class LateLearningRate:
    """包装原step后scheduler；更新后存储的lr严格供下一次optimizer.step使用。"""
    def __init__(self,model,start,end,enabled=True):
        multiplier(0,start,end)  # 构造时核对边界，不等到首步才报错。
        self.model=model;self.start=start;self.end=end;self.enabled=bool(enabled)
        self.original=model.update_learning_rate
        self.base={g['name']:float(g['lr']) for g in model.optimizer.param_groups}
        assert len(self.base)==len(model.optimizer.param_groups), '参数组名称必须唯一。'
        assert set(ACTIVE_GROUPS).issubset(self.base)
        self.last=None

    def factor(self,step):
        return multiplier(step,self.start,self.end) if self.enabled else 1.

    def __call__(self,completed_step):
        inactive={g['name']:float(g['lr']) for g in self.model.optimizer.param_groups if g['name'] not in ACTIVE_GROUPS}
        result=self.original(completed_step)
        factor=self.factor(completed_step+1)
        nominal={g['name']:(float(g['lr']) if g['name']=='xyz' or g['name'] not in ACTIVE_GROUPS
                            else self.base[g['name']]) for g in self.model.optimizer.param_groups}
        for group in self.model.optimizer.param_groups:
            name=group['name']
            if name in ACTIVE_GROUPS:
                group['lr']=nominal[name]*factor
                assert math.isfinite(group['lr']) and group['lr']>0
        assert inactive=={g['name']:float(g['lr']) for g in self.model.optimizer.param_groups if g['name'] not in ACTIVE_GROUPS}
        self.last=dict(completed_step=completed_step,next_step=completed_step+1,multiplier=factor,
                       nominal_next_lrs=nominal,next_lrs={g['name']:float(g['lr']) for g in self.model.optimizer.param_groups})
        return result


def self_check():
    from types import SimpleNamespace
    groups=[dict(name=name,lr=value) for name,value in
            [('xyz',.8),('albedo_dc',.01),('opacity',.05),('scaling',.01),('rotation',.001),('roughness',.123)]]
    model=SimpleNamespace(optimizer=SimpleNamespace(param_groups=groups))
    def original(k):
        groups[0]['lr']=.8/(k+1)
        return groups[0]['lr']
    model.update_learning_rate=original;hook=LateLearningRate(model,16000,20000)
    for k in (15998,15999,16000,17999,19999,20000,23999):
        hook(k)
        expected=1 if k<16000 else 10**(-min((k+1-16000)/4000,1))
        assert abs(groups[0]['lr']-(.8/(k+1))*expected)<1e-15
        assert abs(groups[2]['lr']-.05*expected)<1e-15
        assert groups[-1]['lr']==.123
    # 20k后反复调用仍是base×.1，不会形成.1的逐步连乘；下一步语义无off-by-one。
    for k in range(24000,24010):hook(k)
    assert groups[2]['lr']==.05*.1 and multiplier(16000)==1 and multiplier(20000)==.1
    hook(24009);assert groups[2]['lr']==.05*.1
    # 真实CPU Adam：hook只写lr，不改参数、动量、step或Python/NumPy/Torch RNG。
    import copy,random
    import numpy as np
    import torch
    for enabled in (False,True):
        params=[torch.nn.Parameter(torch.tensor(float(i+1))) for i in range(6)]
        optimizer=torch.optim.Adam([dict(params=[p],name=g['name'],lr=g['lr']) for p,g in zip(params,groups)])
        for p in params:p.grad=torch.ones_like(p)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        actual=SimpleNamespace(optimizer=optimizer)
        def schedule(k):
            optimizer.param_groups[0]['lr']=.8/(k+1)
            return optimizer.param_groups[0]['lr']
        actual.update_learning_rate=schedule;policy=LateLearningRate(actual,8,16,enabled=enabled)
        state=copy.deepcopy(optimizer.state_dict()['state']);values=[p.detach().clone() for p in params]
        rng=(random.getstate(),repr(np.random.get_state()),torch.get_rng_state().clone())
        for k in (6,7,8,14,15,16,17,17):
            policy(k);factor=multiplier(k+1,8,16) if enabled else 1.
            assert optimizer.param_groups[0]['lr']==(.8/(k+1))*factor
            for group in optimizer.param_groups[1:]:
                assert group['lr']==policy.base[group['name']]*(factor if group['name'] in ACTIVE_GROUPS else 1.)
        for key,entry in state.items():
            for name,value in entry.items():assert torch.equal(value,optimizer.state_dict()['state'][key][name])
        assert all(torch.equal(p,value) for p,value in zip(params,values))
        assert random.getstate()==rng[0] and repr(np.random.get_state())==rng[1] and torch.equal(torch.get_rng_state(),rng[2])
    print('PASS：正式及smoke下一步边界、control原语义、五组基准无累乘、inactive/真实Adam/参数/RNG不变。')


if __name__=='__main__':self_check()
