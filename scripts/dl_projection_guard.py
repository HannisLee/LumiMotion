"""固定T05训练支持的批量投影代理；无GT、无连续阴影导数。"""
import torch
import torch.nn.functional as F

class ProjectionGuard:
    def __init__(self, cache, state, prior, device='cuda'):
        assert state['cache_sha256']==prior['cache_sha256'] and not prior.get('gt_read',False)
        self.ids=torch.where(state['train'])[0].to(device)
        xyz=cache['canonical_xyz'].to(device);positions=cache['positions'].to(device)
        train=state['train'].to(device);valid=cache['valid'].to(device)
        self.origin=state['plane_origin'].to(device);self.normal=state['plane_normal'].to(device);self.basis=state['basis'].to(device)
        extent=prior['scene_extent'];support=valid[train].sum(0)>=10
        motion=(positions[train]-positions[train].mean(0)[None]).square().sum(-1).mean(0).sqrt()
        receiver=support&(((xyz-self.origin)@self.normal).abs()<extent*.01)&(motion<extent*.01)
        uv=(xyz[receiver]-self.origin)@self.basis.T
        lower=uv.amin(0);span=(uv.amax(0)-lower).clamp_min(.01)
        self.lower=lower-.05*span;self.span=span*1.1
        self.occluders=positions[train][:,state['occluder'].to(device)]-self.origin
        self.heights=self.occluders@self.normal
        sample_xy=(positions[train]-self.origin)@self.basis.T
        index=((sample_xy-self.lower)/self.span*128).floor().long()
        self.sample_index=index[...,1].clamp(0,127)*128+index[...,0].clamp(0,127)
        self.positive=state['positive'][state['train']].to(device)
        self.negative=state['negative'][state['train']].to(device)
        self.usable=state['usable'][state['train']].to(device)
        self.reference=state['predicted']['smooth'][state['train']].to(device)

    @torch.no_grad()
    def prediction(self,directions):
        directions=directions[self.ids]
        projected=self.occluders-self.heights[...,None]/(directions@self.normal).clamp_min(.001)[:,None,None]*directions[:,None,:]
        xy=projected@self.basis.T
        index=((xy-self.lower)/self.span*128).floor().long()
        inside=((index>=0)&(index<128)).all(-1)
        flat=index[...,1].clamp(0,127)*128+index[...,0].clamp(0,127)
        grid=torch.zeros((len(directions),128*128),device=directions.device)
        grid.scatter_add_(1,flat,inside.float())
        grid=F.max_pool2d(grid.clamp_max(1).view(-1,1,128,128),3,stride=1,padding=1)[:,0].flatten(1)
        return grid.gather(1,self.sample_index)

    @torch.no_grad()
    def cost(self,directions):
        pred=self.prediction(directions)
        error=.5*((1-pred)*self.positive).sum(-1)/self.positive.sum(-1).clamp_min(1)+.5*(pred*self.negative).sum(-1)/self.negative.sum(-1).clamp_min(1)
        return float(error[self.usable].mean())

    @torch.no_grad()
    def verify(self,directions):
        pred=self.prediction(directions)
        mismatches=int((pred!=self.reference).sum())
        assert mismatches==0, f'T05批量代理与已存预测不一致：{mismatches}'
        return dict(status='PASS',mismatches=mismatches,training_frames=len(self.ids),cost=self.cost(directions))
