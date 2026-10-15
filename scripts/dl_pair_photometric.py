"""由固定图像对应建立两端共同反射率消元损失，无GT。"""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F


def collect(frames, train):
    root=Path('output/smoke_test/1015-68-LH-dynamic-DL-T76同源锚点')
    meta=root/'probe/result.json';sources=[meta,Path(__file__)];rows=[];report=[]
    def sample(x,uv):
        h,w=x.shape[-2:];grid=uv.clone();grid[:,0]=(grid[:,0]+.5)*2/w-1;grid[:,1]=(grid[:,1]+.5)*2/h-1
        return F.grid_sample(x.float()[None],grid[None,None],align_corners=False,mode='bilinear')[0,:,0].T
    for j,_ in enumerate(json.loads(meta.read_text())['rows']):
        pp=root/'probe'/f'pair_{j:02d}.npz';hp=root/'homography'/f'residual_{j:02d}.npz';sources.extend([pp,hp]);d=np.load(pp);m=np.load(hp)['moving'];a,b=map(int,d['indices'])
        if not m.any():continue
        assert a in frames and b in frames
        pa=torch.as_tensor(d['pa'][m],device='cuda');pb=torch.as_tensor(d['pb'][m],device='cuda')
        qa,qb=frames[a],frames[b];na=sample(qa['normal'],pa);nb=sample(qb['normal'],pb)
        ya=sample(qa['target'],pa);yb=sample(qb['target'],pb);aa=sample(qa['alpha'],pa);ab=sample(qb['alpha'],pb)
        # 要求双线性插值涉及的像素都在原固定有效掩码中，不按灯向/GT筛选。
        valid=(sample(qa['mask'],pa)[:,0]>.999)&(sample(qb['mask'],pb)[:,0]>.999)&(aa[:,0]>.99)&(ab[:,0]>.99)
        valid &= torch.isfinite(na).all(-1)&torch.isfinite(nb).all(-1)&(na.norm(dim=-1)>.5)&(nb.norm(dim=-1)>.5)
        is_train=a in train and b in train
        report.append(dict(edge=j,a=a,b=b,train=is_train,candidates=int(m.sum()),retained=int(valid.sum())))
        for k in torch.where(valid)[0].tolist():
            rows.append(dict(a=a,b=b,train=is_train,na=F.normalize(na[k],dim=0),nb=F.normalize(nb[k],dim=0),ya=ya[k]/aa[k],yb=yb[k]/ab[k],pa=pa[k],pb=pb[k],edge=j))
    assert rows,'固定有效支持没有对应，不能启动无对应训练'
    pairs={k:torch.tensor([r[k] for r in rows],device='cuda') if k in ['a','b','train','edge'] else torch.stack([r[k] for r in rows]) for k in rows[0]}
    assert pairs['train'].any() and (~pairs['train']).any()
    return pairs,report,sources


def pair_loss(pairs,D,E,train=True):
    m=pairs['train'] if train else ~pairs['train'];a=pairs['a'][m];b=pairs['b'][m]
    sa=E[a]*((pairs['na'][m]*D[a]).sum(-1).clamp_min(0)+.1)
    sb=E[b]*((pairs['nb'][m]*D[b]).sum(-1).clamp_min(0)+.1)
    ya=pairs['ya'][m];yb=pairs['yb'][m]
    rho=(sa[:,None]*ya+sb[:,None]*yb)/(sa.square()+sb.square()).clamp_min(1e-12)[:,None]
    # rho由两次观测共同消元；不是可学习逐帧颜色或无条件物理albedo真值。
    return ((sa[:,None]*rho-ya).double().square()+(sb[:,None]*rho-yb).double().square()).mean()/2
