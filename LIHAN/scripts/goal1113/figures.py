"""生成四类无 GT 可视化；目检结论由执行方独立记录。"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import fit, inputs as I, physics as P


def run(root,device):
    root=Path(root); out=root/'figures'; out.mkdir(exist_ok=False)
    status=json.loads((root/'fitting/status.json').read_text()); stage=status.get('selected') or status.get('stage') or 'A'
    checkpoint=(root/'fitting'/stage/'checkpoint.pt') if stage=='A' else sorted((root/'fitting'/stage).glob('checkpoint_*.pt'))[-1]
    state=torch.load(checkpoint,map_location=device,weights_only=False); params=state['params']
    src,data,_,meta=fit.setup(device)
    for k in params: params[k]=params[k].to(device)
    t=74
    source=cv2.imread(str(I.RUNTIME/f'images/image_{t+1:04d}.png'),cv2.IMREAD_UNCHANGED)
    if source is None: raise FileNotFoundError('原RGB图')
    source=source[...,2::-1].astype(np.float64)/(65535 if source.dtype==np.uint16 else 255)
    with np.load(root/f'render/linear_{t+1:04d}.npz') as z: image=z['total']
    if source.shape[:2]!=image.shape[:2]: source=cv2.resize(source,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_AREA)
    linear=np.where(source<=.04045,source/12.92,((source+.055)/1.055)**2.4)
    residual=image-linear
    fig,ax=plt.subplots(2,3,figsize=(12,8))
    from .render import srgb
    ax[0,0].imshow(np.clip(source,0,1)); ax[0,0].set_title('真实 RGB')
    ax[0,1].imshow(np.clip(srgb(image),0,1)); ax[0,1].set_title('合成 RGB')
    ax[0,2].imshow(np.clip(abs(residual)*3,0,1)); ax[0,2].set_title('线性残差 ×3')
    for i,c in enumerate('RGB'):
        ax[1,i].imshow(residual[...,i],cmap='coolwarm',vmin=-.2,vmax=.2); ax[1,i].set_title(c+' 通道残差')
    for a in ax.flat:a.axis('off')
    fig.tight_layout(); fig.savefig(out/'01-rgb.png',dpi=130); plt.close(fig)
    with np.load(root/f'render/linear_{t+1:04d}.npz') as z: maps={k:z[k] for k in ('albedo','normal','direct','static')}
    fig,ax=plt.subplots(2,3,figsize=(12,8))
    for i,(k,v) in enumerate(maps.items()):
        ax.flat[i].imshow(np.clip(srgb(v) if k!='normal' else v,0,1)); ax.flat[i].set_title(k); ax.flat[i].axis('off')
    diag=json.loads((root/'fitting'/stage/'diagnostic.json').read_text())
    ax.flat[4].plot(diag['energy_frames']); ax.flat[4].set_title('静态能量占比')
    ax.flat[5].axis('off'); fig.tight_layout(); fig.savefig(out/'02-components.png',dpi=130); plt.close(fig)
    with torch.no_grad():
        l=P.direction(params['phi'],params['a']); parts=[]
        for j in range(0,data['n'].shape[1],128):
            sl=slice(j,min(j+128,data['n'].shape[1])); basis=(data['basis'][0][:,sl],data['basis'][1][:,sl])
            _,s=P.components(data['n'][:,sl],data['v'][:,sl],params['rho'][sl],params['env'],l,meta['K'],basis)
            parts.append((data['w'][:,sl]*(s@s.new_tensor(P.Y))).sum(0)/data['w'][:,sl].sum(0))
        sbar=torch.cat(parts).cpu().numpy(); b=data['b'].cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(12,8)); ax[0,0].imshow(np.clip(srgb(params['env'].detach().cpu().numpy().reshape(8,16,3)),0,1)); ax[0,0].set_title(stage+' 环境图')
    for i,name in enumerate(('A','B','C1','C2','C3')):
        d=root/'fitting'/name/'diagnostic.json'
        if d.exists():
            z=json.loads(d.read_text()); ax[0,1].scatter(i,z['energy'],label=name); ax[0,2].scatter(i,z['b_mean'])
    ax[0,1].set_title('阶段静态能量'); ax[0,2].set_title('阶段 b 残差')
    ax[1,0].scatter(b,sbar,s=2,alpha=.3); ax[1,0].set_xlabel('b'); ax[1,0].set_ylabel('时间平均 S')
    ax[1,1].hist((sbar-b)/meta['scale'],bins=50); ax[1,1].set_title('带符号 b 残差 / s'); ax[1,2].axis('off')
    fig.tight_layout(); fig.savefig(out/'03-env-b.png',dpi=130); plt.close(fig)
    l=l.cpu().numpy(); fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].plot(np.rad2deg(np.unwrap(np.arctan2(l[:,1],l[:,0])))); ax[0,0].set_title('方位角')
    ax[0,1].plot(np.rad2deg(np.arcsin(np.clip(l[:,2],-1,1)))); ax[0,1].set_title('仰角')
    for name in ('B','C1','C2','C3'):
        p=root/'fitting'/name/'history.json'
        if p.exists(): ax[1,0].plot([z['total'] for z in json.loads(p.read_text())],label=name)
    ax[1,0].legend(); ax[1,0].set_title('目标收敛')
    ax[1,1].plot(diag['energy_frames'],label='环境占比')
    ax[1,1].axhline(max(0,meta['e0']-.2),color='r'); ax[1,1].axhline(min(.65,meta['e0']+.2),color='r')
    ax[1,1].set_title(f"能量阈值；b均值={diag['b_mean']:.3f} (≤.10)，P90={diag['b_p90']:.3f} (≤.25)")
    fig.tight_layout(); fig.savefig(out/'04-trajectory.png',dpi=130); plt.close(fig)
