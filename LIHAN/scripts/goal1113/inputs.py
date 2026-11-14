"""训练输入白名单及 E4 闭式先验核验。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

BASE = Path('output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角/fitting')
OBS = Path('output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz')
SOURCE = Path('data/LH-data/static/LH-static-jumpingjacks-ELDL-V3')
RUNTIME = Path('output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/source_all')
MODEL = Path('output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp')
PLAN = Path('LIHAN/DOC/规划/1113-01-T01-B引导环境贴图初始化Stage2规划.md')
EXPECTED = {OBS:'eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47',
            PLAN:'9dc58c5bb4af6da99d2f4dccb0ac696b85f94d15dcbfa46e6b42b74c62d58ff8',
            RUNTIME/'rgb_only_manifest.json':'12bb327641df0d0221188416e3f4eb078bc4fe0deb27ec30446f8c1a1eb60e72',
            RUNTIME/'transforms_all.json':'8db4bdab9fc12a4cb7e043019e081f800346debd97413e973c2124a5423cd90e'}
KEYS = ('frames','gaussian_ids','xyz','frame_xyz','normals','intensity','weights','rgb','projection_uv','sign_consistency')


def sha(path):
    if 'lights.json' in str(path).lower():
        raise ValueError('禁止读取光照真值')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''): h.update(buf)
    return h.hexdigest()


def whitelist():
    if (RUNTIME/'rgba').exists(): raise ValueError('rgba 自动覆盖风险')
    paths = list(EXPECTED) + [BASE/'support.npz',BASE/'predictions.npz',BASE/'photometric_diagnostics.npz',
             BASE/'runs/a1e+00_e1e+01_full/direction.npy',
             MODEL/'point_cloud/iteration_35000/point_cloud.ply',MODEL/'deform/iteration_35000/deform.pth',
             SOURCE/'camera.json']
    paths += [RUNTIME/f'images/image_{i:04d}.png' for i in range(1,151)]
    paths += [SOURCE/f'image/image_{i:04d}.png' for i in range(1,151)]
    paths += [Path(p) for p in ('LIHAN/scripts/goal1113/__init__.py','LIHAN/scripts/goal1113/physics.py',
              'LIHAN/scripts/goal1113/optimizer.py','LIHAN/scripts/goal1113/inputs.py',
              'LIHAN/scripts/goal1113/fit.py','LIHAN/scripts/goal1113/render.py','LIHAN/scripts/goal1113/supplement.py',
              'LIHAN/scripts/goal1113/figures.py','LIHAN/scripts/goal1113/freeze.py','LIHAN/scripts/goal1113/evaluate.py',
              'LIHAN/scripts/goal1113/run.py','LIHAN/scripts/goal1113/selftest.py',
              'LIHAN/scripts/static_rgb_light_extract.py','LIHAN/scripts/static_rgb_light_fit_r3.py',
              'scene/dataset_readers.py','utils/camera_utils.py','scene/deform_model.py',
              'gaussian_renderer/__init__.py','utils/general_utils.py')]
    hashes = {str(p):sha(p) for p in paths}
    for p,h in EXPECTED.items():
        if hashes[str(p)] != h: raise ValueError(f'哈希不符：{p}')
    manifest = json.loads((RUNTIME/'rgb_only_manifest.json').read_text())
    if manifest['gt_read'] or manifest['original_alpha_used']: raise ValueError('输入边界不符')
    for i in range(1,151):
        raw = f'image/image_{i:04d}.png'
        runtime = f'images/image_{i:04d}.png'
        if hashes[str(SOURCE/raw)] != manifest['raw_rgb_files_sha256'][Path(raw).name]: raise ValueError(raw)
        if hashes[str(RUNTIME/runtime)] != manifest['files'][runtime]: raise ValueError(runtime)
    if hashes[str(SOURCE/'camera.json')] != manifest['camera_sha256']: raise ValueError('camera.json')
    return hashes


def load_observations():
    from LIHAN.scripts.static_rgb_light_fit_r3 import kkt
    with np.load(OBS,allow_pickle=False) as z: d={k:z[k] for k in KEYS}
    n=d['normals'].astype(np.float64); w=d['weights'].astype(np.float64)
    rgb=d['rgb'].astype(np.float64); y=d['intensity'].astype(np.float64)
    if not np.array_equal(d['frames'],np.arange(1,151)): raise ValueError('帧号不完整')
    if not np.isfinite(w).all() or (w<0).any() or (w.sum()<=0): raise ValueError('无效权重')
    valid=w>0
    if not np.isfinite(rgb[valid]).all() or not np.isfinite(y[valid]).all(): raise ValueError('有效观测非有限')
    rgb=np.where(valid[...,None],rgb,0.)
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    intensity=np.where(valid,y,0.)
    err=np.max(np.abs((linear@np.array([.2126,.7152,.0722]))[valid]-intensity[valid]))
    if err>1e-5: raise ValueError('亮度不一致')
    if not np.isfinite(n).all() or not np.isfinite(d['frame_xyz']).all(): raise ValueError('几何非有限')
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            q20=np.nanquantile(y,.2,axis=0); q95=np.nanquantile(y,.95,axis=0)
    eligible=(np.isfinite(y).sum(0)>=8)&(q95-q20>.015)
    mean_n=n.mean(0) if n.ndim==3 else n
    xyz=d['xyz']; body=(xyz[:,2]>=.05)&(abs(mean_n[:,2])<=.85)
    cells=np.floor(xyz*20).astype(np.int64)
    hval=cells[:,0]*73856093 ^ cells[:,1]*19349663 ^ cells[:,2]*83492791
    held=hval%5==0
    fi=np.flatnonzero(eligible&body&~held)
    support=dict(body_fit=fi,body_val=np.flatnonzero(eligible&body&held),all_val=np.flatnonzero(eligible&held),
                 split_A=fi[(hval[fi]//5)%2==0],split_B=fi[(hval[fi]//5)%2!=0],
                 gaussian_ids=d['gaussian_ids'],hval=hval,body_mask=body)
    if tuple(len(support[k]) for k in ('body_fit','body_val','split_A','split_B')) != (2683,648,1450,1233): raise ValueError('支持数量不符')
    with np.load(BASE/'support.npz',allow_pickle=False) as old:
        for k,v in support.items():
            if not np.array_equal(v,old[k]): raise ValueError('支持不符：'+k)
    with np.load(BASE/'predictions.npz',allow_pickle=False) as z: l=z['E4'].astype(np.float64)
    ref=np.load(BASE/'runs/a1e+00_e1e+01_full/direction.npy',allow_pickle=False)
    if not np.array_equal(l,ref) or np.max(np.abs(np.linalg.norm(l,axis=1)-1))>1e-8: raise ValueError('E4方向不一致')
    nt=n if n.ndim==3 else np.broadcast_to(n[None],(150,*n.shape))
    x=np.maximum(np.einsum('tgc,tc->tg',nt,l),0)
    rr,bb=kkt(torch.from_numpy(x),torch.from_numpy(intensity),torch.from_numpy(w))
    rr,bb=rr.numpy(),bb.numpy()
    fi=support['body_fit']
    with np.load(BASE/'photometric_diagnostics.npz',allow_pickle=False) as z:
        if not np.array_equal(z['body_fit'],fi) or max(np.max(abs(rr[fi]-z['rho'])),np.max(abs(bb[fi]-z['b'])))>1e-10: raise ValueError('E4系数不一致')
    missing = {k:int((~np.isfinite(d[k])).sum()) for k in ('rgb','intensity','projection_uv')}
    return dict(raw=d,n=nt,w=w,target=linear,intensity=intensity,support=support,direction=l,rho_y=rr,b=bb,consistency=err,
                missing=missing, valid_target_change_max=0.)
