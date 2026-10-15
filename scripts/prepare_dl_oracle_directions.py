"""仅为显式参考组生成训练时间GT灯向与开发时间插值；不读取其他监督。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.fit_dl_shadow_tracks import interpolate

p=argparse.ArgumentParser(description=__doc__)
for key in ['estimate','gt-lights','output']:p.add_argument('--'+key,required=True)
a=p.parse_args();out=Path(a.output);assert not out.exists()
base=json.loads(Path(a.estimate).read_text());assert not base.get('gt_read',False)
gt=json.loads(Path(a.gt_lights).read_text())['frames'];nodes=[]
for index in base['train_indices']:
    key=str(int(base['frames'][index].rsplit('_',1)[1])).zfill(4);vectors=[]
    for lamp in gt[key]['lights']:
        assert lamp['type']=='directional' and 'surface toward light' in lamp['direction_convention']
        vector=np.asarray(lamp['light_dir_world'],dtype=np.float64)
        assert np.isfinite(vector).all() and np.linalg.norm(vector)>0
        vectors.append(vector/np.linalg.norm(vector))
    assert vectors and all(np.allclose(vectors[0],v,atol=1e-5) for v in vectors)
    nodes.append(vectors[0])
directions=interpolate(torch.tensor(base['times']),torch.tensor(base['times'])[base['train_indices']],torch.tensor(np.asarray(nodes),dtype=torch.float32))
result={k:base[k] for k in ['frames','times','train_indices','held_indices','cache_sha256']}
result.update(directions=directions.tolist(),gt_read=True,gt_direction_reference=True,control='oracle',protocol='仅训练120时间读取GT方向；15开发保留时间按训练节点归一化线性插值；不使用GT强度/材质/几何/运动',sources={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [a.estimate,a.gt_lights,__file__]})
out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(dict(status='PASS',train=len(base['train_indices']),held=len(base['held_indices']),output=str(out)),ensure_ascii=False))
