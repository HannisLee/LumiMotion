"""冻结后的独立一次性评测；训练入口不导入此模块。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from . import freeze, physics

SOURCE=Path('data/LH-data/static/LH-static-jumpingjacks-ELDL-V3')


def metrics(pred, truth):
    err=np.rad2deg(np.arccos(np.clip((pred*truth).sum(-1),-1,1)))
    az=np.rad2deg(np.angle(np.exp(1j*(np.arctan2(pred[:,1],pred[:,0])-np.arctan2(truth[:,1],truth[:,0])))))
    el=np.rad2deg(np.arcsin(np.clip(pred[:,2],-1,1))-np.arcsin(np.clip(truth[:,2],-1,1)))
    return dict(mean_deg=float(err.mean()),p50_deg=float(np.percentile(err,50)),p90_deg=float(np.percentile(err,90)),
        within_5deg_count=int((err<=5).sum()),azimuth_signed_mean_deg=float(az.mean()),azimuth_mae_deg=float(abs(az).mean()),
        elevation_signed_bias_deg=float(el.mean()),elevation_mae_deg=float(abs(el).mean()),
        angle_deg=err.tolist(),azimuth_deg=az.tolist(),elevation_deg=el.tolist())


def run(root):
    root=Path(root); manifest=freeze.verify(root)
    out=root/'evaluation'; out.mkdir(exist_ok=False)
    if manifest['selected'] is None:
        result=dict(status='FAILED',reason='未完成C3',gt_read=False)
    else:
        ck=torch.load(root/'fitting/C3/checkpoint_1000.pt',map_location='cpu',weights_only=False)
        pred=physics.direction(ck['params']['phi'],ck['params']['a']).numpy()
        with np.load(root/'inputs/prior.npz',allow_pickle=False) as z: e4=z['direction'].copy()
        payload=(SOURCE/'lights.json').read_bytes(); record=json.loads(payload)
        if set(record)!={f'{i:04d}' for i in range(1,151)}: raise ValueError('GT 帧集合不符')
        truth=[]
        for i in range(1,151):
            entry=record[f'{i:04d}']
            if entry.get('source_type')!='SUN': raise ValueError('光源类型不符')
            v=np.asarray(entry['light_dir_world'],dtype=np.float64)
            if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-12: raise ValueError('GT方向无效')
            truth.append(v/np.linalg.norm(v))
        truth=np.stack(truth); main=metrics(pred,truth); base=metrics(e4,truth)
        result=dict(status='PASS' if main['mean_deg']<4.248 else 'FAILED',gt_read=True,
                    gt_sha256=hashlib.sha256(payload).hexdigest(),main=main,E4=base,
                    mean_difference_vs_E4=main['mean_deg']-base['mean_deg'])
    with (out/'metrics.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    run(args.output)


if __name__=='__main__': main()
