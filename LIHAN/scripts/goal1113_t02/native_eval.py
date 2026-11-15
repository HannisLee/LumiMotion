"""T02-native 冻结 + 冻结后一次 GT 评测（独立入口，训练不导入）。
用法：python -m LIHAN.scripts.goal1113_t02.native_eval --output <ROOT> 。
先写 freeze/manifest.json（SHA256，gt_read=false），再读一次 lights.json；evaluation 目录 exist_ok=False 保证只评一次。"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from LIHAN.scripts.goal1113 import physics
from LIHAN.scripts.goal1113.evaluate import metrics, SOURCE


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def run(root):
    root = Path(root)
    status = json.loads((root/'fitting/status.json').read_text())
    rec = json.loads((root/'fit_execution.json').read_text())
    if status.get('selected') != 'C3' or rec.get('exit_code') != 0: raise ValueError('C3 未完整完成')
    fr = root/'freeze'; fr.mkdir(exist_ok=False)
    files = {str(p.relative_to(root)): sha(p) for d in ('inputs','fitting') for p in sorted((root/d).rglob('*')) if p.is_file()}
    for n in ('run.log','fit_execution.json'): files[n] = sha(root/n)
    (fr/'manifest.json').write_text(json.dumps(dict(gt_read=False, selected='C3', files=files), indent=2, ensure_ascii=False)+'\n')
    out = root/'evaluation'; out.mkdir(exist_ok=False)
    ck = torch.load(root/'fitting/C3/checkpoint_1000.pt', map_location='cpu', weights_only=False)
    pred = physics.direction(ck['params']['phi'], ck['params']['a']).numpy()
    with np.load(root/'inputs/prior.npz', allow_pickle=False) as z: e4 = z['direction'].copy()
    payload = (SOURCE/'lights.json').read_bytes(); record = json.loads(payload)
    if set(record) != {f'{i:04d}' for i in range(1,151)}: raise ValueError('GT 帧集合不符')
    truth = []
    for i in range(1,151):
        e = record[f'{i:04d}']
        if e.get('source_type') != 'SUN': raise ValueError('光源类型不符')
        v = np.asarray(e['light_dir_world'], dtype=np.float64)
        if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12: raise ValueError('GT方向无效')
        truth.append(v/np.linalg.norm(v))
    truth = np.stack(truth); main = metrics(pred, truth); base = metrics(e4, truth)
    res = dict(status='PASS' if main['mean_deg'] < 4.248 else 'FAILED', gt_read=True, gt_sha256=hashlib.sha256(payload).hexdigest(),
               main=main, E4=base, mean_difference_vs_E4=main['mean_deg']-base['mean_deg'])
    with (out/'metrics.json').open('x') as f: json.dump(res, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: res['main'][k] for k in ('mean_deg','p50_deg','p90_deg','within_5deg_count')}), res['status'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); run(ap.parse_args().output)
