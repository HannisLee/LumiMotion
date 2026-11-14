"""冻结无 GT 拟合和决策产物。"""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify(root):
    root=Path(root); manifest=json.loads((root/'freeze'/'manifest.json').read_text())
    if manifest['gt_read'] is not False: raise ValueError('冻结状态非法')
    for rel,digest in manifest['files'].items():
        p=(root/rel).resolve()
        if root.resolve() not in p.parents or sha(p)!=digest: raise ValueError('冻结文件变化：'+rel)
    return manifest


def run(root):
    root=Path(root); status=json.loads((root/'fitting/status.json').read_text())
    if status.get('selected') not in (None,'C3'): raise ValueError('仅完整C3可以为主结果')
    required=['inputs/manifest.json','inputs/config.json','inputs/prior.npz','fitting/status.json',
              'render/summary.json','render/total.mp4']
    required += [f'render/linear_{t:04d}.npz' for t in range(1,151)]
    required += [f'figures/{name}.png' for name in ('01-rgb','02-components','03-env-b','04-trajectory')]
    for name in required:
        if not (root/name).is_file(): raise FileNotFoundError(name)
    files={}
    for folder in ('inputs','fitting','render','figures'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file(): files[str(p.relative_to(root))]=sha(p)
    out=root/'freeze'; out.mkdir(exist_ok=False)
    manifest=dict(gt_read=False,selected=status.get('selected'),status=status['status'],files=files)
    with (out/'manifest.json').open('x') as f: json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    return manifest
