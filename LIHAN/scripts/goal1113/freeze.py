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
    for name,digest in manifest['inputs'].items():
        p=Path(name)
        if p.name.lower()=='lights.json': raise ValueError('冻结清单包含 GT')
        if sha(p)!=digest: raise ValueError('冻结输入变化：'+name)
    return manifest


def run(root):
    root=Path(root); status=json.loads((root/'fitting/status.json').read_text())
    if status.get('selected') not in (None,'C3'): raise ValueError('仅完整C3可以为主结果')
    if status.get('selected')=='C3':
        if status['status']!='FIT_COMPLETE': raise ValueError('C3 状态不符')
        for name,step in [('B',2000),('C1',500),('C2',1500),('C3',1000)]:
            diag=root/f'fitting/{name}/diagnostic.json'
            if not diag.is_file() or not json.loads(diag.read_text())['pass_guard']: raise ValueError('阶段防护未通过：'+name)
            if not (root/f'fitting/{name}/checkpoint_{step:04d}.pt').is_file(): raise ValueError('阶段终点缺失：'+name)
    elif status['status'] not in ('INIT_GUARD_FAILED','PRIOR_GUARD_CONFLICT'):
        raise ValueError('无效失败状态')
    elif status['status']=='INIT_GUARD_FAILED':
        stage=status.get('stage')
        if stage not in ('B','C1','C2','C3'): raise ValueError('失败阶段缺失')
        diagnostic=root/f'fitting/{stage}/diagnostic.json'
        if not diagnostic.is_file() or json.loads(diagnostic.read_text())['pass_guard']: raise ValueError('失败防护证据缺失')
    required=['inputs/manifest.json','inputs/config.json','inputs/prior.npz','fitting/status.json',
              'render/summary.json','render/total.mp4','fit_execution.json','render_execution.json']
    required += [f'render/linear_{t:04d}.npz' for t in range(1,151)]
    required += [f'figures/{name}.png' for name in ('01-rgb','02-components','03-env-b','04-trajectory')]
    for name in required:
        if not (root/name).is_file(): raise FileNotFoundError(name)
    for mode in ('fit','render'):
        receipt=json.loads((root/f'{mode}_execution.json').read_text())
        if receipt.get('schema_version')!=1 or receipt.get('mode')!=mode or receipt.get('exit_code')!=0 or not receipt.get('command') or not receipt.get('argv'):
            raise ValueError('执行回执不完整：'+mode)
        if receipt.get('log_path') != ('run.log' if mode=='fit' else 'render.log'):
            raise ValueError('日志名称不符：'+mode)
        if not all(receipt.get(k) for k in ('cwd','hostname','pid','started_utc','finished_utc')): raise ValueError('回执元数据不完整')
        p=(root/receipt['log_path']).resolve()
        if root.resolve() not in p.parents or not p.is_file(): raise ValueError('日志路径非法：'+mode)
    inputs=json.loads((root/'inputs/manifest.json').read_text())['hashes']
    if not inputs: raise ValueError('空输入清单')
    for name,digest in inputs.items():
        p=Path(name)
        if p.name.lower()=='lights.json' or sha(p)!=digest: raise ValueError('输入哈希不一致：'+name)
    files={}
    for folder in ('inputs','fitting','render','figures'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file(): files[str(p.relative_to(root))]=sha(p)
    out=root/'freeze'; out.mkdir(exist_ok=False)
    for mode in ('fit','render'):
        receipt_path=root/f'{mode}_execution.json'; receipt=json.loads(receipt_path.read_text())
        files[receipt_path.name]=sha(receipt_path)
        p=(root/receipt['log_path']).resolve(); files[str(p.relative_to(root.resolve()))]=sha(p)
    manifest=dict(gt_read=False,selected=status.get('selected'),status=status['status'],files=files,inputs=inputs)
    with (out/'manifest.json').open('x') as f: json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    return manifest
