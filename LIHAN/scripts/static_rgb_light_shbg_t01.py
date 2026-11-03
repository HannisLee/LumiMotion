"""T01 预注册流程入口：只读取冻结观测、相机、SH 与无 GT 基线。"""
import argparse
import json
import os
import socket
from pathlib import Path
import sys
import time
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest

OBS_HASH='eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47'
PLAN_HASH='a71d513f797465e7509a3b5144aef600adf3152e902986dd1b56cb52bd30d5c5'


def inventory(a):
    root=Path(__file__).resolve().parents[2]; baseline=a.baseline.resolve(); model=a.model.resolve(); source=a.source.resolve()
    files=[a.plan.resolve(),a.observations.resolve(),source/'transforms_all.json',model/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply',model/'deform'/f'iteration_{a.iteration}'/'deform.pth',baseline/'freeze_manifest.json',root/'utils'/'sh_utils.py']
    files += list((baseline/'fitting').rglob('*'))+list((baseline/'frozen_code').rglob('*'))
    for folder in ('images','rgba'):
        if (source/folder).exists(): files += [p for p in (source/folder).rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.exr')]
    files += list((root/'LIHAN'/'DOC'/'修改').glob('*T01*.md')) + list((root/'LIHAN'/'DOC'/'训练').glob('*T01*.md'))
    files += list((root/'LIHAN'/'scripts').glob('static_rgb_light_shbg_t01*.py'))
    files += [root/'LIHAN'/'scripts'/'static_rgb_light_fit_r3.py',root/'LIHAN'/'scripts'/'static_rgb_light_fit_r5b.py']
    return sorted(set(p.resolve() for p in files if p.is_file()))


def lock(a):
    if digest(a.observations)!=OBS_HASH or digest(a.plan)!=PLAN_HASH: raise ValueError('观测或规划 SHA256 不匹配')
    baseline=a.baseline.resolve(); root=Path(__file__).resolve().parents[2]
    for script in ('static_rgb_light_fit_r3.py','static_rgb_light_fit_r5b.py'):
        if digest(root/'LIHAN'/'scripts'/script)!=digest(baseline/'frozen_code'/script): raise ValueError('导入的 E4 代码与冻结基线不一致')
    files=inventory(a)
    for folder in ('修改','训练'):
        if not any((root/'LIHAN'/'DOC'/folder).glob('*T01*.md')): raise FileNotFoundError(f'缺少 T01 {folder}文档')
    for required in (a.observations.resolve(),a.plan.resolve(),a.source.resolve()/'transforms_all.json',a.model.resolve()/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply',a.model.resolve()/'deform'/f'iteration_{a.iteration}'/'deform.pth',baseline/'freeze_manifest.json'):
        if required not in files: raise FileNotFoundError(required)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    allowed={'allocation.json','command.txt','environment.json','run.log','launcher.pid'}
    if any(p.name not in allowed or not p.is_file() for p in out.iterdir()): raise FileExistsError('输出目录非空或已分配')
    rec={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'argv':sys.argv,'files':{str(p):{'sha256':digest(p),'size':p.stat().st_size} for p in files},'gt_read':False}
    (out/'input_lock.json').write_text(json.dumps(rec,indent=2)); return rec


def verify(a):
    rec=json.loads((a.output/'input_lock.json').read_text()); current=inventory(a)
    if set(map(str,current))!=set(rec['files']): raise ValueError('输入清单发生变化')
    for p in current:
        if digest(p)!=rec['files'][str(p)]['sha256'] or p.stat().st_size!=rec['files'][str(p)]['size']: raise ValueError(f'输入已变化: {p}')
    if digest(a.observations)!=OBS_HASH or digest(a.plan)!=PLAN_HASH: raise ValueError('观测或规划 SHA256 不匹配')


def code_hashes():
    root=Path(__file__).resolve().parents[2]
    files=list((root/'LIHAN'/'scripts').glob('static_rgb_light_shbg_t01*.py'))
    files += [root/'LIHAN'/'scripts'/'static_rgb_light_fit_r3.py',root/'LIHAN'/'scripts'/'static_rgb_light_fit_r5b.py',root/'utils'/'sh_utils.py']
    return {str(p.resolve()):digest(p) for p in sorted(files)}


def self_test(out):
    import contextlib,traceback
    from LIHAN.scripts import static_rgb_light_shbg_t01_selftest as unit
    from LIHAN.scripts import static_rgb_light_shbg_t01_integration_selftest as integration
    out=out.resolve(); out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'FAILED','gt_read':False,'code_sha256':code_hashes()}
    try:
        with (out/'selftest.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            unit.main(); integration.main()
        receipt['status']='PASS'
    except Exception:
        with (out/'selftest.log').open('a') as log: traceback.print_exc(file=log)
        raise
    finally: (out/'receipt.json').write_text(json.dumps(receipt,indent=2))


def check_selftest(path):
    if path is None: raise ValueError('必须提供独立合成自检目录')
    receipt=json.loads((path/'receipt.json').read_text())
    if receipt.get('status')!='PASS' or receipt.get('code_sha256')!=code_hashes(): raise ValueError('合成自检未通过或代码已变化')


def check_smoke(a):
    if a.smoke_dir is None or a.smoke_dir.resolve()==a.output.resolve(): raise ValueError('正式流程需要独立成功冒烟目录')
    smoke_lock=json.loads((a.smoke_dir/'input_lock.json').read_text()); formal_lock=json.loads((a.output/'input_lock.json').read_text())
    if smoke_lock['files']!=formal_lock['files']: raise ValueError('冒烟和正式流程输入不一致')
    smoke=json.loads((a.smoke_dir/'fit'/'fit_results.json').read_text())
    if set(smoke)!={'C01'} or not smoke['C01'].get('full',{}).get('converged') or smoke['C01'].get('failure') is not None: raise ValueError('真实 C01 冒烟未通过')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['self-test','lock','extract','fit','choose','figures','freeze']); p.add_argument('--output',type=Path,required=True); p.add_argument('--baseline',type=Path); p.add_argument('--observations',type=Path); p.add_argument('--source',type=Path); p.add_argument('--model',type=Path); p.add_argument('--iteration',type=int); p.add_argument('--plan',type=Path); p.add_argument('--self-test-dir',type=Path); p.add_argument('--smoke-dir',type=Path); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if socket.gethostname().split('.')[0]!='ushas' or os.environ.get('CONDA_DEFAULT_ENV')!='lumimotion-ushas': raise RuntimeError('服务器或 Conda 环境不符合预注册规划')
    if a.command=='self-test': self_test(a.output); return
    if any(v is None for v in (a.baseline,a.observations,a.source,a.model,a.iteration,a.plan)) or a.iteration!=35000: p.error('须指定全部固定输入且 iteration 为 35000')
    check_selftest(a.self_test_dir)
    if a.command=='lock': lock(a); return
    verify(a)
    if a.command=='extract':
        import numpy as np
        from LIHAN.scripts.static_rgb_light_shbg_t01_extract import extract
        with np.load(a.baseline/'fitting'/'support.npz',allow_pickle=False) as d: ids=d['gaussian_ids'].copy()
        extract(a.observations,a.source,a.model,a.iteration,a.output/'extraction',OBS_HASH,ids)
    elif a.command=='fit':
        from LIHAN.scripts.static_rgb_light_shbg_t01_fit import fit_candidates
        if not a.smoke: check_smoke(a)
        start=time.monotonic()
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        result=fit_candidates(a.observations,a.baseline,a.output/'extraction'/'sh_values.npz',a.output/'extraction'/'extraction.json',a.output/'fit',a.smoke)
        if a.smoke:
            import resource
            record={'elapsed_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,'fit':result,'gt_read':False}
            (a.output/'smoke_result.json').write_text(json.dumps(record,indent=2))
    elif a.command=='choose':
        from LIHAN.scripts.static_rgb_light_shbg_t01_choose import choose
        from LIHAN.scripts.static_rgb_light_shbg_t01_templates import combine_templates
        check_smoke(a); choose(a.observations,a.baseline,a.output/'extraction'/'sh_values.npz',a.output/'fit',a.output/'selection')
        combine_templates(a.output/'fit',a.output/'selection',a.output/'sh_templates.npz')
    elif a.command=='figures':
        from LIHAN.scripts.static_rgb_light_shbg_t01_figures import figures
        check_smoke(a); figures(a.observations,a.output/'fit',a.output/'selection',a.output/'figures')
    elif a.command=='freeze':
        from LIHAN.scripts.static_rgb_light_shbg_t01_freeze import freeze
        check_smoke(a); freeze(a.output)

if __name__=='__main__': main()
