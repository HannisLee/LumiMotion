"""T02 预注册入口。完整合成自检通过前禁止真实数据运行。"""
import argparse
import ast
import contextlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest

ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/han.li/reproduce/LumiMotion-stage2-perlight')
OBS_HASH='eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47'
PLAN_HASH='795b14b37e8ad7292710df1fc75ccc5df94920ba1ba26932f28c678680cc3bbe'
T01_EXTRACT_HASH='90317f8ce41237ecab6ad6db24b2f1bfab2a11b0b7184008ea280cd1ce769f7f'
T01_CORE_HASH='caca2fb62bc81ea3e1a80f8e70274f1699333d5b76e07a12d932957dac39d138'
EXPECTED=dict(baseline=PROJECT/'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角', observations=PROJECT/'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz',source=PROJECT/'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3',model=PROJECT/'output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp',plan=PROJECT/'LIHAN/DOC/规划/1103-02-T02低频轨迹切空间保护SH背景代理规划.md')


def code_hashes():
    todo=list((ROOT/'LIHAN/scripts').glob('static_rgb_light_shbg_t02*.py')); seen=set()
    while todo:
        path=todo.pop().resolve()
        if path in seen: continue
        seen.add(path)
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
                names=[node.module]+[node.module+'.'+n.name for n in node.names]
            for name in names:
                candidate=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if candidate.is_file() and candidate.resolve() not in seen: todo.append(candidate)
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(seen)}


def allowed_baseline(b):
    f=b/'fitting'
    return [f/'support.npz',f/'bootstrap_103108_sample.npy',f/'bootstrap_103109_sample.npy',f/'runs/E3/direction.npy']+[f/'runs'/f'a1e+00_e1e+01_{k}'/'direction.npy' for k in ('full','A','B','boot103108','boot103109')]+[b/'frozen_code'/s for s in ('static_rgb_light_fit_r3.py','static_rgb_light_fit_r5b.py')]


def inventory(a):
    derived=a.observations.resolve().parents[1]/'source_all'; b=a.baseline.resolve()
    files=[a.plan.resolve(),a.observations.resolve(),a.source.resolve()/'camera.json',derived/'transforms_all.json',derived/'rgb_only_manifest.json',derived.parent/'preparation_commands.json',a.model.resolve()/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply',b/'freeze_manifest.json']+allowed_baseline(b)
    files += [ROOT/p for p in code_hashes()]
    return sorted(set(files))


def lock(a):
    if any(getattr(a,k).resolve()!=v.resolve() for k,v in EXPECTED.items()) or a.iteration!=35000: raise ValueError('固定输入路径不一致')
    if digest(a.observations)!=OBS_HASH or digest(a.plan)!=PLAN_HASH: raise ValueError('观测或规划哈希不一致')
    if digest(ROOT/'LIHAN/scripts/static_rgb_light_shbg_t01_extract.py')!=T01_EXTRACT_HASH or digest(ROOT/'LIHAN/scripts/static_rgb_light_shbg_t01_core.py')!=T01_CORE_HASH: raise ValueError('T01 提取代码哈希不一致')
    b=a.baseline.resolve(); manifest=json.loads((b/'freeze_manifest.json').read_text())['files']
    for path in allowed_baseline(b):
        key=str(path.relative_to(PROJECT))
        if key not in manifest or digest(path)!=manifest[key]: raise ValueError('白名单冻结文件哈希不一致: '+key)
    for script in ('static_rgb_light_fit_r3.py','static_rgb_light_fit_r5b.py'):
        if digest(ROOT/'LIHAN/scripts'/script)!=digest(b/'frozen_code'/script): raise ValueError('导入 E4 代码哈希不一致')
    files=inventory(a)
    if not all(p.is_file() for p in files): raise FileNotFoundError('锁定输入或代码缺失')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if any(p.name not in {'allocation.json','command.txt','environment.json','run.log','launcher.pid'} for p in out.iterdir()): raise FileExistsError('输出目录已使用')
    rec=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),argv=sys.argv,files={str(p):dict(sha256=digest(p),size=p.stat().st_size) for p in files},gt_read=False)
    (out/'input_lock.json').write_text(json.dumps(rec,indent=2)); return rec


def verify(a):
    rec=json.loads((a.output/'input_lock.json').read_text()); files=inventory(a)
    if set(map(str,files))!=set(rec['files']): raise ValueError('输入清单变化')
    for p in files:
        if digest(p)!=rec['files'][str(p)]['sha256'] or p.stat().st_size!=rec['files'][str(p)]['size']: raise ValueError('输入或代码哈希变化: '+str(p))
    if digest(a.observations)!=OBS_HASH or digest(a.plan)!=PLAN_HASH: raise ValueError('观测或规划哈希变化')


def self_test(out, plan):
    from LIHAN.scripts import static_rgb_light_shbg_t02_selftest as test
    if plan.resolve()!=EXPECTED['plan'].resolve() or digest(plan)!=PLAN_HASH: raise ValueError('规划路径或哈希不一致')
    out=out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if any(p.name not in {'allocation.json','command.txt','environment.json','run.log','launcher.pid'} for p in out.iterdir()): raise FileExistsError('自检目录已使用')
    receipt=dict(status='FAILED',gt_read=False,argv=sys.argv,code_sha256=code_hashes(),plan_sha256=PLAN_HASH)
    try:
        with (out/'selftest.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log): test.main()
        receipt['status']='PASS'
    except Exception:
        with (out/'selftest.log').open('a') as log: traceback.print_exc(file=log)
        raise
    finally: (out/'receipt.json').write_text(json.dumps(receipt,indent=2))


def check_selftest(path):
    if path is None: raise ValueError('缺少合成自检目录')
    receipt=json.loads((path/'receipt.json').read_text())
    if receipt.get('status')!='PASS' or receipt.get('code_sha256')!=code_hashes() or receipt.get('plan_sha256')!=PLAN_HASH: raise ValueError('自检失败或代码、规划变化')


def check_smoke(a):
    if a.smoke_dir is None or a.smoke_dir.resolve()==a.output.resolve(): raise ValueError('缺少独立冒烟')
    smoke_lock=json.loads((a.smoke_dir/'input_lock.json').read_text()); formal_lock=json.loads((a.output/'input_lock.json').read_text())
    if smoke_lock['files']!=formal_lock['files']: raise ValueError('冒烟输入不同')
    smoke=json.loads((a.smoke_dir/'fit/fit_results.json').read_text())
    result=smoke.get('C01',{})
    if result.get('failure')!='SH_DEGENERATE' and not (result.get('failure') is None and result.get('full',{}).get('converged')): raise ValueError('冒烟未通过或异常')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['self-test','lock','extract','fit','choose','figures','freeze']); p.add_argument('--output',type=Path,required=True)
    for name in ('baseline','observations','source','model','plan','self-test-dir','smoke-dir'): p.add_argument('--'+name,type=Path)
    p.add_argument('--iteration',type=int); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if socket.gethostname().split('.')[0]!='ushas' or os.environ.get('CONDA_DEFAULT_ENV')!='lumimotion-ushas': raise RuntimeError('服务器或 Conda 环境不符')
    if any(getattr(a,k) is None for k in EXPECTED) or a.iteration!=35000: p.error('须指定全部固定输入及 iteration 35000')
    if a.command=='self-test': self_test(a.output,a.plan); return
    check_selftest(a.self_test_dir)
    if a.command=='lock': lock(a); return
    verify(a)
    with (a.output/'commands.jsonl').open('a') as log:
        log.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),argv=sys.argv),ensure_ascii=False)+'\n')
    if a.command=='extract':
        import numpy as np
        from LIHAN.scripts.static_rgb_light_shbg_t01_extract import extract
        with np.load(a.baseline/'fitting/support.npz',allow_pickle=False) as s: ids=s['gaussian_ids'].copy()
        extract(a.observations,a.source,a.model,a.iteration,a.output/'extraction',OBS_HASH,ids)
    elif a.command=='fit':
        from LIHAN.scripts.static_rgb_light_shbg_t02_fit import fit_candidates
        if not a.smoke:
            check_smoke(a)
            from LIHAN.scripts.static_rgb_light_shbg_t02_choose import baseline_scores
            baseline_scores(a.observations,a.baseline,a.output/'baseline_scoring')
        start=time.monotonic(); result=fit_candidates(a.observations,a.baseline,a.output/'extraction/sh_values.npz',a.output/'extraction/extraction.json',a.output/'fit',a.smoke)
        if a.smoke:
            import resource,torch
            (a.output/'smoke_result.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,fit=result,gt_read=False),indent=2))
    elif a.command=='choose':
        check_smoke(a)
        from LIHAN.scripts.static_rgb_light_shbg_t02_choose import choose
        choose(a.observations,a.baseline,a.output/'extraction/sh_values.npz',a.output/'fit',a.output/'selection')
    elif a.command=='figures':
        check_smoke(a)
        from LIHAN.scripts.static_rgb_light_shbg_t02_figures import figures
        figures(a.observations,a.baseline,a.output/'fit',a.output/'selection',a.output/'figures')
    elif a.command=='freeze':
        check_smoke(a)
        from LIHAN.scripts.static_rgb_light_shbg_t02_freeze import freeze
        freeze(a.output)

if __name__=='__main__': main()
