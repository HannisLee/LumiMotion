"""ELDL/DL完整路线双卡训练、共同冻结后逐预登记检查点评测。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import threading

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'


def write(p,d):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def main(control):
    assert socket.gethostname()=='ushas'
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve()
    plans=json.loads((control/'experiments.json').read_text())
    state=dict(status='RUNNING',started_unix=time.time(),pid=os.getpid(),jobs={})
    write(control/'state.json',state)
    mutex=threading.Lock()
    def persist():
        with mutex:write(control/'state.json',state)
    def job(name,cmd,out,gpu,expected):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),DYNAMIC_V3_REPO_ROOT=str(ROOT),PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
        row=dict(command=cmd,shell=shlex.join(cmd),gpu=gpu,started_unix=time.time(),status='RUNNING')
        state['jobs'][name]=row;persist()
        with (out/'README.md').open('a') as f:f.write('\n## '+name+'命令\n\n```bash\n'+shlex.join(cmd)+'\n```\n\n日志：'+name+'.log\n')
        with (out/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=proc.pid;persist();rc=proc.wait()
        row.update(exit_code=rc,status='COMPLETED' if rc==0 else 'FAILED',finished_unix=time.time())
        persist()
        if rc or not expected.is_file():raise RuntimeError(name+'失败；全部产物保留')
    def cmd(module,params,target=ROOT):
        return [PYTHON,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(target),module]+list(map(str,params))
    try:
        # 两个数据集都完成全部18000步后，才冻结所有候选并读取TEST。
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(job,'train_'+p['dataset'],cmd('LIHAN.scripts.dynamic_full_route',
                ['--experiment',p['output']],control/'snapshot_v1'),Path(p['output']),p['gpu'],
                Path(p['output'])/'training_finished.json') for p in plans]
            for f in futures:f.result()
        entries=[]
        for p in plans:
            out=Path(p['output']);(out/'eval_only').mkdir(exist_ok=False)
            for step in (16000,16500,18000):
                dest=out/f'eval_only/freeze_{step}'
                job('freeze_'+p['dataset']+'_'+str(step),cmd('LIHAN.scripts.dynamic_v3_evaluate',
                    ['--action','freeze','--source',p['evaluation_source'],'--model',out/'model_mlp',
                     '--iteration',step,'--stage','stage2','--stage2-module','LIHAN.scripts.dynamic_full_route',
                     '--output',dest,'--confirm-frozen']),out,p['gpu'],dest/'freeze.json')
                entries.append(dict(dataset=p['dataset'],iteration=step,freeze=str(dest/'freeze.json')))
        write(control/'evaluation_lock.json',dict(all_six_frozen_before_test=True,entries=entries,
            primary_iteration=16500,other_iterations=[16000,18000],selection_uses_test_or_gt=False))
        def evaluations(p):
            out=Path(p['output'])
            for step in (16000,16500,18000):
                dest=out/f'eval_only/iteration_{step}'
                job('evaluate_'+p['dataset']+'_'+str(step),cmd('LIHAN.scripts.dynamic_v3_evaluate',
                    ['--action','run','--freeze',out/f'eval_only/freeze_{step}/freeze.json','--output',dest]),
                    out,p['gpu'],dest/'COMPLETED.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluations,p) for p in plans]
            for f in futures:f.result()
        state.update(status='READY_FOR_REVIEW',finished_unix=time.time());write(control/'state.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),finished_unix=time.time());write(control/'state.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,required=True)
    main(p.parse_args().control.resolve())
