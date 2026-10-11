#!/usr/bin/env python3
"""为已完成的检查点补齐评测；保留原始训练失败状态，不重启训练。"""
import argparse,json,subprocess,traceback
from pathlib import Path
import batch_1011 as b

def main(index):
    exp=b.read_json(b.CONFIG)['experiments'][index-1]
    out=Path(exp['output']);attempt=out/'review_1011'
    attempt.mkdir(exist_ok=False)
    tasks=[]
    iteration=35000 if exp['kind']=='native' else 55000
    if exp['kind']=='native':
        tasks.append(b.native_commands(exp)[2])
    tasks.extend(b.eval_commands(exp,iteration))
    tasks.append(('artifact_audit',[b.PY,str(b.ROOT/'LIHAN/scripts/review_1011_artifacts.py'),str(out)]))
    results=[]
    for name,command in tasks:
        gpu=None
        try:
            if name not in {'artifact_audit','verify_all_mae','verify_roughness'}:
                gpu=b.gpu_acquire(exp, update_waiting=False)
            logfile=attempt/(name+'.log')
            with logfile.open('x') as log:
                log.write('开始：'+b.now()+'\n命令：'+b.shell_line(command)+'\nGPU：'+str(gpu)+'\n');log.flush()
                code=subprocess.run(command,cwd=b.ROOT,env=b.env(gpu if gpu is not None else ''),stdout=log,stderr=subprocess.STDOUT).returncode
                log.write('\n结束：'+b.now()+'；退出码：'+str(code)+'\n')
            results.append(dict(step=name,command=command,gpu=gpu,exit_code=code,log=str(logfile)))
        except Exception:
            results.append(dict(step=name,command=command,error=traceback.format_exc()))
        finally:
            if gpu is not None:b.gpu_release()
        b.write_json(attempt/'executed.json',results)
    b.write_json(attempt/'completed.json',dict(time=b.now(),iteration=iteration,results=results))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=int);main(p.parse_args().index)
