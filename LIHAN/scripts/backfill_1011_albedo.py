#!/usr/bin/env python3
"""补评四个原生35k检查点的get_albedo；保留旧汇总，不执行训练。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import batch_1011 as b


def evaluate(index):
    exp = b.read_json(b.CONFIG)['experiments'][index - 1]
    if exp['kind'] != 'native':
        raise ValueError('此入口仅补评原生组。')
    out = Path(exp['output'])
    review = out / 'review_albedo_1011'
    review.mkdir(exist_ok=False)
    for path in [out/'evaluation_35000/mae_summary.json', out/'README.md', Path(exp['document'])]:
        shutil.copy2(path, review / ('原文档.md' if path == Path(exp['document']) else path.name))
    tasks = [(name, command) for name, command in b.eval_commands(exp, 35000)
             if name in {'albedo', 'verify_all_mae'}]
    results = []
    for name, command in tasks:
        gpu = None
        try:
            if name == 'albedo':
                gpu = b.gpu_acquire(exp, update_waiting=False)
            logfile = review / (name + '.log')
            with logfile.open('x') as log:
                log.write('开始：'+b.now()+'\n命令：'+b.shell_line(command)+'\nGPU：'+str(gpu)+'\n')
                log.flush()
                code = subprocess.run(command, cwd=b.ROOT, env=b.env(gpu if gpu is not None else ''),
                                      stdout=log, stderr=subprocess.STDOUT).returncode
                log.write('\n结束：'+b.now()+'；退出码：'+str(code)+'\n')
            results.append(dict(step='原生Stage1补评_'+name, command=command, gpu=gpu,
                                exit_code=code, log=str(logfile)))
        except Exception:
            results.append(dict(step='原生Stage1补评_'+name, command=command, error=traceback.format_exc()))
        finally:
            if gpu is not None:
                b.gpu_release()
        b.write_json(review/'executed.json', results)
        if results[-1].get('exit_code') != 0:
            b.write_json(review/'FAILED.json', results[-1])
            return 1
    metrics = b.read_json(out/'evaluation_35000/mae_summary.json')
    b.write_json(review/'completed.json', dict(time=b.now(), iteration=35000, training_restarted=False,
        parameter_semantics=metrics['details']['albedo']['parameter_semantics'], results=results, status=metrics['status']))
    print(json.dumps(dict(experiment=index, albedo=metrics['albedo'], status=metrics['status']), ensure_ascii=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(1,5))
    args = parser.parse_args()
    if args.index:
        return evaluate(args.index)
    backup = b.CONTROL/'mae_summary_before_native_albedo.json'
    if backup.exists():
        raise FileExistsError('已经执行过原生Albedo补评，禁止覆盖现场。')
    shutil.copy2(b.CONTROL/'mae_summary.json', backup)
    shutil.copy2(b.REPORT, b.CONTROL/'1011-07-原生Albedo补评前总报告.md')
    workers = [subprocess.Popen([b.PY, str(Path(__file__).resolve()), '--index', str(index)],
        cwd=b.ROOT, env=b.env()) for index in range(1,5)]
    codes = [worker.wait() for worker in workers]
    b.write_json(b.CONTROL/'native_albedo_backfill.json', dict(time=b.now(), exit_codes=codes,
        training_restarted=False, status='PASS' if not any(codes) else 'FAILED'))
    return int(any(codes))


if __name__ == '__main__':
    sys.exit(main())
