"""独立进程监督新三阶段任务，保存真实PID、完整命令、日志和退出码。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--entry', choices=['scripts.train_three_stage', 'scripts.train_three_stage_native_pbr'], default='scripts.train_three_stage')
    p.add_argument('training_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert socket.gethostname() == 'ushas'
    python = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
    output = a.output.resolve(); assert output.is_relative_to(root/'output')
    output.mkdir(parents=True, exist_ok=False)
    tail = a.training_args[1:] if a.training_args and a.training_args[0] == '--' else a.training_args
    for option in ('--binding', '--parent'):
        if option in tail:
            index = tail.index(option)+1
            tail[index] = str(Path(tail[index]).resolve())
    command = [python, str(root/'LIHAN/scripts/bootstrap_1011.py'), str(a.snapshot or root),
               a.entry, '--output', str(output), *tail]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        TORCH_EXTENSIONS_DIR=str(root/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/native_cache'),
        THREE_STAGE_REPO_ROOT=str(root), PYTHONUNBUFFERED='1')
    def write(name, data):
        (output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    write('launch.json', dict(argv=command, command=shlex.join(command), cwd=str(root), gpu=a.gpu,
                              env={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'TORCH_EXTENSIONS_DIR', 'THREE_STAGE_REPO_ROOT')}))
    with (output/'train.log').open('w') as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        write('job_state.json', dict(status='RUNNING', pid=child.pid, supervisor=os.getpid(), started=time.time()))
        code = child.wait()
    write('job_state.json', dict(status='FINISHED' if code == 0 else 'FAILED', pid=child.pid,
                                supervisor=os.getpid(), finished=time.time(), exit_code=code))
    if code and not (output/'README.md').exists():
        (output/'README.md').write_text('# 三阶段工程失败记录\n\n完整训练命令：`'+shlex.join(command)+'`。\n\n'
            'source／父model／iteration见launch.json的binding与parent参数；当前模型目录model_mlp。日志train.log。\n\n'
            '渲染命令：训练入口保存后内置完整BVH重算；失败前是否执行见日志。输出previews，存在的checkpoint、统计和日志全部保留。\n\n'
            '定量指标：metrics.jsonl（若未生成则尚未执行）。代表图片／视频：失败前未完成，按实际文件保留。\n\n'
            'RGB／Alpha／Normal／Separation及分支分量目检：未完成。最终结论：FAILED（退出码'+str(code)+'）。\n')
    print(json.dumps(dict(exit_code=code, output=str(output)), ensure_ascii=False), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
