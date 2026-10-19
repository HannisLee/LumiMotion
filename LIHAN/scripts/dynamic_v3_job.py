"""独立运行已预登记的单个动态V3训练，持久保存PID、退出码与失败现场。"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    experiment=args.experiment.resolve()
    assert socket.gethostname()=='ushas'
    assert experiment.is_relative_to(root/'output')
    assert json.loads(args.gate.read_text())['status']=='PASS_RELOAD_AND_DYNAMIC_GRADIENT_ENGINEERING'
    command=json.loads((experiment/'command.json').read_text())
    assert command[0]=='/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
    assert not (experiment/'job.json').exists() and not (experiment/'model_mlp').exists()
    resource=subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    used,total,util=map(int,resource.split(','))
    assert used<1024 and total-used>16000,'首次启动需使用空闲卡，禁止干预其他任务'
    state=dict(status='STARTING',supervisor_pid=os.getpid(),hostname=socket.gethostname(),
               gpu=args.gpu,resource_at_launch=resource,command=command,started_unix=time.time())
    write(experiment/'job.json',state)
    with (experiment/'train.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1'),
                                 stdout=log,stderr=subprocess.STDOUT)
        state.update(status='RUNNING',training_pid=process.pid)
        write(experiment/'job.json',state)
        code=process.wait()
    finished=experiment/'model_mlp/training_finished.json'
    state.update(status='TRAINING_FINISHED_PENDING_EVALUATION' if code==0 and finished.is_file() else 'FAILED',
                 exit_code=code,finished_unix=time.time(),finished_marker_present=finished.is_file())
    write(experiment/'job.json',state)
    raise SystemExit(code if code else (0 if finished.is_file() else 1))


if __name__=='__main__':main()
