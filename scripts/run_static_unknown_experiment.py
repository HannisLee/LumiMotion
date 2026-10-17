"""持续保存未知光实验的真实PID、退出码和运行模块；不重启失败实验。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory.resolve()
    plan = json.loads((out/'plan.json').read_text())
    if (out/'state.json').exists():
        raise FileExistsError('已有运行状态，禁止重复或覆盖。')
    env = dict(os.environ, **plan['env'])
    with (out/'train.log').open('xb') as log:
        child = subprocess.Popen(plan['argv'], cwd=plan['cwd'], env=env,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT)
        state = {'status':'RUNNING','supervisor_pid':os.getpid(),'child_pid':child.pid,
                 'started':datetime.now(timezone.utc).isoformat(), 'exit_code':None}
        write(out/'state.json',state)
        while child.poll() is None:
            state['heartbeat'] = datetime.now(timezone.utc).isoformat()
            try:
                maps = Path(f'/proc/{child.pid}/maps').read_text()
                modules = sorted({line.split()[-1] for line in maps.splitlines()
                                  if any(x in line for x in ('lumimotion_pbr_cuda','surfel_tracer')) and '/' in line})
                if modules:
                    state['runtime_modules'] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                                for p in modules if Path(p).is_file()}
                    state['captured_while_alive'] = True
                state['actual_cmdline'] = Path(f'/proc/{child.pid}/cmdline').read_bytes().decode().split('\0')[:-1]
            except (OSError, ProcessLookupError):
                pass
            write(out/'state.json',state)
            time.sleep(5)
        state.update(exit_code=child.wait(),finished=datetime.now(timezone.utc).isoformat())
        state['status'] = 'TRAINING_FINISHED_PENDING_EVALUATION' if state['exit_code']==0 else 'FAILED'
        write(out/'state.json',state)
    print(json.dumps(state,ensure_ascii=False))
    raise SystemExit(state['exit_code'])


if __name__=='__main__':
    main()
