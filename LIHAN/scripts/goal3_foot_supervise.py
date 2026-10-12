#!/usr/bin/env python3
"""直接启动训练并自动捕获存活进程映射，避免短冒烟先于人工轮询结束。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        parser.error('输出必须在本仓库output中。')
    spec = json.loads(args.command.read_text())
    module = spec['argv'][3]
    if module not in {'scripts.train_static_dl_foot_recovery', 'scripts.train_static_dl_foot_resume'}:
        parser.error('仅支持本Goal的独立训练入口。')
    if (output / 'state.json').exists():
        parser.error('已有启动状态，禁止重复启动或覆盖。')
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    capture_path = output / 'actual_runtime_capture.json'
    if capture_path.exists():
        parser.error('已有运行库证据，禁止覆盖。')
    with (logs / 'supervisor.log').open('x') as supervisor_log, (logs / 'runtime_capture.log').open('x') as capture_log:
        launch = subprocess.Popen([sys.executable, str(ROOT / 'LIHAN/scripts/goal3_foot_train_job.py'),
            '--command', str(args.command.resolve()), '--output', str(output)], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=supervisor_log, stderr=subprocess.STDOUT)
        capture = None
        pid = None
        while launch.poll() is None:
            state_path = output / 'state.json'
            if capture is None and state_path.exists():
                state = json.loads(state_path.read_text())
                pid = state.get('pid')
                if pid:
                    capture = subprocess.Popen([sys.executable, str(ROOT / 'LIHAN/scripts/goal2_capture_runtime.py'),
                        '--pid', str(pid), '--module', module,
                        '--output', str(capture_path), '--timeout', '60'], cwd=ROOT,
                        stdin=subprocess.DEVNULL, stdout=capture_log, stderr=subprocess.STDOUT)
            time.sleep(.1)
        launch_code = launch.wait()
        capture_code = capture.wait() if capture is not None else None
    evidence = json.loads(capture_path.read_text()) if capture_path.exists() else {}
    passed = launch_code == 0 and capture_code == 0 and evidence.get('status') == 'PASS_CAPTURED'
    report = dict(status='PASS_TRAINING_AND_LIVE_RUNTIME' if passed else 'FAILED_OR_INCOMPLETE',
        launcher_exit_code=launch_code, capture_exit_code=capture_code, training_pid=pid,
        capture_status=evidence.get('status', 'MISSING'), actual_runtime_capture=str(capture_path),
        goal_result='尚未评测；监督成功不是足部修复或法线成功。')
    with (output / 'supervisor_result.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
