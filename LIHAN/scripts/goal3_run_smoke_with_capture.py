#!/usr/bin/env python3
"""启动单个脚部恢复smoke，并在训练PID存活期间抓取native/tracer来源。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", default="scripts.train_static_dl_foot_recovery")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "output" / "smoke_test")
    command = args.command.resolve()
    if output.joinpath("state.json").exists():
        parser.error("状态文件已存在；重试必须使用新的smoke编号。")
    spec = json.loads(command.read_text())
    python = spec["argv"][0]
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    supervisor_log_path = logs / "supervisor.log"
    capture_log_path = logs / "runtime_capture.log"
    capture_output = output / "runtime_module_audit.json"
    if supervisor_log_path.exists() or capture_log_path.exists() or capture_output.exists():
        parser.error("日志或runtime记录已存在；拒绝覆盖。")

    with supervisor_log_path.open("x") as supervisor_log:
        launcher = subprocess.Popen(
            [python, str(ROOT / "LIHAN/scripts/goal3_foot_train_job.py"),
             "--command", str(command), "--output", str(output)],
            cwd=ROOT, stdout=supervisor_log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 60
        training_pid = None
        while time.monotonic() < deadline:
            if launcher.poll() is not None:
                break
            try:
                state = json.loads((output / "state.json").read_text())
                training_pid = state.get("pid")
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if training_pid:
                break
            time.sleep(0.1)

        capture = None
        if training_pid:
            with capture_log_path.open("x") as capture_log:
                capture = subprocess.Popen(
                    [python, str(ROOT / "LIHAN/scripts/goal2_capture_runtime.py"),
                     "--pid", str(training_pid), "--module", args.module,
                     "--output", str(capture_output), "--timeout", "45"],
                    cwd=ROOT, stdout=capture_log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                train_exit = launcher.wait()
                capture_exit = capture.wait()
        else:
            train_exit = launcher.wait()
            capture_exit = 125
            capture_output.write_text(json.dumps(
                {"status": "UNAVAILABLE_NO_TRAIN_PID", "supervisor_exit_code": train_exit},
                ensure_ascii=False, indent=2) + "\n")

    (output / "supervisor_exit_code.txt").write_text(f"{train_exit}\n")
    (output / "runtime_capture_exit_code.txt").write_text(f"{capture_exit}\n")
    state_path = output / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    print(json.dumps({"supervisor_exit_code": train_exit,
                      "runtime_capture_exit_code": capture_exit,
                      "state": state}, ensure_ascii=False, indent=2), flush=True)
    return train_exit


if __name__ == "__main__":
    raise SystemExit(main())
