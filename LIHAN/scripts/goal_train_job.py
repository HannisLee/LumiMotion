#!/usr/bin/env python3
"""按物理 GPU UUID 启动独立训练，并持久记录真实设备与退出状态。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = {name: f"lumimotion-{name}" for name in
                ("mahadevi", "minakshi", "parvati", "ushas", "garuda")}
ENVIRONMENTS["venus"] = "lumimotion"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def query(kind, fields):
    result = subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"], text=True)
    return [[value.strip() for value in row] for row in csv.reader(result.splitlines())]


def ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "output")
    spec = json.loads(args.command.read_text())
    argv = spec["argv"]
    host = socket.gethostname().split(".")[0]
    expected_python = Path.home() / "miniconda3/envs" / ENVIRONMENTS[host] / "bin/python"
    if Path(argv[0]).absolute() != expected_python:
        raise ValueError(f"当前服务器必须使用 {expected_python}")
    if Path(argv[1]).resolve() != ROOT / "LIHAN/scripts/bootstrap_1011.py" or argv[3] != "scripts.train_static_dl":
        raise ValueError("必须通过仓库 bootstrap 加载原生训练入口")
    model = Path(argv[argv.index("--model_path") + 1]).resolve()
    if model.parent != output:
        raise ValueError("模型必须输出到本任务目录")
    devices = {int(row[0]): row for row in query("gpu", "index,uuid,memory.free")}
    gpu = int(spec["gpu_physical"])
    device = devices[gpu]
    child_env = os.environ.copy()
    allowed = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONUNBUFFERED"}
    effective = {key: str(spec.get("child_env", {}).get(key, "1" if key == "PYTHONUNBUFFERED" else "4"))
                 for key in allowed}
    effective.update(CUDA_VISIBLE_DEVICES=device[1], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    child_env.update(effective)
    if int(device[2]) < args.min_free_mib:
        raise RuntimeError(f"GPU{gpu} 空闲 {device[2]} MiB，低于启动余量")
    # 按 compute PID 保守计数；同一进程的多个 context 只计一次。
    occupants = {row[0] for row in query("compute-apps", "pid,gpu_uuid") if row[1] == device[1]}
    if len(occupants) >= 2:
        raise RuntimeError(f"GPU{gpu} 已有 {len(occupants)} 个计算进程，拒绝新增训练")
    proof = dict(host=host, gpu_physical=gpu, gpu_uuid=device[1], child_env=effective,
                 argv=argv, existing_compute_pids=sorted(occupants), checked_at_utc=now())
    if args.dry_run:
        print(json.dumps(proof, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # 已启动任务只允许新编号重试，不覆盖日志、检查点或旧状态。
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    logfile = (logs / "train.log").open("x")
    state_path = output / "state.json"
    if state_path.exists():
        previous = state_path.read_text()
        if json.loads(previous).get("status") not in ("PREPARED", "PLANNED"):
            raise RuntimeError("状态不是 PREPARED/PLANNED，拒绝覆盖")
        (output / "state.prelaunch.json").write_text(previous)
    atomic_json(output / "launch_provenance.json", proof)
    state = dict(status="STARTING", started_at_utc=now(), supervisor_pid=os.getpid(),
                 supervisor_start_ticks=ticks(os.getpid()), gpu_physical=gpu,
                 gpu_uuid=device[1], argv=argv, child_env=effective,
                 command_sha256=hashlib.sha256(args.command.read_bytes()).hexdigest(),
                 gpu_verified=False, training_exit_code=None)
    child = subprocess.Popen(argv, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                             stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    state.update(status="RUNNING", pid=child.pid, start_ticks=ticks(child.pid))
    atomic_json(state_path, state)

    def stop(signum, _frame):
        state["stop_signal"] = signum
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failure = None
    start = time.monotonic()
    try:
        actual = dict(item.split("=", 1) for item in Path(f"/proc/{child.pid}/environ").read_bytes().decode().split("\0") if "=" in item)
        observed = {key: actual.get(key) for key in effective}
        atomic_json(output / "child_environment_audit.json", dict(expected=effective, observed=observed, pid=child.pid))
        if observed != effective:
            raise RuntimeError("子进程环境变量与设备绑定不一致")
        while child.poll() is None:
            rows = query("compute-apps", "pid,gpu_uuid,used_gpu_memory")
            mine = [row for row in rows if row[0] == str(child.pid)]
            if mine:
                if any(row[1] != device[1] for row in mine):
                    raise RuntimeError(f"实际 GPU 不符：{mine}")
                state.update(gpu_verified=True, actual_gpu_rows=mine)
            elif not state["gpu_verified"] and time.monotonic() - start > 900:
                raise RuntimeError("15分钟内未验证到训练进程的GPU，停止排查")
            state["last_checked_at_utc"] = now()
            atomic_json(state_path, state)
            time.sleep(10)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
    finally:
        code = child.wait()
        state.update(training_exit_code=code, exit_code=code, ended_at_utc=now(), launcher_error=failure,
                     status="TRAINING_COMPLETED_PENDING_EVAL" if code == 0 and not failure and state["gpu_verified"] else "FAILED")
        atomic_json(state_path, state)
        logfile.close()
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if state["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
