#!/usr/bin/env python3
"""为外观消融训练绑定物理GPU，并记录真实退出与运行时模块来源。"""
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
TRAIN_MODULE = "scripts.train_static_dl_appearance"
RUNTIME_CAPTURE = ROOT / "LIHAN/scripts/goal2_capture_runtime.py"
NATIVE_CACHE = ROOT / "output/1013-02-LH-static-jumpingjacks_DL_V2-Stage2外观优化总控/native_cache"
NATIVE_LIBS = ROOT / "output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def query(kind: str, fields: str):
    result = subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"], text=True)
    return [[value.strip() for value in row] for row in csv.reader(result.splitlines())]


def ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def verify_snapshot(spec: dict, argv: list[str]) -> dict:
    if len(argv) < 4 or Path(argv[1]).resolve() != ROOT / "LIHAN/scripts/bootstrap_1011.py" or argv[3] != TRAIN_MODULE:
        raise ValueError("训练必须由bootstrap加载scripts.train_static_dl_appearance")
    snapshot = Path(argv[2]).resolve()
    if not snapshot.is_relative_to(ROOT / "output") or snapshot == ROOT / "output":
        raise ValueError("训练必须使用output内独立冻结快照")
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha != spec.get("snapshot_manifest_sha256"):
        raise ValueError("命令未绑定冻结快照manifest SHA")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "READY" or Path(manifest.get("destination", "")).resolve() != snapshot:
        raise ValueError("快照状态或destination不匹配")
    entry = TRAIN_MODULE.replace(".", "/") + ".py"
    files = manifest.get("file_sha256", {})
    if files.get(entry) != spec.get("entry_sha256"):
        raise ValueError("外观训练入口SHA未被快照清单绑定")
    checked = 0
    for name, expected in files.items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("快照文件SHA不匹配：" + name)
        checked += 1
    return dict(snapshot=str(snapshot), manifest_sha256=manifest_sha, entry=entry,
                entry_sha256=files[entry], verified_files=checked)


def runtime_snapshot(pid: int, gpu_uuid: str, module_name: str) -> dict:
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    maps = Path(f"/proc/{pid}/maps").read_text(errors="replace").splitlines()
    mapped = sorted({line.split()[-1] for line in maps if "/" in line.split()[-1]
                     and (line.split()[-1].endswith(".so") or ".so." in line.split()[-1])})
    mapped_records = []
    for name in mapped:
        path = Path(name.replace(" (deleted)", ""))
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, PermissionError):
            digest = None
        mapped_records.append(dict(path=name, sha256=digest))
    rows = query("compute-apps", "pid,gpu_uuid,used_gpu_memory")
    own_gpu_rows = [row for row in rows if row[0] == str(pid)]
    return dict(captured_at_utc=now(), pid=pid, start_ticks=ticks(pid),
                argv=[part.decode(errors="replace") for part in cmdline if part],
                expected_training_module=module_name, gpu_rows=own_gpu_rows,
                gpu_uuid=gpu_uuid, mapped_shared_objects=mapped_records)


def verify_fresh_output(output: Path, model: Path) -> None:
    if model.exists():
        raise FileExistsError("model_mlp已存在，拒绝覆盖：" + str(model))
    state_path = output / "state.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("status") not in ("PREPARED", "PLANNED"):
        raise RuntimeError("已有state不是PREPARED/PLANNED，拒绝覆盖")
    owned = ("launch_provenance.json", "state.prelaunch.json", "child_environment_audit.json",
             "actual_runtime_capture.json", "supervisor_result.json",
             "runtime_shared_library_audit.json", "runtime_capture_exit_code.txt")
    collisions = [str(output / name) for name in owned if (output / name).exists()]
    collisions.extend(str(output / "logs" / name) for name in ("train.log", "runtime_capture.log")
                      if (output / "logs" / name).exists())
    if collisions:
        raise FileExistsError("已有训练现场文件，拒绝覆盖：" + ", ".join(collisions))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "output")
    if output == ROOT / "output":
        raise ValueError("训练输出必须使用独立实验子目录")
    spec = json.loads(args.command.read_text())
    argv = spec["argv"]
    host = socket.gethostname().split(".")[0]
    if host not in ENVIRONMENTS:
        raise ValueError("当前服务器没有配置Conda环境：" + host)
    expected_python = Path.home() / "miniconda3/envs" / ENVIRONMENTS[host] / "bin/python"
    if Path(argv[0]).resolve() != expected_python.resolve():
        raise ValueError(f"当前服务器必须使用 {expected_python}")
    snapshot_proof = verify_snapshot(spec, argv)
    model = Path(argv[argv.index("--model_path") + 1]).resolve()
    if model.parent != output:
        raise ValueError("模型必须输出到本任务独立目录的model_mlp")
    verify_fresh_output(output, model)
    command_sha = hashlib.sha256(args.command.read_bytes()).hexdigest()
    recorded_command = output / "commands/train.json"
    if recorded_command.exists() and hashlib.sha256(recorded_command.read_bytes()).hexdigest() != command_sha:
        raise ValueError("实验目录commands/train.json与实际启动spec不同")
    gpu = int(spec["gpu_physical"])
    devices = {int(row[0]): row for row in query("gpu", "index,uuid,memory.free")}
    if gpu not in devices:
        raise ValueError(f"不存在物理GPU{gpu}")
    device = devices[gpu]
    if int(device[2]) < args.min_free_mib:
        raise RuntimeError(f"GPU{gpu}空闲{device[2]}MiB，低于启动余量{args.min_free_mib}MiB")
    occupants = {row[0] for row in query("compute-apps", "pid,gpu_uuid") if row[1] == device[1]}
    # 包括服务器上的既有服务，避免通过不同实验绕开每卡并行上限。
    if len(occupants) >= 2:
        raise RuntimeError(f"GPU{gpu}已有{len(occupants)}个计算进程，拒绝增加任务")
    defaults = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "PYTHONUNBUFFERED": "1", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                "TORCH_CUDA_ARCH_LIST": "8.6", "MAX_JOBS": "4"}
    effective = {key: str(spec.get("child_env", {}).get(key, value)) for key, value in defaults.items()}
    conda_env = Path.home() / "miniconda3/envs" / ENVIRONMENTS[host]
    effective.update(CXX=str(conda_env / "bin/x86_64-conda-linux-gnu-g++"),
                     CC=str(conda_env / "bin/x86_64-conda-linux-gnu-gcc"),
                     CUDA_HOME=str(conda_env), TORCH_EXTENSIONS_DIR=str(NATIVE_CACHE),
                     TORCH_CUDA_ARCH_LIST="8.6", MAX_JOBS="4",
                     LD_LIBRARY_PATH=":".join((str(NATIVE_LIBS), "/usr/local/cuda/lib64",
                                                str(conda_env / "lib"))))
    effective.update(CUDA_VISIBLE_DEVICES=device[1], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    proof = dict(host=host, conda_environment=ENVIRONMENTS[host], gpu_physical=gpu,
                 gpu_uuid=device[1], free_mib_at_launch=int(device[2]), child_env=effective,
                 argv=argv, snapshot_verification=snapshot_proof,
                 native_tracer_capture_script=str(RUNTIME_CAPTURE),
                 native_tracer_capture_sha256=hashlib.sha256(RUNTIME_CAPTURE.read_bytes()).hexdigest(),
                 existing_compute_pids=sorted(occupants), checked_at_utc=now())
    if args.dry_run:
        print(json.dumps(proof, ensure_ascii=False, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / "state.json"
    if state_path.exists():
        previous = state_path.read_text()
        if json.loads(previous).get("status") not in ("PREPARED", "PLANNED"):
            raise RuntimeError("状态不是PREPARED/PLANNED，拒绝覆盖旧结果")
        (output / "state.prelaunch.json").open("x").write(previous)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    (output / "commands").mkdir(exist_ok=True)
    if not recorded_command.exists():
        with recorded_command.open("xb") as stream:
            stream.write(args.command.read_bytes())
    capture_log = (logs / "runtime_capture.log").open("x")
    with (logs / "train.log").open("x") as logfile:
        atomic_json(output / "launch_provenance.json", proof)
        state = dict(status="STARTING", started_at_utc=now(), supervisor_pid=os.getpid(),
                     supervisor_start_ticks=ticks(os.getpid()), gpu_physical=gpu, gpu_uuid=device[1],
                     argv=argv, child_env=effective,
                     command_sha256=command_sha,
                     gpu_verified=False, training_exit_code=None)
        child_env = os.environ.copy()
        child_env.update(effective)
        child = subprocess.Popen(argv, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                                 stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status="RUNNING", pid=child.pid, start_ticks=ticks(child.pid))
        atomic_json(state_path, state)
        capture = None

        def stop(signum, _frame):
            state["stop_signal"] = signum
            if child.poll() is None:
                child.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        failure = None
        start = time.monotonic()
        runtime_records = []
        runtime_capture_started = False
        try:
            if hashlib.sha256(RUNTIME_CAPTURE.read_bytes()).hexdigest() != proof["native_tracer_capture_sha256"]:
                raise RuntimeError("native/tracer运行时采集脚本在启动后发生变化")
            capture = subprocess.Popen(
                [argv[0], str(RUNTIME_CAPTURE), "--pid", str(child.pid), "--module", TRAIN_MODULE,
                 "--output", str(output / "actual_runtime_capture.json"), "--timeout", "45"],
                cwd=ROOT, stdout=capture_log, stderr=subprocess.STDOUT, start_new_session=True)
            actual = dict(item.split("=", 1) for item in Path(f"/proc/{child.pid}/environ").read_bytes().decode().split("\0") if "=" in item)
            observed = {key: actual.get(key) for key in effective}
            atomic_json(output / "child_environment_audit.json",
                        dict(expected=effective, observed=observed, pid=child.pid))
            if observed != effective:
                raise RuntimeError("子进程环境变量或物理GPU绑定不同")
            while child.poll() is None:
                rows = query("compute-apps", "pid,gpu_uuid,used_gpu_memory")
                mine = [row for row in rows if row[0] == str(child.pid)]
                if mine:
                    if any(row[1] != device[1] for row in mine):
                        raise RuntimeError(f"实际GPU不符：{mine}")
                    state.update(gpu_verified=True, actual_gpu_rows=mine)
                    if not runtime_records:
                        runtime_records.append(runtime_snapshot(child.pid, device[1], TRAIN_MODULE))
                        atomic_json(output / "runtime_shared_library_audit.json",
                                    dict(status="CAPTURED", expected_training_module=TRAIN_MODULE,
                                         snapshots=runtime_records))
                    runtime_capture_started = True
                elif not state["gpu_verified"] and time.monotonic() - start > 900:
                    raise RuntimeError("15分钟内未验证到训练PID使用指定GPU")
                state["last_checked_at_utc"] = now()
                atomic_json(state_path, state)
                time.sleep(2)
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
            capture_code = 125
            if capture is not None:
                try:
                    capture_code = capture.wait(timeout=55)
                except subprocess.TimeoutExpired:
                    capture.terminate()
                    capture_code = capture.wait(timeout=5)
            capture_log.close()
            (output / "runtime_capture_exit_code.txt").write_text(f"{capture_code}\n")
            state.update(training_exit_code=code, exit_code=code, ended_at_utc=now(), launcher_error=failure,
                         status="TRAINING_COMPLETED_PENDING_EVAL" if code == 0 and not failure and state["gpu_verified"] else "FAILED")
            atomic_json(state_path, state)
            if not runtime_records:
                atomic_json(output / "runtime_shared_library_audit.json",
                            dict(status="UNAVAILABLE_NO_GPU_RUNTIME_CAPTURE", expected_training_module=TRAIN_MODULE,
                                 training_exit_code=code, launcher_error=failure))
            else:
                atomic_json(output / "runtime_shared_library_audit.json",
                            dict(status="CAPTURED", expected_training_module=TRAIN_MODULE,
                                 snapshots=runtime_records, training_exit_code=code))
            if not (output / "actual_runtime_capture.json").exists():
                atomic_json(output / "actual_runtime_capture.json", dict(
                    status="UNAVAILABLE_CAPTURE_PROCESS", capture_exit_code=capture_code,
                    capture_started_after_gpu_verification=runtime_capture_started))
            runtime = json.loads((output / "actual_runtime_capture.json").read_text())
            capture_passed = (capture_code == 0 and runtime.get("status") == "PASS_CAPTURED"
                              and runtime.get("process_alive_during_hash") is True
                              and runtime.get("pid") == child.pid and runtime.get("actual_module") == TRAIN_MODULE
                              and runtime.get("argv") == argv)
            supervisor_code = 0 if code == 0 and failure is None and state["gpu_verified"] and capture_passed else 1
            supervisor = dict(
                status="PASS_TRAINING_AND_LIVE_RUNTIME" if supervisor_code == 0 else "FAILED_OR_INCOMPLETE",
                launcher_exit_code=supervisor_code, training_exit_code=code,
                capture_exit_code=capture_code, training_pid=child.pid,
                capture_status=runtime.get("status", "MISSING"),
                actual_runtime_capture=str(output / "actual_runtime_capture.json"),
                gpu_physical=gpu, gpu_uuid=device[1], gpu_verified=state["gpu_verified"],
                started_at_utc=state["started_at_utc"], ended_at_utc=now(),
                launcher_error=failure, command_sha256=command_sha)
            atomic_json(output / "supervisor_result.json", supervisor)
            print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
            print(json.dumps(supervisor, ensure_ascii=False, indent=2), flush=True)
        if supervisor_code != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
