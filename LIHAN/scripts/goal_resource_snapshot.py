#!/usr/bin/env python3
"""保存Goal训练调度前后的GPU与进程占用快照。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROL = ROOT / "output/1011-09-静态DL-Goal训练总控"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="快照时点标签")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.control / "resource_snapshots" / f"{stamp}-{args.label}.json"
    if output.exists():
        raise FileExistsError(f"拒绝覆盖资源快照：{output}")
    gpu_csv = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    app_csv = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    processes = []
    for line in app_csv.splitlines():
        fields = [part.strip() for part in line.split(",", 3)]
        if len(fields) != 4:
            continue
        uuid, pid, name, memory = fields
        ps = subprocess.run(["ps", "-p", pid, "-o", "pid=,user=,etime=,args="],
                            text=True, stdout=subprocess.PIPE, check=False)
        processes.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name,
                          "memory_mib": memory, "owner_command": ps.stdout.strip()})
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "label": args.label,
               "hostname": subprocess.check_output(["hostname"], text=True).strip(),
               "gpus": gpu_csv.splitlines(), "compute_processes": processes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
