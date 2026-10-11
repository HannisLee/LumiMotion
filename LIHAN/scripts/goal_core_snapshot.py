#!/usr/bin/env python3
"""为单个Goal训练复制最小核心Python运行时并记录逐文件SHA256。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_DIRS = ("arguments", "scene", "gaussian_renderer", "utils")
CORE_FILES = ("scripts/train_stage1.py", "scripts/train_static_dl.py",
              "LIHAN/scripts/goal_smoke_static_dl.py")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="快照子目录名")
    parser.add_argument("--control", type=Path,
                        default=ROOT / "output/1011-09-静态DL-Goal训练总控")
    args = parser.parse_args()
    destination = (args.control / "snapshots" / args.name).resolve()
    if destination.exists():
        raise FileExistsError(f"快照已存在，拒绝覆盖：{destination}")
    destination.mkdir(parents=True)
    for directory in CORE_DIRS:
        shutil.copytree(ROOT / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in CORE_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (destination / "scripts/__init__.py").touch(exist_ok=True)
    smoke_source = destination / "LIHAN/scripts/goal_smoke_static_dl.py"
    smoke_destination = destination / "scripts/goal_smoke_static_dl.py"
    smoke_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smoke_source, smoke_destination)
    shutil.rmtree(destination / "LIHAN")
    bootstrap = ROOT / "LIHAN/scripts/bootstrap_1011.py"
    for runner_name, entry in (("run_static_dl.py", "scripts.train_static_dl"),
                               ("run_smoke_static_dl.py", "scripts.goal_smoke_static_dl")):
        runner = destination / runner_name
        runner.write_text(
            "#!/usr/bin/env python3\n"
            "import subprocess, sys\n"
            f"BOOTSTRAP = {str(bootstrap)!r}\n"
            f"SNAPSHOT = {str(destination)!r}\n"
            f"ENTRY = {entry!r}\n"
            "raise SystemExit(subprocess.call([sys.executable, BOOTSTRAP, SNAPSHOT, ENTRY, *sys.argv[1:]]))\n",
            encoding="utf-8")
    files = {str(path.relative_to(destination)): digest(path)
             for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "snapshot_manifest.json"}
    payload = {
        "status": "READY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT),
        "destination": str(destination),
        "included_directories": list(CORE_DIRS),
        "included_files": list(CORE_FILES),
        "excluded": ["LIHAN/DOC", "archive", "output", "data", "large history artifacts"],
        "runtime_dependencies": ["lumimotion-ushas Conda environment", "bootstrap_1011.py repository native_module CUDA rasterizer"],
        "file_sha256": files,
    }
    (destination / "snapshot_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot": str(destination), "file_count": len(files),
                      "manifest": str(destination / "snapshot_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
