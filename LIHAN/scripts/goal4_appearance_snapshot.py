#!/usr/bin/env python3
"""为Stage2外观实验创建不可覆盖的独立运行快照。"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    control = args.control.resolve()
    if not control.is_relative_to(ROOT / 'output'):
        parser.error('总控必须位于本仓库output。')
    if Path(args.name).name != args.name:
        parser.error('名称只能是单级目录。')
    entry = ROOT / 'scripts/train_static_dl_appearance.py'
    if not entry.is_file():
        parser.error('新的独立训练入口尚未完成。')
    destination = control / 'snapshots' / args.name
    if destination.exists():
        raise FileExistsError('快照已存在，拒绝覆盖：' + str(destination))
    destination.mkdir(parents=True)
    for directory in ('arguments', 'scene', 'gaussian_renderer', 'utils', 'assets'):
        shutil.copytree(ROOT / directory, destination / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination / 'scripts').mkdir()
    (destination / 'scripts/__init__.py').touch()
    shutil.copy2(entry, destination / 'scripts' / entry.name)
    files = {str(p.relative_to(destination)): digest(p)
             for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest = dict(status='READY', created_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_root=str(ROOT), destination=str(destination),
                    entry='scripts.train_static_dl_appearance', file_sha256=files,
                    bootstrap=str(ROOT / 'LIHAN/scripts/bootstrap_1011.py'),
                    bootstrap_sha256=digest(ROOT / 'LIHAN/scripts/bootstrap_1011.py'),
                    scope='只复制运行依赖；不包含数据、GT、旧输出及文档归档。')
    path = destination / 'snapshot_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(destination), manifest=str(path),
                         manifest_sha256=digest(path), file_count=len(files)), ensure_ascii=False))


if __name__ == '__main__':
    main()
