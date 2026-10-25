"""把训练所需源码冻结复制到实验目录，生成逐文件SHA清单；正式训练通过bootstrap_1011.py从快照运行。"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ['arguments', 'assets', 'gaussian_renderer', 'scene', 'scripts', 'utils']


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for d in DIRS:
        shutil.copytree(ROOT / d, out / d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for p in sorted(out.rglob('*')):
        if p.is_file():
            files[str(p.relative_to(out))] = hashlib.sha256(p.read_bytes()).hexdigest()
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--'] + DIRS,
                            capture_output=True, text=True).stdout.splitlines()
    manifest = dict(source_root=str(ROOT), snapshot=str(out), commit=head, worktree_status_of_code_dirs=status,
                    files=files, key_files={k: files[k] for k in args.key if k in files})
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (out / 'snapshot_manifest.json').write_text(text)
    print(json.dumps(dict(snapshot=str(out), files=len(files), manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          key_files=manifest['key_files']), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--key', nargs='*', default=[])
    main(p.parse_args())
