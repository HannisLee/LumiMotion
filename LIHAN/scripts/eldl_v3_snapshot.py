"""生成独立V3源码快照；不带历史权重和数据，逐文件记录SHA。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    a=p.parse_args();out=a.destination.resolve()
    assert out.is_relative_to(ROOT/'output')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('scene','utils','arguments','gaussian_renderer','assets','scripts'):
        shutil.copytree(ROOT/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    paths=[f for f in out.rglob('*') if f.is_file()]
    value=dict(source_root=str(ROOT),snapshot=str(out),
               commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               files={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(paths)})
    (out/'snapshot_manifest.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(out),files=len(paths)),ensure_ascii=False))


if __name__=='__main__':main()
