#!/usr/bin/env python3
"""训练结束后生成代表图并核验视频，刷新总报告和全分支共享资料。"""
import importlib.util
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / 'LIHAN/scripts/batch_1011.py'
spec = importlib.util.spec_from_file_location('batch_1011', path)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
completed = set()
experiments = batch.read_json(batch.CONFIG)['experiments']
while len(completed) < len(experiments):
    for exp in experiments:
        if exp['id'] in completed:
            continue
        state = batch.read_json(Path(exp['output']) / 'state.json')
        if state.get('status') not in {'FAILED', 'AWAITING_VISUAL_REVIEW', 'PASS'}:
            continue
        if state.get('status') != 'FAILED':
            try:
                with (Path(exp['output']) / 'logs/artifact_audit.log').open('x') as log:
                    result = subprocess.run([batch.PY, str(ROOT / 'LIHAN/scripts/review_1011_artifacts.py'), exp['output']],
                        cwd=ROOT, env=batch.env(''), stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    batch.update(exp, status='FAILED', artifact_error='完整产物核验失败，见logs/artifact_audit.log')
            except FileExistsError:
                pass
        completed.add(exp['id'])
    batch.refresh()
    time.sleep(30)
batch.refresh()
with (batch.CONTROL / 'logs/shared_sync_final.log').open('w') as log:
    result = subprocess.run([batch.PY, str(ROOT / 'LIHAN/scripts/sync_1011_shared.py')], cwd=ROOT,
                            env=batch.env(''), stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print('最终共享资料同步失败，详见shared_sync_final.log', flush=True)
