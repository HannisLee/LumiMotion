#!/usr/bin/env python3
"""提交本批共享资料，并以普通快进原子推送同步全部分支，保留各分支实现。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'output/1011-07-六组训练总控'


def git(*args, env=None, input=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env, input=input).decode().strip()


with (CONTROL / 'sync.lock').open('a') as sync_lock:
    fcntl.flock(sync_lock, fcntl.LOCK_EX)
    git('fetch', 'origin', '+refs/heads/*:refs/remotes/origin/*')
    state_path = CONTROL / 'shared_sync.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    allowed = previous.get('shared_tree') or git('rev-parse', 'HEAD:LIHAN')
    refs = git('for-each-ref', '--format=%(refname:strip=3)', 'refs/remotes/origin').splitlines()
    refs = [ref for ref in refs if ref != 'HEAD']
    parents = {ref: git('rev-parse', 'refs/remotes/origin/' + ref) for ref in refs}
    for branch, parent in parents.items():
        if git('rev-parse', parent + ':LIHAN') != allowed:
            raise RuntimeError('远端共享资料出现并发改动，保留现场：' + branch)
    with (CONTROL / 'report.lock').open('a') as report_lock:
        fcntl.flock(report_lock, fcntl.LOCK_EX)
        git('add', '--', 'LIHAN', 'LIHAN.md',
            'submodules/2dgs_rasterizer_lumimotion/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h')
        if git('diff', '--cached', '--name-only'):
            git('commit', '-m', '1011：六组训练、全部MAE与共享资料更新')
        source = git('rev-parse', 'HEAD')
        shared = git('rev-parse', 'HEAD:LIHAN')
        li_md = git('rev-parse', 'HEAD:LIHAN.md')
    current = git('branch', '--show-current')
    candidates = {}
    for branch, parent in parents.items():
        if branch == current:
            candidates[branch] = source
            continue
        if git('rev-parse', parent + ':LIHAN') == shared and git('rev-parse', parent + ':LIHAN.md') == li_md:
            candidates[branch] = parent
            continue
        with tempfile.TemporaryDirectory(dir=CONTROL) as temp:
            isolated_env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / 'index'))
            git('read-tree', parent, env=isolated_env)
            # 只清空临时索引中的共享路径；不删除工作区文件。
            git('rm', '-r', '-f', '--cached', '--ignore-unmatch', '--', 'LIHAN', 'LIHAN.md', env=isolated_env)
            git('read-tree', '--prefix=LIHAN/', shared, env=isolated_env)
            git('update-index', '--add', '--cacheinfo', '100644,' + li_md + ',LIHAN.md', env=isolated_env)
            tree = git('write-tree', env=isolated_env)
            candidate = git('commit-tree', tree, '-p', parent,
                            input='1011：同步六组训练及全部MAE共享资料\n'.encode())
            changed = git('diff-tree', '--no-commit-id', '--name-only', '-r', parent, candidate).splitlines()
            if any(not (p.startswith('LIHAN/') or p.startswith('"LIHAN/') or p == 'LIHAN.md') for p in changed):
                raise AssertionError('共享同步涉及实现文件：' + branch)
            candidates[branch] = candidate
    git('push', '--atomic', 'origin', *[sha + ':refs/heads/' + branch for branch, sha in candidates.items()])
    for branch, sha in candidates.items():
        if branch != current:
            git('update-ref', 'refs/heads/' + branch, sha)
        git('update-ref', 'refs/remotes/origin/' + branch, sha)
        if git('rev-parse', sha + ':LIHAN') != shared or git('rev-parse', sha + ':LIHAN.md') != li_md:
            raise AssertionError('分支共享资料对象不一致：' + branch)
    result = dict(time=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), source_commit=source,
                  shared_tree=shared, LIHAN_md=li_md, branches=candidates, status='PASS')
    state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
