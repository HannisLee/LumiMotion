#!/usr/bin/env python3
"""仅同步 LIHAN 与 LIHAN.md，保留各分支独立实现和并发修改。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[2]


def git(*args, env=None, input=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env,
                                   input=input).decode().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--message', default='1011：静态DL目标实验与共享资料更新')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    control = Path(args.control).resolve()
    if not control.is_relative_to(ROOT / 'output'):
        raise ValueError('同步状态必须保存在本仓库output内。')
    control.mkdir(parents=True, exist_ok=True)
    with (control / 'shared_sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if git('diff', '--cached', '--name-only'):
            raise RuntimeError('索引已有改动；请先处理，禁止误提交。')
        git('fetch', 'origin', '+refs/heads/*:refs/remotes/origin/*')
        state_path = control / 'shared_sync.json'
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        expected = previous.get('shared_tree') or git('rev-parse', 'HEAD:LIHAN')
        expected_md = previous.get('LIHAN_md') or git('rev-parse', 'HEAD:LIHAN.md')
        current = git('branch', '--show-current')
        if not current:
            raise RuntimeError('禁止在游离HEAD同步。')
        names = git('for-each-ref', '--format=%(refname:strip=3)',
                    'refs/remotes/origin').splitlines()
        names = [name for name in names if name != 'HEAD']
        parents = {name: git('rev-parse', 'refs/remotes/origin/' + name) for name in names}
        local_names = set(git('for-each-ref', '--format=%(refname:strip=2)',
                              'refs/heads').splitlines())
        worktrees = git('worktree', 'list', '--porcelain').splitlines()
        checked_out = {line.removeprefix('branch refs/heads/') for line in worktrees
                       if line.startswith('branch refs/heads/')}
        for name, parent in parents.items():
            if (git('rev-parse', parent + ':LIHAN') != expected or
                    git('rev-parse', parent + ':LIHAN.md') != expected_md):
                raise RuntimeError('远端共享资料出现独立改动，保留现场：' + name)
            if name != current and name in local_names:
                if git('rev-parse', 'refs/heads/' + name) != parent:
                    raise RuntimeError('本地分支与远端存在独立差异，保留现场：' + name)
                if name in checked_out:
                    raise RuntimeError('其他分支已被工作树使用，不能直接更新：' + name)
        untracked_local = local_names - set(parents)
        if untracked_local:
            raise RuntimeError('存在未纳入远端同步的本地分支：' + ','.join(sorted(untracked_local)))
        if args.dry_run:
            print(json.dumps(dict(status='READY', branches=list(parents)), ensure_ascii=False))
            return
        git('add', '--', 'LIHAN', 'LIHAN.md')
        if git('diff', '--cached', '--name-only'):
            git('commit', '-m', args.message)
        source = git('rev-parse', 'HEAD')
        shared = git('rev-parse', 'HEAD:LIHAN')
        li_md = git('rev-parse', 'HEAD:LIHAN.md')
        candidates = {}
        for name, parent in parents.items():
            if name == current:
                candidates[name] = source
                continue
            if shared == expected and li_md == expected_md:
                candidates[name] = parent
                continue
            # 直接替换顶层树的两项，不扫描巨大的历史归档工作区。
            entries = git('ls-tree', parent).splitlines()
            entries = [line for line in entries if line.split('\t', 1)[1] not in {'LIHAN', 'LIHAN.md'}]
            entries.extend(['040000 tree ' + shared + '\tLIHAN',
                            '100644 blob ' + li_md + '\tLIHAN.md'])
            tree = git('mktree', input=('\n'.join(entries) + '\n').encode())
            candidate = git('commit-tree', tree, '-p', parent,
                            input=(args.message + '\n').encode())
            changed = git('-c', 'core.quotepath=false', 'diff-tree',
                          '--no-commit-id', '--name-only', '-r', parent, candidate).splitlines()
            if any(not (p.startswith('LIHAN/') or p == 'LIHAN.md') for p in changed):
                raise AssertionError('共享同步涉及实现文件：' + name)
            candidates[name] = candidate
        git('push', '--atomic', 'origin',
            *[sha + ':refs/heads/' + name for name, sha in candidates.items()])
        for name, sha in candidates.items():
            if name != current:
                git('update-ref', 'refs/heads/' + name, sha, parents[name])
            git('update-ref', 'refs/remotes/origin/' + name, sha)
            if (git('rev-parse', sha + ':LIHAN') != shared or
                    git('rev-parse', sha + ':LIHAN.md') != li_md):
                raise AssertionError('共享对象不一致：' + name)
        result = dict(time=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                      source_commit=source, shared_tree=shared, LIHAN_md=li_md,
                      branches=candidates, status='PASS')
        state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
