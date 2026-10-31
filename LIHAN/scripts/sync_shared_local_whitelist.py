#!/usr/bin/env python3
"""通过精确清单仅同步本地分支共享资料；默认只检查，绝不联网。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def git(*args, data=None, env=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, input=data, env=env)


def txt(*args, **kwargs):
    return git(*args, **kwargs).decode().strip()


def fingerprint(excluded):
    """只检查用户文件的状态和元数据，不读取清单以外文件内容。"""
    records = git('status', '--porcelain=v1', '-z', '--untracked-files=all').split(b'\0')
    result = []
    i = 0
    while i < len(records):
        record = records[i]
        i += 1
        if not record:
            continue
        path = os.fsdecode(record[3:])
        rename_source = None
        if record[:1] in (b'R', b'C') or record[1:2] in (b'R', b'C'):
            rename_source = os.fsdecode(records[i])
            i += 1
        if path in excluded or any(path.startswith(p) for p in excluded if p.endswith('/')):
            continue
        full = ROOT / path
        try:
            meta = full.lstat()
            metadata = [meta.st_mode, meta.st_size, meta.st_mtime_ns, meta.st_ino]
        except FileNotFoundError:
            metadata = None
        result.append([os.fsdecode(record), rename_source, metadata])
    return result


def index_without(excluded):
    return [os.fsdecode(r) for r in git('ls-files', '--stage', '-z').split(b'\0')
            if r and os.fsdecode(r.split(b'\t', 1)[1]) not in excluded]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSON清单：{"paths": [仓库相对路径]}')
    parser.add_argument('--control', required=True, help='output内部的状态目录')
    parser.add_argument('--message', default='1030：本轮共享资料本地全分支同步')
    parser.add_argument('--apply', action='store_true', help='执行本地提交和原子refs更新；默认只读检查')
    args = parser.parse_args()
    control = Path(args.control).resolve()
    if not control.is_relative_to(ROOT / 'output'):
        raise ValueError('状态目录必须位于本仓库output内部')
    paths = json.loads(Path(args.manifest).read_text())['paths']
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('清单必须非空且无重复')
    for path in paths:
        if (not isinstance(path, str) or Path(path).is_absolute() or '..' in Path(path).parts
                or path != str(Path(path)) or not (path.startswith('LIHAN/') or path == 'LIHAN.md')):
            raise ValueError('只允许精确LIHAN文件路径：' + str(path))
        full = ROOT / path
        if full.is_symlink() or not full.is_file() or not full.resolve().is_relative_to(ROOT):
            raise ValueError('清单只能包含仓库内部普通文件：' + path)
    if txt('symbolic-ref', '--quiet', 'HEAD') not in txt('for-each-ref', '--format=%(refname)', 'refs/heads').splitlines():
        raise RuntimeError('必须在本地分支上运行')
    if txt('ls-files', '--unmerged'):
        raise RuntimeError('索引存在冲突，停止')
    refs = {}
    for line in txt('for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads').splitlines():
        ref, sha = line.split()
        refs[ref] = sha
    current = txt('symbolic-ref', 'HEAD')
    worktrees = txt('worktree', 'list', '--porcelain').splitlines()
    occupied = [line[7:] for line in worktrees if line.startswith('branch ')]
    if any(ref != current for ref in occupied):
        raise RuntimeError('其他分支正被工作树使用，停止')
    shared = {(txt('rev-parse', sha + ':LIHAN'), txt('rev-parse', sha + ':LIHAN.md')) for sha in refs.values()}
    if len(shared) != 1:
        raise RuntimeError('分支共享树已有分歧，禁止覆盖')
    baseline = refs[current]
    payloads = {path: (ROOT / path).read_bytes() for path in paths}
    blobs = {path: txt('hash-object', '--stdin', data=data) for path, data in payloads.items()}
    modes = {path: ('100755' if (ROOT / path).stat().st_mode & stat.S_IXUSR else '100644') for path in paths}
    protected_exclusions = set(paths) | {str(control.relative_to(ROOT)) + '/'}
    before_status = fingerprint(protected_exclusions)
    before_index = index_without(set(paths))
    report = dict(status='READY', utc=datetime.now(timezone.utc).isoformat(), refs_before=refs,
                  baseline_shared=list(next(iter(shared))), paths=paths, blobs=blobs,
                  apply=args.apply, network=False)
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    control.mkdir(parents=True, exist_ok=True)
    # 互斥只覆盖本脚本；refs事务仍以旧OID校验其他进程的并发更新。
    import fcntl
    with (control / 'local_shared_sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_whitelist = [r for r in git('ls-files', '--stage', '-z').split(b'\0')
                              if r and os.fsdecode(r.split(b'\t', 1)[1]) in set(paths)]
        with tempfile.TemporaryDirectory(prefix='共享同步-', dir=control) as temporary:
            env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
            git('read-tree', baseline, env=env)
            for path in paths:
                blob = txt('hash-object', '-w', '--stdin', data=payloads[path])
                git('update-index', '--add', '--cacheinfo', modes[path], blob, path, env=env)
            source_tree = txt('write-tree', env=env)
            source_shared = txt('rev-parse', source_tree + ':LIHAN')
            source_md = txt('rev-parse', source_tree + ':LIHAN.md')
            candidates = {}
            for ref, parent in refs.items():
                entries = git('ls-tree', '-z', parent).split(b'\0')
                entries = [entry for entry in entries if entry and entry.split(b'\t', 1)[1] not in (b'LIHAN', b'LIHAN.md')]
                entries += [('040000 tree ' + source_shared + '\tLIHAN').encode(), ('100644 blob ' + source_md + '\tLIHAN.md').encode()]
                tree = txt('mktree', '-z', data=b'\0'.join(entries) + b'\0')
                changed = {os.fsdecode(p) for p in git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', parent + '^{tree}', tree).split(b'\0') if p}
                if not changed.issubset(set(paths)):
                    raise RuntimeError('候选提交包含清单以外路径：' + ref)
                candidates[ref] = parent if not changed else txt('commit-tree', tree, '-p', parent, data=(args.message + '\n').encode())
            if fingerprint(protected_exclusions) != before_status or index_without(set(paths)) != before_index:
                raise RuntimeError('准备期间用户工作区或索引发生变化，保留候选但不更新refs')
            if any((ROOT / p).read_bytes() != payloads[p] for p in paths):
                raise RuntimeError('准备期间清单文件发生变化，停止')
            report['refs_after'] = candidates
            # 写入可恢复的提交/父节点清单，随后一次事务更新所有本地分支。
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            report_path = control / ('local_shared_sync_' + stamp + '.json')
            report['status'] = 'PREPARED'
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            transaction = 'start\n' + ''.join('update ' + ref + ' ' + candidates[ref] + ' ' + parent + '\n' for ref, parent in refs.items()) + 'prepare\ncommit\n'
            git('update-ref', '--stdin', data=transaction.encode())
            try:
                # 仅刷新本轮文件的正常索引；用户的其他暂存和工作区内容原样保留。
                git('reset', '--quiet', 'HEAD', '--', *paths)
                if fingerprint(protected_exclusions) != before_status or index_without(set(paths)) != before_index:
                    raise RuntimeError('同步后用户状态校验失败')
                for ref in refs:
                    if (txt('rev-parse', ref + ':LIHAN') != source_shared or txt('rev-parse', ref + ':LIHAN.md') != source_md):
                        raise RuntimeError('同步后共享树不一致：' + ref)
            except Exception:
                rollback = 'start\n' + ''.join('update ' + ref + ' ' + parent + ' ' + candidates[ref] + '\n' for ref, parent in refs.items()) + 'prepare\ncommit\n'
                git('update-ref', '--stdin', data=rollback.encode())
                removal = [b'0 ' + b'0' * 40 + b'\t' + os.fsencode(p) for p in paths]
                git('update-index', '-z', '--index-info', data=b'\0'.join(removal + original_whitelist) + b'\0')
                report['status'] = 'ROLLED_BACK'
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                raise
            report['status'] = 'PASS'
            report['shared_tree'] = source_shared
            report['LIHAN_md'] = source_md
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
