#!/usr/bin/env python3
"""执行 1015-01 两阶段实验，持续记录状态、退出码和实验说明。"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def update_readme(root, plan, state):
    sections = [(root / 'references/1015-01-实验说明正文.md').read_text()]
    sections.append('\n## 实时执行状态\n\n```json\n' +
                    json.dumps(state, ensure_ascii=False, indent=2) + '\n```\n')
    sections.append('\n## 完整执行命令\n\n工作目录：`' + plan['cwd'] + '`。\n')
    for step in plan['steps']:
        sections.append('\n### ' + step['name'] + '\n\n```bash\n' +
                        shlex.join(['env'] + [k + '=' + v for k, v in plan['env'].items()] +
                                   step['argv']) + '\n```\n')
    sections.append('\n## 已生成的定量指标\n')
    metrics = sorted((root / 'model_mlp').glob('results*.json'))
    if not metrics:
        sections.append('\n尚未生成最终评测结果；训练期间指标见日志与 TensorBoard。\n')
    for path in metrics:
        sections.append('\n`' + str(path.relative_to(root)) + '`\n\n```json\n' +
                        path.read_text() + '\n```\n')
    sections.append('\n## 已生成的图片和视频索引\n')
    for pattern in ['renders_stage1_insights/ours_35000/*',
                    'trained_materials/ours_55000/*',
                    'eval_nvs_dynamic/ours_55000/*',
                    'render_relight_with_hdr_*/ours_55000/*']:
        paths = sorted((root / 'model_mlp').glob(pattern))
        png = [p for p in paths if p.suffix == '.png']
        movies = [p for p in paths if p.suffix == '.mp4']
        if png:
            selected = sorted(set([0, len(png)//2, len(png)-1]))
            sections.extend('\n- `' + str(png[i].relative_to(root)) + '`' for i in selected)
        sections.extend('\n- `' + str(p.relative_to(root)) + '`' for p in movies)
    sections.append('\n\n四类可视化目检：RGB、Alpha、基色/动静分离、Normal 均待实际检查。'
                    '脚本不根据退出码自动判定图像质量。\n\n最终结论：' +
                    ('FAILED（执行中断，全部产物保留）。' if state['status'] == 'FAILED' else
                     '待验收；完成实际目检后填写 PASS 或 FAILED。') + '\n')
    temp = root / 'README.md.tmp'
    temp.write_text(''.join(sections))
    temp.replace(root / 'README.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists():
            raise RuntimeError('目录已有执行记录，禁止重复启动或覆盖。')
        state = {'status': 'RUNNING', 'start_time': now(), 'controller_pid': os.getpid(),
                 'current_step': None, 'completed_steps': [], 'child_pid': None}
        child = None

        def interrupted(signum, frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            raise RuntimeError('总控收到信号 ' + str(signum))

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGHUP, interrupted)

        def save():
            state['heartbeat'] = now()
            write_json(root / 'status.json', state)

        with (root / 'logs/pipeline.log').open('x', buffering=1) as pipeline_log:
            try:
                for step in plan['steps']:
                    name = step['name']
                    state.update(current_step=name, step_start=now(), child_pid=None)
                    save()
                    update_readme(root, plan, state)
                    pipeline_log.write('[' + now() + '] START ' + name + '\n')
                    env = dict(os.environ, **plan['env'])
                    with (root / ('logs/' + name + '.log')).open('xb') as log:
                        child = subprocess.Popen(step['argv'], cwd=plan['cwd'], env=env,
                                                 stdin=subprocess.DEVNULL, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        state['child_pid'] = child.pid
                        save()
                        while True:
                            try:
                                code = child.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                save()
                    result = dict(name=name, start=state['step_start'], end=now(),
                                  exit_code=code, pid=child.pid, argv=step['argv'])
                    write_json(root / ('logs/' + name + '_result.json'), result)
                    pipeline_log.write('[' + now() + '] EXIT ' + name + ' code=' + str(code) + '\n')
                    if code:
                        state['exit_code'] = code
                        raise RuntimeError(name + ' 执行失败，退出码 ' + str(code))
                    state['completed_steps'].append(name)
                    child = None
                    save()
                state.update(status='COMPLETED_PENDING_REVIEW', current_step=None,
                             child_pid=None, end_time=now())
                pipeline_log.write('[' + now() + '] COMPLETED_PENDING_REVIEW\n')
            except BaseException as exc:
                state.update(status='FAILED', error=str(exc), end_time=now())
                pipeline_log.write('[' + now() + '] FAILED ' + str(exc) + '\n')
                raise
            finally:
                save()
                update_readme(root, plan, state)


if __name__ == '__main__':
    main()
