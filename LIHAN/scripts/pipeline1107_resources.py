"""T01(1107-03) I7：训练资源采集与耗时/显存汇总。

  sample     python pipeline1107_resources.py sample --pid <训练PID> --output <csv> [--interval 30]
             每 interval 秒用 nvidia-smi 按 PID 采样 memory.used，同时记录整卡 memory.used；PID 结束自动退出。
  summarize  python pipeline1107_resources.py summarize --run <臂目录> [--run ...] [--samples csv ...] --output <json>
             汇总 state.json 墙钟、metrics.jsonl 单步毫秒(均值/中位数)、训练期 max_memory_allocated、采样峰值。
单步毫秒在不同卡型/共驻条件下不可直接比较：本脚本把 GPU 名称与采样到的共驻进程数写进结果。
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def query(args):
    return subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, check=False).stdout.strip()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def sample_once(pid):
    apps = [line.split(',') for line in query(['--query-compute-apps=pid,used_memory,gpu_uuid',
                                               '--format=csv,noheader,nounits']).splitlines() if line.strip()]
    mine = [int(x[1]) for x in apps if int(x[0]) == pid]
    gpus = [line.split(',') for line in query(['--query-gpu=index,uuid,name,memory.used', '--format=csv,noheader,nounits']).splitlines()]
    return dict(time=time.time(), pid_memory_mb=sum(mine) if mine else None, processes_on_node=len(apps),
                gpu_memory_used_mb={g[0].strip(): int(g[3]) for g in gpus}, gpu_names={g[0].strip(): g[2].strip() for g in gpus})


def sample(a):
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('a', newline='') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['time', 'pid_memory_mb', 'processes_on_node', 'gpu_memory_used_mb_json', 'gpu_names_json'])
        while alive(a.pid):
            row = sample_once(a.pid)
            writer.writerow([row['time'], row['pid_memory_mb'], row['processes_on_node'],
                             json.dumps(row['gpu_memory_used_mb']), json.dumps(row['gpu_names'])])
            handle.flush()
            time.sleep(a.interval)


def read_samples(path):
    rows = list(csv.DictReader(Path(path).open()))
    pid = [float(r['pid_memory_mb']) for r in rows if r['pid_memory_mb'] not in ('', 'None')]
    node = [max(json.loads(r['gpu_memory_used_mb_json']).values()) for r in rows]
    return dict(samples=len(rows), pid_peak_mb=max(pid) if pid else None,
                node_peak_mb=max(node) if node else None,
                max_processes_on_node=max(int(r['processes_on_node']) for r in rows) if rows else None,
                gpu_names=sorted({n for r in rows for n in json.loads(r['gpu_names_json']).values()}))


def summarize_run(run, samples=None):
    run = Path(run)
    state = json.loads((run / 'state.json').read_text())
    out = dict(run=str(run), arm=state.get('arm'), status=state.get('status'), gpu=state.get('gpu'),
               wall_seconds=(state['finished'] - state['started']) if 'finished' in state else None,
               steps=state.get('steps'))
    path = run / 'model_mlp/metrics.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ms = [r['step_ms'] for r in rows if r.get('step_ms') is not None]
        out['step_ms'] = dict(n=len(ms), mean=statistics.mean(ms) if ms else None,
                              median=statistics.median(ms) if ms else None,
                              note='每100循环取一次单步 CUDA event 毫秒，非全步平均')
        mem = [r['cuda_max_allocated_mb'] for r in rows if r.get('cuda_max_allocated_mb') is not None]
        out['torch_max_allocated_mb'] = max(mem) if mem else None
        out['last_iteration_logged'] = rows[-1]['iteration'] if rows else None
    if samples:
        out['nvidia_smi'] = read_samples(samples)
    return out


def summarize(a):
    samples = a.samples or [None] * len(a.run)
    if len(samples) != len(a.run):
        raise ValueError('--samples 数量须与 --run 一致。')
    result = [summarize_run(r, s) for r, s in zip(a.run, samples)]
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for r in result:
        wall = r['wall_seconds']
        print(r['arm'], r['status'], f'{wall/3600:.2f} h' if wall else 'N/A', r.get('torch_max_allocated_mb'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--pid', type=int, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--interval', type=float, default=30.)
    s = sub.add_parser('summarize')
    s.add_argument('--run', type=Path, action='append', required=True)
    s.add_argument('--samples', type=Path, action='append')
    s.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sample(a) if a.action == 'sample' else summarize(a)


if __name__ == '__main__':
    main()
