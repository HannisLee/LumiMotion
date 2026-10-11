#!/usr/bin/env python3
"""只用CPU等待checkpoint完整且稳定；不启动GPU评测。"""
import argparse
from pathlib import Path
import time
from goal_eval_protocol import write, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--physical', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=3600)
    args = p.parse_args()
    required = [args.model / 'point_cloud' / f'iteration_{args.iteration}' / 'point_cloud.ply',
                args.model / 'deform' / f'iteration_{args.iteration}' / 'deform.pth']
    if args.physical:
        required.append(args.model / 'static_dl_lighting' / f'iteration_{args.iteration}' / 'lighting.json')
    if args.output.exists():
        raise FileExistsError('监视记录禁止覆盖。')
    previous = None
    stable_start = None
    start = time.monotonic()
    while time.monotonic()-start < args.timeout:
        if all(f.is_file() and f.stat().st_size > 0 for f in required):
            current = [(f.stat().st_size, f.stat().st_mtime_ns) for f in required]
            if current != previous:
                previous, stable_start = current, time.monotonic()
            elif time.monotonic()-stable_start >= 10:
                hashes = {str(f.resolve()): digest(f) for f in required}
                if [(f.stat().st_size, f.stat().st_mtime_ns) for f in required] != current:
                    stable_start = None
                    previous = None
                    continue
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write(args.output, dict(status='READY_STABLE', model=str(args.model.resolve()),
                    iteration=args.iteration, files=hashes, elapsed_seconds=time.monotonic()-start,
                    stable_seconds=10, gpu_evaluation_started=False))
                print('checkpoint稳定并已保存SHA：' + str(args.output), flush=True)
                return
        time.sleep(5)
    raise TimeoutError('checkpoint等待超时；未启动GPU评测。')


if __name__ == '__main__':
    main()
