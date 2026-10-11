#!/usr/bin/env python3
"""限时只读捕获真实训练PID的native/tracer映射；缺失绝不使用fresh替代。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--module',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=45)
    args=p.parse_args()
    if args.output.exists() or not 0 < args.timeout <= 60:
        p.error('禁止覆盖，限时须1至60秒。')
    begin=time.monotonic();base=Path(f'/proc/{args.pid}');report=None
    while time.monotonic()-begin < args.timeout:
        try:
            argv=base.joinpath('cmdline').read_bytes().decode().split('\0')
            if args.module not in argv:
                raise AssertionError('实际PID模块与授权任务不符。')
            mapped=sorted({line.split()[-1] for line in base.joinpath('maps').read_text().splitlines()
                if ('lumimotion_pbr_cuda' in line or 'surfel_tracer/' in line) and line.split()[-1].endswith('.so')})
            if len(mapped)==2 and any('surfel_tracer/' in path for path in mapped):
                hashes={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in mapped}
                expected={'f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3',
                          '5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e'}
                alive=base.joinpath('maps').exists()
                status=('PASS_CAPTURED' if alive else 'PASS_PATH_CAPTURED_HASH_AFTER_EXIT') if set(hashes.values())==expected else 'FAILED_UNEXPECTED_BINARY'
                report=dict(status=status,pid=args.pid,actual_module=args.module,argv=[v for v in argv if v],
                    actual_runtime_modules=hashes,captured_at_utc=datetime.now(timezone.utc).isoformat(),
                    process_alive_during_hash=alive,
                    scope='实际训练PID存活时读取maps与soSHA，不以snapshot路径猜测，不以fresh替代。')
                break
        except FileNotFoundError:
            report=dict(status='UNAVAILABLE_PROCESS_EXITED',pid=args.pid,actual_runtime_modules={})
            break
        time.sleep(.25)
    if report is None:
        report=dict(status='UNAVAILABLE_MAPPING_TIMEOUT',pid=args.pid,actual_runtime_modules={})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
