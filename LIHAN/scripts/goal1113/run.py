"""T01 无 GT 入口；评测入口独立。"""
import argparse
import json
import os
import shlex
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('selftest','fit','render','freeze'))
    parser.add_argument('--output',required=True); parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume')
    args=parser.parse_args(); out=Path(args.output)
    if args.resume and args.mode!='fit': raise ValueError('仅 fit 支持续算')
    if args.mode in ('fit','render'):
        started=datetime.now(timezone.utc).isoformat(); code=1
        try:
            if args.mode=='fit':
                from .fit import run
                run(out,args.device,args.resume)
            else:
                from .render import run
                from .figures import run as make_figures
                run(out,args.device); make_figures(out,args.device)
            code=0
        finally:
            receipt=dict(schema_version=1,mode=args.mode,command=shlex.join(sys.orig_argv),argv=sys.orig_argv,
                cwd=os.getcwd(),hostname=socket.gethostname(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                pid=os.getpid(),started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),exit_code=code,
                log_path='run.log' if args.mode=='fit' else 'render.log')
            out.mkdir(parents=True,exist_ok=True)
            path=out/f'{args.mode}_execution.json'
            if path.exists(): path.rename(out/f'{args.mode}_execution_previous_{os.getpid()}.json')
            with path.open('x') as f: json.dump(receipt,f,ensure_ascii=False,indent=2); f.write('\n')
        return
    if args.mode=='selftest':
        if 'smoke_test' not in out.parts or 'SYNTH-T01-BguidedEnvMap' not in out.name:
            raise ValueError('合成自检必须在指定冒烟输出目录')
        out.mkdir(parents=True,exist_ok=True)
        if any(p.name!='selftest.log' for p in out.iterdir()): raise FileExistsError('自检目录已使用')
        from .selftest import run
        result=run(out)
        if args.device.startswith('cuda'):
            from .raster_selftest import run as raster_test
            result['合成光栅']=raster_test()
        with (out/'selftest.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    else:
        from .freeze import run
        run(out)


if __name__=='__main__': main()
