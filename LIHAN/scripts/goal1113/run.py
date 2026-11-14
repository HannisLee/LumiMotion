"""T01 无 GT 入口；评测入口独立。"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('selftest','fit','render','freeze'))
    parser.add_argument('--output',required=True); parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume')
    args=parser.parse_args(); out=Path(args.output)
    if args.resume and args.mode!='fit': raise ValueError('仅 fit 支持续算')
    if args.mode=='selftest':
        if 'smoke_test' not in out.parts or 'SYNTH-T01-BguidedEnvMap' not in out.name:
            raise ValueError('合成自检必须在指定冒烟输出目录')
        out.mkdir(parents=True,exist_ok=True)
        if any(p.name!='selftest.log' for p in out.iterdir()): raise FileExistsError('自检目录已使用')
        from .selftest import run
        result=run(out)
        with (out/'selftest.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    elif args.mode=='fit':
        from .fit import run
        run(out,args.device,args.resume)
    elif args.mode=='render':
        from .render import run
        from .figures import run as make_figures
        run(out,args.device); make_figures(out,args.device)
    else:
        from .freeze import run
        run(out)


if __name__=='__main__': main()
