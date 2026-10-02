"""既有材质/HDR 命令自动识别独立 Lambertian checkpoint。"""
import argparse
import json
import sys
from pathlib import Path


def dispatch(task):
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--model_path',default='');probe.add_argument('--deform-type',dest='deform_type',default='mlp')
    known,_=probe.parse_known_args()
    if not known.model_path:return False
    model=Path(known.model_path)
    if not (model/'stage2_config.json').exists():model=Path(str(model)+'_'+known.deform_type)
    config=model/'stage2_config.json'
    if not config.exists() or json.loads(config.read_text()).get('render_mode')!='photometric_lambertian':return False
    parser=argparse.ArgumentParser(description='Stage2 Lambertian '+task)
    parser.add_argument('--model_path',required=True);parser.add_argument('--load_iter',type=int,required=True)
    parser.add_argument('--deform-type',dest='deform_type',default='mlp')
    parser.add_argument('--hdr_filepath','--hdr',default='example_envmaps/golden_bay_4k_32x16_rot330.hdr')
    parser.add_argument('--output_path',default='');parser.add_argument('--hdr_samples',type=int,default=2048)
    parser.add_argument('--hdr_yaw',type=float,default=0.0);parser.add_argument('--hdr_exposure',type=float,default=1.0)
    # 旧入口惯用模型/数据参数由 checkpoint 配置负责；显式未识别参数拒绝，避免静默丢弃。
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args()
    from scripts.render_stage2 import run_cli
    command=['--model_path',str(model),'--load_iter',str(args.load_iter),'--task',task,
             '--hdr_filepath',args.hdr_filepath,'--hdr_samples',str(args.hdr_samples),
             '--hdr_yaw',str(args.hdr_yaw),'--hdr_exposure',str(args.hdr_exposure)]
    command+=['--output_path',args.output_path or str(model/('stage2_'+task)/('ours_'+str(args.load_iter)))]
    run_cli(command)
    return True
