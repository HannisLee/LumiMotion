"""静态DL从头训练：仅normal开始2000或8000，dist始终8000。"""
from argparse import ArgumentParser
from pathlib import Path
import torch
from arguments import ModelParams,OptimizationParams,PipelineParams
from scene.static_dl_early_geometry import EarlyGeometryTrainer
from utils.general_utils import safe_state

def main():
    parser = ArgumentParser(description=__doc__)
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[10000, 20000, 35000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[10000, 20000, 35000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--deform-type', dest='deform_type', type=str, default='mlp')
    parser.add_argument('--static_dl_mode', choices=['sh', 'lambertian', 'lambertian_visibility'], default='sh')
    parser.add_argument('--static_dl_directions', default='')
    parser.add_argument('--static_dl_physical_start', type=int, default=1000)
    parser.add_argument('--static_dl_visibility_start', type=int, default=5000)
    parser.add_argument('--static_dl_light_lr', type=float, default=.001)
    parser.add_argument('--static_dl_normal_weight', type=float, default=.1)
    parser.add_argument('--static_dl_shadow_weight', type=float, default=.25)
    parser.add_argument('--static_dl_highlight_weight', type=float, default=.5)
    parser.add_argument('--static_dl_residual_threshold', type=float, default=.08)
    parser.add_argument('--static_dl_shadow_bias', type=float, default=.03)
    parser.add_argument('--static_dl_bvh_interval', type=int, default=50)
    parser.add_argument('--dist_start',type=int,default=8000)
    parser.add_argument('--loss_model',choices=['censored'],default='censored')
    parser.add_argument('--normal_depth_statistic',choices=['expected'],default='expected')
    parser.add_argument('--smoke_schedule',action='store_true',help='仅output/smoke_test内短程物理分支测试，禁止用作正式日程。')
    args = parser.parse_args()
    if not args.eval:
        parser.error('必须指定--eval以保持135训练帧和15测试帧分开。')
    if args.depth_ratio != 1 or args.static_dl_mode != 'lambertian_visibility':
        parser.error('固定物理photo depth_ratio=1；expected只作用独立几何。')
    if args.smoke_schedule:
        if 'smoke_test' not in Path(args.model_path).parts or args.iterations>50:
            parser.error('smoke_schedule仅允许独立smoke_test中最多50步。')
    elif (args.start_normal_reg not in (2000,8000) or args.dist_start!=8000
          or args.static_dl_physical_start!=1000 or args.static_dl_visibility_start!=5000
          or args.iterations!=10000):
        parser.error('正式日程固定10k、normal2000/8000、dist8000、physical1000、V5000。')
    if args.deform_type != 'mlp':
        parser.error('该入口仅支持mlp格式，以保持标准评测加载一致。')
    if not args.model_path:
        parser.error('必须显式提供独立的output实验模型路径。')
    if not args.model_path.endswith('_mlp'):
        args.model_path = args.model_path.rstrip('/') + '_mlp'
    if Path(args.model_path).exists():
        parser.error('只允许从头训练；已有模型目录不能覆盖，包括失败现场。')
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    args.test_iterations = sorted(set(args.test_iterations + [args.iterations]))
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = EarlyGeometryTrainer(args=args, dataset=lp.extract(args), opt=op.extract(args),
                              pipe=pp.extract(args), testing_iterations=args.test_iterations,
                              saving_iterations=args.save_iterations)
    trainer.train(args.iterations)
    print('静态DL训练完成。')


if __name__ == '__main__':
    main()
