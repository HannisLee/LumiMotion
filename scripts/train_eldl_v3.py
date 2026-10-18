"""V3严格无GT的固定EL+DL混合光照几何训练。"""
from argparse import ArgumentParser
from pathlib import Path
import torch
from arguments import ModelParams,OptimizationParams,PipelineParams
from scene.eldl_v3_trainer import ELDLTrainer
from utils.general_utils import safe_state

def main():
    parser = ArgumentParser(description=__doc__)
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--eldl_lighting',required=True)
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
    parser.add_argument('--normal_detach_start',type=int,default=15001)
    parser.add_argument('--skip_opacity_reset_at',type=int,default=0,
                        help='仅跳过指定一步的opacity值截断及其Adam动量清零，0保持原流程。')
    parser.add_argument('--skip_additional_opacity_reset_at',type=int,default=0,
                        help='在已有末次跳过策略上再跳一次；正式仅允许12000与原15000组合。')
    parser.add_argument('--dist_opacity_route',choices=['full','detach'],default='full',
                        help='detach只阻断dist对opacity参数的梯度，保留几何与屏幕余切。')
    parser.add_argument('--plane_normal_weight',type=float,choices=[0.,.005],default=0.)
    parser.add_argument('--plane_start',type=int,default=8000)
    parser.add_argument('--preserve_screen_radii_event',action='store_true',
                        help='仅在一次增密剪枝事件内传递父点屏幕半径，联合剪枝后清零。')
    parser.add_argument('--normal_screen_exclude_from',type=int,default=0,
                        help='仅从指定步起移除normal增密屏幕代理，保留真实参数全部梯度；0关闭。')
    parser.add_argument('--train_alpha_hull',action='store_true',
                        help='仅接纳唯一已冻结135 TRAIN Alpha保守外壳位置初始化产物。')
    parser.add_argument('--sparse_rgb_surface_weight',type=float,default=0.)
    parser.add_argument('--sparse_rgb_surface_file',default='')
    args = parser.parse_args()
    if args.sparse_rgb_surface_weight not in (0.,.1):
        parser.error('稀疏Median表面项只允许关闭或预登记0.1，不扫描系数。')
    if bool(args.sparse_rgb_surface_weight)!=bool(args.sparse_rgb_surface_file):
        parser.error('稀疏Median开关和锁定资产必须同时给出或同时关闭。')
    if args.sparse_rgb_surface_weight:
        from scene.static_sparse_rgb_surface import validate_asset
        validate_asset(args.sparse_rgb_surface_file)
    if args.skip_opacity_reset_at < 0 or (not args.smoke_schedule and args.skip_opacity_reset_at not in (0,15000)):
        parser.error('正式实验只允许关闭开关或仅跳过15000 reset。')
    extra = args.skip_additional_opacity_reset_at
    if extra < 0 or (extra and (not args.skip_opacity_reset_at or extra >= args.skip_opacity_reset_at)):
        parser.error('新增跳过点须为已有末次跳过点之前的正整数。')
    if not args.smoke_schedule and extra not in (0,12000):
        parser.error('正式新增跳过点仅允许12000，与15000组成固定集合。')
    for skipped in (args.skip_opacity_reset_at, extra):
        if skipped and (skipped % args.opacity_reset_interval
                        or skipped >= min(args.iterations,args.densify_until_iter)):
            parser.error('跳过点必须是非终点、实际会触发的opacity reset边界。')
    if args.dist_opacity_route == 'detach' and args.normal_detach_start > args.dist_start+1:
        parser.error('dist隔离要求启用dist时已有detach_opacity几何副本。')
    if args.plane_normal_weight and args.normal_detach_start > args.plane_start+1:
        parser.error('接收面先验启用时必须已有detach_opacity几何副本。')
    if args.normal_screen_exclude_from < 0 or (not args.smoke_schedule and args.normal_screen_exclude_from not in (0,8001)):
        parser.error('正式normal-screen隔离仅允许关闭或8001起，不按GT扫描时程。')
    if not args.eval:
        parser.error('必须指定--eval以保持135训练帧和15测试帧分开。')
    if args.depth_ratio != 1 or args.static_dl_mode != 'lambertian_visibility':
        parser.error('固定物理photo depth_ratio=1；expected只作用独立几何。')
    if args.smoke_schedule:
        if 'smoke_test' not in Path(args.model_path).parts or args.iterations>50:
            parser.error('smoke_schedule仅允许独立smoke_test中最多50步。')
    elif (args.start_normal_reg not in (2000,8000) or args.dist_start!=8000
          or args.static_dl_physical_start!=1000 or args.static_dl_visibility_start!=5000
          or args.iterations<10000 or args.densify_until_iter!=20000):
        parser.error('正式日程至少10k、增密截至20k、normal2000/8000、dist8000、physical1000、V5000。')
    if args.plane_normal_weight and not args.smoke_schedule and args.plane_start != 8000:
        parser.error('当前正式接收面先验固定8001启用，不按GT调整起点。')
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
    import json, hashlib
    source=Path(args.source_path).resolve()
    manifest=json.loads((source/'no_gt_manifest.json').read_text())
    if (source/'transforms_test.json').exists() or (source/'rgbd').exists():
        parser.error('训练数据目录不得包含测试相机、RGBD或GT资产。')
    if set(p.name for p in source.iterdir()) != {'images','points3d.ply','transforms_train.json','estimated_directions_train.json','estimated_eldl_train.json','no_gt_manifest.json'}:
        parser.error('训练源只允许预注册的六项V3无GT输入。')
    if Path(args.static_dl_directions).resolve()!=source/'estimated_directions_train.json':
        parser.error('仅允许该源绑定的估计灯向。')
    for relative,expected in manifest['files'].items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('训练输入SHA变化：'+relative)
    if args.train_alpha_hull or args.sparse_rgb_surface_weight or args.plane_normal_weight:
        parser.error('V3首轮不混入历史外壳/表面/平面消融。')
    if manifest['gt_read'] or manifest['init_weights_loaded'] or manifest['dataset']!='LH-static-jumpingjacks-ELDL-V3':
        parser.error('不是V3从头无GT输入。')
    if Path(args.eldl_lighting).resolve()!=source/'estimated_eldl_train.json':
        parser.error('只允许源绑定的TRAIN估计混合灯光。')
    asset=json.loads(Path(args.eldl_lighting).read_text())
    if asset['gt_read'] or asset['source_split']!='TRAIN' or (not args.smoke_schedule and not asset.get('internal_validation_passed',False)):
        parser.error('必须先完成TRAIN内部估光验证。')
    if asset.get('engineering_only',False) and not args.smoke_schedule:
        parser.error('合成工程灯光禁止正式训练。')
    # 拒绝本训练进程打开训练源、当前输出之外的实验数据与模型文件。
    import sys, os
    output=Path(args.model_path).resolve()
    def guard(event,values):
        if event!='open' or not isinstance(values[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(values[0])).resolve()
        if path.is_relative_to(source) or path.is_relative_to(output):
            return
        if path.suffix.lower() in {'.ply','.pth','.pt','.exr','.blend','.npz'} or path.name in {'lights.json','bone_pose.json','object_pose.json'}:
            raise PermissionError('无GT训练禁止读取外部模型或GT：'+str(path))
        if path.suffix.lower() in {'.png','.jpg','.jpeg'}:
            raise PermissionError('无GT训练禁止读取外部图像：'+str(path))
    sys.addaudithook(guard)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = ELDLTrainer(args=args, dataset=lp.extract(args), opt=op.extract(args),
                              pipe=pp.extract(args), testing_iterations=args.test_iterations,
                              saving_iterations=args.save_iterations)
    trainer.train(args.iterations)
    (Path(args.model_path)/'training_finished.json').write_text(json.dumps({'status':'TRAINING_FINISHED','actual_updates':args.iterations,'gt_supervision':False},ensure_ascii=False,indent=2))
    print('V3固定EL+DL训练完成。')


if __name__ == '__main__':
    main()
