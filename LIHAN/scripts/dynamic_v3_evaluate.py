"""隔离动态评测：先准备/冻结，后统一评价；禁止用TEST/GT选择模型。

CPU入口prepare读取TRAIN观测与独立GT输入，禁止读取TEST RGB或候选；freeze只登记模型。
GPU入口须通过bootstrap_1011执行；所有产物写入output内新的eval_only目录。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import shlex
import socket
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.geometry_eval_utils import (camera_rays, depth_normals, erode,
    file_sha256, load_normal_exr, normal_errors, unit_normal)
from LIHAN.scripts.eldl_v3_eval import (METHODS, DEPTHS, HUMAN, GROUND,
    normal_item, depth_item, stat, aggregate, mean_present)
from LIHAN.scripts.dynamic_v3_radiometry import estimate_background,black_observation

CONTROL = ROOT/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控'
OLD = ROOT/'data/LH-data/danamic/LH-dynamic-jumpingjacks_EL_V2'
NEW = ROOT/'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3'
OLD_GT = CONTROL/'eval_only/old_dynamic_gt_v1/gt_depth'
NEW_GT = CONTROL/'eval_only/dynamic_gt_v1/gt_depth'
OLD_MODEL = ROOT/'output/1011-04-LH-dynamic-jumpingjacks_EL_V2-原生LumiMotion两阶段/stage1_frozen'
DL_V2 = ROOT/'data/LH-data/danamic/LH-dynamic-jumpingjacks_DL_V2'
DL_V2_GT = ROOT/'output/archive/1011-07-六组训练总控/GT-LH-dynamic-jumpingjacks_DL_V2/gt_depth'
DL_V2_MODEL = ROOT/'output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/stage1_frozen'
TEST = list(range(10, 151, 10))
DEV = [n for n in range(61, 80) if n % 10]
TRAIN = [n for n in range(1, 151) if n not in TEST + DEV]


def supplementary_protocol(bundle,track_protocol='triangle_barycentric_v1',correspondence=None):
    """执行前固定预算和定义，不依据候选或GT误差改变阈值。"""
    fps=read(Path(bundle['raw'])/'camera.json').get('fps')
    fps=float(fps) if isinstance(fps,(int,float)) and math.isfinite(fps) and fps>0 else None
    return dict(format_version=1,track_protocol=track_protocol,
        correspondence=str(Path(correspondence).resolve()) if correspondence else None,
        correspondence_sha256=sha(correspondence) if correspondence else None,
        frames=TEST,video_frames=list(range(1,151)),video_fps=fps or 30.,
        video_fps_evidence='camera.json.fps' if fps else '仅显示播放速度，非物理时间证据',
        time_unit='second' if fps else 'normalized_time',physical_fps=fps,
        surface_samples=16384,track_samples=8192,seed=1019,
        distances_world_units=[.01,.02,.05,.10],missing_distance_world_units=1.,
        geometry_unit='原始world单位；单位换算仅由GT manifest另行记录，不擅称米或厘米',
        temporal='15TEST同GT三角面/重心坐标运输' if track_protocol=='triangle_barycentric_v1' else '15TEST经独立150帧polygon/loop与保序modifier证明的固定GT vertex ID；均匀采样参考时刻被三角面引用的顶点，非面积加权',
        temporal_residual='扣除真实GT运动后的world残差一/二阶差；非逐原始帧抖动',
        visibility='GT射线首命中误差<=max(1e-4 world,1e-3*距离)，Alpha>=.99且2像素内缩；与候选无关',
        missing_temporal='位置漏预测罚1 world；速度上限2/dt，二阶上限2*(1/dt1+1/dt2)/mean(dt)；有效异常同样截断并另报未截断误差',
        topology='三角协议须全部TEST triangles/object_ids/object_names相同；持久顶点协议须提供PROVEN独立收据并逐帧核对vertex-world SHA；三角协议失效诊断始终单列',
        vertex_anatomy='持久顶点局部分区固定为首TEST相邻三角语义多数票；这是边界近似，不等同原始骨骼权重分区',
        videos='RGB含Alpha、GS/Expected法线、Expected/Median/Mesh深度、静动态门控；Stage2另存EL/DL',
        mesh_video='只有15TEST有真实独立TSDF，其余135帧显示NOT_RECONSTRUCTED，禁止插值或静态替代',
        missing_mesh='空预测mesh：GT召回与预测精度都设0，F-score=0，双向距离各罚1world；精度端16384为预登记虚拟缺失计数，非实际采样面',
        visual_colors='深度/误差固定OpenCV TURBO；无支持黑色、TEST GT前景漏预测误差为品红；法线RGB=.5*n_world+.5',
        visual_only='PNG/视频仅显示时clip与8位量化；定量始终使用原float数组，禁止从PNG反推指标',
        visual_depth_range_world=[0.,10.],visual_error_range_world=[0.,.1],
        local_review='15TEST固定GT人体/腿/足/鞋包围框加16像素；四类及Stage2 EL/DL全列，等比缩放后补黑边，不改变数值指标',
        limitations=['GT mesh距离为三角几何表面，不模拟透明纹理孔洞',
            '表面采样完整性是Monte Carlo估计，精度与召回均须报告，空mesh不得跳过',
            '轨迹是GT可见表面沿相机射线的几何残差，不是模型物质点或切向运动精度',
            '15TEST时间间距10原始帧，无法识别该采样Nyquist频率以上的高频抖动',
            '条件误差依有效预测，必须同时引用固定GT分母惩罚和覆盖率',
            '视频自动解码检查只证明文件完整，四类目检仍需人工填写'])


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


class LogStream:
    """把本次完整stdout/stderr同时保存在实验目录，失败也保留。"""
    def __init__(self, stream, log):
        self.stream,self.log=stream,log

    def write(self, value):
        self.stream.write(value);self.log.write(value);self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush();self.log.flush()


def sha(path):
    return file_sha256(Path(path))


def fresh(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT/'output') or 'eval_only' not in path.parts:
        raise ValueError('隔离评测输出必须在output/.../eval_only下。')
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify(files):
    for name, expected in files.items():
        if sha(name) != expected:
            raise ValueError('来源SHA变化：'+name)


def linearize(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    # 原生SH预测可以超出[0,1]；保留其误差，不在指标前clip。
    return np.where(rgb <= .04045, rgb/12.92, ((np.maximum(rgb, 0)+.055)/1.055)**2.4)


def raw_frame(raw, number):
    image = Path(raw)/f'image/image_{number:04d}.png'
    alpha_path = Path(raw)/f'albedo/albedo_{number:04d}.png'
    image_array = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    alpha_array = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
    for array in (image_array, alpha_array):
        if array is None or array.dtype != np.uint16 or array.shape != (800, 800, 4):
            raise ValueError('要求原16位800×800 RGBA：'+str(image))
    rgb = image_array[..., [2, 1, 0]].astype(np.float64)/65535.
    alpha = alpha_array[..., 3].astype(np.float64)/65535.
    return rgb, alpha, image_array[..., [2, 1, 0]] == 65535, (image, alpha_path)


def target_frame(raw, number, background_linear):
    rgb, alpha, saturated, files = raw_frame(raw, number)
    # 原image合成固定灰背景；只移除已知背景透射项，保留物体环境照明。
    target = cv2.resize(black_observation(rgb,alpha,background_linear), (400, 400), interpolation=cv2.INTER_AREA)
    source_alpha = cv2.resize(alpha, (400, 400), interpolation=cv2.INTER_AREA)
    straight = cv2.resize(linearize(rgb), (400, 400), interpolation=cv2.INTER_AREA)
    saturation = cv2.resize(saturated.astype(np.float32), (400, 400), interpolation=cv2.INTER_AREA)
    return target, source_alpha, straight, saturation, files


def audit_gt_inputs(bundle):
    """只核查GT输入；不加载候选，不读取TEST RGB，不提供调参依据。"""
    rows=[];files={};gt_dir=Path(bundle['gt']);manifest=read(gt_dir/'manifest.json')
    if manifest['depth_convention']!='positive camera Z; unnormalized pinhole rays':
        raise ValueError('GT深度单位/约定不符。')
    for n in TEST:
        alpha_path=Path(bundle['raw'])/f'albedo/albedo_{n:04d}.png'
        rgba=cv2.imread(str(alpha_path),cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.dtype!=np.uint16 or rgba.shape!=(800,800,4):
            raise ValueError('GT原Alpha必须16位800×800。')
        alpha=cv2.resize(rgba[...,3].astype(np.float64)/65535.,(400,400),interpolation=cv2.INTER_AREA)
        gt,sources=load_gt(bundle,n,alpha);files.update(sources);files[str(alpha_path)]=sha(alpha_path)
        record=bundle['frames'][str(n)]
        if not np.allclose(gt['camera_to_world'],record['transform_matrix'],atol=1e-5,rtol=0) or not np.allclose(gt['intrinsics'],record['intrinsics'],atol=1e-5,rtol=0):
            raise ValueError('原相机与GT相机不符。')
        foreground=alpha>=.5;hit=gt['valid']&np.isfinite(gt['depth_z'])&(gt['depth_z']>0)
        eligible=foreground&hit&gt['gt_exr_valid'];masks,status=masks_for(gt)
        rows.append(dict(frame=n,time=record['time'],gt_pixels=int(eligible.sum()),
            alpha_pixels=int(foreground.sum()),alpha_hit_iou=float((foreground&hit).sum()/max(1,(foreground|hit).sum())),
            excluded_alpha_pixels_without_valid_depth_or_normal=int((foreground&~eligible).sum()),
            region_pixels={k:int((eligible&mask).sum()) for k,mask in masks.items()},region_status=status,
            mask_sha256=__import__('hashlib').sha256(eligible.tobytes()).hexdigest(),
            mesh=manifest['frames'][str(n)].get('mesh')))
        if not eligible.any():raise ValueError('GT时刻分母为空。')
    return dict(status='PASS_GT_INPUT_ASSOCIATION_ONLY',candidate_loaded=False,test_rgb_read=False,
        frames=rows,source_files=files,
        correspondence='本数据内时刻/相机/Alpha/GT区域关联；不宣称新旧骨骼、顶点或逐像素GT区域相等')


def prepare(args):
    raw = {'elv2': OLD, 'v3': NEW, 'dlv2': DL_V2}[args.dataset]
    gt = (args.gt or {'elv2': OLD_GT, 'v3': NEW_GT, 'dlv2': DL_V2_GT}[args.dataset]).resolve()
    if not (gt/'manifest.json').is_file():
        raise FileNotFoundError('先定位真实GT目录：'+str(gt))
    camera_path = raw/'camera.json'
    camera = read(camera_path)
    if sorted(map(int, camera['frames'])) != list(range(1, 151)):
        raise ValueError('相机必须包含全部150时刻。')
    train = [n for n in range(1, 151) if n not in TEST] if args.dataset == 'elv2' else TRAIN
    records = {}
    for n in range(1, 151):
        c = camera['frames'][f'{n:04d}']; i = c['intrinsics']
        records[str(n)] = dict(source_frame=n, time=n/150 if args.dataset in ('elv2','dlv2') else (n-1)/149,
            transform_matrix=c['extrinsics']['camera_to_world'],
            intrinsics=[i['fx']/2, i['fy']/2, i['cx']/2, i['cy']/2])
    rng = np.random.default_rng(1019); samples=[]
    files = {str(camera_path): sha(camera_path), str(gt/'manifest.json'): sha(gt/'manifest.json')}
    for n in train:
        rgb, alpha, _, paths = raw_frame(raw, n)
        lum = linearize(rgb)@np.array([.2126, .7152, .0722]); values=lum[alpha >= .5]
        if not values.size:
            raise ValueError('TRAIN前景为空。')
        samples.append(values[rng.integers(0, values.size, 8192)])
        files.update({str(p): sha(p) for p in paths})
    distribution = np.concatenate(samples)
    radiometry=estimate_background(raw,train)
    files.update(radiometry['source_files'])
    audit_bundle=dict(raw=str(raw),gt=str(gt),frames=records)
    gt_audit=audit_gt_inputs(audit_bundle)
    files.update(gt_audit['source_files'])
    out = fresh(args.output)
    write(out/'gt_input_audit.json',gt_audit)
    write(out/'source.json', dict(format_version=1, evaluation_only=True, dataset=args.dataset,
        raw=str(raw), gt=str(gt), frames=records, train_frames=train, test_frames=TEST,
        thresholds=dict(q_base=float(np.quantile(distribution, .5)), q_bright=float(np.quantile(distribution, .9)),
            probabilities=[.5,.9], source_split='TRAIN', source_frames=train, samples_per_frame=8192, seed=1019,
            definition='原16位sRGB逆变换，原Alpha>=.5，等帧抽样；候选间固定，TEST不重取分位'),
        source_files=files, radiometry=radiometry, raw_test_rgb_read=False, gt_input_audit='gt_input_audit.json',
        note='隔离引用源，禁止把此目录作为训练source；未复制GT到训练。'))
    print('PASS：TRAIN阈值与独立GT输入关联审计；未读TEST RGB或候选。', flush=True)


def source_bundle(path):
    path = Path(path).resolve()
    bundle = read(path)
    if bundle['dataset']=='elv2' and Path(bundle['gt']).resolve()!=OLD_GT.resolve():
        raise ValueError('最终EL_V2只使用old_dynamic_gt_v1语义GT，请重新prepare；historical仅供历史诊断。')
    if bundle['dataset']=='dlv2' and (Path(bundle['gt']).resolve()!=DL_V2_GT.resolve() or Path(bundle['raw']).resolve()!=DL_V2.resolve()):
        raise ValueError('DL_V2必须绑定自身原数据与归档GT，不可复用ELDL或EL_V2。')
    verify(bundle['source_files'])
    if bundle['test_frames'] != TEST or len(bundle['frames']) != 150 or not bundle['evaluation_only']:
        raise ValueError('评测source需要完整固定15TEST和150时刻。')
    expected = [n for n in range(1,151) if n not in TEST] if bundle['dataset']=='elv2' else TRAIN
    if bundle['train_frames'] != expected or bundle['thresholds']['source_frames'] != expected:
        raise ValueError('TRAIN划分或亮度阈值来源不符。')
    thresholds=bundle['thresholds']
    if not (0<=thresholds['q_base']<=thresholds['q_bright']<=1):
        raise ValueError('TRAIN亮度阈值非法。')
    return bundle


def assets(model, step, stage, dataset):
    from scripts.eval_geometry_gt import read_config
    model = Path(model).resolve(); config, config_path = read_config(model)
    paths=[config_path, model/f'point_cloud/iteration_{step}/point_cloud.ply',
        model/f'deform/iteration_{step}/deform.pth']
    marker = None
    if stage == 'stage2' and (model/'dynamic_full_route.json').is_file():
        marker_path=model/'dynamic_full_route.json';marker=read(marker_path)
        if marker.get('route')!='dynamic_full_route_v1' or step not in (16000,16500,18000):
            raise ValueError('完整路线只允许预登记的三个检查点。')
        if (marker.get('gt_supervision') is not False or marker.get('external_test_read') is not False
                or marker.get('train_frames')!=TRAIN or marker.get('dataset')!=dataset
                or marker.get('actual_optimizer_updates')!=18000 or not marker.get('fresh_random_initialization')):
            raise ValueError('完整路线训练登记不符。')
        receipt_path=model/f'dynamic_full_route/iteration_{step}/receipt.json';receipt=read(receipt_path)
        if receipt['actual_optimizer_updates']!=step or receipt['gt_read'] is not False:
            raise ValueError('完整路线实际更新数不符。')
        paths += [marker_path,receipt_path,model/f'dynamic_full_route/iteration_{step}/state.pt']
        for relative,expected in receipt['checkpoint_sha256'].items():
            path=(model/relative).resolve()
            if not path.is_relative_to(model) or sha(path)!=expected:raise ValueError('完整路线checkpoint SHA不符。')
        config.update(marker['config'])
    elif stage == 'stage2':
        if step != 8100:
            raise ValueError('本轮Stage2仅允许预登记真实新增8100更新终点。')
        marker_path=model/'dynamic_v3_stage2.json';marker=read(marker_path)
        receipt_path=model/f'dynamic_v3/iteration_{step}/receipt.json'
        receipt=read(receipt_path)
        paths += [marker_path, model/f'dynamic_v3/iteration_{step}/state.pt',receipt_path]
        if marker.get('gt_supervision') is not False or marker.get('external_test_read') is not False:
            raise ValueError('Stage2需要明确无GT监督/无外部TEST读取。')
        if marker.get('train_frames')!=TRAIN or dataset not in ('v3','dlv2'):
            raise ValueError('本轮Stage2只能使用新V3固定117TRAIN。')
        if dataset=='dlv2' and Path(marker['parent']).resolve()!=DL_V2_MODEL.resolve():
            raise ValueError('DL_V2 Stage2必须从其真实原生Stage1出发。')
        if marker.get('actual_optimizer_updates')!=step or marker.get('parent_iteration')!=35000:
            raise ValueError('Stage2真实新增更新数/Stage1父步数不符。')
        for relative,expected in receipt['checkpoint_sha256'].items():
            path=(model/relative).resolve()
            if not path.is_relative_to(model) or sha(path)!=expected:
                raise ValueError('Stage2终点receipt SHA不符：'+relative)
        config.update(marker.get('config', {}))
    elif dataset == 'v3':
        if step != 35000:
            raise ValueError('本轮Stage1仅允许预登记35000终点。')
        marker_path=model/'training_finished.json';marker=read(marker_path);paths.append(marker_path)
        if marker.get('gt_supervision') is not False or marker.get('train_frames') != TRAIN:
            raise ValueError('Stage1无GT/117TRAIN登记不符。')
        if marker['args']['iterations'] != step:
            raise ValueError('Stage1预算不符。')
        for relative, expected in marker['checkpoint_sha256'].items():
            if sha(model/relative) != expected:
                raise ValueError('训练终点checkpoint SHA不符。')
    elif dataset == 'dlv2':
        if model != DL_V2_MODEL.resolve() or step != 35000:
            raise ValueError('DL_V2仅允许已核验stage1_frozen@35000。')
        original=model.parent/'model_mlp'
        for relative in ('cfg_args', 'point_cloud/iteration_35000/point_cloud.ply', 'deform/iteration_35000/deform.pth'):
            if sha(model/relative)!=sha(original/relative):
                raise ValueError('DL_V2冻结Stage1与真实原生训练资产不一致：'+relative)
    else:
        if model != OLD_MODEL.resolve() or step != 35000:
            raise ValueError('EL_V2仅允许已核验真实stage1_frozen@35000。')
    return config, marker, {str(p):sha(p) for p in paths}


def freeze(args):
    bundle=source_bundle(args.source)
    if not args.confirm_frozen:
        raise ValueError('freeze需--confirm-frozen，声明选择/预算已固定且未用TEST/GT选择。')
    config, marker, files=assets(args.model,args.iteration,args.stage,bundle['dataset'])
    if args.track_protocol=='persistent_vertices_v1':
        if args.correspondence is None:raise ValueError('持久顶点后继协议需要显式--correspondence独立证明收据。')
        correspondence=verify_correspondence(args.correspondence,bundle)
        files[str(args.correspondence.resolve())]=sha(args.correspondence)
        files.update(correspondence['source_sha256'])
    elif args.correspondence is not None:
        raise ValueError('不能给三角重心协议暗中附加持久顶点替代。')
    code_paths=[Path(__file__).resolve(),ROOT/'utils/geometry_eval_utils.py',
        ROOT/'LIHAN/scripts/eldl_v3_eval.py',ROOT/'scripts/eval_geometry_gt.py',
        ROOT/'utils/predicted_mesh.py',ROOT/'gaussian_renderer/__init__.py',
        ROOT/'scene/gaussian_model.py',ROOT/'scene/deform_model.py',ROOT/'scene/cameras.py',
        ROOT/'LIHAN/scripts/dynamic_v3_radiometry.py',ROOT/'utils/time_utils.py',
        ROOT/'scene/pbr_rasterizer.py',ROOT/'scene/dataset_readers.py',ROOT/'utils/graphics_utils.py',
        ROOT/'utils/general_utils.py',ROOT/'utils/point_utils.py',ROOT/'utils/sh_utils.py']
    if args.stage=='stage2':
        spec=importlib.util.find_spec(args.stage2_module)
        if spec is None or spec.origin is None:raise ValueError('Stage2模块尚未实现，不能冻结为可评测。')
        code_paths.append(Path(spec.origin))
    out=fresh(args.output)
    write(out/'freeze.json',dict(evaluation_allowed=True,selection_uses_test_or_gt=False,
        source=str(args.source.resolve()),source_sha256=sha(args.source),model=str(args.model.resolve()),
        iteration=args.iteration,stage=args.stage,source_files=files,config=config,
        stage2_module=args.stage2_module,marker_snapshot=marker,
        code_sha256={str(p):sha(p) for p in code_paths},
        time_rule='frame/150' if bundle['dataset'] in ('elv2','dlv2') else '(frame-1)/149',
        test_frames=TEST,normal_missing_penalty=180,normal_aggregation='帧等权、三方法等权',
        supplementary_protocol=supplementary_protocol(bundle,args.track_protocol,args.correspondence),
        gt_mesh_manifest=str(gt_mesh_directory(bundle)/'manifest.json'),
        gt_mesh_manifest_sha256=sha(gt_mesh_directory(bundle)/'manifest.json'),
        mesh=dict(static_scene=False,voxel_size=.02,sdf_trunc=.08,depth_trunc=20.,alpha_threshold=.5),
        purpose='冻结后一次性独立最终评价；不得反馈训练、选checkpoint或停止点'))
    print('PASS：只冻结配置与资产，未评价候选。',flush=True)


def make_camera(record):
    import torch
    from scene.cameras import Camera
    from scene.dataset_readers import blender_c2w_to_camera_extrinsics
    fx,fy,cx,cy=record['intrinsics']
    if abs(cx-200)>1e-5 or abs(cy-200)>1e-5:
        raise ValueError('当前光栅器仅支持居中主点。')
    r,t=blender_c2w_to_camera_extrinsics(record['transform_matrix'])
    n=record['source_frame']
    return Camera(colmap_id=n,R=r,T=t,FoVx=2*math.atan(200/fx),FoVy=2*math.atan(200/fy),
        image_train_light=torch.zeros((3,400,400)),gt_alpha_mask=torch.ones((1,400,400)),
        image_name_train_light=f'image_{n:04d}',image_path_train_light='',uid=n,data_device='cuda',fid=record['time'])


def runtime(frozen):
    import torch
    import diff_surfel_rasterization as backend
    from scripts.eval_geometry_gt import GeometryRun
    from gaussian_renderer import render
    if getattr(backend,'provenance',{}).get('method')!='repository_lumimotion_cuda_rasterizer':
        raise ValueError('run必须经bootstrap_1011绑定原生光栅器。')
    model=Path(frozen['model']);step=frozen['iteration'];config=dict(frozen['config'])
    run=GeometryRun(model,step)
    run.effective_rasterizer=backend.provenance
    recorded_backend=(frozen.get('marker_snapshot') or {}).get('rasterizer')
    if recorded_backend is not None and recorded_backend['source_sha256']!=backend.provenance['source_sha256']:
        raise ValueError('原生Stage1训练与评测光栅器来源SHA不同。')
    photo=None
    if frozen['stage']=='stage2':
        photo=importlib.import_module(frozen['stage2_module']).DynamicSurfaceModel.load(model,step)
        run.pc,run.deform=photo.pc,photo.deform
    run.config=dict(config,compute_cov3D_python=False,convert_SHs_python=False,
                    pbr_rasterizer_backend='lumimotion',depth_ratio=config.get('depth_ratio',1.))
    parent_iteration=int((frozen.get('marker_snapshot') or {}).get('parent_iteration',35000))
    run.iteration=step if (frozen.get('marker_snapshot') or {}).get('route')=='dynamic_full_route_v1' else parent_iteration if photo else step
    run.pc.active_sh_degree=int(config.get('active_sh_degree',min(run.pc.max_sh_degree,run.iteration//config.get('oneupSHdegree_step',1000))))
    run.pc.no_binary_separation=bool(config.get('no_binary_separation',False))
    run.deform.deform.eval()

    def deformation(camera):
        # 与原生Trainer/Stage2实际canonical查询一致，保留get_binary_feature(eval=True)语义。
        return run.deform.step(run.pc._xyz,camera.fid.reshape(1,1).expand(len(run.pc._xyz),1),
            iteration=run.iteration,feature=run.pc.get_binary_feature(eval=True),
            camera_center=camera.camera_center,is_training=False)
    run.deformation=deformation

    def predict(camera,c2w,k,with_rgb=True,with_separation=False):
        pred=run.render(camera,c2w,k)
        if not with_rgb:
            return pred
        if photo:
            package=photo.render(camera,components=True)
            rgb=package['render_linear'].permute(1,2,0).detach().cpu().numpy()
            components={name:package[name].permute(1,2,0).detach().cpu().numpy()
                for name in ('environment_linear','direct_linear') if name in package}
        else:
            ds=deformation(camera)
            package=render(camera,run.pc,SimpleNamespace(**run.config),torch.zeros(3,device='cuda'),
                **{k:ds[k] for k in ('d_xyz','d_rotation','d_scaling','d_opacity','d_color')})
            rgb=linearize(package['render'].permute(1,2,0).detach().cpu().numpy())
            components={}
        normal,valid=unit_normal(package['rend_normal'].permute(1,2,0).detach().cpu().numpy())
        alpha=package['rend_alpha'].squeeze(0).detach().cpu().numpy()
        surf=package['surf_depth'].squeeze(0).detach().cpu().numpy()
        ratio=run.config['depth_ratio'];depth=(1-ratio)*pred['expected']+ratio*pred['median']
        visible=(pred['alpha']>=.5)&np.isfinite(depth)
        parity=dict(alpha_max_abs=float(np.max(np.abs(alpha-pred['alpha']))),
            normal_max_abs=float(np.max(np.abs(normal[valid]-pred['normal'][valid]))) if valid.any() else 0.,
            surface_depth_max_abs=float(np.max(np.abs(surf[visible]-depth[visible]))) if visible.any() else 0.)
        if any(not math.isfinite(v) or v>2e-5 for v in parity.values()):
            raise ValueError('着色与独立几何渲染不一致：'+str(parity))
        pred.update(rgb_linear=rgb,**components)
        if with_separation:
            gate=run.pc.get_binary_feature(eval=True)
            colors=torch.cat((gate,1-gate,torch.zeros_like(gate)),dim=1)
            ds=deformation(camera)
            split=render(camera,run.pc,SimpleNamespace(**run.config),torch.zeros(3,device='cuda'),
                **{key:ds[key] for key in ('d_xyz','d_rotation','d_scaling','d_opacity','d_color')},override_color=colors)
            pred['separation']=split['render'].permute(1,2,0).detach().cpu().numpy()
        return pred,parity
    return run,predict


def build_meshes(bundle,frozen,out,predict):
    import torch
    import open3d as o3d
    folder=out/'predicted_meshes';folder.mkdir()
    manifest=dict(static_scene=False,gt_geometry_used=False,test_camera_used_for_reconstruction=False,
        model=frozen['model'],iteration=frozen['iteration'],camera_frames=bundle['train_frames'],
        method='逐target time，全部TRAIN相机固定同一fid的Expected Z独立TSDF',settings=frozen['mesh'],frames={})
    basis=np.diag([1.,-1.,-1.,1.])
    # 逐相机构造避免把全部TRAIN图像/相机GPU张量常驻显存。
    for n in TEST:
        time=bundle['frames'][str(n)]['time']
        volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.02,sdf_trunc=.08,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        integrations=[]
        for fid in bundle['train_frames']:
            record=dict(bundle['frames'][str(fid)],time=time);cam=make_camera(record)
            c2w=np.asarray(record['transform_matrix']);k=np.asarray(record['intrinsics'])
            pred=predict(cam,c2w,k,with_rgb=False)
            valid=(pred['alpha']>=.5)&np.isfinite(pred['expected'])&(pred['expected']>0)&(pred['expected']<20)
            depth=np.where(valid,pred['expected'],0).astype(np.float32)
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.zeros((400,400,3),np.uint8)),
                o3d.geometry.Image(depth),depth_scale=1.,depth_trunc=20.,convert_rgb_to_intensity=False)
            intrinsic=o3d.camera.PinholeCameraIntrinsic(400,400,k[0],k[1],k[2]-.5,k[3]-.5)
            volume.integrate(rgbd,intrinsic,basis@np.linalg.inv(c2w))
            integrations.append(dict(source_frame=fid,render_time=time,integrated_pixels=int(valid.sum())))
            del cam,pred
        mesh=volume.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
        available=mesh.has_triangles() and np.isfinite(np.asarray(mesh.vertices)).all()
        entry=dict(source_frame=n,time=time,available=bool(available),integrations=integrations,
            vertices=len(mesh.vertices),triangles=len(mesh.triangles))
        if available:
            path=folder/f'predicted_mesh_{n:04d}.ply'
            if not o3d.io.write_triangle_mesh(str(path),mesh,write_vertex_colors=False):
                raise IOError('预测mesh写入失败。')
            entry.update(mesh=path.name,sha256=sha(path))
        else:
            entry['missing_reason']='TSDF无有限三角网格；保留该帧并按缺失计分'
        manifest['frames'][str(n)]=entry;write(folder/'manifest.json',manifest)
        print(f'预测mesh {n:04d} available={bool(available)}',flush=True)
        del mesh,volume
    return manifest


def load_gt(bundle,n,alpha):
    gt_dir=Path(bundle['gt']);manifest=read(gt_dir/'manifest.json');entry=manifest['frames'][str(n)]
    path=gt_dir/entry['file']
    if sha(path)!=entry['sha256']:
        raise ValueError('GT文件SHA不符。')
    with np.load(path,allow_pickle=False) as z:
        gt={k:z[k] for k in z.files}
    if gt['depth_z'].shape!=(400,400):
        raise ValueError('GT必须400×400。')
    normal_path=Path(bundle['raw'])/f'normal_exr/normal_{n:04d}.exr'
    normal,normal_valid=load_normal_exr(normal_path,400,400)
    if 'alpha' in gt and not np.allclose(alpha,gt['alpha'],atol=1e-7,rtol=0):
        raise ValueError('GT Alpha与原16位AREA不符。')
    gt_time=entry.get('time_normalized',entry.get('time'))
    if gt_time is not None and abs(gt_time-bundle['frames'][str(n)]['time'])>1e-9:
        raise ValueError('GT时间映射不符。')
    gt.update(alpha=alpha,gt_exr_normal=normal,gt_exr_valid=normal_valid)
    return gt,{str(path):entry['sha256'],str(normal_path):sha(normal_path)}


def masks_for(gt):
    names=[str(n) for n in gt['object_names']];oid=gt['object_id'];sid=gt.get('semantic_id')
    by_name=lambda wanted:np.isin(oid,[i for i,name in enumerate(names) if name in wanted])
    human=by_name(HUMAN);shoe=by_name(['Character_Mesh_Ch02_Sneakers']);sock=by_name(['Character_Mesh_Ch02_Socks'])
    masks=dict(foreground=np.ones_like(oid,bool),interior=erode(gt['alpha']>=.99,2),human=human,platform=by_name(GROUND))
    status={k:'AVAILABLE' for k in masks}
    if sid is None:
        masks.update(leg=np.zeros_like(human),foot=np.zeros_like(human),shoe=shoe,
            footwear=shoe|sock,sock=sock)
        status.update(leg='NOT_AVAILABLE_NO_BONE_SEMANTICS',foot='NOT_AVAILABLE_NO_BONE_SEMANTICS',
            shoe='AVAILABLE_OBJECT_ID',footwear='AVAILABLE_SHOE_AND_SOCK_OBJECT_ID',sock='AVAILABLE_OBJECT_ID')
    else:
        masks.update(leg=sid==2,foot=sid==3,shoe=sid==4,footwear=np.isin(sid,[3,4]),sock=sock)
        status.update({k:'AVAILABLE_SEMANTIC_ID' for k in ('leg','foot','shoe','footwear')})
        status['sock']='AVAILABLE_OBJECT_ID'
    return masks,status


def geometry_metrics(pred,gt):
    depth=gt['depth_z'];depth_base=(gt['alpha']>=.5)&gt['valid']&np.isfinite(depth)&(depth>0)
    _,normal_valid=unit_normal(gt['gt_exr_normal']);base=depth_base&normal_valid&gt['gt_exr_valid']
    visible=np.isfinite(pred['alpha'])&(pred['alpha']>=.5)
    errors={};valids={}
    for method,field in METHODS.items():
        error,valid=normal_errors(pred[field],gt['gt_exr_normal']);support=pred['mesh_valid'] if method=='mesh' else visible
        if method=='depth_normal':support=erode(base&visible,1)
        errors[method]=error['angular_deg'];valids[method]=support&valid
    masks,status=masks_for(gt);rays=camera_rays(gt['camera_to_world'],gt['intrinsics'],400,400);regions={}
    for name,mask in masks.items():
        eligible=base&mask;eligible_depth=depth_base&mask;common=eligible.copy()
        for valid in valids.values():common&=valid
        regions[name]=dict(gt_pixels=int(eligible.sum()),gt_depth_pixels=int(eligible_depth.sum()),
            alpha_valid=int((eligible_depth&visible).sum()),
            normal={m:normal_item(errors[m],valids[m],eligible) for m in METHODS},
            depth={field:depth_item(pred[field],pred['mesh_valid'] if field=='mesh_depth' else visible,
                depth,rays,eligible_depth) for field in DEPTHS},
            common=dict(valid_pixels=int(common.sum()),normal={m:stat(errors[m][common]) for m in METHODS}))
    return regions,status,base


def rgb_metrics(pred,target,alpha,straight,saturation,thresholds):
    foreground=alpha>=.5;lum=straight@np.array([.2126,.7152,.0722])
    lo,hi=thresholds['q_base'],thresholds['q_bright']
    masks=dict(all=np.ones_like(foreground),foreground=foreground,
        dark=foreground&(lum<lo),middle=foreground&(lum>=lo)&(lum<hi),bright=foreground&(lum>=hi))
    rows={};valid=np.isfinite(pred).all(-1)
    for name,mask in masks.items():
        count=int(mask.sum());available=int((mask&valid).sum())
        if available!=count:
            # 非有限RGB属于完整验收失败；不通过排除难像素降低误差。
            raise ValueError('RGB非有限，不能改变固定分母：'+name)
        error=pred-target;values=np.mean(error**2,-1)[mask];mae=np.mean(np.abs(error),-1)[mask]
        mse=float(values.mean()) if count else None
        rows[name]=dict(eligible_pixels=count,valid_pixels=available,coverage=available/count if count else None,
            squared_sum=float(values.sum()),absolute_sum=float(mae.sum()),mse=mse,
            mae=float(mae.mean()) if count else None,psnr=(-10*math.log10(mse)) if mse and mse>0 else None,
            exact_prediction=bool(mse==0) if count else None,
            saturated_channel_fraction=float(saturation[mask].mean()) if count else None)
    return rows


def aggregate_rgb(rows):
    result={}
    for region in rows[0]['rgb']:
        values=[r['rgb'][region] for r in rows];n=sum(v['eligible_pixels'] for v in values)
        mse=sum(v['squared_sum'] for v in values)/n if n else None
        result[region]=dict(eligible_pixels=n,eligible_frames=sum(v['eligible_pixels']>0 for v in values),
            mse_frame_mean=mean_present([v['mse'] for v in values]),mse_pixel_mean=mse,
            mae_frame_mean=mean_present([v['mae'] for v in values]),
            mae_pixel_mean=sum(v['absolute_sum'] for v in values)/n if n else None,
            psnr_frame_mean=None if any(v['exact_prediction'] is True for v in values) else mean_present([v['psnr'] for v in values]),
            psnr_status='INFINITE_FRAME_PRESENT' if any(v['exact_prediction'] is True for v in values) else 'FINITE_OR_EMPTY',
            psnr_from_pixel_mse=(-10*math.log10(mse)) if mse and mse>0 else None,
            exact_prediction_frames=sum(v['exact_prediction'] is True for v in values))
    return result


def gt_mesh_directory(bundle):
    """已实际核验的两种归档布局；不使用manifest中失效的旧绝对路径。"""
    gt=Path(bundle['gt'])
    historical=ROOT/'output/archive/1011-07-六组训练总控/GT-LH-dynamic-jumpingjacks_EL_V2/gt_depth'
    folder=gt.parent/('gt_meshes' if gt==historical or bundle['dataset']=='dlv2' else 'meshes')
    if not (folder/'manifest.json').is_file():
        raise FileNotFoundError('未登记该GT的真实mesh目录：'+str(folder))
    return folder


def verify_correspondence(path,bundle):
    receipt=read(path)
    if receipt.get('status')!='PROVEN_PERSISTENT_VERTEX_INDICES' or receipt.get('dataset')!=bundle['dataset'] or receipt.get('evaluation_only') is not True:
        raise ValueError('持久顶点GT索引对应尚未证明。')
    if receipt.get('candidate_model_loaded') is not False or receipt.get('training_or_selection_use') is not False:
        raise ValueError('对应收据需要独立GT输入审计声明。')
    if sorted(map(int,receipt['frames']))!=list(range(1,151)) or receipt['errors'] or receipt['unsupported_modifiers']:
        raise ValueError('对应证明需要完整150帧、无不稳定拓扑/modifier。')
    manifest=gt_mesh_directory(bundle)/'manifest.json'
    if receipt['source_sha256'].get(str(manifest))!=sha(manifest):raise ValueError('持久顶点收据与本次GT版本不同。')
    if any(not f['topology_equal_reference'] or not f['gt_vertices_exact_equal'] for f in receipt['frames'].values()):
        raise ValueError('对应收据逐帧检查不全通过。')
    verify(receipt['source_sha256'])
    return receipt


def gt_mesh(bundle,n,files):
    folder=gt_mesh_directory(bundle);manifest=read(folder/'manifest.json')
    path=folder/manifest['frames'][str(n)]['mesh'];digest=manifest['meshes'][path.name]['sha256']
    if sha(path)!=digest:raise ValueError('当时GT mesh SHA不符。')
    files[str(folder/'manifest.json')]=sha(folder/'manifest.json');files[str(path)]=digest
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def ray_scene(vertices,triangles):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,np.float32)),
        o3d.core.Tensor(np.asarray(triangles,np.uint32)))
    return scene


def sample_surface(vertices,triangles,count,seed):
    """确定性面积采样，返回三角面索引与重心坐标，禁止按候选误差筛种子。"""
    faces=np.asarray(vertices,dtype=np.float64)[triangles]
    area=np.linalg.norm(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]),axis=1)*.5
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('GT/预测表面没有有限正面积。')
    rng=np.random.default_rng(seed);ids=rng.choice(len(area),count,p=area/area.sum())
    u=np.sqrt(rng.random(count));v=rng.random(count)
    bary=np.stack((1-u,u*(1-v),u*v),axis=1)
    points=(faces[ids]*bary[...,None]).sum(1)
    return points,ids,bary


def sample_regions(mesh,ids):
    names=[str(x) for x in mesh['object_names']];oid=mesh['object_ids'][ids];sid=mesh.get('semantic_ids')
    by_name=lambda wanted:np.isin(oid,[i for i,name in enumerate(names) if name in wanted])
    masks=dict(foreground=np.ones(len(ids),bool),human=by_name(HUMAN),platform=by_name(GROUND),
        shoe=by_name(['Character_Mesh_Ch02_Sneakers']),sock=by_name(['Character_Mesh_Ch02_Socks']))
    if sid is None:
        masks.update(leg=np.zeros(len(ids),bool),foot=np.zeros(len(ids),bool),footwear=masks['shoe']|masks['sock'])
    else:
        sid=sid[ids];masks.update(leg=sid==2,foot=sid==3,shoe=sid==4,footwear=np.isin(sid,[3,4]))
    return masks


def vertex_regions(mesh,vertex_ids):
    triangles=mesh['triangles'];count=len(mesh['vertices'])
    object_ids=np.full(count,-1,dtype=np.int32)
    np.maximum.at(object_ids,triangles.reshape(-1),np.repeat(mesh['object_ids'],3))
    synthetic=dict(object_names=mesh['object_names'],object_ids=object_ids)
    if 'semantic_ids' in mesh:
        votes=np.zeros((count,6),np.int32)
        np.add.at(votes,(triangles.reshape(-1),np.repeat(mesh['semantic_ids'],3)),1)
        synthetic['semantic_ids']=votes.argmax(axis=1)
    return sample_regions(synthetic,vertex_ids)


def distance_summary(values,eligible,available,protocol):
    n=int(eligible.sum());valid=eligible&available&np.isfinite(values);count=int(valid.sum())
    cap=protocol['missing_distance_world_units'];clipped=np.minimum(values[valid],cap)
    return dict(status='EVALUATED' if n else 'NOT_AVAILABLE_NO_ELIGIBLE_GT_SAMPLES',eligible_samples=n,valid_samples=count,coverage=count/n if n else None,
        conditional_distance_world=stat(values[valid]),
        capped_with_missing_mean_world=(float(clipped.sum())+(n-count)*cap)/n if n else None,
        close_counts=[int((valid&(values<=d)).sum()) for d in protocol['distances_world_units']],
        close_fractions=[float((valid&(values<=d)).sum()/n) if n else None for d in protocol['distances_world_units']])


def surface_completeness(mesh,predicted_path,protocol):
    import open3d as o3d
    count=protocol['surface_samples'];points,ids,_=sample_surface(mesh['vertices'],mesh['triangles'],count,protocol['seed'])
    masks=sample_regions(mesh,ids);available=predicted_path is not None
    distance=np.full(count,np.inf);reverse=np.full(count,np.inf)
    if available:
        predicted=o3d.io.read_triangle_mesh(str(predicted_path))
        vertices=np.asarray(predicted.vertices);triangles=np.asarray(predicted.triangles)
        if not len(triangles):raise ValueError('标记available的预测mesh无三角面。')
        distance=ray_scene(vertices,triangles).compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
        reverse_points,_,_=sample_surface(vertices,triangles,count,protocol['seed'])
        reverse=ray_scene(mesh['vertices'],mesh['triangles']).compute_distance(o3d.core.Tensor(reverse_points.astype(np.float32))).numpy()
    recall={name:distance_summary(distance,mask,np.full(count,available),protocol) for name,mask in masks.items()}
    precision=distance_summary(reverse,np.ones(count,bool),np.full(count,available),protocol)
    r=recall['foreground']['close_fractions'];p=precision['close_fractions']
    return dict(gt_samples=count,prediction_available=available,recall_gt_to_prediction=recall,
        precision_prediction_to_gt=precision,
        fscore=[2*x*y/(x+y) if x+y else 0. for x,y in zip(r,p)],
        symmetric_capped_missing_chamfer_world=.5*(recall['foreground']['capped_with_missing_mean_world']+precision['capped_with_missing_mean_world']),
        anatomical_regions='旧GT无semantic_ids时leg/foot为零eligible样本，明确不可用；sock不等同解剖foot')


def bilinear_samples(array,u,v):
    h,w=array.shape[:2];inside=np.isfinite(u)&np.isfinite(v)&(u>=0)&(v>=0)&(u<w-1)&(v<h-1)
    x=np.clip(np.nan_to_num(u),0,w-2).astype(int);y=np.clip(np.nan_to_num(v),0,h-2).astype(int)
    du=np.clip(np.nan_to_num(u,nan=0.,posinf=0.,neginf=0.),0,w-1)-x
    dv=np.clip(np.nan_to_num(v,nan=0.,posinf=0.,neginf=0.),0,h-1)-y
    values=np.stack((array[y,x],array[y,x+1],array[y+1,x],array[y+1,x+1]),axis=1)
    weights=np.stack(((1-du)*(1-dv),du*(1-dv),(1-du)*dv,du*dv),axis=1)
    return np.sum(np.nan_to_num(values)*weights,axis=1),inside,values


def track_observation(points,mesh,gt,pred,c2w,k):
    import open3d as o3d
    local=(points-c2w[:3,3])@c2w[:3,:3];z=-local[:,2]
    u=k[0]*local[:,0]/np.maximum(z,1e-12)+k[2]-.5
    v=-k[1]*local[:,1]/np.maximum(z,1e-12)+k[3]-.5
    alpha,inside,alpha_neighbors=bilinear_samples(erode(gt['alpha']>=.99,2).astype(float),u,v)
    direction=points-c2w[:3,3];rays=np.concatenate((np.broadcast_to(c2w[:3,3],points.shape),direction),axis=1)
    hit=ray_scene(mesh['vertices'],mesh['triangles']).cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
    ray_length=np.linalg.norm(direction,axis=1)
    visible=inside&(z>0)&(alpha_neighbors>=1).all(1)&np.isfinite(hit)&(np.abs(hit-1)*ray_length<=np.maximum(1e-4,1e-3*ray_length))
    residuals={};valids={}
    for field in DEPTHS:
        depth,_,neighbors=bilinear_samples(pred[field],u,v)
        support=pred['mesh_valid'] if field=='mesh_depth' else np.isfinite(pred['alpha'])&(pred['alpha']>=.5)
        _,_,support_neighbors=bilinear_samples(support.astype(float),u,v)
        valid=inside&(support_neighbors>=1).all(1)&np.isfinite(neighbors).all(1)&(neighbors>0).all(1)
        residuals[field]=direction*(depth/np.maximum(z,1e-12)-1)[:,None]
        valids[field]=valid
    return visible,residuals,valids


def temporal_penalty(values,valid,eligible,cap):
    caps=np.broadcast_to(np.asarray(cap),values.shape);n=int(eligible.sum());mask=eligible&valid&np.isfinite(values)
    count=int(mask.sum());penalty=np.where(mask,np.minimum(values,caps),caps)
    interval_means=[float(penalty[i][eligible[i]].mean()) for i in range(len(values)) if eligible[i].any()]
    return dict(eligible_observations=n,valid_observations=count,coverage=count/n if n else None,
        conditional_raw=stat(values[mask]),
        fixed_denominator_capped_missing_mean=float(penalty[eligible].mean()) if n else None,
        equal_interval_capped_missing_mean=mean_present(interval_means),missing_observations=n-count)


def temporal_metrics(visible,residuals,valids,times,masks,protocol):
    dt=np.diff(times)
    if len(times)!=len(TEST) or not np.all(dt>0):raise ValueError('动态评测必须完整15递增TEST时间。')
    cap=protocol['missing_distance_world_units'];result={}
    for name,region in masks.items():
        region_result={}
        for field in DEPTHS:
            r=residuals[field];valid=valids[field];eligible=visible&region[None]
            speed=np.linalg.norm(np.diff(r,axis=0)/dt[:,None,None],axis=2)
            accel=np.linalg.norm((np.diff(r,axis=0)[1:]/dt[1:,None,None]-np.diff(r,axis=0)[:-1]/dt[:-1,None,None])/
                ((dt[1:]+dt[:-1])*.5)[:,None,None],axis=2)
            region_result[field]=dict(position=temporal_penalty(np.linalg.norm(r,axis=2),valid,eligible,cap),
                velocity_residual=temporal_penalty(speed,valid[1:]&valid[:-1],eligible[1:]&eligible[:-1],(2*cap/dt)[:,None]),
                acceleration_residual=temporal_penalty(accel,valid[2:]&valid[1:-1]&valid[:-2],eligible[2:]&eligible[1:-1]&eligible[:-2],
                    (2*cap*(1/dt[1:]+1/dt[:-1])/((dt[1:]+dt[:-1])*.5))[:,None]))
        result[name]=region_result
    return dict(status='EVALUATED_GT_VISIBLE_RESIDUAL_TRACKS' if (visible[1:]&visible[:-1]).any() else 'NOT_AVAILABLE_NO_GT_VISIBLE_PAIRS',time_unit=protocol['time_unit'],times=times.tolist(),
        units=dict(position='world',velocity='world/'+protocol['time_unit'],acceleration='world/'+protocol['time_unit']+'^2'),
        metrics=result,gt_visible_observations=int(visible.sum()),candidate_gt_feedback_allowed=False)


def supplementary_geometry(bundle,meshes,out,files,protocol):
    """固定15TEST表面完整性及GT轨迹残差；只在冻结run内部调用。"""
    rows=[];visible=[];residuals={f:[] for f in DEPTHS};valids={f:[] for f in DEPTHS};times=[]
    reference=None;topology_errors=[];vertex_errors=[]
    persistent=protocol['track_protocol']=='persistent_vertices_v1'
    correspondence=verify_correspondence(protocol['correspondence'],bundle) if persistent else None
    if correspondence:
        files[str(Path(protocol['correspondence']).resolve())]=protocol['correspondence_sha256']
        files.update(correspondence['source_sha256'])
    for n in TEST:
        mesh=gt_mesh(bundle,n,files);record=bundle['frames'][str(n)]
        entry=meshes['frames'][str(n)];predicted_path=out/'predicted_meshes'/entry['mesh'] if entry['available'] else None
        row=dict(frame=n,**surface_completeness(mesh,predicted_path,protocol));rows.append(row)
        if reference is None:
            reference=mesh
            if persistent:
                referenced=np.unique(mesh['triangles'])
                vertex_ids=np.random.default_rng(protocol['seed']).choice(referenced,protocol['track_samples'],replace=len(referenced)<protocol['track_samples'])
                region_masks=vertex_regions(mesh,vertex_ids)
            else:
                _,track_ids,bary=sample_surface(mesh['vertices'],mesh['triangles'],protocol['track_samples'],protocol['seed'])
                region_masks=sample_regions(mesh,track_ids)
        topology_ok=(len(reference['vertices'])==len(mesh['vertices']) and all(np.array_equal(reference[k],mesh[k]) for k in ('triangles','object_ids','object_names')))
        if not topology_ok:
            topology_errors.append(n)
        vertex_ok=False
        if persistent:
            vertex_ok=(__import__('hashlib').sha256(mesh['vertices'].tobytes()).hexdigest()==correspondence['frames'][str(n)]['world_vertices_sha256'] and
                np.array_equal(mesh['object_names'],reference['object_names']) and len(mesh['vertices'])==len(reference['vertices']))
            if not vertex_ok:vertex_errors.append(n)
        if (persistent and vertex_ok) or (not persistent and topology_ok):
            points=mesh['vertices'][vertex_ids] if persistent else (mesh['vertices'][reference['triangles'][track_ids]]*bary[...,None]).sum(1)
            with np.load(out/'arrays'/f'prediction_{n:04d}.npz',allow_pickle=False) as z:pred={k:z[k] for k in ('alpha','expected','median','mesh_depth','mesh_valid','source_alpha')}
            gt,sources=load_gt(bundle,n,pred['source_alpha']);files.update(sources)
            observation=track_observation(points,mesh,gt,pred,np.asarray(record['transform_matrix']),np.asarray(record['intrinsics']))
            visible.append(observation[0])
            for field in DEPTHS:residuals[field].append(observation[1][field]);valids[field].append(observation[2][field])
        fps=protocol['physical_fps'];times.append((n-1)/fps if fps else record['time'])
        write(out/'surface_completeness_partial.json',rows)
    surface=dict(protocol=protocol,rows=rows,
        fscore_frame_mean=[float(np.mean([r['fscore'][i] for r in rows])) for i in range(len(protocol['distances_world_units']))],
        symmetric_capped_missing_chamfer_frame_mean_world=float(np.mean([r['symmetric_capped_missing_chamfer_world'] for r in rows])),
        gt_to_prediction_regions={name:dict(eligible_samples=sum(r['recall_gt_to_prediction'][name]['eligible_samples'] for r in rows),
            close_fraction_frame_mean=[mean_present([r['recall_gt_to_prediction'][name]['close_fractions'][i] for r in rows]) for i in range(len(protocol['distances_world_units']))],
            capped_missing_distance_frame_mean_world=mean_present([r['recall_gt_to_prediction'][name]['capped_with_missing_mean_world'] for r in rows]))
            for name in rows[0]['recall_gt_to_prediction']})
    failed=vertex_errors if persistent else topology_errors
    temporal=dict(status='NOT_AVAILABLE_VERTEX_PROOF_MISMATCH' if persistent else 'NOT_AVAILABLE_TOPOLOGY_MISMATCH',frames=failed) if failed else temporal_metrics(
        np.asarray(visible),{k:np.asarray(v) for k,v in residuals.items()},{k:np.asarray(v) for k,v in valids.items()},np.asarray(times),region_masks,protocol)
    temporal['active_track_protocol']=protocol['track_protocol']
    temporal['triangle_barycentric_diagnostic']=dict(status='NOT_AVAILABLE_TOPOLOGY_MISMATCH' if topology_errors else 'TOPOLOGY_EQUAL',mismatch_frames=topology_errors)
    write(out/'surface_completeness.json',surface);write(out/'temporal_residuals.json',dict(protocol=protocol,**temporal))
    if not failed:
        ids=dict(vertex_ids=vertex_ids) if persistent else dict(triangle_ids=track_ids,barycentric=bary)
        np.savez_compressed(out/'temporal_tracks.npz',gt_visible=np.asarray(visible),times=np.asarray(times),**ids,
            **{'residual_'+k:np.asarray(v) for k,v in residuals.items()},**{'valid_'+k:np.asarray(v) for k,v in valids.items()})
    return surface,temporal


def display_rgb(linear):
    values=np.maximum(np.asarray(linear),0)
    return np.clip(np.where(values<=.0031308,12.92*values,1.055*values**(1/2.4)-.055),0,1)


def scalar_display(values,valid,maximum):
    scaled=np.clip(np.nan_to_num(values)/maximum,0,1)
    color=cv2.applyColorMap(np.rint(scaled*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1].astype(np.float32)/255.
    color[~valid|~np.isfinite(values)]=0
    return color


def panel(label,array):
    rgb=np.clip(np.nan_to_num(array),0,1)
    pixels=np.rint(rgb*255).astype(np.uint8)
    if pixels.ndim==2:pixels=np.repeat(pixels[...,None],3,axis=2)
    canvas=np.zeros((pixels.shape[0]+32,pixels.shape[1],3),np.uint8);canvas[32:]=pixels
    cv2.putText(canvas,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    return canvas


def save_visual(path,pixels):
    if not cv2.imwrite(str(path),pixels[...,::-1]):raise IOError('可视化PNG写入失败：'+str(path))


def local_review_crops(folder,n,pred,target,gt,mesh_normal,mesh_depth,depth_error):
    masks,status=masks_for(gt);result={}
    visible=np.isfinite(pred['alpha'])&(pred['alpha']>=.5)
    normal=np.where((visible&(np.linalg.norm(pred['normal'],axis=-1)>1e-6))[...,None],pred['normal']*.5+.5,0)
    derived=np.where((visible&(np.linalg.norm(pred['depth_normal'],axis=-1)>1e-6))[...,None],pred['depth_normal']*.5+.5,0)
    rows=[('rgb',[
        ('GT corrected RGB',display_rgb(target)),('Predicted RGB',display_rgb(pred['rgb_linear'])),
        ('GT Alpha',np.repeat(gt['alpha'][...,None],3,axis=2)),('Predicted Alpha',np.repeat(pred['alpha'][...,None],3,axis=2))]),
        ('normal',[
            ('GT world normal',np.where(gt['gt_exr_valid'][...,None],gt['gt_exr_normal']*.5+.5,0)),
            ('GS world normal',normal),('Expected Depth normal',derived),('Mesh world normal',mesh_normal)]),
        ('depth_mesh',[
            ('Expected Z [0,10]',scalar_display(pred['expected'],visible,10.)),
            ('Median Z [0,10]',scalar_display(pred['median'],visible,10.)),
            ('Mesh Z [0,10]',mesh_depth),('Z error; missing: magenta',depth_error)]),
        ('separation_components',[
            ('Red dynamic; green static',pred['separation'])])]
    if 'environment_linear' in pred and 'direct_linear' in pred:
        rows[-1][1].extend([
            ('Environment (no directional V)',display_rgb(pred['environment_linear'])),
            ('Direct (unweighted)',display_rgb(pred['direct_linear'])),
            ('Complete forward RGB',display_rgb(pred['rgb_linear']))])
    for name in ('human','leg','foot','shoe'):
        mask=masks[name]&(gt['alpha']>=.5)&gt['valid'];ys,xs=np.where(mask)
        if not len(xs):
            result[name]=dict(status=status.get(name,'NO_GT_PIXELS'),path=None);continue
        x0=max(0,int(xs.min())-16);x1=min(400,int(xs.max())+17);y0=max(0,int(ys.min())-16);y1=min(400,int(ys.max())+17)
        scale=min(256/(x1-x0),256/(y1-y0));width=max(1,round((x1-x0)*scale));height=max(1,round((y1-y0)*scale))
        def tile(label,array):
            resized=cv2.resize(array[y0:y1,x0:x1],(width,height),interpolation=cv2.INTER_AREA)
            canvas=np.zeros((256,256,3),dtype=resized.dtype)
            top,left=(256-height)//2,(256-width)//2
            canvas[top:top+height,left:left+width]=resized
            return panel(label,canvas)
        row_images=[]
        for category,entries in rows:
            panels=[tile(label,array) for label,array in entries]
            while len(panels)<4:panels.append(panel('Not applicable',np.zeros((256,256,3))))
            row_images.append(np.concatenate(panels,axis=1))
        path=folder/f'{name}_{n:04d}.png';save_visual(path,np.concatenate(row_images,axis=0))
        result[name]=dict(status='GENERATED_FIXED_GT_BBOX',path=str(path),bbox_xyxy=[x0,y0,x1,y1],margin_pixels=16,
            resize='等比缩放后居中补黑边',content_size_wh=[width,height],
            panel_rows={category:[label for label,_ in entries] for category,entries in rows},
            limitation='双腿或双鞋仍可共用同语义总框；不宣称左右肢体独立分割，PNG仅显示量化，不能反推指标')
    return result


def continuous_visuals(bundle,frozen,run,predict,meshes,out,files,protocol):
    """固定150连续时刻；Mesh只展示真正重建的15TEST，不生成额外TSDF。"""
    import torch
    from utils.predicted_mesh import PredictedMesh
    folder=out/'four_classes';folder.mkdir();crop_folder=out/'review_crops';crop_folder.mkdir()
    categories=['rgb','normal','depth_mesh','separation']+(['components'] if frozen['stage']=='stage2' else [])
    for category in categories:(folder/category).mkdir()
    writers={};paths={};rows=[];black=np.zeros((400,400,3))
    try:
        with torch.no_grad():
            for n in protocol['video_frames']:
                record=bundle['frames'][str(n)];camera=make_camera(record);c2w=np.asarray(record['transform_matrix']);k=np.asarray(record['intrinsics'])
                pred,parity=predict(camera,c2w,k,with_separation=True)
                target,alpha,_,_,source_files=target_frame(bundle['raw'],n,bundle['radiometry']['background_linear'])
                files.update({str(p):sha(p) for p in source_files})
                mesh_label='Mesh: NOT_RECONSTRUCTED';mesh_normal=black;mesh_depth=black;gt_normal=black;depth_error=black
                crops=None
                if n in TEST:
                    with np.load(out/'arrays'/f'prediction_{n:04d}.npz',allow_pickle=False) as z:
                        valid=z['mesh_valid'];mesh_normal=np.where(valid[...,None],z['mesh_normal']*.5+.5,0)
                        mesh_depth=scalar_display(z['mesh_depth'],valid,10.)
                    mesh_label='Mesh: TEST TSDF' if meshes['frames'][str(n)]['available'] else 'Mesh: MISSING PREDICTION'
                    gt,sources=load_gt(bundle,n,alpha);files.update(sources)
                    gt_normal=np.where(gt['gt_exr_valid'][...,None],gt['gt_exr_normal']*.5+.5,0)
                    gt_region=gt['valid']&(alpha>=.5)
                    predicted_depth_valid=np.isfinite(pred['alpha'])&(pred['alpha']>=.5)&np.isfinite(pred['expected'])&(pred['expected']>0)
                    depth_error=scalar_display(np.abs(pred['expected']-gt['depth_z']),gt_region&predicted_depth_valid,.1)
                    depth_error[gt_region&~predicted_depth_valid]=[1.,0.,1.]
                    crops=local_review_crops(crop_folder,n,pred,target,gt,mesh_normal,mesh_depth,depth_error)
                visible=np.isfinite(pred['alpha'])&(pred['alpha']>=.5)
                normal=np.where((visible&(np.linalg.norm(pred['normal'],axis=-1)>1e-6))[...,None],pred['normal']*.5+.5,0)
                derived=np.where((visible&(np.linalg.norm(pred['depth_normal'],axis=-1)>1e-6))[...,None],pred['depth_normal']*.5+.5,0)
                panels=dict(
                    rgb=[panel('GT corrected linear -> sRGB',display_rgb(target)),panel('Prediction linear -> sRGB',display_rgb(pred['rgb_linear'])),
                        panel('GT Alpha [0,1]',np.repeat(alpha[...,None],3,axis=2)),panel('Prediction Alpha [0,1]',np.repeat(pred['alpha'][...,None],3,axis=2))],
                    normal=[panel('GT normal: TEST only',gt_normal),panel('GS world normal',normal),panel('Expected Depth normal',derived),panel(mesh_label,mesh_normal)],
                    depth_mesh=[panel('Expected Z [0,10] world',scalar_display(pred['expected'],visible,10.)),
                        panel('Median Z [0,10] world',scalar_display(pred['median'],visible,10.)),panel(mesh_label+' Z [0,10]',mesh_depth),panel('TEST err [0,.1]; missing: magenta',depth_error)],
                    separation=[panel('Red: dynamic gate; green: static',pred['separation'])])
                if frozen['stage']=='stage2':
                    for key in ('environment_linear','direct_linear'):
                        if key not in pred:raise ValueError('Stage2四类分量可视化缺少：'+key)
                    panels['components']=[panel('Environment (no directional V)',display_rgb(pred['environment_linear'])),
                        panel('Direct component (unweighted)',display_rgb(pred['direct_linear'])),panel('Complete forward RGB',display_rgb(pred['rgb_linear']))]
                for category,images in panels.items():
                    body=np.concatenate(images,axis=1);canvas=np.zeros((body.shape[0]+24,body.shape[1],3),np.uint8);canvas[:body.shape[0]]=body
                    cv2.putText(canvas,f'frame {n:04d} / 0150',(8,canvas.shape[0]-7),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                    path=folder/category/f'frame_{n:04d}.png';save_visual(path,canvas)
                    if category not in writers:
                        video=folder/f'{category}.mp4';paths[category]=video
                        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),protocol['video_fps'],(canvas.shape[1],canvas.shape[0]))
                        if not writer.isOpened():raise IOError('视频编码器不可用，PNG已保留：'+str(video))
                        writers[category]=writer
                    writers[category].write(canvas[...,::-1])
                rows.append(dict(frame=n,time=record['time'],geometry_parity=parity,
                    mesh_reconstruction='TEST_ONLY' if n in TEST else 'NOT_RECONSTRUCTED',crops=crops))
                if n%10==0:
                    write(folder/'manifest_partial.json',dict(frames=rows,manual_review='PENDING'))
                    print(f'连续四类可视化 {n:04d}/0150',flush=True)
                del camera,pred
    finally:
        for writer in writers.values():writer.release()
    videos={}
    for category,path in paths.items():
        capture=cv2.VideoCapture(str(path));count=0
        try:
            while True:
                ok,_=capture.read()
                if not ok:break
                count+=1
        finally:capture.release()
        png_count=len(list((folder/category).glob('frame_*.png')))
        if count!=150 or png_count!=150:raise ValueError('连续视频/PNG不完整：'+category)
        representatives=[]
        for n in (10,80,150):
            array=cv2.imread(str(folder/category/f'frame_{n:04d}.png'))
            if array is None:raise IOError('固定代表帧读取失败。')
            representatives.append(array[...,::-1])
        contact=folder/f'{category}_contact_0010_0080_0150.png';save_visual(contact,np.concatenate(representatives,axis=0))
        videos[category]=dict(path=str(path),decoded_frames=count,pngs=png_count,sha256=sha(path),contact_sheet=str(contact))
    result=dict(status='PASS_FILES_AND_DECODING_ONLY',frames=rows,videos=videos,protocol=protocol,
        four_class_manual_review={k:'PENDING' for k in ('rgb','normal','depth_mesh','separation')},
        stage2_component_review='PENDING' if frozen['stage']=='stage2' else 'NOT_APPLICABLE',
        limitations='目检未自动判PASS；正侧背与鞋尖应检查完整环绕视频和固定GT局部裁剪，不宣称三个代表帧已覆盖所有方向',
        model_metrics_computed_only_on_test=TEST,candidate_gt_feedback_allowed=False)
    write(folder/'manifest.json',result)
    return result


def evaluate(args):
    import torch
    from utils.predicted_mesh import PredictedMesh
    frozen=read(args.freeze)
    if frozen.get('evaluation_allowed') is not True or frozen.get('selection_uses_test_or_gt') is not False:
        raise ValueError('缺少冻结且无TEST选择声明。')
    if sha(frozen['source'])!=frozen['source_sha256'] or frozen['test_frames']!=TEST:
        raise ValueError('冻结source或TEST集合变化。')
    verify(frozen['source_files']);verify(frozen['code_sha256'])
    bundle=source_bundle(frozen['source']);frozen_protocol=frozen.get('supplementary_protocol',{})
    protocol=supplementary_protocol(bundle,frozen_protocol.get('track_protocol','triangle_barycentric_v1'),frozen_protocol.get('correspondence'))
    if frozen.get('supplementary_protocol')!=protocol:
        raise ValueError('补充指标/视频预算未冻结或发生变化，必须先重新freeze，不兼容旧冻结收据。')
    if sha(frozen['gt_mesh_manifest'])!=frozen['gt_mesh_manifest_sha256']:
        raise ValueError('冻结GT动态网格manifest改变。')
    out=fresh(args.output)
    log=(out/'evaluation.log').open('x');original_streams=sys.stdout,sys.stderr
    sys.stdout,sys.stderr=LogStream(sys.stdout,log),LogStream(sys.stderr,log)
    launcher_argv=list(getattr(sys,'orig_argv',[sys.executable]+sys.argv))
    environment={key:os.environ[key] for key in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'] if key in os.environ}
    command=shlex.join(['env']+[key+'='+value for key,value in environment.items()]+launcher_argv)
    (out/'README.md').write_text('# 冻结动态模型统一评价\n\n完整执行/渲染命令：\n\n```bash\n'+command+
        '\n```\n\nsource：`'+frozen['source']+'`；model：`'+frozen['model']+'`；iteration：'+str(frozen['iteration'])+
        '。\n渲染输出：arrays；预测网格：predicted_meshes；日志：evaluation.log。'+
        '\n定量：metrics.json、surface_completeness.json、temporal_residuals.json；逐帧：per_frame_partial.json。'+
        '\n代表图片：four_classes/*_contact_0010_0080_0150.png；视频：four_classes/{rgb,normal,depth_mesh,separation}.mp4；局部：review_crops。'+
        '\nRGB（含Alpha）/法线/Depth-Mesh/静动态分离四类可视化目检：均PENDING，Stage2额外EL/DL分量PENDING。'+
        '\n150连续帧以原动态相机展示；只有15TEST重建Mesh，其余帧明确标注未重建。完整指标定义/预算/缺失惩罚/局限保存在supplementary_protocol.json。'+
        '\n最终结论：PENDING；工程执行通过不等于科学目标通过。\n')
    write(out/'supplementary_protocol.json',protocol)
    write(out/'freeze_snapshot.json',frozen)
    write(out/'command.json',dict(argv=sys.argv,launcher_argv=launcher_argv,environment=environment,shell=command,source=frozen['source'],
        model=frozen['model'],iteration=frozen['iteration'],output=str(out),test_gt_feedback_allowed=False))
    try:
        run,predict=runtime(frozen)
        with torch.no_grad():
            meshes=build_meshes(bundle,frozen,out,predict)
            rows=[];files={};(out/'arrays').mkdir()
            for n in TEST:
                record=bundle['frames'][str(n)];camera=make_camera(record)
                c2w=np.asarray(record['transform_matrix']);k=np.asarray(record['intrinsics'])
                pred,parity=predict(camera,c2w,k)
                entry=meshes['frames'][str(n)]
                if entry['available']:
                    mesh_path=out/'predicted_meshes'/entry['mesh']
                    if sha(mesh_path)!=entry['sha256']:raise ValueError('预测mesh SHA变化。')
                    pred.update(PredictedMesh(mesh_path).render(c2w,k,400,400))
                else:
                    pred.update(mesh_normal=np.zeros((400,400,3)),mesh_depth=np.zeros((400,400)),mesh_valid=np.zeros((400,400),bool))
                target,alpha,straight,saturation,paths=target_frame(bundle['raw'],n,bundle['radiometry']['background_linear'])
                files.update({str(p):sha(p) for p in paths});gt,gt_files=load_gt(bundle,n,alpha);files.update(gt_files)
                if not np.allclose(gt['camera_to_world'],c2w,atol=1e-5,rtol=0) or not np.allclose(gt['intrinsics'],k,atol=1e-5,rtol=0):
                    raise ValueError('模型/GT相机不一致。')
                regions,status,base=geometry_metrics(pred,gt)
                row=dict(source_frame=n,time=record['time'],regions=regions,region_status=status,geometry_parity=parity,
                    rgb=rgb_metrics(pred['rgb_linear'],target,alpha,straight,saturation,bundle['thresholds']))
                rows.append(row);write(out/'per_frame_partial.json',rows)
                np.savez_compressed(out/'arrays'/f'prediction_{n:04d}.npz',**pred,target_rgb_linear=target,
                    source_alpha=alpha,normal_eligible=base,camera_to_world=c2w,intrinsics=k)
                print(f'冻结TEST评测 {n:04d}',flush=True);del camera,pred,gt
            surface,temporal=supplementary_geometry(bundle,meshes,out,files,protocol)
            visuals=continuous_visuals(bundle,frozen,run,predict,meshes,out,files,protocol)
        if [r['source_frame'] for r in rows]!=TEST:raise ValueError('完整15TEST分母未保留。')
        summary=aggregate(rows);normal=summary['foreground']['normal']
        composite=float(np.mean([normal[m]['missing180_frame_mean_deg'] for m in METHODS]))
        gt_metadata=read(Path(bundle['gt'])/'manifest.json')
        verify(files);verify(frozen['source_files']);verify(bundle['source_files']);verify(frozen['code_sha256'])
        write(out/'metrics.json',dict(status='PASS_EXECUTION_ONLY',scientific_goal='NOT_DECIDED',
            model=frozen['model'],iteration=frozen['iteration'],stage=frozen['stage'],dataset=bundle['dataset'],
            unit_system=gt_metadata.get('unit_system'),meters_per_world_unit=gt_metadata.get('meters_per_world_unit'),
            effective_rasterizer=run.effective_rasterizer,
            frames=TEST,source_files=files,source_bundle_sha256=frozen['source_sha256'],
            protocol=dict(alpha_threshold=.5,resolution=[400,400],signed_world_normal=True,missing_normal_degrees=180,
                normal_primary='三法线固定GT前景分母，逐帧等权，三方法等权；Expected Depth保留1像素有效邻域',
                primary_region='GT Alpha>=.5且GT ray hit且GT EXR normal有效；不依候选预测改变',
                rgb='先逆sRGB，减去(1-原Alpha)*固定TRAIN背景B，再AREA；候选SH码值逆sRGB；Stage2直接render_linear',
                thresholds=bundle['thresholds'],depth='正camera Z，无缩放/对齐；条件误差、覆盖、固定分母阈值命中/缺失相对罚分'),
            normal_composite_fixed_denominator_mae=composite,summary=summary,rgb=aggregate_rgb(rows),rows=rows,
            supplementary_protocol=protocol,
            surface_completeness=dict(file='surface_completeness.json',fscore_frame_mean=surface['fscore_frame_mean'],
                symmetric_capped_missing_chamfer_frame_mean_world=surface['symmetric_capped_missing_chamfer_frame_mean_world']),
            temporal_residuals=dict(file='temporal_residuals.json',status=temporal['status']),
            four_classes_manifest='four_classes/manifest.json',
            pending=['四类及EL/DL分量人工目检','科学目标最终判定']+
                (['局部GT语义不可用，详见逐帧region_status'] if any(str(r['region_status'].get(k,'')).startswith('NOT_AVAILABLE') for r in rows for k in ('leg','foot','shoe')) else []),
            candidate_gt_feedback_allowed=False))
        write(out/'COMPLETED.json',dict(status='PASS_EXECUTION_ONLY',frames=15,video_frames=150,
            videos={k:v['decoded_frames'] for k,v in visuals['videos'].items()},temporal_status=temporal['status'],
            manual_review='PENDING',scientific_goal='NOT_DECIDED'))
        with (out/'README.md').open('a') as stream:
            stream.write('\n统一三法线固定分母综合MAE：'+str(composite)+'度。\n表面完整性：surface_completeness.json；GT轨迹残差：'+temporal['status']+
                '，详见temporal_residuals.json。四类连续视频均已解码验证150帧，固定代表图及人体/腿足鞋裁剪路径见four_classes/manifest.json。'+
                '\n目检结论：RGB待人工；法线待人工；Depth/Mesh待人工；静动态/EL-DL待人工。'+
                '\n最终结论：**PENDING**（指标与产物执行PASS，四类人工目检及科学目标判定待完成）。\n')
        print('PASS：统一指标执行完成；不代表科学目标达成。',flush=True)
    except Exception as exc:
        traceback.print_exc()
        write(out/'FAILED.json',dict(status='FAILED',error=str(exc),traceback=traceback.format_exc(),assets_retained=True))
        with (out/'README.md').open('a') as stream:
            stream.write('\n最终结论：**FAILED**；checkpoint保持原位，渲染/统计/日志保留。原因：'+str(exc)+'\n')
        raise
    finally:
        sys.stdout,sys.stderr=original_streams;log.close()


def self_check():
    # 仅合成CPU数值，不读取任何候选或GT数据。
    errors=np.array([[10.,20.]]);eligible=np.ones((1,2),bool)
    item=normal_item(errors,np.array([[True,False]]),eligible)
    assert item['missing180_mae']==95 and item['coverage']==.5
    missing=normal_item(errors,np.zeros((1,2),bool),eligible)
    assert missing['missing180_mae']==180 and missing['valid_pixels']==0
    depth=depth_item(np.array([[1.,0.]]),np.ones((1,2),bool),np.ones((1,2)),
        np.array([[[0.,0.,0.,0.,0.,-1.],[0.,0.,0.,0.,0.,-1.]]]),eligible)
    assert depth['coverage']==.5 and depth['clipped_relative_missing_mean']==.5
    def synthetic_row(n,item):
        return dict(source_frame=n,regions={'foreground':dict(gt_pixels=2,gt_depth_pixels=2,
            alpha_valid=item['valid_pixels'],normal={m:item for m in METHODS},depth={m:depth for m in DEPTHS},
            common=dict(valid_pixels=0,normal={m:stat([]) for m in METHODS}))})
    result=aggregate([synthetic_row(10,item),synthetic_row(20,missing)])['foreground']['normal']['gs_render']
    assert result['missing180_frame_mean_deg']==137.5 and result['missing_prediction_frames']==[20]
    assert np.allclose(linearize(np.array([0.,.04045,1.])),[0.,.0031308049535603713,1.])
    alpha=np.ones((2,2));target=np.ones((2,2,3))*.5
    rgb=rgb_metrics(np.zeros_like(target),target,alpha,target,np.zeros_like(target),dict(q_base=.4,q_bright=.7))
    assert rgb['foreground']['eligible_pixels']==4 and rgb['foreground']['mse']==.25
    assert rgb['middle']['eligible_pixels']==4 and rgb['dark']['eligible_pixels']==0
    sampled,face_ids,bary=sample_surface(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]]),128,1019)
    assert np.allclose(bary.sum(1),1) and (face_ids==0).all() and (sampled>=0).all()
    assert np.array_equal(sampled,sample_surface(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]]),128,1019)[0])
    interpolated,inside,_=bilinear_samples(np.array([[0.,2.],[2.,4.]]),np.array([.5]),np.array([.5]))
    assert inside.all() and np.allclose(interpolated,2.)
    times=np.arange(15,dtype=float);residual=np.zeros((15,2,3));residual[:,1,0]=times
    present=np.ones((15,2),bool);present[6]=False
    temporal=temporal_metrics(np.ones((15,2),bool),{k:residual for k in DEPTHS},{k:present for k in DEPTHS},
        times,{'foreground':np.ones(2,bool)},dict(missing_distance_world_units=1.,time_unit='second'))
    velocity=temporal['metrics']['foreground']['expected']['velocity_residual']
    acceleration=temporal['metrics']['foreground']['expected']['acceleration_residual']
    assert velocity['eligible_observations']==28 and velocity['missing_observations']==4
    assert velocity['conditional_raw']['mean']==.5 and acceleration['conditional_raw']['mean']==0
    assert acceleration['fixed_denominator_capped_missing_mean']>0
    assert all(TEST[i]<TEST[i+1] for i in range(14)) and len(TRAIN)==117 and not set(TRAIN)&set(TEST+DEV)
    print('PASS：合成CPU检查，未读取候选/GT、未初始化CUDA。',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',required=True,choices=('prepare','freeze','check','self-check','run'))
    parser.add_argument('--dataset',choices=('elv2','v3','dlv2'),default='v3')
    parser.add_argument('--source',type=Path,help='隔离prepare产物source.json')
    parser.add_argument('--gt',type=Path,help='真实gt_depth目录；默认已定位新/旧目录')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--iteration',type=int)
    parser.add_argument('--stage',choices=('stage1','stage2'),default='stage1')
    parser.add_argument('--stage2-module',default='LIHAN.scripts.dynamic_v3_surface')
    parser.add_argument('--freeze',type=Path,help='freeze产物freeze.json')
    parser.add_argument('--confirm-frozen',action='store_true')
    parser.add_argument('--track-protocol',choices=('triangle_barycentric_v1','persistent_vertices_v1'),default='triangle_barycentric_v1',
        help='freeze时显式选择预登记协议；三角失败不得隐式替换')
    parser.add_argument('--correspondence',type=Path,help='persistent_vertices_v1需独立150帧correspondence.json证明')
    args=parser.parse_args()
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve():
        parser.error('本入口登记ushas/lumimotion-ushas。')
    required=dict(prepare=('output',),freeze=('source','model','iteration','output'),check=('source',),run=('freeze','output'),**{'self-check':()})
    if any(getattr(args,k) is None for k in required[args.action]):
        parser.error('缺少参数：'+','.join(required[args.action]))
    if args.action=='prepare':prepare(args)
    elif args.action=='freeze':freeze(args)
    elif args.action=='check':source_bundle(args.source);print('PASS：隔离source SHA及固定划分。')
    elif args.action=='self-check':self_check()
    else:evaluate(args)


if __name__=='__main__':
    main()
