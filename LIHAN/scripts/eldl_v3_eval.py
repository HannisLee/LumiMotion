#!/usr/bin/env python3
"""V3独立评测：16位隔离输入、预测几何、固定分母统计及四类可视化。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import traceback

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONTROL = ROOT / 'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控'
RAW = ROOT / 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3'
SOURCE = CONTROL / 'eval_only/eval_source_v1'
GT = CONTROL / 'eval_only/gt_v1'
PYTHON = Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python')
TEST = list(range(10, 151, 10))
TRAIN = [n for n in range(1, 151) if n not in TEST]
HELD = [n for n in TRAIN if 61 <= n <= 79]
FIT = [n for n in TRAIN if n not in HELD]
SPLITS = dict(train=TRAIN, test=TEST, fit=FIT, held=HELD, all=list(range(1, 151)))
REPRESENTATIVES = [10, 80, 150]
METHODS = dict(gs_render='normal', depth_normal='depth_normal', mesh='mesh_normal')
DEPTHS = ('expected', 'median', 'mesh_depth')
DISTANCES = [.01, .02, .05, .10]
RELATIVE = [.01, .02, .05, .10]
HUMAN = ['Character_Mesh_Ch02_' + n for n in ('Body', 'Cloth', 'Eyelashes', 'Hair', 'Sneakers', 'Socks')]
GROUND = ['Reference_Checkered_Platform', 'Reference_Platform_Base']
PROTOCOL_DOC = ROOT / 'LIHAN/DOC/报告/1019-02-ELDL-V3固定评测协议.md'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise AssertionError('来源SHA变化：' + path)


def output_path(path, create=False):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'output'):
        raise ValueError('所有输出必须位于仓库output。')
    if create:
        path.mkdir(parents=True, exist_ok=False)
    return path


def save_png(path, array):
    pixels = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    if pixels.ndim == 3:
        pixels = pixels[..., ::-1]
    if not cv2.imwrite(str(path), pixels):
        raise IOError('PNG写入失败。')


def prepare_source(args):
    source = output_path(args.source)
    if not source.is_relative_to(CONTROL / 'eval_only'):
        raise ValueError('评测输入只允许建立在eval_only。')
    source.mkdir(parents=True, exist_ok=False)
    (source / 'images').mkdir()
    camera_path = RAW / 'camera.json'
    cameras = read(camera_path)
    template_path = CONTROL / 'data_train_v2/transforms_train.json'
    template = read(template_path)
    metadata = {k: v for k, v in template.items() if k != 'frames'}
    metadata['rgba16'] = True
    records = {}
    manifest = dict(version=1, evaluation_only=True, rgba16=True, raw_rgb_preserved=True,
        rgb_source='原image16位RGB；Alpha仅来自原albedo透明度，不使用材质颜色',
        camera_json=str(camera_path.resolve()), camera_json_sha256=sha(camera_path),
        train_template=str(template_path), train_template_sha256=sha(template_path),
        source_files={}, files={}, splits=SPLITS, frames={})
    for n in range(1, 151):
        rgb_path = RAW / f'image/image_{n:04d}.png'
        alpha_path = RAW / f'albedo/albedo_{n:04d}.png'
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        rgba_alpha = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
        if any(a is None or a.dtype != np.uint16 or a.shape != (800, 800, 4) for a in (rgb, rgba_alpha)):
            raise ValueError('原输入必须是800×800 uint16 RGBA。')
        combined = np.concatenate((rgb[..., :3], rgba_alpha[..., 3:4]), axis=-1)
        filename = f'image_{n:04d}.png'
        destination = source / 'images' / filename
        if not cv2.imwrite(str(destination), combined):
            raise IOError('16位评测输入写入失败。')
        restored = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        if not np.array_equal(restored, combined):
            raise AssertionError('16位输入发生量化或颜色变更。')
        camera = cameras['frames'][f'{n:04d}']; intrinsic = camera['intrinsics']
        records[n] = dict(file_path=filename, source_frame=n, time=(n-1)/149.,
            time_seconds_from_first_frame=(n-1)/cameras['fps'],
            fl_x=intrinsic['fx'], fl_y=intrinsic['fy'], cx=intrinsic['cx'], cy=intrinsic['cy'],
            camera_angle_x=intrinsic['fov_x_rad'], camera_angle_y=intrinsic['fov_y_rad'],
            transform_matrix=camera['extrinsics']['camera_to_world'])
        manifest['source_files'][str(rgb_path.resolve())] = sha(rgb_path)
        manifest['source_files'][str(alpha_path.resolve())] = sha(alpha_path)
        manifest['files']['images/' + filename] = sha(destination)
        manifest['frames'][str(n)] = dict(rgb_exact=True, alpha_exact=True, dtype='uint16')
    for split, ids in SPLITS.items():
        name = f'transforms_{split}.json'
        write(source / name, dict(metadata, frames=[records[n] for n in ids]))
        manifest['files'][name] = sha(source / name)
    # 原训练帧的时间与相机必须完全沿用，不能为评测重排时间。
    for record in template['frames']:
        actual = records[record['source_frame']]
        if actual['time'] != record['time'] or not np.array_equal(actual['transform_matrix'], record['transform_matrix']):
            raise AssertionError('评测源与训练时间/相机不同。')
    write(source / 'manifest.json', manifest)
    (source / 'README.md').write_text('# V3隔离评测输入\n\n原始image的16位RGB与原albedo的16位Alpha组成RGBA；不使用albedo RGB。150张均逐像素无损验证。保留原始c2w与全序列time，transforms声明rgba16=true。外部test15、TRAIN135、内部held18、fit117固定。仅评测使用，禁止作为训练source。\n\n完整命令：`' + shlex.join([str(PYTHON), str(Path(__file__).resolve()), '--action', 'prepare-source', '--source', str(source)]) + '`。\n\n最终结论：PASS（输入准备；未执行模型评测）。\n')
    print('PASS：150帧16位隔离评测输入完成。', flush=True)


def check_source(source, full=True, scope='test'):
    source = Path(source).resolve(); manifest = read(source / 'manifest.json')
    if manifest['splits'] != SPLITS or not manifest['evaluation_only'] or not manifest['rgba16']:
        raise ValueError('评测源协议不符。')
    if full:
        allowed=set(TRAIN) if scope=='held' else set(range(1,151))
        allowed_splits={'train','fit','held'} if scope=='held' else set(SPLITS)
        files={}
        for path,expected in manifest['files'].items():
            if path.startswith('images/'):
                if int(Path(path).stem.split('_')[-1]) not in allowed:continue
            elif Path(path).stem.removeprefix('transforms_') not in allowed_splits:continue
            files[str(source/path)]=expected
        verify_files(files)
    camera = read(manifest['camera_json'])
    if sha(manifest['camera_json']) != manifest['camera_json_sha256']:
        raise ValueError('原相机发生变化。')
    for split, ids in SPLITS.items():
        if scope=='held' and split in ('test','all'):continue
        transforms = read(source / f'transforms_{split}.json')
        if transforms.get('rgba16') is not True or [r['source_frame'] for r in transforms['frames']] != ids:
            raise ValueError('固定split或16位声明不符。')
        for record in transforms['frames']:
            expected = camera['frames'][f"{record['source_frame']:04d}"]['extrinsics']['camera_to_world']
            if not np.array_equal(record['transform_matrix'], expected) or record['time'] != (record['source_frame']-1)/149.:
                raise ValueError('评测相机/时间不是原始值。')
    return manifest


def freeze_protocol(args):
    source = check_source(args.source,scope=args.split)
    protocol=dict(version=1, dataset='LH-static-jumpingjacks-ELDL-V3', splits=SPLITS,
        evaluation_split=args.split, resolution=2, alpha_threshold=.5, interior_radius=2,
        source=str(args.source.resolve()), source_manifest_sha256=sha(args.source / 'manifest.json'),
        gt=str(args.gt.resolve()), representatives=REPRESENTATIVES, video_frames=150, video_fps=24,
        normal_methods=METHODS, depth_fields=DEPTHS, aggregation=['pixel', 'equal_frame'],
        signed_normal=True, missing_normal_penalty_deg=180, relative_distance_missing_penalty=1,
        distance_thresholds_world_units=DISTANCES, relative_distance_thresholds=RELATIVE,
        distance_denominator='固定GT分区深度像素；缺失/低Alpha/非有限/非正深度均失败',
        tsdf=dict(voxel_size=.02, sdf_trunc=.08, depth_trunc=20, alpha_threshold=.5,
                  expected_depth=True, train_cameras_only=True, test_camera_reconstruction=False),
        scientific_thresholds=None, historical_dl_v2_thresholds_reused=False,
        protocol_document=str(PROTOCOL_DOC), protocol_document_sha256=sha(PROTOCOL_DOC),
        evaluator_sha256=sha(Path(__file__)), source_splits_verified=True)
    protocol['test_light_prediction']='TRAIN directions component-wise linear interpolation then normalization (nlerp); nearest endpoint hold'
    if args.action!='smoke' and args.split=='test':
        protocol['gt_manifest_sha256']=sha(args.gt/'gt_depth/manifest.json')
        protocol['partition_definition_sha256']=sha(args.gt/'partition_definition.json')
    return protocol


def model_record(model, iteration, split, smoke=False):
    model = Path(model).resolve()
    marker = model / 'eldl_v3_training.json'
    record = read(marker)
    from LIHAN.scripts.eldl_v3_model_record import fixed_eldl_record_files
    extra_record_files = fixed_eldl_record_files(model, record)
    if extra_record_files is not None:
        # 混合模型使用另一种训练记录；训练白名单决定实际相机，不能假定135。
        manifest = read(Path(record['args']['source_path']) / 'no_gt_manifest.json')
        if manifest.get('gt_read') is not False or manifest.get('init_weights_loaded') is not False:
            raise ValueError('混合模型训练输入未通过无GT/无历史初始化声明。')
        static = read(model / 'static_dl_training.json')
        record = dict(record, baseline='eldl_fixed', init_weights_loaded=False,
            train_frames=manifest['train_frames'], rasterizer_provenance=static['rasterizer_provenance'],
            source_code_root=str(Path(next(iter(record['new_sources']))).parents[1]))
    if record.get('gt_supervision') is not False or record.get('init_weights_loaded') is not False:
        raise ValueError('需要V3无GT/无历史初始化训练记录。')
    if record['baseline'] not in ('native', 'static_sh', 'eldl_fixed'):
        raise ValueError('尚未登记该模型adapter，禁止猜测渲染语义。')
    expected = FIT if split == 'held' else TRAIN
    if record['train_frames'] != expected and not smoke:
        raise ValueError('内部117与外部135模型必须按实际训练划分分别评测。')
    if not smoke and ('smoke_test' in model.parts or record['args'].get('smoke')):
        raise ValueError('冒烟模型只允许action=smoke，不可进入正式精度表。')
    files = [model / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply',
             model / 'deform' / f'iteration_{iteration}' / 'deform.pth', model / 'cfg_args']
    files += extra_record_files or []
    if record['baseline']=='eldl_fixed':
        files += [model/'static_dl_training.json',model/f'static_dl_lighting/iteration_{iteration}/lighting.json']
    return record, {str(p): sha(p) for p in files}


def interpolate_train_direction(frames,number):
    """仅TRAIN估计方向作分量线性插值后归一化；序列边界保持最近TRAIN。"""
    ids=sorted(map(int,frames))
    if any(n in TEST for n in ids):raise ValueError('估计光资产不能含外部test方向。')
    if number<=ids[0]:left=right=ids[0]
    elif number>=ids[-1]:left=right=ids[-1]
    elif number in ids:left=right=number
    else:
        right=next(n for n in ids if n>number);left=ids[ids.index(right)-1]
    a=np.asarray(frames[f'{left:04d}']['light_dir_world'],dtype=np.float64).reshape(3)
    b=np.asarray(frames[f'{right:04d}']['light_dir_world'],dtype=np.float64).reshape(3)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or abs(np.linalg.norm(a)-1)>1e-4 or abs(np.linalg.norm(b)-1)>1e-4:
        raise ValueError('TRAIN估计方向必须为有限单位向量。')
    weight=(number-left)/(right-left) if left!=right else 0.
    value=(1-weight)*a+weight*b
    norm=np.linalg.norm(value)
    if norm<1e-6:raise ValueError('TRAIN方向线性插值退化为零，不能任意选方向。')
    value/=norm
    return value,dict(left_train=left,right_train=right,weight=weight,
        method='TRAIN_exact' if number==left==right else ('TRAIN_endpoint_hold' if left==right else 'TRAIN_nlerp'))


def adapter(model, iteration, source, split, smoke=False):
    """延迟导入GPU依赖；CPU准备/校验不触发CUDA构建。"""
    import torch
    import diff_surfel_rasterization as backend
    from scripts.eval_geometry_gt import GeometryRun
    from scene.cameras import Camera
    from scene.dataset_readers import blender_c2w_to_camera_extrinsics
    from utils.geometry_eval_utils import unit_normal
    from gaussian_renderer import render
    from argparse import Namespace
    if getattr(backend, 'provenance', {}).get('method') != 'repository_lumimotion_cuda_rasterizer':
        raise RuntimeError('必须经bootstrap_1011绑定原生CUDA光栅器。')
    record, files = model_record(model, iteration, split, smoke)

    class V3Run(GeometryRun):
        def __init__(self):
            super().__init__(model, iteration, source)
            self.record = record
            self.config = dict(record['args'])
            self.config.update(train_light_folder='images', source_path=str(source))
            self.files.update(files)
            self.training_marker_snapshot = record
            self.backend = backend.provenance
            self.source_manifest=read(self.source/'manifest.json')
            self.files[str(self.source/'manifest.json')]=sha(self.source/'manifest.json')
            for split_name in (SPLITS if split=='test' else ('train','fit','held')):
                relative=f'transforms_{split_name}.json';path=self.source/relative
                if sha(path)!=self.source_manifest['files'][relative]:raise ValueError('评测相机split文件SHA变化。')
                self.files[str(path)]=self.source_manifest['files'][relative]
            degree_step = self.config.get('oneupSHdegree_step', 1000)
            self.pc.active_sh_degree = min(self.pc.max_sh_degree, iteration // degree_step)
            self.pc.no_binary_separation = bool(self.config.get('no_binary_separation', False) or
                iteration <= self.config.get('binarization_warm_up', 1000))
            # 加载训练时快照的核心算法哈希；不把不同版本的输入缩放混为一个训练。
            self.training_core_comparison = {}
            code_root = Path(record.get('source_code_root', ROOT))
            relatives=['scene/gaussian_model.py', 'scene/deform_model.py', 'utils/time_utils.py',
                       'gaussian_renderer/__init__.py', 'scene/cameras.py']
            if record['baseline']=='eldl_fixed':relatives+=['scene/eldl_v3_photometric.py','scene/static_dl_photometric.py']
            for relative in relatives:
                old = code_root / relative; current = ROOT / relative
                equal = old.is_file() and sha(old) == sha(current)
                self.training_core_comparison[relative] = equal
                if not equal:
                    raise ValueError('训练快照与运行核心不一致，需要专门adapter：' + relative)
                self.files[str(current)] = sha(current)
            if record['rasterizer_provenance']['source_sha256'] != self.backend['source_sha256']:
                raise ValueError('训练/评测CUDA光栅来源不同。')
            if record['baseline'] in ('static_sh','eldl_fixed'):
                for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
                    layer = getattr(self.deform.deform, name)
                    if torch.count_nonzero(layer.weight) or torch.count_nonzero(layer.bias):
                        raise ValueError('静态模型几何头不是严格全零。')
            if record['baseline']=='eldl_fixed':
                from scene.static_dl_photometric import StaticVisibility
                lighting=read(model/f'static_dl_lighting/iteration_{iteration}/lighting.json')
                asset=lighting['lighting_asset']
                if (not lighting['lighting_frozen'] or lighting['frozen_light_check']!='PASS'
                        or asset['gt_read'] is not False or asset['source_split']!='TRAIN'
                        or asset['environment_mode']!='shared_rgb_constant'):
                    raise ValueError('需要checkpoint内保存的冻结TRAIN估计EL+DL资产。')
                if sorted(map(int,asset['frames']))!=record['train_frames']:
                    raise ValueError('混合照明资产含非TRAIN帧或缺失TRAIN帧。')
                directions={};provenance={}
                for n in range(1,151):
                    value,info=interpolate_train_direction(asset['frames'],n)
                    directions[f'{n:04d}']=torch.as_tensor(value,dtype=torch.float32,device='cuda').reshape(1,3)
                    provenance[str(n)]=info
                class Lights:
                    def __init__(self):
                        self.ambient_rgb=torch.as_tensor(asset['ambient_rgb'],dtype=torch.float32,device='cuda')
                        self.intensity=torch.as_tensor(asset['direct_intensity'],dtype=torch.float32,device='cuda').reshape(1)
                        if not torch.isfinite(self.ambient_rgb).all() or (self.ambient_rgb<0).any() or not torch.isfinite(self.intensity).all() or (self.intensity<=0).any():
                            raise ValueError('冻结EL/DL估计资产不合法。')
                    def for_camera(self,camera):
                        from scene.stage2_photometric import frame_key
                        return directions[frame_key(camera.image_name_train_light)],self.intensity
                self.physical_lights=Lights()
                self.visibility=StaticVisibility(bias=self.config['static_dl_shadow_bias'],update_interval=self.config['static_dl_bvh_interval'])
                self.light_prediction_provenance=dict(method='TRAIN-only nlerp; endpoint hold',
                    gt_light_read=False,checkpoint_asset=True,per_frame=provenance)

        def camera(self, metadata, frame, divisor=2):
            path = self.source / 'images' / frame['file_path']
            expected_sha=self.source_manifest['files']['images/'+frame['file_path']]
            if sha(path)!=expected_sha:raise ValueError('评测图像SHA变化。')
            self.files[str(path)]=expected_sha
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if metadata.get('rgba16') is not True or raw is None or raw.dtype != np.uint16:
                raise ValueError('V3相机只接受显式16位输入。')
            oh, ow = raw.shape[:2]; w, h = round(ow/divisor), round(oh/divisor)
            alpha = cv2.resize(raw[..., 3].astype(np.float32)/65535., (w,h), interpolation=cv2.INTER_AREA)
            # 仅RGB指标用原始RGB作黑背景合成，保持与FloatRGBA相同先合成再AREA。
            rgba = raw[..., [2,1,0,3]].astype(np.float32)/65535.
            target_rgb = cv2.resize(rgba[..., :3]*rgba[..., 3:4], (w,h), interpolation=cv2.INTER_AREA)
            fx, fy = frame['fl_x'], frame['fl_y']
            if abs(frame['cx']-ow/2)>1e-5 or abs(frame['cy']-oh/2)>1e-5:
                raise ValueError('当前光栅器只登记居中主点。')
            r,t = blender_c2w_to_camera_extrinsics(frame['transform_matrix'])
            cam = Camera(colmap_id=frame['source_frame'], R=r, T=t,
                FoVx=2*math.atan(ow/(2*fx)), FoVy=2*math.atan(oh/(2*fy)),
                image_train_light=torch.zeros((3,h,w)), gt_alpha_mask=torch.from_numpy(alpha[None]),
                image_name_train_light=path.stem, image_path_train_light=str(path), uid=frame['source_frame'],
                data_device='cuda', fid=frame['time'])
            return cam, alpha, np.asarray([fx*w/ow, fy*h/oh, frame['cx']*w/ow, frame['cy']*h/oh]), target_rgb

        def deformation(self, camera):
            if self.record['baseline']=='eldl_fixed':
                return dict(d_xyz=0.,d_rotation=0.,d_scaling=0.,d_opacity=None,d_color=None)
            if iteration < self.config.get('warm_up', 1000):
                return dict(d_xyz=0., d_rotation=0., d_scaling=0., d_opacity=None, d_color=None)
            return self.deform.step(self.pc.get_xyz.detach(), camera.fid.unsqueeze(0).expand(len(self.pc.get_xyz), -1),
                iteration=iteration, feature=self.pc.get_binary_feature(), camera_center=camera.camera_center,
                is_training=False)

        def rgb(self, camera, colors=None):
            if self.record['baseline']=='eldl_fixed' and colors is None and iteration>self.config['static_dl_physical_start']:
                from scene.eldl_v3_photometric import render_photometric
                visibility=self.visibility if iteration>=self.config['static_dl_visibility_start'] else None
                return render_photometric(camera,self.pc,Namespace(**self.config),torch.zeros(3,device='cuda'),
                    self.physical_lights,visibility,iteration)
            deformation = self.deformation(camera)
            package = render(camera, self.pc, Namespace(**self.config), torch.zeros(3,device='cuda'),
                **{k:deformation[k] for k in ('d_xyz','d_rotation','d_scaling','d_opacity','d_color')},
                override_color=colors)
            return package

        def prediction(self, camera, c2w, intrinsic):
            prediction = self.render(camera,c2w,intrinsic)
            package = self.rgb(camera)
            prediction['rgb'] = package['render'].permute(1,2,0).cpu().numpy()
            normal, valid = unit_normal(package['rend_normal'].permute(1,2,0).cpu().numpy())
            ratio = self.config['depth_ratio']; visible = prediction['alpha'] >= .5
            parity = dict(normal_max_abs=float(np.max(np.abs(normal[valid]-prediction['normal'][valid]))) if valid.any() else 0.,
                alpha_max_abs=float(np.max(np.abs(package['rend_alpha'].squeeze(0).cpu().numpy()-prediction['alpha']))),
                surf_depth_max_abs=float(np.max(np.abs(package['surf_depth'].squeeze(0).cpu().numpy()[visible]-
                    ((1-ratio)*prediction['expected']+ratio*prediction['median'])[visible]))) if visible.any() else 0.)
            if any(v > 2e-5 for v in parity.values()):
                raise AssertionError('RGB与独立几何光栅map不一致。')
            return prediction, parity
    return V3Run()


def predicted_arrays(args, smoke=False):
    import torch
    run = adapter(args.model, args.iteration, args.source, args.split, smoke)
    out = args.output / ('smoke_render' if smoke else 'predictions')
    out.mkdir(exist_ok=False)
    meta = read(args.source / f'transforms_{args.split}.json')
    frames = [r for r in meta['frames'] if not smoke or r['source_frame'] in REPRESENTATIVES]
    # 内部smoke不借外部test，固定内部held首中末。
    if smoke and args.split == 'held':
        frames = [meta['frames'][i] for i in (0,len(meta['frames'])//2,-1)]
    report = dict(model=str(args.model), iteration=args.iteration, baseline=run.record['baseline'],
        source=str(args.source), source_files=run.files, split=args.split,
        smoke=smoke, precision_claim=False if smoke else None, frames={},
        training_record_snapshot=run.record, core_code_matches=run.training_core_comparison,
        effective_rasterizer=run.backend, active_sh_degree=run.pc.active_sh_degree,
        warmup=bool(args.iteration < run.config.get('warm_up',1000)))
    if hasattr(run,'light_prediction_provenance'):report['light_prediction_provenance']=run.light_prediction_provenance
    with torch.no_grad():
        for record in frames:
            n = record['source_frame']; camera, alpha, k, rgb = run.camera(meta,record)
            c2w = np.asarray(record['transform_matrix'])
            prediction, parity = run.prediction(camera,c2w,k)
            filename = f'prediction_{n:04d}.npz'
            np.savez_compressed(out / filename, **prediction, target_rgb=rgb, source_alpha=alpha,
                camera_to_world=c2w, intrinsics=k)
            save_png(out / f'rgb_{n:04d}.png',prediction['rgb'])
            save_png(out / f'normal_{n:04d}.png',prediction['normal']*.5+.5)
            report['frames'][str(n)] = dict(file=filename, sha256=sha(out/filename), geometry_parity=parity)
            print(f'预测渲染 {n:04d}',flush=True)
            del camera
    verify_files(run.files)
    write(out / 'manifest.json',report)
    if smoke:
        write(args.output/'SMOKE_COMPLETED.json',dict(status='PASS_LOAD_RENDER_ONLY', model_metrics='NOT_RUN',
            gt_read=False, frames=list(report['frames']), output=str(out)))


def predicted_meshes(args):
    """此阶段不打开GT，仅TRAIN相机在目标time的预测expected深度参与TSDF。"""
    import torch
    import open3d as o3d
    run = adapter(args.model,args.iteration,args.source,args.split)
    out = args.output/'predicted_meshes'; out.mkdir(exist_ok=False)
    train_split = 'fit' if args.split == 'held' else 'train'
    train = read(args.source/f'transforms_{train_split}.json')
    targets = read(args.source/f'transforms_{args.split}.json')['frames']
    if [r['source_frame'] for r in train['frames']] != run.record['train_frames']:
        raise ValueError('TSDF相机不等于模型实际训练帧。')
    report = dict(kind='predicted_mesh',model_path=str(args.model),iteration=args.iteration,
        source_path=str(args.source),static_scene=run.record['baseline'] in ('static_sh','eldl_fixed'),
        method='Open3D ScalableTSDFVolume; predicted expected camera-Z only',
        gt_geometry_used=False,test_camera_used_for_reconstruction=False,fit_scale_or_pose=False,
        camera_frames=run.record['train_frames'],target_metadata_usage='只取目标time和编号，不融合目标相机',
        settings=dict(resolution=2,voxel_size=.02,sdf_trunc=.08,depth_trunc=20.,alpha_threshold=.5),
        source_files=run.files,frames={})
    cameras=[]
    for record in train['frames']:
        camera,_,k,_=run.camera(train,record)
        intrinsic=o3d.camera.PinholeCameraIntrinsic(camera.image_width,camera.image_height,k[0],k[1],k[2]-.5,k[3]-.5)
        cameras.append((record,camera,k,np.asarray(record['transform_matrix']),intrinsic))
    basis=np.diag([1.,-1.,-1.,1.]); cached=None
    with torch.no_grad():
        for target in targets:
            n=target['source_frame']
            if cached is not None and report['static_scene']:
                report['frames'][str(n)]=dict(cached,source_frame=n,time=target['time'])
                continue
            volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.02,sdf_trunc=.08,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
            integrations=[]
            for record,camera,k,c2w,intrinsic in cameras:
                camera.fid=torch.tensor([target['time']],dtype=torch.float32,device='cuda')
                prediction=run.render(camera,c2w,k)
                depth=prediction['expected'];valid=np.isfinite(depth)&(depth>0)&(depth<20)&(prediction['alpha']>=.5)
                depth=np.where(valid,depth,0).astype(np.float32);h,w=depth.shape
                rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.zeros((h,w,3),np.uint8)),
                    o3d.geometry.Image(depth),depth_scale=1.,depth_trunc=20.,convert_rgb_to_intensity=False)
                volume.integrate(rgbd,intrinsic,basis@np.linalg.inv(c2w))
                integrations.append(dict(source_frame=record['source_frame'],split=train_split,
                    render_time=target['time'],valid_prediction_pixels=int(valid.sum())))
            mesh=volume.extract_triangle_mesh()
            mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
            entry=dict(source_frame=n,time=target['time'],vertices=len(mesh.vertices),triangles=len(mesh.triangles),
                integrations=integrations,available=bool(mesh.has_triangles()))
            if mesh.has_triangles():
                filename=f'predicted_mesh_{n:04d}.ply'
                if not o3d.io.write_triangle_mesh(str(out/filename),mesh,write_vertex_colors=False):
                    raise IOError('预测Mesh写入失败。')
                entry.update(mesh=filename,sha256=sha(out/filename))
            else:
                entry.update(mesh=None,status='EMPTY_PREDICTION_RETAINED')
            report['frames'][str(n)]=entry;cached=entry
            write(out/'manifest.json',report)
            print(f'预测TSDF {n:04d}，三角面{entry["triangles"]}',flush=True)
    verify_files(run.files);write(out/'manifest.json',report)


def stat(values):
    a=np.asarray(values,dtype=np.float64)
    a=a[np.isfinite(a)]
    return dict(count=int(a.size),sum=float(a.sum()),mean=float(a.mean()) if a.size else None,
        median=float(np.median(a)) if a.size else None,p95=float(np.quantile(a,.95)) if a.size else None)


def normal_item(errors, valid, eligible):
    mask=np.asarray(valid,bool)&eligible;n=int(eligible.sum());s=stat(errors[mask]);count=s['count']
    return dict(eligible_pixels=n,valid_pixels=count,coverage=count/n if n else None,angular=s,
        missing180_sum=s['sum']+180*(n-count),missing180_mae=(s['sum']+180*(n-count))/n if n else None)


def depth_item(predicted, support, gt_z, rays, eligible):
    valid=eligible&support&np.isfinite(predicted)&(predicted>0)
    n=int(eligible.sum());count=int(valid.sum());absolute=np.abs(predicted-gt_z)
    ray_norm=np.linalg.norm(rays[...,3:],axis=-1)
    distance=absolute*ray_norm
    gt_distance=gt_z*ray_norm
    relative=distance/np.maximum(gt_distance,1e-12)
    ratios=np.minimum(relative[valid],1.)
    return dict(eligible_pixels=n,valid_pixels=count,coverage=count/n if n else None,
        absolute=stat(absolute[valid]),abs_rel=stat((absolute/np.maximum(gt_z,1e-12))[valid]),
        bias=stat((predicted-gt_z)[valid]),point_distance=stat(distance[valid]),
        relative_ray_distance=stat(relative[valid]),
        close_counts=[int((valid&(distance<=threshold)).sum()) for threshold in DISTANCES],
        relative_close_counts=[int((valid&(relative<=threshold)).sum()) for threshold in RELATIVE],
        close_fractions=[int((valid&(distance<=threshold)).sum())/n if n else None for threshold in DISTANCES],
        relative_close_fractions=[int((valid&(relative<=threshold)).sum())/n if n else None for threshold in RELATIVE],
        clipped_relative_missing_sum=float(ratios.sum())+n-count,
        clipped_relative_missing_mean=(float(ratios.sum())+n-count)/n if n else None)


def weighted(items,field,denominator):
    count=sum(i[denominator] for i in items)
    return sum(i[field] for i in items)/count if count else None


def mean_present(values):
    v=[v for v in values if v is not None]
    return float(np.mean(v)) if v else None


def aggregate(rows):
    result={}
    for region in rows[0]['regions']:
        items=[row['regions'][region] for row in rows]
        entry=dict(normal={},depth={},alpha={},gt_pixels=sum(i['gt_pixels'] for i in items),
            gt_depth_pixels=sum(i['gt_depth_pixels'] for i in items))
        for name in METHODS:
            methods=[i['normal'][name] for i in items]
            n=sum(i['eligible_pixels'] for i in methods);count=sum(i['valid_pixels'] for i in methods)
            entry['normal'][name]=dict(mae_pixel_mean_deg=sum(i['angular']['sum'] for i in methods)/count if count else None,
                mae_frame_mean_deg=mean_present([i['angular']['mean'] for i in methods]),
                coverage=count/n if n else None,coverage_frame_mean=mean_present([i['coverage'] for i in methods]),
                missing180_pixel_mean_deg=sum(i['missing180_sum'] for i in methods)/n if n else None,
                missing180_frame_mean_deg=mean_present([i['missing180_mae'] for i in methods]),
                eligible_pixels=n,valid_pixels=count,eligible_frames=sum(i['eligible_pixels']>0 for i in methods),
                valid_frames=sum(i['valid_pixels']>0 for i in methods),
                missing_prediction_frames=[row['source_frame'] for row,i in zip(rows,methods) if i['eligible_pixels'] and not i['valid_pixels']])
        for name in DEPTHS:
            methods=[i['depth'][name] for i in items];n=sum(i['eligible_pixels'] for i in methods);count=sum(i['valid_pixels'] for i in methods)
            value=dict(eligible_pixels=n,valid_pixels=count,coverage=count/n if n else None,
                coverage_frame_mean=mean_present([i['coverage'] for i in methods]),
                eligible_frames=sum(i['eligible_pixels']>0 for i in methods),valid_frames=sum(i['valid_pixels']>0 for i in methods),
                distance_thresholds_world_units=DISTANCES,relative_distance_thresholds=RELATIVE,
                close_fraction_pixel=[sum(i['close_counts'][j] for i in methods)/n if n else None for j in range(len(DISTANCES))],
                close_fraction_frame=[mean_present([i['close_fractions'][j] for i in methods]) for j in range(len(DISTANCES))],
                relative_close_fraction_pixel=[sum(i['relative_close_counts'][j] for i in methods)/n if n else None for j in range(len(RELATIVE))],
                relative_close_fraction_frame=[mean_present([i['relative_close_fractions'][j] for i in methods]) for j in range(len(RELATIVE))],
                clipped_relative_missing_pixel_mean=sum(i['clipped_relative_missing_sum'] for i in methods)/n if n else None,
                clipped_relative_missing_frame_mean=mean_present([i['clipped_relative_missing_mean'] for i in methods]),
                missing_prediction_frames=[row['source_frame'] for row,i in zip(rows,methods) if i['eligible_pixels'] and not i['valid_pixels']])
            for metric in ('absolute','abs_rel','bias','point_distance','relative_ray_distance'):
                value[metric]=dict(pixel_mean=sum(i[metric]['sum'] for i in methods)/count if count else None,
                    frame_mean=mean_present([i[metric]['mean'] for i in methods]))
            entry['depth'][name]=value
        alpha_count=sum(i['alpha_valid'] for i in items)
        entry['alpha']=dict(valid_pixels=alpha_count,coverage=alpha_count/entry['gt_depth_pixels'] if entry['gt_depth_pixels'] else None)
        common=[i['common'] for i in items];cc=sum(i['valid_pixels'] for i in common)
        entry['common']=dict(valid_pixels=cc,coverage=cc/entry['gt_pixels'] if entry['gt_pixels'] else None,
            normal={name:dict(mae_pixel_mean_deg=sum(i['normal'][name]['sum'] for i in common)/cc if cc else None,
                mae_frame_mean_deg=mean_present([i['normal'][name]['mean'] for i in common])) for name in METHODS})
        result[region]=entry
    return result


def metric_arrays(pred,gt):
    from utils.geometry_eval_utils import unit_normal,normal_errors,erode,camera_rays
    gt_z=gt['depth_z'];depth_base=(gt['alpha']>=.5)&gt['valid']&np.isfinite(gt_z)&(gt_z>0)
    target,target_valid=unit_normal(gt['gt_exr_normal']);base=depth_base&target_valid&gt['gt_exr_valid']
    visible=np.isfinite(pred['alpha'])&(pred['alpha']>=.5)
    names=[str(n) for n in gt['object_names']];oid=gt['object_id'];sid=gt['semantic_id']
    human_ids=[i for i,n in enumerate(names) if n in HUMAN];platform_ids=[i for i,n in enumerate(names) if n in GROUND]
    region_masks=dict(foreground=np.ones_like(base),interior=erode(gt['alpha']>=.99,2),
        human=np.isin(oid,human_ids),leg=sid==2,foot=sid==3,shoe=sid==4,footwear=np.isin(sid,[3,4]),
        platform=np.isin(oid,platform_ids),human_other=sid==1,unknown=sid==0)
    region_masks.update({f'object_{i:02d}':oid==i for i in range(len(names))})
    if (depth_base&(sid==0)).any():
        raise ValueError('GT前景存在未覆盖语义，不能静默丢弃。')
    if any(not np.allclose(pred[k],gt[k],atol=1e-5,rtol=0) for k in ('camera_to_world','intrinsics')):
        raise ValueError('预测和GT相机不一致。')
    if not np.allclose(pred['source_alpha'],gt['alpha'],atol=1e-7,rtol=0):
        raise ValueError('预测和GT的16位Alpha/AREA不一致。')
    errors={};valids={}
    for name,field in METHODS.items():
        error,valid=normal_errors(pred[field],target)
        support=pred['mesh_valid'] if name=='mesh' else visible
        if name=='depth_normal':support=erode(base&visible,1)
        errors[name]=error['angular_deg'];valids[name]=support&valid
    rays=camera_rays(gt['camera_to_world'],gt['intrinsics'],*base.shape)
    regions={}
    for name,mask in region_masks.items():
        eligible=base&mask;depth_eligible=depth_base&mask
        common=eligible.copy()
        for valid in valids.values():common&=valid
        regions[name]=dict(gt_pixels=int(eligible.sum()),gt_depth_pixels=int(depth_eligible.sum()),
            alpha_valid=int((depth_eligible&visible).sum()),
            normal={method:normal_item(errors[method],valids[method],eligible) for method in METHODS},
            depth={field:depth_item(pred[field],pred['mesh_valid'] if field=='mesh_depth' else visible,
                gt_z,rays,depth_eligible) for field in DEPTHS},
            common=dict(valid_pixels=int(common.sum()),normal={method:stat(errors[method][common]) for method in METHODS}))
    alpha_iou=float((visible&depth_base).sum()/max(1,(visible|depth_base).sum()))
    mse=float(np.mean((np.clip(pred['rgb'],0,1)-pred['target_rgb'])**2))
    return regions,dict(alpha_iou=alpha_iou,rgb_mse_black_background=mse,
        rgb_psnr_black_background=(-10*math.log10(mse)) if mse>0 else None),errors,base


def preview_errors(out,n,pred,gt,errors,eligible):
    from matplotlib import colormaps
    panels=[]
    for key in ('normal','depth_normal','mesh_normal'):
        a=(pred[key]*.5+.5).clip(0,1);a[~eligible]=0;panels.append(a)
    for key in METHODS:
        a=colormaps['turbo'](np.clip(errors[key]/90.,0,1))[...,:3];a[~eligible]=0;panels.append(a)
    for key in DEPTHS:
        a=colormaps['magma'](np.clip(np.abs(pred[key]-gt['depth_z'])/.2,0,1))[...,:3]
        support=(pred['mesh_valid'] if key=='mesh_depth' else pred['alpha']>=.5)&np.isfinite(pred[key])&(pred[key]>0)
        a[eligible&~support]=[0,1,1];a[~eligible]=0;panels.append(a)
    canvas=np.concatenate([np.concatenate(panels[i:i+3],axis=1) for i in range(0,9,3)],axis=0)
    save_png(out/f'geometry_{n:04d}.png',canvas)
    # 所有方法同一GT足鞋bbox，仅由固定GT分区计算，不随预测变化。
    yy,xx=np.where(eligible&np.isin(gt['semantic_id'],[2,3,4]))
    if len(xx):
        x0,x1=max(0,int(xx.min())-8),min(eligible.shape[1],int(xx.max())+9)
        y0,y1=max(0,int(yy.min())-8),min(eligible.shape[0],int(yy.max())+9)
        crops=[p[y0:y1,x0:x1] for p in panels]
        save_png(out/f'leg_foot_shoe_{n:04d}.png',np.concatenate(crops,axis=1))
        write(out/f'crop_{n:04d}.json',dict(source='仅固定GT腿足鞋掩膜',box_xyxy=[x0,y0,x1,y1]))


def metrics(args):
    if args.split != 'test':
        raise ValueError('受限GT仅供外部test评测；内部held只允许RGB/已有Alpha验证。')
    from utils.predicted_mesh import PredictedMesh
    out=args.output/'metrics';out.mkdir(exist_ok=False)
    (out/'arrays').mkdir();(out/'previews').mkdir()
    predictions=read(args.output/'predictions/manifest.json');meshes=read(args.output/'predicted_meshes/manifest.json')
    if (predictions['model']!=str(args.model) or predictions['iteration']!=args.iteration
            or meshes['model_path']!=str(args.model) or meshes['iteration']!=args.iteration
            or meshes['gt_geometry_used'] is not False or meshes['test_camera_used_for_reconstruction'] is not False):
        raise ValueError('预测/重建来源不是当前模型或违反仅TRAIN预测重建。')
    gt_manifest=read(args.gt/'gt_depth/manifest.json')
    if gt_manifest['depth_convention']!='positive camera Z; unnormalized pinhole rays':raise ValueError('GT深度定义不符。')
    if predictions['smoke']:raise ValueError('冒烟预测禁止进入正式精度表。')
    if list(map(int,predictions['frames']))!=SPLITS[args.split]:raise ValueError('预测帧不完整。')
    rows=[];cache={};hashes={};all_mesh_available=True
    for n in SPLITS[args.split]:
        pp=args.output/'predictions'/predictions['frames'][str(n)]['file']
        gp=args.gt/'gt_depth'/gt_manifest['frames'][str(n)]['file']
        for path,expected in [(pp,predictions['frames'][str(n)]['sha256']),(gp,gt_manifest['frames'][str(n)]['sha256'])]:
            if sha(path)!=expected:raise ValueError('预测或GT数组SHA不符。')
            hashes[str(path)]=expected
        pred=dict(np.load(pp,allow_pickle=False));gt=dict(np.load(gp,allow_pickle=False))
        entry=meshes['frames'][str(n)]
        if entry['available']:
            path=args.output/'predicted_meshes'/entry['mesh']
            if sha(path)!=entry['sha256']:raise ValueError('预测Mesh SHA不符。')
            hashes[str(path)]=entry['sha256']
            if str(path) not in cache:cache[str(path)]=PredictedMesh(path)
            pred.update(cache[str(path)].render(pred['camera_to_world'],pred['intrinsics'],*pred['alpha'].shape))
        else:
            all_mesh_available=False
            pred.update(mesh_valid=np.zeros_like(pred['alpha'],bool),mesh_depth=np.zeros_like(pred['alpha']),mesh_normal=np.zeros_like(pred['normal']))
        regions,other,errors,eligible=metric_arrays(pred,gt)
        rows.append(dict(source_frame=n,split=args.split,regions=regions,**other))
        np.savez_compressed(out/'arrays'/f'evaluated_{n:04d}.npz',mesh_valid=pred['mesh_valid'],
            mesh_normal=pred['mesh_normal'],mesh_depth=pred['mesh_depth'],eligible=eligible,
            semantic_id=gt['semantic_id'],**{name+'_error_deg':error for name,error in errors.items()})
        # 外部固定15帧全部保留几何与足鞋图，不仅保存最优视角。
        preview_errors(out/'previews',n,pred,gt,errors,eligible)
        print(f'固定分母统计 {n:04d}',flush=True)
    verify_files(hashes)
    report=dict(status='PASS_METRIC_EXECUTION',model=str(args.model),iteration=args.iteration,split=args.split,
        scientific_goal='NOT_DECIDED_NO_V3_THRESHOLDS',all_predicted_meshes_available=all_mesh_available,
        protocol_sha256=sha(args.output/'protocol.json'),source_hashes=hashes,
        gt_depth_manifest_sha256=sha(args.gt/'gt_depth/manifest.json'),
        partition_definition=read(args.gt/'partition_definition.json'),
        unit_system=gt_manifest['unit_system'],meters_per_world_unit=gt_manifest['meters_per_world_unit'],
        rows=rows,summary=aggregate(rows),alpha_iou_frame_mean=float(np.mean([r['alpha_iou'] for r in rows])),
        rgb_psnr_frame_mean=mean_present([r['rgb_psnr_black_background'] for r in rows]),
        rgb_reference='原16位RGB先以已有Alpha合成黑背景再AREA；预测裁到[0,1]，无曝光/颜色/尺度拟合。',
        preview_legend='3列：GS/expected法线/Mesh；第2行0～90度角误差；第3行expected/median/Mesh深度绝对误差0～0.2世界单位；青色为缺失预测。',
        no_gt_used_in_training_or_mesh=True)
    write(out/'metrics.json',report)
    csv_rows=[]
    for row in rows:
        for region,item in row['regions'].items():
            for method,value in item['normal'].items():
                csv_rows.append(dict(frame=row['source_frame'],region=region,method=method,
                    eligible=value['eligible_pixels'],valid=value['valid_pixels'],coverage=value['coverage'],
                    mae_deg=value['angular']['mean'],missing180_deg=value['missing180_mae']))
    with (out/'normal_per_frame_regions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write(out/'METRICS_COMPLETED.json',dict(status='PASS_EXECUTION',frames=len(rows),mesh_complete=all_mesh_available))


def visuals(args):
    import torch
    import imageio.v2 as imageio
    run=adapter(args.model,args.iteration,args.source,args.split)
    out=args.output/'four_classes';out.mkdir(exist_ok=False)
    categories=('rgb','alpha','normal','separation');writers={};checks=[]
    for key in categories:
        (out/key).mkdir();writers[key]=imageio.get_writer(str(out/(key+'.mp4')),fps=24,macro_block_size=1,codec='libx264')
    metadata=read(args.source/('transforms_all.json' if args.split=='test' else 'transforms_held.json'))
    expected_frames=len(metadata['frames'])
    representatives=REPRESENTATIVES if args.split=='test' else [HELD[0],HELD[len(HELD)//2],HELD[-1]]
    try:
        with torch.no_grad():
            for record in metadata['frames']:
                n=record['source_frame'];camera,_,k,_=run.camera(metadata,record)
                pred,parity=run.prediction(camera,np.asarray(record['transform_matrix']),k)
                feature=run.pc.get_binary_feature();colors=torch.cat((feature,1-feature,torch.zeros_like(feature)),dim=1)
                separation=run.rgb(camera,colors)['render'].permute(1,2,0).cpu().numpy()
                alpha=pred['alpha'][...,None]
                arrays=dict(rgb=pred['rgb'],alpha=np.repeat(alpha,3,axis=-1),normal=(pred['normal']*.5+.5)*alpha,separation=separation)
                for key,array in arrays.items():
                    if not np.isfinite(array).all():raise ValueError('四类可视化出现非有限值。')
                    save_png(out/key/f'{n:04d}.png',array)
                    writers[key].append_data((np.clip(array,0,1)*255).round().astype(np.uint8))
                checks.append(dict(frame=n,**parity));del camera
                if n%10==0:print(f'四类可视化 {n:04d}',flush=True)
    finally:
        for writer in writers.values():writer.close()
    videos={}
    for key in categories:
        path=out/(key+'.mp4');cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();pngs=len(list((out/key).glob('*.png')))
        if count!=expected_frames or pngs!=expected_frames:raise ValueError('四类视频或PNG不完整。')
        videos[key]=dict(path=str(path),decoded_frames=count,pngs=pngs)
    contact=[]
    for key in categories:
        contact.append(np.concatenate([cv2.imread(str(out/key/f'{n:04d}.png')) for n in representatives],axis=1))
    contact_name='four_classes_10_80_150.png' if args.split=='test' else 'four_classes_held.png'
    cv2.imwrite(str(out/contact_name),np.concatenate(contact,axis=0))
    verify_files(run.files)
    write(out/'artifacts.json',dict(status='PASS_COMPLETENESS',videos=videos,geometry_parity=checks,
        frames=expected_frames,representatives=representatives,manual_review='PENDING',source_files=run.files,
        separation_semantics='原binary feature红高绿低；静态场景仅为特征门控，不能宣称运动语义。'))


def make_plan(args):
    out=output_path(args.output,create=True);(out/'logs').mkdir()
    protocol=freeze_protocol(args);write(out/'protocol.json',protocol)
    snapshot=PROTOCOL_DOC.read_bytes()
    if hashlib.sha256(snapshot).hexdigest()!=protocol['protocol_document_sha256']:
        raise ValueError('协议文档在建立计划时变化，请保留现场并新建计划。')
    (out/PROTOCOL_DOC.name).write_bytes(snapshot)
    record,files=model_record(args.model,args.iteration,args.split,args.action=='smoke')
    write(out/'training_record_snapshot.json',record)
    actions=(['render','mesh','metrics','visuals'] if args.split=='test' else ['render','internal-metrics','visuals']) if args.action!='smoke' else ['smoke-render']
    commands=[]
    for action in actions:
        command=[str(PYTHON),str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),
            'LIHAN.scripts.eldl_v3_eval','--action',action,'--model',str(args.model),
            '--iteration',str(args.iteration),'--source',str(args.source),'--output',str(out),
            '--gt',str(args.gt),'--split',args.split]
        # 纯CPU聚合无需bootstrap或CUDA上下文。
        if action in ('metrics','internal-metrics'):command=[str(PYTHON),str(Path(__file__).resolve())]+command[4:]
        commands.append(dict(action=action,command=command,shell=shlex.join(command)))
    write(out/'plan.json',dict(model=str(args.model),iteration=args.iteration,split=args.split,
        gpu=args.gpu,model_sha256=files,steps=commands,protocol_sha256=sha(out/'protocol.json'),
        evaluator_sha256=sha(Path(__file__)),training_record_mutability='原marker可随训练结束更新；本次记录已快照，checkpoint必须保持不变'))
    contact_name='four_classes_10_80_150.png' if args.split=='test' else 'four_classes_held.png'
    lines=['# V3固定评测记录','','source：`'+str(args.source)+'`；model：`'+str(args.model)+'`；iteration：'+str(args.iteration)+'。',
        '划分：'+args.split+'；模型实际训练帧数：'+str(len(record['train_frames']))+'。',
        'GT仅供离线聚合；重建Mesh只用模型实际TRAIN相机预测expected深度。',
        '输出：`'+str(out)+'`；日志：`logs`；指标：`metrics/metrics.json`；数组：`predictions`与`metrics/arrays`。',
        '代表图：`four_classes/'+contact_name+'`；视频：`four_classes/{rgb,alpha,normal,separation}.mp4`。',
        '完整命令（GPU步骤由计划的CUDA_VISIBLE_DEVICES固定）：','']
    for item in commands:lines+=['```bash',item['shell'],'```','']
    lines+=['定量指标：PENDING。RGB目检：PENDING；Alpha目检：PENDING；Normal目检：PENDING；分离目检：PENDING。',
        '最终结论：PENDING。V3科学目标阈值未提供，不沿用旧DL_V2阈值；数值完整性PASS不代表完整重建。',
        '失败必须保留checkpoint、渲染、数组、统计和日志；重试新目录。','']
    (out/'README.md').write_text('\n'.join(lines))
    return commands


def execute(args,commands):
    if args.gpu is None:raise ValueError('执行GPU步骤前必须显式登记--gpu。')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    outcomes=[]
    for item in commands:
        with (args.output/'logs'/(item['action']+'.log')).open('x') as log:
            log.write(item['shell']+'\n');log.flush()
            rc=subprocess.run(item['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        outcomes.append(dict(action=item['action'],exit_code=rc));write(args.output/'executed.json',outcomes)
        if rc:
            write(args.output/'FAILED.json',dict(step=item['action'],exit_code=rc,retained=True))
            with (args.output/'README.md').open('a') as f:f.write('\n最终结论：**FAILED**；失败现场保留，步骤'+item['action']+'。\n')
            raise RuntimeError('评测步骤失败：'+item['action'])
    if args.action=='smoke':
        with (args.output/'README.md').open('a') as f:f.write('\n最终结论：**PASS（仅加载/渲染冒烟，无GT指标、无精度声明）**。\n')
        return
    if args.split=='held':
        write(args.output/'PIPELINE_COMPLETED.json',dict(status='PASS_INTERNAL_VALIDATION_EXECUTION',
            external_test_read=False,restricted_gt_read=False,manual_review='PENDING'))
        with (args.output/'README.md').open('a') as f:
            f.write('\n内部held18验证执行通过，仅RGB/已有Alpha；不读取受限GT，不构成外部135训练候选验收。目检PENDING。\n')
        return
    metric=read(args.output/'metrics/metrics.json')
    write(args.output/'PIPELINE_COMPLETED.json',dict(status='PASS_PIPELINE',numeric_execution='PASS',
        mesh_complete=metric['all_predicted_meshes_available'],manual_review='PENDING',scientific_goal='NOT_DECIDED'))
    with (args.output/'README.md').open('a') as f:
        f.write('\n定量表（前景，角度为度；完整分区/距离/缺失惩罚见metrics.json）：\n\n| 方法 | 像素MAE | 等帧MAE | coverage | 缺失180像素MAE |\n|---|---:|---:|---:|---:|\n')
        for method,value in metric['summary']['foreground']['normal'].items():
            f.write('| '+method+' | '+' | '.join(str(value[k]) for k in ['mae_pixel_mean_deg','mae_frame_mean_deg','coverage','missing180_pixel_mean_deg'])+' |\n')
        f.write('\n流水线完整性PASS；四类实际目检尚未回填，最终模型验收PENDING。\n')


def cpu_self_check():
    from utils.geometry_eval_utils import camera_rays
    errors=np.array([[10.,30.]])
    normal=normal_item(errors,np.array([[True,False]]),np.ones((1,2),bool))
    assert normal['coverage']==.5 and normal['missing180_mae']==95.
    rays=camera_rays(np.eye(4),np.array([1e6,1e6,1.,.5]),1,2)
    depth=depth_item(np.array([[2.,0.]]),np.ones((1,2),bool),np.ones((1,2)),rays,np.ones((1,2),bool))
    assert depth['coverage']==.5 and depth['relative_close_fractions']==[0.,0.,0.,0.]
    assert depth['clipped_relative_missing_mean']==1.
    zero=normal_item(errors,np.zeros((1,2),bool),np.ones((1,2),bool))
    assert zero['angular']['mean'] is None and zero['missing180_mae']==180.
    assert len(FIT)==117 and len(HELD)==18 and set(TRAIN).isdisjoint(TEST)
    frames={'0009':{'light_dir_world':[1,0,0]},'0011':{'light_dir_world':[0,1,0]}}
    direction,info=interpolate_train_direction(frames,10)
    np.testing.assert_allclose(direction,np.array([1,1,0])/math.sqrt(2),atol=1e-12)
    assert info['left_train']==9 and info['right_train']==11
    direction,info=interpolate_train_direction(frames,150)
    np.testing.assert_array_equal(direction,[0,1,0]);assert info['method']=='TRAIN_endpoint_hold'
    direction,info=interpolate_train_direction({'0001':{'light_dir_world':[1,0,0]},'0009':{'light_dir_world':[0,1,0]}},3)
    np.testing.assert_allclose(direction,np.array([.75,.25,0])/np.linalg.norm([.75,.25,0]),atol=1e-12)
    assert info['method']=='TRAIN_nlerp' and info['weight']==.25
    # 零有效帧保留在缺失惩罚等帧平均中，条件均值必须同时报告缺帧。
    def toyrow(n,item):
        return dict(source_frame=n,regions={'foreground':dict(gt_pixels=2,gt_depth_pixels=2,
            alpha_valid=item['valid_pixels'],normal={k:item for k in METHODS},
            depth={k:depth for k in DEPTHS},common=dict(valid_pixels=0,normal={k:stat([]) for k in METHODS}))})
    result=aggregate([toyrow(1,normal),toyrow(2,zero)])['foreground']['normal']['gs_render']
    assert result['coverage']==.25 and result['missing180_frame_mean_deg']==137.5
    assert result['missing_prediction_frames']==[2] and result['mae_pixel_mean_deg']==10.
    print('PASS：固定分母/无效惩罚/零有效帧/划分CPU检查。')


def internal_metrics(args):
    if args.split!='held':raise ValueError('该动作仅供内部held。')
    out=args.output/'metrics';out.mkdir(exist_ok=False)
    manifest=read(args.output/'predictions/manifest.json');rows=[]
    if list(map(int,manifest['frames']))!=HELD:raise ValueError('内部held帧不完整。')
    for n in HELD:
        entry=manifest['frames'][str(n)];path=args.output/'predictions'/entry['file']
        if sha(path)!=entry['sha256']:raise ValueError('预测数组SHA不符。')
        pred=dict(np.load(path,allow_pickle=False));foreground=pred['source_alpha']>=.5;visible=pred['alpha']>=.5
        mse=float(np.mean((np.clip(pred['rgb'],0,1)-pred['target_rgb'])**2))
        rows.append(dict(frame=n,rgb_mse_black_background=mse,
            rgb_psnr_black_background=-10*math.log10(mse) if mse>0 else None,
            alpha_iou=float((foreground&visible).sum()/max(1,(foreground|visible).sum()))))
    write(out/'metrics.json',dict(status='PASS_INTERNAL_RGB_ALPHA_EXECUTION',restricted_gt_read=False,
        external_test_read=False,rows=rows,rgb_psnr_frame_mean=mean_present([r['rgb_psnr_black_background'] for r in rows]),
        alpha_iou_frame_mean=float(np.mean([r['alpha_iou'] for r in rows]))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',required=True,choices=['prepare-source','check','self-check','plan','run','smoke','smoke-render','render','mesh','metrics','internal-metrics','visuals'])
    parser.add_argument('--model',type=Path);parser.add_argument('--iteration',type=int)
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--gt',type=Path,default=GT)
    parser.add_argument('--output',type=Path);parser.add_argument('--split',choices=['test','held'],default='test')
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if socket.gethostname()!='ushas':parser.error('当前入口仅登记ushas。')
    if Path(sys.executable).resolve()!=PYTHON.resolve():parser.error('请使用lumimotion-ushas Python。')
    args.source=args.source.resolve();args.gt=args.gt.resolve()
    if args.action=='prepare-source':prepare_source(args);return
    if args.action=='self-check':cpu_self_check();return
    if args.action=='check':
        check_source(args.source,scope=args.split);cpu_self_check()
        print('PASS：评测源split、相机、time、rgba16与文件SHA。');return
    if args.model is None or args.iteration is None or args.output is None:parser.error('模型动作需要--model --iteration --output。')
    args.model=args.model.resolve();args.output=output_path(args.output)
    if args.action in ('plan','run','smoke'):
        if args.action=='smoke' and not args.output.is_relative_to(ROOT/'output/smoke_test'):parser.error('冒烟输出必须位于output/smoke_test。')
        commands=make_plan(args)
        if args.action in ('run','smoke'):execute(args,commands)
        return
    if not (args.output/'plan.json').is_file():parser.error('分步执行需要已有独立计划。')
    plan=read(args.output/'plan.json')
    if plan['model']!=str(args.model) or plan['iteration']!=args.iteration or plan['split']!=args.split:parser.error('模型/iteration/split与计划不一致。')
    if sha(Path(__file__))!=plan['evaluator_sha256']:parser.error('评测源码与计划不同。')
    if sha(args.output/'protocol.json')!=plan['protocol_sha256']:parser.error('固定协议已变化。')
    protocol=read(args.output/'protocol.json')
    if sha(args.source/'manifest.json')!=protocol['source_manifest_sha256']:parser.error('评测源manifest与协议不同。')
    if args.action=='metrics':
        if args.split!='test':parser.error('内部held禁止打开受限GT。')
        if sha(args.gt/'gt_depth/manifest.json')!=protocol['gt_manifest_sha256'] or sha(args.gt/'partition_definition.json')!=protocol['partition_definition_sha256']:
            parser.error('GT或分区定义与固定协议不同。')
    verify_files(plan['model_sha256'])
    actions={'render':predicted_arrays,'smoke-render':lambda a:predicted_arrays(a,True),
             'mesh':predicted_meshes,'metrics':metrics,'internal-metrics':internal_metrics,'visuals':visuals}
    actions[args.action](args)


if __name__=='__main__':main()
