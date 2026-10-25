"""静态ELDL-V3 P0：核验117 TRAIN/18 DEV/15 TEST划分与既有净化RGBA，导出仅含TRAIN帧号与单位方向的GT主光方向表。

本脚本是唯一读取原始lights.json/albedo Alpha/camera.json的环节；训练进程只读取本脚本的输出与data_fit_v2。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

DEV_FRAMES = [61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79]
TEST_FRAMES = list(range(10, 151, 10))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read16(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert raw is not None and raw.dtype == np.uint16 and raw.ndim == 3 and raw.shape[-1] == 4, path
    return raw[..., [2, 1, 0, 3]]


def main(args):
    raw_root = args.raw.resolve(); fit = args.fit.resolve(); out = args.output.resolve()
    out.mkdir(parents=False, exist_ok=False)
    all_frames = list(range(1, 151))
    train135 = [f for f in all_frames if f not in TEST_FRAMES]
    train117 = [f for f in train135 if f not in DEV_FRAMES]
    assert len(train135) == 135 and len(train117) == 117 and len(DEV_FRAMES) == 18
    assert not set(DEV_FRAMES) & set(TEST_FRAMES)

    # 1. 既有净化目录 data_fit_v2 的完整性与划分
    manifest = json.loads((fit / 'no_gt_manifest.json').read_text())
    assert manifest['dataset'] == 'LH-static-jumpingjacks-ELDL-V3' and manifest['gt_read'] is False
    assert manifest['train_frames'] == train117 and manifest['internal_fit_frames'] == train117
    assert manifest['internal_held_frames'] == DEV_FRAMES and manifest['test_frames'] == TEST_FRAMES
    assert set(p.name for p in fit.iterdir()) == {'images', 'points3d.ply', 'transforms_train.json', 'no_gt_manifest.json'}
    file_check = {}
    for rel, expected in manifest['files'].items():
        actual = sha(fit / rel); assert actual == expected, rel; file_check[rel] = actual
    assert sorted(p.name for p in (fit / 'images').iterdir()) == [f'image_{f:04d}.png' for f in train117]
    transforms = json.loads((fit / 'transforms_train.json').read_text())
    assert transforms.get('rgba16') is True and transforms['w'] == 800 and transforms['h'] == 800
    tframes = transforms['frames']
    assert [f['source_frame'] for f in tframes] == train117

    camera = json.loads((raw_root / 'camera.json').read_text())
    cam_err = 0.0; time_err = 0.0
    for f in tframes:
        rec = camera['frames'][f"{f['source_frame']:04d}"]
        c2w = np.asarray(rec['extrinsics']['camera_to_world'], float)
        cam_err = max(cam_err, float(np.abs(c2w - np.asarray(f['transform_matrix'], float)).max()))
        assert abs(f['fl_x'] - rec['intrinsics']['fx']) < 1e-9 and abs(f['cx'] - rec['intrinsics']['cx']) < 1e-9
        time_err = max(time_err, abs(f['time'] - (f['source_frame'] - 1) / 149))
    assert cam_err < 1e-6 and time_err < 1e-9

    # 2. 净化RGBA：RGB逐位等于原图RGB，A逐位等于albedo Alpha；不保留albedo RGB
    pixel = dict(rgb_bitexact=True, rgb_zero_where_alpha0=True, alpha_bitexact=True)
    stats = dict(raw_image_alpha_min=65535, alpha_zero_fraction=[], alpha_partial_fraction=[],
                 bg_rgb_mean=[], fg_saturated_channel_fraction=[], bg_rgb_max=0)
    for f in train117:
        a = read16(fit / 'images' / f'image_{f:04d}.png')
        img = read16(raw_root / 'image' / f'image_{f:04d}.png')
        alb_alpha = read16(raw_root / 'albedo' / f'albedo_{f:04d}.png')[..., 3]
        fgm = a[..., 3] > 0
        # data_fit_v2在Alpha=0处已清零RGB；加载器本就按Alpha预乘，故有效输入与原图逐位一致
        pixel['rgb_bitexact'] &= bool((a[..., :3][fgm] == img[..., :3][fgm]).all())
        pixel['rgb_zero_where_alpha0'] &= bool((a[..., :3][~fgm] == 0).all())
        pixel['alpha_bitexact'] &= bool((a[..., 3] == alb_alpha).all())
        stats['raw_image_alpha_min'] = min(stats['raw_image_alpha_min'], int(img[..., 3].min()))
        bg = a[..., 3] == 0
        stats['alpha_zero_fraction'].append(float(bg.mean()))
        stats['alpha_partial_fraction'].append(float(((a[..., 3] > 0) & (a[..., 3] < 65535)).mean()))
        stats['bg_rgb_mean'].append((img[..., :3][bg].astype(np.float64).mean(0) / 65535).tolist())
        stats['bg_rgb_max'] = max(stats['bg_rgb_max'], int(img[..., :3][bg].max()))
        fg = a[..., 3] > 0
        stats['fg_saturated_channel_fraction'].append(float((a[..., :3][fg] == 65535).mean()))
    assert all(pixel.values()), pixel
    image_summary = dict(
        pixel,
        raw_image_alpha_min=stats['raw_image_alpha_min'],
        alpha_zero_fraction_mean=float(np.mean(stats['alpha_zero_fraction'])),
        alpha_partial_fraction_mean=float(np.mean(stats['alpha_partial_fraction'])),
        raw_image_background_rgb_mean_where_alpha0=np.mean(stats['bg_rgb_mean'], 0).tolist(),
        raw_image_background_rgb_max_code_where_alpha0=stats['bg_rgb_max'],
        foreground_saturated_channel_fraction_mean=float(np.mean(stats['fg_saturated_channel_fraction'])),
        foreground_saturated_channel_fraction_max=float(np.max(stats['fg_saturated_channel_fraction'])),
        loader_semantics='utils/rgba16.FloatRGBA：uint16/65535浮点，RGB预乘Alpha（黑背景，white_background=False），'
                         'INTER_AREA缩放至400×400（resolution=2），满码掩膜=原分辨率RGB==65535且Alpha>0再AREA>0；不经8位量化',
        color_space='原PNG为Blender Standard视图变换后的sRGB编码显示值；训练光度在该sRGB域比较，物理着色在线性域计算后转sRGB',
    )

    # 3. GT主光方向：只导出117 TRAIN帧的单位方向
    lights = json.loads((raw_root / 'lights.json').read_text())
    directions = []; max_axis_err = 0.0; max_norm_err = 0.0; elev = []; azim = []
    for f in all_frames:
        rec = lights[f'{f:04d}']
        assert rec['source_type'] == 'SUN' and rec['light_name'] == 'Orbit_Directional_Sun'
        sun = [x for x in rec['lights'] if x['name'] == 'Orbit_Directional_Sun']; assert len(sun) == 1
        d = np.asarray(rec['light_dir_world'], float)
        assert np.allclose(d, np.asarray(sun[0]['light_dir_world'], float), atol=0, rtol=0)
        axis = np.asarray(sun[0]['matrix_world'], float)[:3, 2]
        max_norm_err = max(max_norm_err, abs(float(np.linalg.norm(d)) - 1.0))
        du = d / np.linalg.norm(d); au = axis / np.linalg.norm(axis)
        max_axis_err = max(max_axis_err, float(np.abs(du - au).max()))
        elev.append(math.degrees(math.asin(du[2]))); azim.append(math.degrees(math.atan2(du[1], du[0])))
        if f in train117:
            directions.append(dict(frame=f, image_name=f'image_{f:04d}', direction=du.tolist()))
    assert max_axis_err < 1e-6 and max_norm_err < 1e-5
    az_step = np.diff(np.unwrap(np.radians(azim)))
    direction_audit = dict(
        convention='世界坐标（Blender world，Z向上），表面指向SUN的单位向量，用于max(n·l,0)；等于SUN matrix_world局部+Z轴，光传播方向为其相反数；不翻转、不重定向',
        max_abs_error_vs_sun_local_z=max_axis_err, max_raw_norm_error=max_norm_err,
        elevation_deg_min=min(elev), elevation_deg_max=max(elev),
        azimuth_step_deg_mean=float(np.degrees(az_step).mean()), azimuth_step_deg_std=float(np.degrees(az_step).std()),
        all_150_frames_checked_for_consistency=True, exported_frames=train117,
        dev_or_test_directions_exported=False,
    )
    direction_file = dict(
        format_version=1, dataset='LH-static-jumpingjacks-ELDL-V3', split='TRAIN117',
        content='仅TRAIN帧号、图像名与单位方向；不含光强/颜色/点光位置/环境/阴影或其他GT',
        forbidden_fields_included=[], convention=direction_audit['convention'],
        records=directions)
    dpath = out / 'train117_sun_directions.json'
    dpath.write_text(json.dumps(direction_file, ensure_ascii=False, indent=2) + '\n')

    split = dict(dataset='LH-static-jumpingjacks-ELDL-V3', train_frames=train117, dev_frames=DEV_FRAMES,
                 test_frames=TEST_FRAMES, original_train135=train135,
                 source='沿用1019-01总控data_fit_v2（I00内部fit117）既有划分：DEV=61..79去除TEST帧70，共18帧',
                 rule='新Stage1与后续全部训练只加载train_frames；DEV与TEST在训练全程不可见，只供隔离评测')
    spath = out / 'split_train117_dev18_test15.json'
    spath.write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n')

    audit = dict(
        status='PASS', raw_dataset=str(raw_root),
        raw_sha256={n: sha(raw_root / n) for n in ['lights.json', 'camera.json', 'dataset_manifest.json', 'README.txt']},
        reused_sanitized_source=str(fit), reused_manifest_sha256=sha(fit / 'no_gt_manifest.json'),
        reused_transforms_sha256=sha(fit / 'transforms_train.json'),
        reused_points3d_sha256=sha(fit / 'points3d.ply'),
        initialization=manifest['initialization'], initial_point_cloud_sha256=manifest['initial_point_cloud_sha256'],
        camera_max_abs_c2w_error=cam_err, time_max_abs_error=time_err,
        image_audit=image_summary, direction_audit=direction_audit,
        outputs={str(dpath): sha(dpath), str(spath): sha(spath)},
        script=str(Path(__file__).resolve()), script_sha256=sha(Path(__file__).resolve()))
    (out / 'p0_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', outputs=audit['outputs'], image=pixel,
                          direction=dict(max_axis=max_axis_err, elev=(min(elev), max(elev)))), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
