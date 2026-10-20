"""动态V3的18帧DEV准备、固定终点评分与预登记选择；不访问TEST或GT。"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('DYNAMIC_V3_REPO_ROOT', CODE_ROOT)).resolve()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
DEV = [n for n in range(61, 80) if n != 70]
ORDER = ['A1', 'B1', 'A0', 'B0']
TOLERANCE = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def new_output(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT/'output'), '输出必须位于output内'
    path.mkdir(parents=True, exist_ok=False)
    return path


def prepare(args):
    import cv2
    import numpy as np
    from LIHAN.scripts.dynamic_v3_radiometry import linearize
    raw = args.raw.resolve()
    assert raw.name in ('LH-dynamic-jumpingjacks-ELDL-V3', 'LH-dynamic-jumpingjacks_DL_V2')
    dl_v2 = raw.name == 'LH-dynamic-jumpingjacks_DL_V2'
    radiometry, stats = read(args.radiometry), read(args.stats)
    train_manifest = read(args.train_manifest)
    assert radiometry['status'] == 'PASS_TRAIN_FIXED_BACKGROUND'
    assert radiometry['source_split'] == 'TRAIN' and not radiometry['gt_read']
    assert radiometry['input_manifest_sha256'] == sha(args.train_manifest)
    assert Path(radiometry['source']).resolve() == raw
    assert Path(train_manifest['source']).resolve() == raw
    assert train_manifest['dev_frames'] == DEV and not train_manifest['gt_read']
    assert len(train_manifest['train_frames']) == 117
    assert not set(DEV) & set(train_manifest['train_frames'])
    assert stats['train_count'] == 117
    assert [r['frame'] for r in stats['per_frame']] == train_manifest['train_frames']
    # 只读取camera元数据与下面明确列出的DEV图像，不读取其他图像或GT资产。
    camera = read(raw/'camera.json')
    assert sha(raw/'camera.json') == train_manifest['camera_sha256']
    out = new_output(args.output)
    (out/'images').mkdir(); (out/'targets').mkdir()
    from utils.no_gt_guard import install_no_gt_guard
    allowed_raw = [raw/'camera.json']
    allowed_raw += [raw/f'{kind}/{kind}_{n:04d}.png' for n in DEV for kind in ('image', 'albedo')]
    install_no_gt_guard(allowed_raw+[args.radiometry.resolve(), args.stats.resolve(),
                                   args.train_manifest.resolve(), out])
    records, provenance, diagnostics = [], {}, []
    background = np.asarray(radiometry['background_linear'], dtype=np.float64)
    for n in DEV:
        image, albedo = raw/f'image/image_{n:04d}.png', raw/f'albedo/albedo_{n:04d}.png'
        rgb_file = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        alpha_file = cv2.imread(str(albedo), cv2.IMREAD_UNCHANGED)
        assert rgb_file is not None and alpha_file is not None
        assert rgb_file.dtype == alpha_file.dtype == np.uint16
        assert rgb_file.shape == alpha_file.shape == (800, 800, 4)
        # albedo RGB只被解码器读取，不进入任何计算或产物。
        alpha16 = alpha_file[..., 3].copy()
        merged = rgb_file.copy(); merged[..., 3] = alpha16
        assert cv2.imwrite(str(out/'images'/image.name), merged)
        rgb = rgb_file[..., [2, 1, 0]].astype(np.float64)/65535.
        alpha = alpha16.astype(np.float64)/65535.
        preclamp = linearize(rgb)-(1-alpha)[..., None]*background
        target = cv2.resize(preclamp.clip(0), (400, 400), interpolation=cv2.INTER_AREA)
        alpha400 = cv2.resize(alpha, (400, 400), interpolation=cv2.INTER_AREA)
        saturated = (rgb_file[..., [2, 1, 0]] == 65535) & (alpha16[..., None] > 0)
        saturated = cv2.resize(saturated.astype(np.float32), (400, 400), interpolation=cv2.INTER_AREA) > 0
        np.savez_compressed(out/f'targets/{n:04d}.npz', linear=target.astype(np.float32),
                            alpha=alpha400.astype(np.float32), saturated=saturated)
        f = camera['frames'][f'{n:04d}']; intr = f['intrinsics']
        c2w = np.asarray(f['extrinsics']['camera_to_world'])
        w2c = np.asarray(f['extrinsics']['world_to_camera'])
        assert np.max(np.abs(c2w@w2c-np.eye(4))) < 1e-5
        assert abs(intr['cx']-400) < 1e-6 and abs(intr['cy']-400) < 1e-6
        records.append(dict(file_path=image.name, source_frame=n, time=n/150 if dl_v2 else (n-1)/149,
            camera_angle_x=intr['fov_x_rad'], camera_angle_y=intr['fov_y_rad'],
            fl_x=intr['fx'], fl_y=intr['fy'], cx=intr['cx'], cy=intr['cy'], transform_matrix=c2w.tolist()))
        provenance[str(image)] = dict(sha256=sha(image), used_channels=['R', 'G', 'B'])
        provenance[str(albedo)] = dict(sha256=sha(albedo), used_channels=['Alpha'])
        diagnostics.append(dict(frame=n, preclamp_negative_channels=int((preclamp < 0).sum()),
            preclamp_minimum=float(preclamp.min()), foreground_pixels=int((alpha400 >= .5).sum())))
    common = {k: records[0][k] for k in ('camera_angle_x', 'camera_angle_y', 'fl_x', 'fl_y', 'cx', 'cy')}
    write(out/'transforms_dev.json', dict(**common, w=800, h=800, rgba16=True,
          per_frame_intrinsics=True, frames=records))
    write(out/'dev_manifest.json', dict(format_version=1, status='PASS_DEV_PREPARATION', split='DEV',
        frames=DEV, gt_read=False, test_read=False, raw=str(raw), source_files=provenance,
        camera_sha256=sha(raw/'camera.json'), radiometry_sha256=sha(args.radiometry),
        train_manifest_sha256=sha(args.train_manifest), statistics_sha256=sha(args.stats),
        frozen_background_linear=background.tolist(), frozen_q_base=stats['q_base'],
        frozen_q_bright=stats['q_bright'], per_frame=diagnostics,
        target_rule='原RGB逆sRGB减冻结TRAIN透射背景后max0，再AREA至400；不乘或除Alpha',
        saturation_rule='原满码且原Alpha>0的通道，AREA支持内任一满码即为下界',
        files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        source_code_sha256=sha(__file__)))
    print(json.dumps(dict(status='PASS_DEV_PREPARATION', source=str(out), frames=DEV), ensure_ascii=False))


def validate_source(source):
    source = Path(source).resolve()
    m = read(source/'dev_manifest.json')
    assert m['split'] == 'DEV' and m['frames'] == DEV
    assert not m['gt_read'] and not m['test_read']
    transforms = read(source/'transforms_dev.json')
    assert [f['source_frame'] for f in transforms['frames']] == DEV
    dl_v2 = Path(m['raw']).name == 'LH-dynamic-jumpingjacks_DL_V2'
    assert all(abs(f['time']-(f['source_frame']/150 if dl_v2 else (f['source_frame']-1)/149)) < 1e-12 for f in transforms['frames'])
    assert [f['file_path'] for f in transforms['frames']] == [f'image_{n:04d}.png' for n in DEV]
    expected = {'transforms_dev.json'}
    expected.update(f'images/image_{n:04d}.png' for n in DEV)
    expected.update(f'targets/{n:04d}.npz' for n in DEV)
    assert set(m['files']) == expected, 'DEV只能包含18帧固定观测与相机'
    for name, digest in m['files'].items():
        p = (source/name).resolve()
        assert p.is_relative_to(source) and sha(p) == digest, name
    return m


def load_cameras(source):
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    infos = readCamerasFromTransforms(str(source), 'transforms_dev.json', 'images', False, no_bg=False)
    config = SimpleNamespace(resolution=2, data_device='cpu', load2gpu_on_the_fly=True)
    cameras = [loadCam(config, i, info, 1.) for i, info in enumerate(infos)]
    assert [int(c.image_name_train_light.split('_')[-1]) for c in cameras] == DEV
    return cameras


def finite_parameters(model, kind):
    import torch
    pc = model.pc
    values = [x for x in vars(pc).values() if isinstance(x, torch.Tensor)]
    values += list(model.deform.deform.parameters()) + list(model.deform.deform.buffers())
    if kind == 'stage2':
        values += list(model.radiance.parameters()) + list(model.radiance.buffers())
    return all(bool(torch.isfinite(x).all()) for x in values)


def load_native(folder, iteration):
    from scene import GaussianModel, DeformModel
    from scripts.probe_dynamic_dl_selfcal import read_config
    cfg = read_config(folder/'cfg_args')
    pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
    pc.load_ply(str(folder/f'point_cloud/iteration_{iteration}/point_cloud.ply'))
    pc.active_sh_degree = min(cfg.sh_degree, iteration//1000)
    deform = DeformModel(deform_type=cfg.deform_type, is_blender=cfg.is_blender,
                        hyper_dim=cfg.hyper_dim, pred_color=cfg.pred_color)
    assert deform.load_weights(str(folder), iteration)
    deform.deform.eval().requires_grad_(False)
    return SimpleNamespace(pc=pc, deform=deform, pipe=SimpleNamespace(
        compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.))


def score(args):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from gaussian_renderer import render
    from scene.stage2_photometric import srgb_to_linear
    from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel
    from utils.no_gt_guard import install_no_gt_guard
    source, folder = args.source.resolve(), args.model.resolve()
    out = new_output(args.output); (out/'renders').mkdir()
    install_no_gt_guard([source, folder, out])
    manifest = validate_source(source)
    assert args.iteration == (35000 if args.kind == 'stage1' else 8100)
    model = load_native(folder, args.iteration) if args.kind == 'stage1' else DynamicSurfaceModel.load(folder, args.iteration)
    if args.kind == 'stage2':
        assert model.iteration == 35000 and model.loaded_step == 8100
        assert model.config['iterations'] == 8100
    parameter_finite = finite_parameters(model, args.kind)
    rows = []
    for n, camera in zip(DEV, load_cameras(source)):
        camera.load2device()
        with torch.no_grad():
            if args.kind == 'stage1':
                values = model.deform.step(model.pc.get_xyz, camera.fid.reshape(1, 1).expand(len(model.pc.get_xyz), 1),
                    iteration=35000, feature=model.pc.get_binary_feature(), camera_center=camera.camera_center,
                    is_training=False)
                package = render(camera, model.pc, model.pipe, model.pc.get_xyz.new_zeros(3),
                    values['d_xyz'], values['d_rotation'], values['d_scaling'],
                    d_opacity=values['d_opacity'], d_color=values['d_color'])
                prediction = srgb_to_linear(package['render'].clamp_min(0))
            else:
                package = model.render(camera)
                prediction = package['render_linear']
            with np.load(source/f'targets/{n:04d}.npz') as target_file:
                target = torch.from_numpy(target_file['linear'].copy()).permute(2, 0, 1).cuda()
                alpha = torch.from_numpy(target_file['alpha'].copy()).cuda()
                saturated = torch.from_numpy(target_file['saturated'].copy()).permute(2, 0, 1).cuda()
            assert prediction.shape == target.shape == (3, 400, 400)
            foreground, predicted = alpha >= .5, package['rend_alpha'][0] > .5
            assert foreground.any()
            interior = -F.max_pool2d(-foreground.float()[None, None], 5, stride=1, padding=2)[0, 0]
            proxy_mask = interior.bool() & predicted
            normal = F.normalize(package['rend_normal'], dim=0)
            depth_normal = F.normalize(package['surf_normal'], dim=0)
            finite = all(bool(torch.isfinite(x).all()) for x in package.values() if isinstance(x, torch.Tensor))
            finite = finite and bool(torch.isfinite(prediction).all())
            difference = prediction-target
            absolute = torch.where(saturated, difference.clamp_max(0), difference).abs()
            coverage = float((foreground & predicted).sum()/foreground.sum())
            union = (foreground | predicted).sum()
            iou = float((foreground & predicted).sum()/union.clamp_min(1))
            # 无代理支持时保留最差2与显式支持数；不将空均值记成完美0。
            consistency = float((1-(normal*depth_normal).sum(0))[proxy_mask].mean()) if proxy_mask.any() and finite else 2.
            rgb_error = float(absolute[:, foreground].mean()) if finite else None
            row = dict(frame=n, time=(n-1)/149, finite=finite, rgb_censored_mae=rgb_error,
                foreground_coverage=coverage, alpha_iou=iou, gs_expected_one_minus_dot=consistency,
                fixed_foreground_pixels=int(foreground.sum()), predicted_foreground_pixels=int(predicted.sum()),
                consistency_supported_pixels=int(proxy_mask.sum()), render=f'renders/{n:04d}.npz')
            arrays = {key: package[key].detach().cpu().numpy() for key in
                      ('rend_alpha', 'rend_normal', 'surf_normal', 'surf_depth')}
            arrays['render_linear'] = prediction.detach().cpu().numpy()
            np.savez_compressed(out/row['render'], **arrays)
            preview = np.clip(arrays['render_linear'].transpose(1, 2, 0), 0, 1)
            preview = np.where(preview <= .0031308, 12.92*preview, 1.055*preview**(1/2.4)-.055)
            assert cv2.imwrite(str(out/f'renders/{n:04d}.png'), np.rint(np.nan_to_num(preview)*255).astype(np.uint8)[..., ::-1])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        camera.load2device('cpu')
    all_finite = parameter_finite and all(r['finite'] for r in rows)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    identity_files = [folder/'cfg_args', folder/f'point_cloud/iteration_{args.iteration}/point_cloud.ply',
                      folder/f'deform/iteration_{args.iteration}/deform.pth']
    if args.kind == 'stage2':
        identity_files += [folder/f'dynamic_v3/iteration_{args.iteration}/state.pt',
                           folder/f'dynamic_v3/iteration_{args.iteration}/receipt.json']
    summary = dict(format_version=1, status='PASS_DEV_SCORE' if all_finite else 'FAILED_NONFINITE',
        split='DEV', frames=DEV, gt_read=False, test_read=False, kind=args.kind, model=str(folder),
        iteration=args.iteration, parameter_finite=parameter_finite, all_finite=all_finite,
        dev_manifest_sha256=sha(source/'dev_manifest.json'), statistics_sha256=manifest['statistics_sha256'],
        radiometry_sha256=manifest['radiometry_sha256'], source_code_sha256=sha(__file__),
        model_sha256={str(p.relative_to(folder)): sha(p) for p in identity_files},
        variant=None if args.kind == 'stage1' else model.path+str(int(model.config['ps_weight'] > 0)),
        direction_lr=None if args.kind == 'stage1' else model.config['direction_lr'],
        rgb_censored_mae=mean('rgb_censored_mae') if all(r['rgb_censored_mae'] is not None for r in rows) else None,
        foreground_coverage=mean('foreground_coverage'), alpha_iou=mean('alpha_iou'),
        gs_expected_one_minus_dot=mean('gs_expected_one_minus_dot'), per_frame=rows,
        geometry_proxy='Expected depth_ratio0；固定Alpha>=0.5前景内缩两像素且predAlpha>0.5上的1-dot',
        aggregation='18帧等权；RGB固定前景分母；原满码通道为删失下界；几何代理非GT准确率',
        argv=sys.argv)
    write(out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'rgb_censored_mae', 'foreground_coverage', 'alpha_iou', 'gs_expected_one_minus_dot')}, ensure_ascii=False))


def select(args):
    baseline = read(args.baseline)
    assert baseline['kind'] == 'stage1' and baseline['iteration'] == 35000
    assert baseline['frames'] == DEV and baseline['all_finite'], '参照必须有限且完整'
    assert not baseline['gt_read'] and not baseline['test_read']
    names = ORDER if args.mode == 'four' else ['freeze', 'directionlr']
    entries = {}
    for argument in args.candidate:
        name, value = argument.split('=', 1)
        assert name in names and name not in entries, name
        path = Path(value).resolve(); summary = read(path)
        assert summary['kind'] == 'stage2' and summary['iteration'] == 8100
        assert summary['frames'] == DEV and not summary['gt_read'] and not summary['test_read']
        if args.mode == 'four':
            assert summary['variant'] == name and summary['direction_lr'] == 0.
        else:
            assert summary['direction_lr'] == (0. if name == 'freeze' else 1e-5)
        for key in ('dev_manifest_sha256', 'statistics_sha256', 'radiometry_sha256'):
            assert summary[key] == baseline[key], key
        failures = []
        if not summary['all_finite']:
            failures.append('参数或渲染非有限')
        if summary['foreground_coverage'] < baseline['foreground_coverage']-.05:
            failures.append('覆盖低于参照减0.05')
        if summary['alpha_iou'] < baseline['alpha_iou']-.05:
            failures.append('IoU低于参照减0.05')
        if summary['gs_expected_one_minus_dot'] > baseline['gs_expected_one_minus_dot']+.05:
            failures.append('法线一致性误差高于参照加0.05')
        if summary['rgb_censored_mae'] is None or not math.isfinite(summary['rgb_censored_mae']):
            failures.append('RGB误差非有限')
        entries[name] = dict(summary_path=str(path), summary_sha256=sha(path), eligible=not failures,
            failures=failures, rgb_censored_mae=summary['rgb_censored_mae'], model=summary['model'],
            model_sha256=summary['model_sha256'], iteration=summary['iteration'])
        entries[name]['variant'] = summary['variant']
    assert set(entries) == set(names), f'要求候选{names}'
    if args.mode == 'final':
        assert entries['freeze']['variant'] == entries['directionlr']['variant'], '灯向对照必须同一主线配置'
    eligible = [n for n in names if entries[n]['eligible']]
    pool = eligible or names
    error = lambda n: entries[n]['rgb_censored_mae'] if entries[n]['rgb_censored_mae'] is not None else math.inf
    best_error = min(error(n) for n in pool)
    tied = [n for n in pool if error(n) <= best_error+TOLERANCE]
    selected = next(n for n in names if n in tied)
    result = dict(format_version=1, status='PASS_DEV_SELECTION' if eligible else 'FAILED_DEV_ELIGIBILITY',
        mode=args.mode, selected=selected, diagnostic_only=not bool(eligible),
        baseline_path=str(args.baseline.resolve()), baseline_sha256=sha(args.baseline),
        dev_manifest_sha256=baseline['dev_manifest_sha256'], frames=DEV, gt_read=False, test_read=False,
        tie_tolerance=TOLERANCE, tie_order=names, candidates=entries,
        source_code_sha256=sha(__file__), argv=sys.argv,
        rule='资格按同Stage1参照覆盖/IoU减0.05，一致性误差加0.05；RGB最小，1e-6内按固定顺序',
        limitation='全部不合格时仅选诊断终点，不能声明通过；不因结果增加训练预算')
    out = new_output(args.output)
    write(out/'selection.json', result)
    print(json.dumps(dict(status=result['status'], selected=selected, diagnostic_only=result['diagnostic_only']), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('prepare', help='只准备18帧DEV，不更改TRAIN源')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--radiometry', type=Path, required=True)
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.set_defaults(run=prepare)
    p = actions.add_parser('score', help='固定Stage1@35000或Stage2@8100只读DEV渲染')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--kind', choices=['stage1', 'stage2'], required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--output', type=Path, required=True); p.set_defaults(run=score)
    p = actions.add_parser('select', help='四组主线或冻结灯向与灯向对照的确定性选择')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--mode', choices=['four', 'final'], default='four')
    p.add_argument('--candidate', action='append', required=True, help='名称=summary.json；four为A0/A1/B0/B1，final为freeze/directionlr')
    p.add_argument('--output', type=Path, required=True); p.set_defaults(run=select)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
