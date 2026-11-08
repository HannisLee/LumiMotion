"""T01(1107-03) I1/I3/I4：完整管线(Stage1->E4->Stage3)总控、snapshot、E0/E1/E2 启动器。

子命令：
  prepare  建总控目录：三份 source、E4 方向 JSON、snapshot（含必要接口补丁）、manifest.json
  run      启动 E0/E1/E2 训练（含冒烟 A/B），写 state.json/README/exitcode；--dry-run 只打印命令
  prereg   冻结预注册 preregistration.json（方案代号、命令、SHA、判定规则）

snapshot 内的补丁全部是“接口适配/只加日志”，每处都用唯一锚点断言，锚点缺失即报错：
  P1 attach 混合光(沿用 stage3_el_eldl_four.py)；P2 关闭内置 TEST 报告；
  P3 pbr_geometry 门禁放宽到 envmap+混合方向光；P4 几何项暂存；P5 metrics.jsonl 日志。
不改损失、渲染方程、优化器与保存规则。仓库工作树中的冻结代码不被修改。
"""
import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
HOST = 'ushas'
STAGE1 = 'output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1'
STAGE1_ITER = 35000
STAGE1_PLY_SHA = 'adf94a9370f2251532d8885ef3f3e5a188ad2cdb6daa41539cb8d227d39222ff'
EVAL_SOURCE = ROOT / 'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/eval_only/eval_source_v1'
CONTROL_NAME = '1107-03-LH-static-ELDL-V3-完整管线去bg总控'
SNAPSHOT_DIRS = ['arguments', 'scene', 'scripts', 'utils', 'gaussian_renderer', 'lpipsPyTorch', 'LIHAN/scripts']
LINKED_DIRS = ['submodules', 'assets', 'example_envmaps']

# 臂定义：名称、是否混合E4方向光、是否几何微调(N1)、是否 --wo_indirect(去bg)
ARMS = {
    'E0': dict(name='1107-04-LH-static-ELDL-V3-E0原始管线环境PBR', mixed=False, geometry=False, wo_indirect=False),
    'E1': dict(name='1107-05-LH-static-ELDL-V3-E1完整管线带bg', mixed=True, geometry=True, wo_indirect=False),
    'E2': dict(name='1107-06-LH-static-ELDL-V3-E2完整管线去bg', mixed=True, geometry=True, wo_indirect=True),
}
SCHEME = 'BG-R2-①③'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


# ---------------------------------------------------------------- snapshot 补丁
def replace_once(text, needle, replacement, label):
    if text.count(needle) != 1:
        raise AssertionError(f'补丁锚点 {label} 出现 {text.count(needle)} 次（应为 1），仓库代码已变化，停止。')
    return text.replace(needle, replacement)


def patch_train_stage2(text):
    # P1 混合方向光 attach（与 stage3_el_eldl_four.py 完全相同的接入点）
    text = replace_once(
        text, '        self.iters_only_diffuse = 2000',
        '        from LIHAN.scripts.stage3_el_eldl_mixed import attach\n        attach(self)\n'
        '        self._pipeline_geometry_terms = {}\n\n        self.iters_only_diffuse = 2000', 'P1 attach')
    # P2 内置 TEST 报告关闭，评测只走独立进程
    text = replace_once(text, 'self.testing_iterations = testing_iterations',
                        'self.testing_iterations = []', 'P2 testing_iterations')
    # P4 暂存几何项供日志读取（不影响计算）
    text = replace_once(
        text, '            extra, geometry_terms = self.pbr_geometry.losses(self.gaussians, render_pkg_re, viewpoint_cam)\n',
        '            extra, geometry_terms = self.pbr_geometry.losses(self.gaussians, render_pkg_re, viewpoint_cam)\n'
        '            self._pipeline_geometry_terms = geometry_terms\n', 'P4 geometry terms')
    # P5 metrics.jsonl 日志（每100循环；异常只写 metrics_error.log）
    text = replace_once(
        text, '            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log\n',
        '            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log\n'
        '            from LIHAN.scripts.pipeline1107_metrics import log_step\n'
        '            log_step(self, loss, loss_result, Ll1)\n', 'P5 metrics')
    return text


def patch_pbr_geometry(text):
    # P3 门禁：原先只放行 gt_directional；放宽为 envmap + 非空 pbr_mixed_lights_path。几何损失本体不改。
    old = ("    if args.render_mode != 'original_ir' or args.pbr_light_mode != 'gt_directional':\n"
           "        raise ValueError('原生PBR几何微调仅适用于GT方向光原生PBR。')\n")
    new = ("    mixed = (args.pbr_light_mode == 'envmap' and bool(getattr(args, 'pbr_mixed_lights_path', '')))\n"
           "    if args.render_mode != 'original_ir' or (args.pbr_light_mode != 'gt_directional' and not mixed):\n"
           "        raise ValueError('原生PBR几何微调仅适用于GT方向光原生PBR，或环境光+固定方向光混合PBR。')\n")
    return replace_once(text, old, new, 'P3 geometry gate')


def build_snapshot(snapshot, root=ROOT):
    snapshot = Path(snapshot)
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in SNAPSHOT_DIRS:
        shutil.copytree(root / name, snapshot / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in LINKED_DIRS:
        (snapshot / name).symlink_to((root / name).resolve(), target_is_directory=True)
    path = snapshot / 'scripts/train_stage2.py'
    path.write_text(patch_train_stage2(path.read_text()))
    path = snapshot / 'scene/pbr_geometry.py'
    path.write_text(patch_pbr_geometry(path.read_text()))
    return {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*.py'))}


# ---------------------------------------------------------------- source 构造
def frame_ids(transforms_path):
    return sorted(int(f['source_frame']) for f in json.loads(Path(transforms_path).read_text())['frames'])


def build_sources(control, eval_source=EVAL_SOURCE):
    """A4：由 eval_source_v1 构造 train/dev/test 三个 source；train 的 TEST 置空。"""
    from LIHAN.scripts.pipeline1107_e4_directions import fit_frames, dev_frames, test_frames
    eval_source = Path(eval_source)
    train = json.loads((eval_source / 'transforms_fit.json').read_text())
    held = json.loads((eval_source / 'transforms_held.json').read_text())
    test = json.loads((eval_source / 'transforms_test.json').read_text())
    if frame_ids(eval_source / 'transforms_fit.json') != fit_frames():
        raise AssertionError('A4：transforms_fit 不是 FIT117。')
    if frame_ids(eval_source / 'transforms_held.json') != dev_frames():
        raise AssertionError('A4：transforms_held 不是 DEV18。')
    if frame_ids(eval_source / 'transforms_test.json') != test_frames():
        raise AssertionError('A4：transforms_test 不是 TEST15。')
    if not (len(train['frames']), len(held['frames']), len(test['frames'])) == (117, 18, 15):
        raise AssertionError('A4：split 帧数不是 117/18/15。')
    for mode, targets in [('train', []), ('dev', held['frames']), ('test', test['frames'])]:
        dest = Path(control) / f'{mode}_source'
        (dest / 'images').mkdir(parents=True)
        for frame in train['frames'] + targets:
            target = dest / 'images' / frame['file_path']
            if target.exists():
                continue
            image = eval_source / 'images' / frame['file_path']
            if not image.exists():
                raise FileNotFoundError(str(image))
            target.symlink_to(image.resolve())
        points = eval_source / 'points3d.ply'
        if points.exists():
            (dest / 'points3d.ply').symlink_to(points.resolve())
        write(dest / 'transforms_train.json', train)
        write(dest / 'transforms_test.json', {**test, 'frames': targets})
    return dict(fit=len(train['frames']), dev=len(held['frames']), test=len(test['frames']))


def audit_stage1_images(control, stage1_model, eval_source=EVAL_SOURCE):
    """A4：逐帧比对 FIT117 图像与 B00 训练源同名图像。

    记录字节 SHA 相同数；并在 Alpha>0 的前景上逐像素比较(PNG 编码/背景 RGB 差异不应算作训练输入差异)。
    只记录差异，不阻断。
    """
    import cv2
    import numpy as np
    cfg = ast.parse((Path(stage1_model) / 'cfg_args').read_text().strip(), mode='eval').body
    values = {kw.arg: ast.literal_eval(kw.value) for kw in cfg.keywords}
    source = Path(values['source_path'])
    train = json.loads((Path(eval_source) / 'transforms_fit.json').read_text())['frames']
    byte_same, fg_same, fg_different, alpha_different, missing = 0, 0, [], [], []
    for frame in train:
        name = frame['file_path']
        a, b = Path(eval_source) / 'images' / name, source / 'images' / name
        if not b.exists():
            missing.append(name)
            continue
        byte_same += sha(a) == sha(b)
        ia, ib = cv2.imread(str(a), -1), cv2.imread(str(b), -1)
        if ia.shape != ib.shape or ia.dtype != ib.dtype:
            fg_different.append(name)
        elif not np.array_equal(ia[..., 3], ib[..., 3]):
            alpha_different.append(name)
        else:
            mask = ia[..., 3] > 0
            if np.array_equal(ia[..., :3][mask], ib[..., :3][mask]):
                fg_same += 1
            else:
                fg_different.append(name)
    return dict(stage1_source=str(source), frames=len(train), byte_identical=int(byte_same),
                foreground_pixel_identical=fg_same, foreground_different=fg_different,
                alpha_different=alpha_different, missing=missing)


# ---------------------------------------------------------------- 总控
def prepare(a):
    from LIHAN.scripts.pipeline1107_e4_directions import (convert, check_camera_consistency, CAMERA_REFERENCE)
    control = Path(a.control)
    control.mkdir(parents=True, exist_ok=False)
    stage1 = ROOT / STAGE1 / 'model_mlp'
    ply = stage1 / f'point_cloud/iteration_{STAGE1_ITER}/point_cloud.ply'
    if sha(ply) != STAGE1_PLY_SHA:
        raise AssertionError('A1：B00 PLY SHA 不匹配。')
    record = dict(server=socket.gethostname(), environment='lumimotion-ushas', plan='LIHAN/DOC/规划/1107-03-T01完整管线与去bg规划.md',
                  stage1=dict(model=str(stage1), iteration=STAGE1_ITER, ply_sha256=sha(ply),
                              deform_sha256=sha(stage1 / f'deform/iteration_{STAGE1_ITER}/deform.pth')))
    record['splits'] = build_sources(control, a.eval_source)
    record['camera_check'] = check_camera_consistency(
        a.camera_reference, {m: control / f'{m}_source/transforms_{t}.json'
                             for m, t in [('train', 'train'), ('dev', 'test'), ('test', 'test')]})
    if not a.skip_image_sha:
        record['stage1_image_audit'] = audit_stage1_images(control, stage1, a.eval_source)
    record['e4'] = convert(control, a.e4_dir)
    record['snapshot'] = build_snapshot(control / 'snapshot')
    record['arms'] = ARMS
    write(control / 'manifest.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'snapshot'}, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- 训练命令
def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                TORCH_EXTENSIONS_DIR=str(ROOT / 'output/1025-02-Stage3-PPT配对总控/native_cache'),
                CUDA_HOME=str(Path(PYTHON).parents[1]), TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='4',
                PYTHONPATH=str(ROOT / 'output/archive/1011-07-六组训练总控/tools/nvdiffrast-sm86/extension'))


def training_command(control, arm, output, steps, smoke, geometry, warmup):
    """规划 2.3 节模板。smoke=None 为正式；'A' 为 24 循环；'B' 为 2040 循环(默认几何 warmup 2000 后激活)。"""
    control = Path(control)
    spec = ARMS[arm]
    snapshot = control / 'snapshot'
    start = STAGE1_ITER
    if smoke == 'A':
        steps = 24
    elif smoke == 'B':
        steps = 2040
    end = start + steps
    if smoke == 'A':
        saves = [end]
    elif smoke == 'B':
        saves = [start + 2000, end]
    else:
        saves = sorted({start + min(4000, steps), start + min(12000, steps), end})
    cmd = [PYTHON, str(snapshot / 'LIHAN/scripts/bootstrap_1011.py'), str(snapshot), 'LIHAN.scripts.stage3_el_eldl_entry',
           '--source_path', str(control / 'train_source'), '--model_path', str(Path(output) / 'model_mlp'),
           '--stage1_model_path', str(ROOT / STAGE1 / 'model_mlp'), '--load_iter', str(start),
           '--iterations', str(end), '--save_iterations', *[str(v) for v in saves],
           '--is_blender', '--eval', '--load2gpu_on_the_fly', '--resolution', '2',
           '--train_light_folder', 'images', '--test_light_folder', 'images', '--gt_alpha_mask_as_scene_mask',
           '--diffuse_sample_num', '128', '--trace_num_rays', '65536', '--depth_ratio', '1',
           '--pbr_rasterizer_backend', 'lumimotion']
    if spec['mixed']:
        cmd += ['--pbr_mixed_lights_path', str(control / 'e4_train_directions.json')]
    if spec['geometry'] and geometry:
        cmd += ['--pbr_optimize_geometry', '--pbr_freeze_opacity']
        if warmup is not None:
            cmd += ['--pbr_geometry_warmup', str(warmup)]
    elif warmup is not None:
        raise ValueError('--geometry-warmup 只对启用几何微调的臂有意义。')
    if spec['wo_indirect']:
        cmd += ['--wo_indirect']
    return cmd, start, end


def run(a):
    control = Path(a.control)
    spec = ARMS[a.arm]
    suffix = f'-冒烟{a.smoke}' if a.smoke else ''
    base = ROOT / 'output' / ('smoke_test' if a.smoke else '')
    name = a.name or spec['name']
    output = base / (name + suffix + a.suffix)
    # 冒烟 A 只有 24 循环，默认几何 warmup 2000 会被 validate_pbr_geometry 拒绝；默认用 8 以走通几何路径
    uses_geometry = spec['geometry'] and not a.no_geometry
    warmup = a.geometry_warmup if a.geometry_warmup is not None else (8 if a.smoke == 'A' and uses_geometry else None)
    cmd, start, end = training_command(control, a.arm, output, a.steps, a.smoke, not a.no_geometry, warmup)
    if a.dry_run:
        print(shlex.join(cmd))
        return 0
    if socket.gethostname() != HOST or Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise AssertionError('必须在 ushas 并使用 lumimotion-ushas 解释器。')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(arm=a.arm, scheme=SCHEME, status='RUNNING', command=cmd, iteration=end, parent_iteration=start,
                 steps=end - start, gpu=a.gpu, smoke=a.smoke, geometry=spec['geometry'] and not a.no_geometry,
                 wo_indirect=spec['wo_indirect'], started=time.time())
    write(output / 'state.json', state)
    readme = output / 'README.md'
    readme.write_text('# 完整管线 Stage3 实验（' + a.arm + '）\n\n完整训练命令：\n```bash\n' + shlex.join(cmd) +
                      '\n```\n\n日志 train.log；输出 model_mlp；metrics.jsonl 每 100 循环；状态 RUNNING。未读 TEST 评分。\n')
    with (output / 'train.log').open('x') as log:
        child = subprocess.Popen(cmd, cwd=control / 'snapshot', env=environment(a.gpu), stdout=log, stderr=subprocess.STDOUT)
        state['pid'] = child.pid
        write(output / 'state.json', state)
        code = child.wait()
    state.update(status='TRAIN_COMPLETED' if code == 0 else 'FAILED', exit_code=code, finished=time.time())
    write(output / 'state.json', state)
    (output / 'exitcode.txt').write_text(f'{code}\n')
    with readme.open('a') as handle:
        handle.write(f'\n训练退出码 {code}，状态 {state["status"]}。\n')
    return code


def prereg(a):
    """冻结预注册：任何正式训练启动前调用一次；已存在则拒绝覆盖。"""
    control = Path(a.control)
    path = control / 'preregistration.json'
    if path.exists():
        raise FileExistsError('预注册已冻结，禁止覆盖：' + str(path))
    manifest = json.loads((control / 'manifest.json').read_text())
    geometry = not a.no_geometry
    record = dict(
        scheme=SCHEME, geometry_mode='N1' if geometry else 'N0',
        definition='E2 = E1 + --wo_indirect（逐高斯SH间接光通道置零，环境光由全局EnvLight承担）；其余逐字一致',
        commands={arm: shlex.join(training_command(control, arm, ROOT / 'output' / ARMS[arm]['name'], 20000, None, geometry, None)[0])
                  for arm in ARMS},
        snapshot_sha256=manifest['snapshot'],
        direction_json_sha256=manifest['e4']['files'],
        decision_rule=dict(
            tau_P='max(0.05 dB, 2*delta_PSNR)', tau_S='max(1e-4, 2*delta_SSIM)', tau_L='max(1e-3, 2*delta_LPIPS)',
            support='dPSNR>=tau_P and dSSIM>=tau_S and dLPIPS<=+tau_L', oppose='dPSNR<=-tau_P and dSSIM<=-tau_S',
            otherwise='不确定/混合', qualifier='支持且M1-M5触发>=2个旗标时写“渲染指标支持，伴随泄漏迹象”'),
        fallback=dict(geometry='冒烟B任一N1回退条件触发则E1/E2同时改N0', deadline='11-10 06:00 保存点降级见规划第10节'),
        frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write(path, record)
    write(control / 'preregistration.sha256.json', dict(sha256=sha(path)))
    print(json.dumps(dict(path=str(path), sha256=sha(path), geometry_mode=record['geometry_mode']), ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'run', 'prereg'):
        s = sub.add_parser(name)
        s.add_argument('--control', type=Path, default=ROOT / 'output' / CONTROL_NAME)
        if name == 'prepare':
            from LIHAN.scripts.pipeline1107_e4_directions import E4_DIR, CAMERA_REFERENCE
            s.add_argument('--eval-source', type=Path, default=EVAL_SOURCE)
            s.add_argument('--e4-dir', type=Path, default=E4_DIR)
            s.add_argument('--camera-reference', type=Path, default=CAMERA_REFERENCE)
            s.add_argument('--skip-image-sha', action='store_true')
        if name == 'run':
            s.add_argument('--arm', choices=ARMS, required=True)
            s.add_argument('--smoke', choices=['A', 'B'])
            s.add_argument('--steps', type=int, default=20000)
            s.add_argument('--gpu', type=int, default=0)
            s.add_argument('--name', default='')
            s.add_argument('--suffix', default='')
            s.add_argument('--no-geometry', action='store_true', help='N0 回退：去掉两个几何开关')
            s.add_argument('--geometry-warmup', type=int)
            s.add_argument('--dry-run', action='store_true')
        if name == 'prereg':
            s.add_argument('--no-geometry', action='store_true')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a)
    elif a.action == 'prereg':
        prereg(a)
    else:
        sys.exit(run(a))


if __name__ == '__main__':
    main()
