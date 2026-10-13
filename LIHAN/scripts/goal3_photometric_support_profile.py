#!/usr/bin/env python3
"""F-T04@16500原135TRAIN光度支持与参数梯度竞争；零更新、无GT几何/测试输入。"""
import argparse
from argparse import Namespace
import ast
import hashlib
import importlib.abc
import importlib.util
import itertools
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import shlex
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp'
SNAPSHOT = ROOT / 'output/1012-20-LH-static-jumpingjacks_DL_V2-F-T04-T05执行总控/snapshots/foot-densify-stable'
OUTPUT = ROOT / 'output/1012-25-LH-static-jumpingjacks_DL_V2-G2-T07光度支持与几何梯度零更新诊断'
PROBE_OUTPUT = ROOT / 'output/smoke_test/1012-23-staticDL-G2-T07真实native零更新探针'
PROBE_FRAMES = ('0001', '0071', '0149')
BOOTSTRAP = ROOT / 'LIHAN/scripts/bootstrap_1011.py'
ITERATION, SEED = 16500, 1012
GRADIENT_TOLERANCE = dict(atol=1e-8, rtol=1e-3, relative_l2_max=1e-3)
PARAMETERS = ('xyz', 'scaling', 'rotation', 'opacity')
BRANCHES = ('photo', 'alpha', 'distortion', 'normal', 'normal_render', 'normal_depth')
PAIRS = tuple(itertools.combinations(BRANCHES, 2))
ASSET_SHA = {
 'foot_densify_training.json': '1fb6429d07649cfc0a5919b60c7fe9f151ee45c18229aadd96a1f97fd1ca2975',
 'foot_densify_finished.json': 'c2e6a24983c3fd8f132f4ee36cbad44b047fca3fb232ccd18a30cd1152ec1b81',
 'static_dl_training.json': '0fb1adec4b73f1d9fd8e348048a231dc86d0d8d0eb021ad8493bf73dd82ab0b0',
 'point_cloud/iteration_16500/point_cloud.ply': 'eb3eafd88af6a54d736541a5c3766410a86dc53df612e432307906f0591e6cdc',
 'deform/iteration_16500/deform.pth': '343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0',
 'static_dl_lighting/iteration_16500/lighting.json': 'a2a046d56c6ead9055b275464ecfb36c3efccb3b4d58076160f3ec0590a7c945',
 'optimizer/iteration_16500.pth': '1f36f92d1745bcd538a4e64f6518d1f4baf4af12c33715cf9cf63720b974bd4e',
}
MANIFEST_SHA = '99ef56495338589d6f78ee7c48f9555e13b33c128bffb56eb84af84ef84d4d83'
TRACER_SHA = '5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def tensor_digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode()); h.update(str(value.device).encode())
        h.update(str(value.requires_grad).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_tensors(pc, lights):
    """冻结全部pc公开/私有Tensor及灯参数、方向；tracer的BVH缓存不算模型参数。"""
    import torch
    values = {f'pc.{n}': v for n, v in vars(pc).items() if isinstance(v, torch.Tensor)}
    values.update({f'lights.parameter.{n}': v for n, v in lights.named_parameters()})
    values.update({f'lights.buffer.{n}': v for n, v in lights.named_buffers()})
    values.update({f'lights.direction.{n}': v for n, v in lights.directions.items()})
    return values


def grad_fields_empty(values):
    return all(v.grad is None for v in values.values() if v.is_leaf or v.retains_grad)


def rng_snapshot():
    import numpy as np
    import torch
    return {'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
            'torch': tensor_digest({'state': torch.get_rng_state()}),
            'cuda': tensor_digest({str(i): s for i, s in enumerate(torch.cuda.get_rng_state_all())})}


def camera_input_paths(cfg, metadata):
    """在图像加载前复现冻结loader路径规则；只解析原TRAIN相机JSON，不打开候选图片。"""
    paths = []
    for frame in metadata['frames']:
        raw = Path(os.path.join(cfg['source_path'], cfg['train_light_folder'], frame['file_path']))
        image = raw if frame['file_path'].endswith(('jpg','png')) else Path(str(raw)+'.png')
        rgba = raw.parent.parent / 'rgba'
        if rgba.exists():
            image = Path(str(rgba / raw.name).replace('.jpg','.png'))
        paths.append(image.resolve())
    return paths


def preflight():
    """仅CPU和原TRAIN输入；不导入评测protocol/audit，不读取GT目录。"""
    inputs = {str(MODEL / k): digest(MODEL / k) for k in ASSET_SHA}
    if inputs != {str(MODEL / k): v for k, v in ASSET_SHA.items()}:
        raise AssertionError('T04@16500冻结资产不符。')
    manifest = SNAPSHOT / 'snapshot_manifest.json'
    if digest(manifest) != MANIFEST_SHA:
        raise AssertionError('原102文件训练快照manifest不符。')
    sources = {str(SNAPSHOT / k): v for k, v in json.loads(manifest.read_text())['file_sha256'].items()}
    if any(digest(p) != h for p, h in sources.items()):
        raise AssertionError('原科学快照内容变化。')
    sources.update({str(manifest): MANIFEST_SHA, str(BOOTSTRAP): digest(BOOTSTRAP), str(Path(__file__).resolve()): digest(__file__)})
    t = json.loads((MODEL / 'foot_densify_training.json').read_text())
    st = json.loads((MODEL / 'static_dl_training.json').read_text())
    light = json.loads((MODEL / 'static_dl_lighting/iteration_16500/lighting.json').read_text())
    cfg = t['inherited_config'].copy()
    expected = {str(i).zfill(4) for i in range(1, 151) if i % 10}
    if (t['smoke'] or t['clock_start'] != 16000 or t['update_budget'] != 2000
            or t['densification_route'] != 'original' or t['normal_opacity_route'] != 'detach_opacity'
            or light['iteration'] != ITERATION or light['optimizer_updates'] != 500
            or light['normal_opacity_route'] != 'detach_opacity' or light['ambient_fraction'] != 0
            or cfg['depth_ratio'] != 1 or cfg['white_background'] or cfg['static_dl_bvh_interval'] != 1):
        raise AssertionError('冻结模型/真实损失路线与预注册不一致。')
    for k, v in {'lambda_alpha_loss': .1, 'lambda_dist': 1000, 'static_dl_normal_weight': .1,
                 'static_dl_shadow_weight': .25, 'static_dl_highlight_weight': .5, 'static_dl_residual_threshold': .08}.items():
        if cfg[k] != v:
            raise AssertionError('真实权重变化：' + k)
    if any(sources.get(p) != h for p, h in t['new_sources'].items()):
        raise AssertionError('12训练科学文件与原快照不一致。')
    metadata_path = Path(cfg['source_path']) / 'transforms_train.json'
    metadata = json.loads(metadata_path.read_text())
    if len(metadata['frames']) != 135 or {str(f['source_frame']).zfill(4) for f in metadata['frames']} != expected:
        raise AssertionError('原135 TRAIN相机集合不一致。')
    # marker固定SHA使此137项白名单不能通过改写marker扩张到GT/test。
    if len(t['training_inputs']) != 137:
        raise AssertionError('原输入不是137项。')
    whitelist = {Path(p).resolve() for p in t['training_inputs']}
    camera_paths = camera_input_paths(cfg, metadata)
    if len(set(camera_paths)) != 135 or any(p not in whitelist for p in camera_paths):
        raise AssertionError('冻结loader实际将读取的图像不在原135 TRAIN白名单，加载前拒绝。')
    for p, h in t['training_inputs'].items():
        if digest(p) != h:
            raise AssertionError('原TRAIN输入变化：' + p)
        inputs[p] = h
    if str(metadata_path) not in t['training_inputs'] or str(Path(cfg['static_dl_directions'])) not in t['training_inputs']:
        raise AssertionError('相机与纯方向未绑定原TRAIN白名单。')
    command = MODEL.parent / 'commands/train.json'
    if Path(json.loads(command.read_text())['argv'][2]).resolve() != SNAPSHOT:
        raise AssertionError('真实训练bootstrap来源不是固定snapshot20。')
    inputs[str(command)] = digest(command)
    return cfg, t, st, light, inputs, sources


def dispatch(args):
    entry, helper = '_lumimotion_goal3_photometric_support_profile', str(Path(__file__).resolve())
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            return importlib.util.spec_from_file_location(fullname, helper) if fullname == entry else None
    sys.meta_path.insert(0, Finder())
    os.environ['LUMI_PHOTO_PROFILE_COMMAND_JSON'] = json.dumps([sys.executable, helper] + sys.argv[1:])
    sys.argv = [str(BOOTSTRAP), str(SNAPSHOT), entry, '--bootstrapped'] + sys.argv[1:]
    runpy.run_path(str(BOOTSTRAP), run_name='__main__')


def initialize():
    import torch
    import diff_surfel_rasterization as native
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms, getNerfppNorm
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_ambient import render_photometric
    from scene.static_dl_normal_opacity import geometry_losses
    from scene.static_dl_censored import censored_photometric_loss
    from scene.stage2_photometric import frame_key
    from utils.camera_utils import cameraList_from_camInfos
    cfg, training, static, lighting, inputs, sources = preflight()
    if Path.cwd().resolve() != SNAPSHOT:
        raise AssertionError('未通过真实原bootstrap进入固定快照。')
    for func in (render_photometric, geometry_losses, censored_photometric_loss,
                 readCamerasFromTransforms, cameraList_from_camInfos, GaussianModel,
                 DirectionOnlyLights, StaticVisibility, frame_key):
        p = str(Path(sys.modules[func.__module__].__file__).resolve())
        if p not in sources or digest(p) != sources[p]:
            raise AssertionError('实际科学函数来源未冻结：' + p)
    for key in ('method', 'source_sha256', 'extension_sha256'):
        if native.provenance[key] != static['rasterizer_provenance'][key]:
            raise AssertionError('原生CUDA来源变化：' + key)
    cfg.update(data_device='cpu', load2gpu_on_the_fly=True)
    pipe = Namespace(**cfg)
    infos = readCamerasFromTransforms(cfg['source_path'], 'transforms_train.json', cfg['train_light_folder'], False, no_bg=True)
    if any(str(Path(i.image_path_train_light)) not in training['training_inputs'] for i in infos):
        raise AssertionError('加载图片不是固定原TRAIN输入。')
    cameras = cameraList_from_camInfos(infos, 1., pipe)
    expected = {str(i).zfill(4) for i in range(1, 151) if i % 10}
    if len(cameras) != 135 or {frame_key(c.image_name_train_light) for c in cameras} != expected:
        raise AssertionError('实际相机不等于原135TRAIN。')
    if abs(float(getNerfppNorm(infos)['radius']) - training['camera_extent']) > 1e-7:
        raise AssertionError('原训练相机extent变化。')
    state = torch.load(MODEL / 'deform/iteration_16500/deform.pth', map_location='cpu', weights_only=True)
    if any(torch.count_nonzero(v) for k, v in state.items() if any(x in k for x in ('gaussian_warp.', 'gaussian_rotation.', 'gaussian_scaling.'))):
        raise AssertionError('静态几何头不是零。')
    pc = GaussianModel(cfg['sh_degree'], no_binary_separation=True, fea_dim=cfg['hyper_dim'])
    pc.load_ply(str(MODEL / 'point_cloud/iteration_16500/point_cloud.ply'))
    del pc._albedo_dc_stage1
    if len(pc.get_xyz) != 63340:
        raise AssertionError('16500实际点数不符。')
    lights = DirectionOnlyLights(cfg['static_dl_directions'])
    if set(lights.directions) != expected or lights.sha256 != lighting['directions_sha256']:
        raise AssertionError('只接受原135 TRAIN纯方向。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    return pc, lights, cameras, pipe, StaticVisibility(cfg['static_dl_shadow_bias'], 1), inputs, sources, native.provenance


def runtime_evidence(provenance):
    import subprocess
    import torch
    maps = Path('/proc/self/maps').read_text()
    modules = sorted({x.split()[-1] for x in maps.splitlines() if ('lumimotion_pbr_cuda' in x or 'surfel_tracer/' in x) and x.split()[-1].endswith('.so')})
    hashes = {p: digest(p) for p in modules}
    if not any('surfel_tracer/' in p and h == TRACER_SHA for p, h in hashes.items()):
        raise AssertionError('实际tracer来源不符。')
    if not any('lumimotion_pbr_cuda' in p and h == provenance['extension_sha256'] for p, h in hashes.items()):
        raise AssertionError('实际CUDA映射来源不符。')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = [x for x in result.stdout.splitlines() if x.split(',')[0].strip() == str(os.getpid())]
    if not rows:
        raise AssertionError('无真实GPU进程记录。')
    return dict(pid=os.getpid(), proc_cmdline=Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode(), runtime_modules=hashes, gpu_rows=rows,
                native_provenance=provenance, python=sys.executable, torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CONDA_PREFIX', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCH_CUDA_ARCH_LIST')})


def describe(a):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    if not a.size:
        return {'count': 0, 'mean': None, 'min': None, 'p10': None, 'median': None, 'p90': None, 'max': None}
    return dict(count=int(a.size), mean=float(a.mean()), min=float(a.min()), p10=float(np.quantile(a, .1)), median=float(np.median(a)), p90=float(np.quantile(a, .9)), max=float(a.max()))


def point_statistics(gradients):
    """CPU float64范数/点积避免小余切平方下溢；原向量保存float32。"""
    import numpy as np
    stats = {}
    for p in PARAMETERS:
        norms = {b: np.linalg.norm(gradients[b][p].astype(np.float64), axis=1) for b in BRANCHES}
        for b, n in norms.items():
            stats[f'{b}__{p}__norm'] = n
        for a, b in PAIRS:
            dot = np.einsum('ij,ij->i', gradients[a][p].astype(np.float64), gradients[b][p].astype(np.float64))
            denom = norms[a] * norms[b]
            stats[f'{a}_vs_{b}__{p}__dot'] = dot
            stats[f'{a}_vs_{b}__{p}__cos'] = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
            stats[f'{a}_vs_{b}__{p}__supported'] = denom > 0
    return stats


def gradient_match(actual, expected):
    import numpy as np
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    diff = a - b
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-30))
    maximum = float(np.max(np.abs(diff)))
    return dict(passed=bool(np.allclose(a, b, atol=GRADIENT_TOLERANCE['atol'],
        rtol=GRADIENT_TOLERANCE['rtol']) and rel <= GRADIENT_TOLERANCE['relative_l2_max']),
        max_abs=maximum, relative_l2=rel, tolerance=GRADIENT_TOLERANCE)


def pixel_diagnostics(image, target, alpha, dphoto, geometry):
    import numpy as np
    def cpu(x): return x.detach().cpu().numpy()
    pred, rgb, mask, derivative = map(cpu, (image, target, alpha, dphoto))
    signed = (pred - rgb).mean(axis=0, keepdims=True)
    fg = mask > .5
    group = np.where(fg & (signed > .08), 1, np.where(fg & (signed < -.08), 2, 0)).astype(np.uint8)
    weights = np.where(group == 1, .25, np.where(group == 2, .5, 1.)).astype(np.float32)
    saturated = (rgb == 1.) & np.broadcast_to(fg, rgb.shape)
    satisfied = saturated & (pred >= 1.)
    # 原相机加载后sRGB通道算术均值；只作观测分组，不称物理辐亮度/阴影语义。
    brightness = rgb.mean(axis=0, keepdims=True)
    bands = np.digitize(brightness, [.05, .2, .8]).astype(np.uint8)
    arrays = dict(target_rgb=rgb, predicted_rgb=pred, target_alpha=mask, target_brightness=brightness,
                  brightness_band=bands, signed_residual=signed, residual_group=group,
                  detached_photo_weight=weights, exact_saturated=saturated, censor_satisfied=satisfied,
                  dphoto_dimage=derivative, predicted_expected_depth=cpu(geometry['surf_depth']))
    summaries = {}
    for name, pix in {'all': np.ones_like(fg), 'foreground': fg, **{f'foreground_brightness_band_{i}': fg & (bands == i) for i in range(4)}}.items():
        ch = np.broadcast_to(pix, rgb.shape)
        summaries[name] = dict(pixel_count=int(pix.sum()), brightness=describe(brightness[pix]),
            shadow_residual_pixels=int(((group == 1) & pix).sum()), highlight_residual_pixels=int(((group == 2) & pix).sum()),
            exact_saturated_channels=int((saturated & ch).sum()), censor_satisfied_channels=int((satisfied & ch).sum()),
            dphoto_dimage_abs=describe(np.abs(derivative[ch])), nonzero_derivative_channels=int(np.count_nonzero(derivative[ch])))
    return arrays, summaries, bool(np.all(derivative[satisfied] == 0))


def cpu_check():
    import numpy as np
    import torch
    cfg, t, st, light, inputs, sources = preflight()
    # 从冻结损失文件提取原两个纯Torch函数，避免CPU预检导入CUDA renderer。
    tree = ast.parse((SNAPSHOT / 'scene/static_dl_censored.py').read_text())
    funcs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('saturated_channels', 'censored_photometric_loss')], type_ignores=[])
    ns = {'torch': torch}; exec(compile(funcs, '<frozen_censored_functions>', 'exec'), ns)
    target = torch.tensor([[[0., .9, 1., 1.]]]).repeat(3, 1, 1)
    image = torch.tensor([[[.2, .5, 1.3, .8]]]).repeat(3, 1, 1).requires_grad_(True)
    loss = ns['censored_photometric_loss'](image, target, torch.ones(1, 1, 4))[0]
    grad = torch.autograd.grad(loss, image)[0]
    residual = image.detach() - target
    ref = residual / torch.sqrt(residual.square() + 1e-6) * torch.tensor([.25, .5, 0., .5]) / image.numel()
    fake = {b: {p: np.zeros((3, 2), np.float32) for p in PARAMETERS} for b in BRANCHES}
    for p in PARAMETERS:
        fake['photo'][p][0] = [1., 0.]; fake['normal'][p][0] = [-2., 0.]
        fake['photo'][p][1] = [1e-25, 0.]; fake['normal'][p][1] = [1e-25, 0.]
    stats = point_statistics(fake)
    p = torch.tensor([[1., 2.]], requires_grad=True)
    rn, dn = p * 2, p.square()
    normal = .1 * ((rn - dn)**2).sum()
    direct = torch.autograd.grad(normal, p, retain_graph=True)[0]
    vr, vd = torch.autograd.grad(normal, (rn, dn), retain_graph=True)
    split = torch.autograd.grad(rn, p, vr.detach(), retain_graph=True)[0] + torch.autograd.grad(dn, p, vd.detach())[0]
    fake_pc = Namespace(weight=torch.nn.Parameter(torch.ones(2)), unused=torch.zeros(3))
    fake_lights = torch.nn.Module(); fake_lights.log_intensity = torch.nn.Parameter(torch.zeros(2))
    fake_lights.directions = {'0001':torch.tensor([[0.,0.,1.]])}
    inventory = model_tensors(fake_pc, fake_lights)
    inventory_before = tensor_digest(inventory)
    fake_pc.weight.grad = torch.ones_like(fake_pc.weight)
    grad_nonempty_detected = not grad_fields_empty(model_tensors(fake_pc, fake_lights))
    fake_pc.weight.grad = None
    calls = {n.func.attr for n in ast.walk(ast.parse(Path(__file__).read_text())) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    checks = dict(frozen_sources_and_137_inputs=True, no_test_frames=t['train_images'] == 135 and not t['test_images_loaded'],
        exact_censor_derivative=bool(torch.allclose(grad, ref, atol=1e-8, rtol=1e-6)), satisfied_fullcode_zero=bool((grad[..., 2] == 0).all()),
        opposing_gradients=stats['photo_vs_normal__xyz__cos'][0] == -1., tiny_gradients_supported=bool(stats['photo_vs_normal__xyz__supported'][1]),
        zero_gradients_unsupported=not bool(stats['photo_vs_normal__xyz__supported'][2]), vjp_sum=bool(torch.equal(direct, split)),
        all_model_tensor_inventory=set(inventory)=={'pc.weight','pc.unused','lights.parameter.log_intensity','lights.direction.0001'},
        grad_fields_complete=grad_nonempty_detected and grad_fields_empty(model_tensors(fake_pc,fake_lights)),
        inventory_digest_stable=inventory_before==tensor_digest(model_tensors(fake_pc,fake_lights)),
        output_scope=OUTPUT.parent==ROOT/'output' and PROBE_OUTPUT.parent==ROOT/'output/smoke_test',
        probe_frames_train_only=len(PROBE_FRAMES)==3 and all(int(k)%10 for k in PROBE_FRAMES),
        loader_paths_whitelisted_before_read=len(camera_input_paths(cfg,json.loads((Path(cfg['source_path'])/'transforms_train.json').read_text())))==135,
        no_optimizer_or_topology_calls=not bool(calls & {'step', 'training_setup', 'densify_and_prune', 'densify_and_split', 'densify_and_clone', 'prune_points', 'reset_opacity'}))
    result=dict(status='PASS_CPU_ONLY' if all(checks.values()) else 'FAILED', checks={k:bool(v) for k,v in checks.items()}, source_count=len(sources), input_count=len(inputs),
        note='仅CPU公式/来源预检，无CUDA初始化；真实135TRAIN梯度分解/显存与科学结论待独立执行。')
    print(json.dumps(result, ensure_ascii=False))
    if not all(checks.values()): raise AssertionError('CPU检查失败。')


def run(output, probe_only=False):
    import numpy as np
    import torch
    from scene.static_dl_ambient import render_photometric
    from scene.static_dl_normal_opacity import geometry_losses
    from scene.static_dl_censored import censored_photometric_loss
    pc, lights, cameras, pipe, visibility, inputs, sources, provenance = initialize()
    params = {n: getattr(pc, '_' + n) for n in PARAMETERS}
    all_params = model_tensors(pc, lights)
    if pc.optimizer is not None or not grad_fields_empty(all_params):
        raise AssertionError('不能存在优化器或遗留参数梯度。')
    before, rng_before = tensor_digest(all_params), rng_snapshot()
    local_rng, remaining, order = random.Random(SEED), list(range(135)), []
    while remaining: order.append(remaining.pop(local_rng.randrange(len(remaining))))
    if probe_only:
        from scene.stage2_photometric import frame_key
        lookup = {frame_key(c.image_name_train_light): i for i, c in enumerate(cameras)}
        order = [lookup[k] for k in PROBE_FRAMES]
    else:
        probe_path = PROBE_OUTPUT / 'summary.json'
        probe_plan = PROBE_OUTPUT / 'plan.json'
        if not probe_path.is_file() or not probe_plan.is_file():
            raise AssertionError('全135诊断前必须完成独立真实native三帧探针。')
        probe, registered = json.loads(probe_path.read_text()), json.loads(probe_plan.read_text())
        if (probe['status'] != 'PASS' or not probe.get('probe_only') or probe['frames'] != 3
                or registered['code_sources'] != sources or registered['inputs_before'] != inputs
                or registered['gradient_tolerance'] != GRADIENT_TOLERANCE):
            raise AssertionError('三帧探针未通过或来源/容差已改变，必须独立重审。')
    original_command = json.loads(os.environ['LUMI_PHOTO_PROFILE_COMMAND_JSON'])
    output.mkdir(parents=True, exist_ok=True); (output / 'per_frame').mkdir(exist_ok=False)
    plan=dict(model=str(MODEL), iteration=ITERATION, source=pipe.source_path, snapshot=str(SNAPSHOT), command=original_command, command_shell=shlex.join(original_command),
        inputs_before=inputs, code_sources=sources, native_provenance=provenance, parameter_digest_before=before, rng_before=rng_before,
        rng_scope='模型初始化完成后、诊断所选原TRAIN帧之前至之后；不冒称重放历史RNG/Adam', frames=[cameras[i].image_name_train_light for i in order],
        camera_indices=order, seed=SEED, loss_weights=dict(photo=1., alpha=.1, distortion=1000., normal=.1),
        probe_only=probe_only, frame_count=len(order), gradient_tolerance=GRADIENT_TOLERANCE,
        model_tensor_inventory={n:dict(shape=list(v.shape), dtype=str(v.dtype), device=str(v.device), requires_grad=v.requires_grad)
                                for n,v in all_params.items()},
        model_tensor_scope='全部GaussianModel直接Tensor属性，灯参数/buffer/全部135方向；原生tracer的重建BVH缓存可变，非模型状态。',
        derivative_coordinates='原可导参数_xyz、log_scaling、未归一化_rotation、opacity_logit；不同参数量纲不混算cos',
        branches=BRANCHES, branch_note='前四项为真实训练总loss；normal_render/normal_depth仅分解normal，不能再次加到总loss',
        low_height_bands=[0., .05, .35], low_height_note='预测点世界Z分组，非GT鞋语义，不选择训练区域',
        brightness_edges=[.05,.2,.8], optimizer_created=False, parameter_updates=0, train_only=True, new_mae=False,
        storage='每帧float32原参数余切/像素数组；CPU float64范数/点积与跨帧累计；GPU无跨帧数组')
    write(output / 'plan.json', plan)
    xyz = pc.get_xyz.detach().cpu().numpy().copy(); z = xyz[:, 2]
    groups={'all':np.ones(len(z),bool),'pred_z_le_0':z<=0,'pred_z_0_to_005':(z>0)&(z<=.05),'pred_z_005_to_035':(z>.05)&(z<=.35),'pred_z_gt_035':z>.35}
    np.savez_compressed(output/'points.npz', xyz=xyz, scaling=pc.get_scaling.detach().cpu().numpy(), opacity=pc.get_opacity.detach().cpu().numpy(), **groups)
    sums, norm_sums, support, pair_sums, pair_cos_sums, pair_support, pair_negative = {}, {}, {}, {}, {}, {}, {}
    rows=[]; background=torch.zeros(3,device='cuda')
    def take_grads(value, grad_outputs=None):
        gs=torch.autograd.grad(value, tuple(params.values()), grad_outputs=grad_outputs, retain_graph=True, allow_unused=True)
        result={}; absent={}
        for (name,p),g in zip(params.items(),gs):
            absent[name]=g is None
            a=np.zeros(tuple(p.shape),np.float32) if g is None else g.detach().cpu().numpy().copy()
            if not np.isfinite(a).all(): raise FloatingPointError('非有限参数余切：'+name)
            result[name]=a
        return result,absent
    for index, ci in enumerate(order,1):
        camera=cameras[ci]; camera.load2device()
        photo=render_photometric(camera,pc,pipe,background,lights,visibility,ITERATION+index,0.)
        target,alpha=camera.original_image_train_light.cuda(),camera.gt_alpha_mask.cuda()
        rgb=censored_photometric_loss(photo['render'],target,alpha,.25,.5,.08)[0]
        losses,geometry=geometry_losses(camera,pc,pipe,background,alpha,photo,'detach_opacity')
        weighted={'photo':rgb,'alpha':.1*losses['alpha'],'distortion':1000.*losses['distortion'],'normal':.1*losses['normal']}
        if not all(torch.isfinite(v) for v in weighted.values()): raise FloatingPointError('真实loss非有限。')
        forward={'rgb':photo['render'],'alpha':photo['rend_alpha'],'expected':geometry['surf_depth'],'rn':geometry['rend_normal'],'dn':geometry['surf_normal']}
        forward_before=tensor_digest(forward)
        dphoto=torch.autograd.grad(rgb,photo['render'],retain_graph=True)[0]
        pixels,pixel_summary,censor_zero=pixel_diagnostics(photo['render'],target,alpha,dphoto,geometry)
        del dphoto
        grads,absent={},{}
        for branch,value in weighted.items(): grads[branch],absent[branch]=take_grads(value)
        vr,vd=torch.autograd.grad(weighted['normal'],(geometry['rend_normal'],geometry['surf_normal']),retain_graph=True)
        grads['normal_render'],absent['normal_render']=take_grads(geometry['rend_normal'],vr.detach())
        grads['normal_depth'],absent['normal_depth']=take_grads(geometry['surf_normal'],vd.detach())
        del vr,vd
        total,_=take_grads(sum(weighted.values()))
        normal_check={p:gradient_match(grads['normal'][p],grads['normal_render'][p]+grads['normal_depth'][p]) for p in PARAMETERS}
        total_check={p:gradient_match(total[p],sum(grads[b][p] for b in BRANCHES[:4])) for p in PARAMETERS}
        repeated,_=take_grads(weighted['normal'])
        repeat_check={p:gradient_match(grads['normal'][p],repeated[p]) for p in PARAMETERS}
        checks=dict(normal_vjp_sum=all(v['passed'] for v in normal_check.values()), total_loss_gradient_sum=all(v['passed'] for v in total_check.values()),
            native_normal_repeated_vjp=all(v['passed'] for v in repeat_check.values()),
            expected_gradient_paths_present=all(not absent[b][p] for b in BRANCHES for p in PARAMETERS
                if not (b in ('normal','normal_render','normal_depth') and p=='opacity')),
            normal_opacity_zero=all(np.count_nonzero(grads[b]['opacity'])==0 for b in ('normal','normal_render','normal_depth')),
            censor_satisfied_derivative_zero=censor_zero, parameter_grad_fields_untouched=grad_fields_empty(model_tensors(pc,lights)),
            all_model_tensors_unchanged=before==tensor_digest(model_tensors(pc,lights)), no_optimizer=pc.optimizer is None,
            forward_unchanged=forward_before==tensor_digest(forward), rng_unchanged=rng_before==rng_snapshot())
        if not all(checks.values()):
            write(output/f'failed_frame_{index:03d}.json',dict(checks=checks,normal_sum=normal_check,total_sum=total_check,normal_repeated_vjp=repeat_check))
            failure_arrays={f'{b}__{p}__gradient':grads[b][p] for b in BRANCHES for p in PARAMETERS}
            failure_arrays.update({f'total__{p}__gradient':total[p] for p in PARAMETERS})
            failure_arrays.update({f'normal_repeat__{p}__gradient':repeated[p] for p in PARAMETERS})
            np.savez_compressed(output/f'failed_frame_{index:03d}_points.npz',**failure_arrays)
            np.savez_compressed(output/f'failed_frame_{index:03d}_pixels.npz',**pixels)
            raise AssertionError('真实帧梯度分解/不变性失败，保留现场。')
        del repeated
        stats=point_statistics(grads)
        for b in BRANCHES:
            for p in PARAMETERS:
                key=f'{b}__{p}'; g=grads[b][p].astype(np.float64); n=stats[key+'__norm']
                if key not in sums: sums[key]=np.zeros_like(g); norm_sums[key]=np.zeros_like(n); support[key]=np.zeros(len(n),np.int32)
                sums[key]+=g; norm_sums[key]+=n; support[key]+=(n>0)
        for a,b in PAIRS:
            for p in PARAMETERS:
                key=f'{a}_vs_{b}__{p}'; dot=stats[key+'__dot']; cos=stats[key+'__cos']; valid=stats[key+'__supported']
                if key not in pair_sums:
                    pair_sums[key]=np.zeros_like(dot); pair_cos_sums[key]=np.zeros_like(dot); pair_support[key]=np.zeros(len(dot),np.int32); pair_negative[key]=np.zeros(len(dot),np.int32)
                pair_sums[key]+=dot; pair_cos_sums[key]+=cos; pair_support[key]+=valid; pair_negative[key]+=(dot<0)&valid
        arrays={f'{b}__{p}__gradient':grads[b][p] for b in BRANCHES for p in PARAMETERS}
        arrays.update(stats); arrays['radii_visible']=photo['visibility_filter'].detach().cpu().numpy()
        gp=output/'per_frame'/f'{index:03d}_{camera.image_name_train_light}_points.npz'
        ip=output/'per_frame'/f'{index:03d}_{camera.image_name_train_light}_pixels.npz'
        np.savez_compressed(gp,**arrays); np.savez_compressed(ip,**pixels)
        row=dict(order=index,camera_index=ci,frame=camera.image_name_train_light,weighted_losses={k:float(v) for k,v in weighted.items()},
            checks=checks,normal_vjp_reconstruction=normal_check,total_reconstruction=total_check,
            normal_repeated_vjp=repeat_check,gradient_absent=absent,pixel_summary=pixel_summary,
            point_arrays=str(gp),point_arrays_sha256=digest(gp),pixel_arrays=str(ip),pixel_arrays_sha256=digest(ip),
            position_groups={name:{'point_count':int(mask.sum()),'norms':{k:describe(v[mask]) for k,v in stats.items() if k.endswith('__norm')}} for name,mask in groups.items()})
        rows.append(row)
        with (output/'per_frame.jsonl').open('a') as stream: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        camera.load2device('cpu')
        if index==1: write(output/'runtime_active.json',runtime_evidence(provenance))
        if index%5==0 or index==1: print(json.dumps(dict(completed=index,total=len(order),frame=row['frame']),ensure_ascii=False),flush=True)
        del photo,geometry,losses,weighted,forward,rgb,target,alpha,grads,total,stats,arrays,pixels,value
    aggregated={}
    for key in sums:
        aggregated[key+'__gradient_mean']=sums[key]/len(order)
        aggregated[key+'__norm_mean']=norm_sums[key]/len(order)
        aggregated[key+'__nonzero_views']=support[key]
        aggregated[key+'__directional_coherence']=np.divide(np.linalg.norm(sums[key],axis=1),norm_sums[key],out=np.zeros_like(norm_sums[key]),where=norm_sums[key]>0)
    for key in pair_sums:
        aggregated[key+'__dot_mean']=pair_sums[key]/len(order)
        aggregated[key+'__cos_mean_supported']=np.divide(pair_cos_sums[key],pair_support[key],out=np.zeros_like(pair_cos_sums[key]),where=pair_support[key]>0)
        aggregated[key+'__both_nonzero_views']=pair_support[key]; aggregated[key+'__opposing_views']=pair_negative[key]
    aggregate_path = output / ('probe_points.npz' if probe_only else 'all135_points.npz')
    np.savez_compressed(aggregate_path,**aggregated)
    invariants=dict(parameters_unchanged=before==tensor_digest(model_tensors(pc,lights)),inputs_unchanged=all(digest(p)==h for p,h in inputs.items()),sources_unchanged=all(digest(p)==h for p,h in sources.items()),
        rng_unchanged=rng_before==rng_snapshot(),no_optimizer=pc.optimizer is None,parameter_grad_fields_untouched=grad_fields_empty(model_tensors(pc,lights)),
        frames_complete_unique=len(rows)==len(order) and len({r['frame'] for r in rows})==len(order),all_frame_checks=all(all(r['checks'].values()) for r in rows))
    summary=dict(status='PASS' if all(invariants.values()) else 'FAILED',invariants=invariants,model=str(MODEL),iteration=ITERATION,frames=len(order),point_count=len(z),
        probe_only=probe_only,gradient_tolerance=GRADIENT_TOLERANCE,parameter_digest_after=tensor_digest(model_tensors(pc,lights)),
        parameter_updates=0,optimizer_created=False,test_images_loaded=False,gt_geometry_or_semantics_loaded=False,new_mae=False,runtime=runtime_evidence(provenance),
        aggregate_arrays=str(aggregate_path),aggregate_arrays_sha256=digest(aggregate_path),position_groups={name:{k:describe(v[mask]) for k,v in aggregated.items() if v.ndim==1} for name,mask in groups.items()},
        interpretation='仅支持/竞争诊断；radii可见不等于无遮挡贡献，低高度非鞋语义，余切不等于Adam实际更新；无自动修改/训练准入结论。')
    write(output/'summary.json',summary)
    previous=(output/'README.md').read_text() if (output/'README.md').exists() else ''
    (output/'README.md').write_text(previous+'\n\n# 光度支持与几何参数梯度零更新诊断\n\n'+f'模型 `{MODEL}`，iteration16500，source `{pipe.source_path}`，处理原135 TRAIN中的{len(order)}帧，probe_only={probe_only}。完整命令：\n\n```bash\n'+shlex.join(original_command)+'\n```\n\nplan.json保存输入/source/环境范围；runtime_active.json为活跃GPU与运行库来源；logs/profile.log保存日志。per_frame保存原像素与逐点向量、范数、点积；'+aggregate_path.name+'保存每点跨视角均值/支持。\n\n无新MAE、无优化器更新。四类视觉/代表图片视频不适用，本诊断不渲染测试结果也不验收脚形。\n\n最终执行与无更新核验：**'+summary['status']+'**。机制解释待独立审查，不代表Goal成功。\n')
    if not all(invariants.values()): raise AssertionError('零更新或来源最终门禁失败。')
    print(f'光度支持{len(order)}TRAIN零更新诊断PASS：'+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check',action='store_true')
    parser.add_argument('--probe-only',action='store_true',help='固定0001/0071/0149的真实native零更新准入探针。')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--bootstrapped',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if socket.gethostname()!='ushas': parser.error('本固定环境只准ushas。')
    output = PROBE_OUTPUT if args.probe_only else OUTPUT
    if args.output is not None and args.output.resolve()!=output: parser.error('输出只准预注册25或smoke23，失败重试须独立登记。')
    if args.cpu_check: cpu_check(); return
    if any((output/p).exists() for p in ('plan.json','per_frame','summary.json')): parser.error('禁止覆盖已有诊断资产。')
    preflight()
    if not args.bootstrapped: dispatch(args); return
    run(output, args.probe_only)


if __name__=='__main__': main()
