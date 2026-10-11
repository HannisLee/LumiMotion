#!/usr/bin/env python3
"""仅原135训练帧的原生贡献、方向信息与遮挡梯度诊断；不建立优化器。"""
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def array_summary(value):
    import numpy as np
    value = np.asarray(value)
    finite = value[np.isfinite(value)]
    return {'count':int(value.size), 'finite_count':int(finite.size),
            'mean':float(finite.mean()) if finite.size else None,
            'quantiles_0_10_25_50_75_90_100':np.quantile(finite, [0,.1,.25,.5,.75,.9,1]).tolist() if finite.size else []}


@contextmanager
def capture_raw_alpha(tracer):
    """只旁观真实native返回，退出后恢复实例方法；不替换trace数学。"""
    old = tracer.trace
    had_local = 'trace' in vars(tracer)
    local = vars(tracer).get('trace')
    captured = []
    def observe(*args, **kwargs):
        result = old(*args, **kwargs)
        captured.append(result[-1].detach())
        return result
    tracer.trace = observe
    try:
        yield captured
    finally:
        if had_local:
            tracer.trace = local
        else:
            del tracer.trace


def verify_sources(model, marker, provenance):
    """绑定训练快照和实际调用源码，不借评测入口加载任何测试图像。"""
    command_path = model.parent/'commands/train.json'
    if not command_path.is_file():
        command_path = model.parent/'commands.json'
    command = json.loads(command_path.read_text())
    snapshot = Path(command['snapshot'])
    files = {str(command_path):digest(command_path), str(Path(__file__)):digest(__file__)}
    for relative in ('scene/static_dl_photometric.py', 'scene/static_dl_shadow_split.py',
                     'scene/static_dl_censored.py', 'scene/stage2_photometric.py',
                     'scene/gaussian_model.py', 'scene/cameras.py', 'scene/dataset_readers.py',
                     'utils/camera_utils.py', 'gaussian_renderer/__init__.py'):
        current, saved = ROOT/relative, snapshot/relative
        if digest(current) != digest(saved):
            raise AssertionError('当前诊断依赖与训练快照不同：'+relative)
        files[str(current)], files[str(saved)] = digest(current), digest(saved)
    trained = marker['rasterizer_provenance']
    for key in ('method', 'source_sha256', 'extension_sha256', 'source_files'):
        if provenance[key] != trained[key]:
            raise AssertionError('实际native来源与训练不一致：'+key)
    if digest(provenance['extension_path']) != trained['extension_sha256']:
        raise AssertionError('实际native动态库SHA不符。')
    files[provenance['extension_path']] = digest(provenance['extension_path'])
    return files


def run(args, output, log):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from scene.static_dl_photometric import DirectionOnlyLights
    from scene.static_dl_shadow_split import render_split_photometric, visibility_xyz_only, require_native_backend
    from scene.static_dl_censored import loss_for_mode
    from scene.stage2_photometric import frame_key, srgb_to_linear
    from utils.camera_utils import cameraList_from_camInfos
    from utils.general_utils import build_rotation

    if (socket.gethostname() != 'ushas' or Path(sys.executable).resolve() !=
            Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve()):
        raise RuntimeError('本次诊断限定ushas及lumimotion-ushas。')
    model = args.model_path.resolve()
    static_path = model/'static_dl_training.json'
    marker = json.loads(static_path.read_text())
    config = marker['args'].copy()
    if config['static_dl_mode'] != 'lambertian_visibility' or config.get('compute_cov3D_python'):
        raise ValueError('仅支持已保存的原生方向Lambertian+visibility检查点。')
    provenance = require_native_backend()
    files = verify_sources(model, marker, provenance)
    from surfel_tracer import _C as trace_extension
    import surfel_tracer.raytracer as trace_python
    for module in (trace_extension,trace_python):
        files[str(Path(module.__file__).resolve())] = digest(module.__file__)
    runtime_record = model.parent/f'evaluation_{args.iteration}/native_binding_audit.json'
    runtime_training_match = None
    if runtime_record.is_file():
        # 只读取已捕获的进程来源元数据，不读取该目录的GT、图像或MAE。
        runtime = json.loads(runtime_record.read_text())['training_actual_runtime_modules']
        actual_hashes = {digest(provenance['extension_path']),digest(trace_extension.__file__)}
        runtime_training_match = actual_hashes == set(runtime.values())
        if not runtime_training_match:
            raise AssertionError('当前raster/tracer动态库与原训练实际映射SHA不同。')
        files[str(runtime_record)] = digest(runtime_record)
    audit_path = next((model/name for name in ('depth_statistic_training.json', 'censored_training.json',
                         'shadow_split_training.json') if (model/name).is_file()), None)
    if audit_path is None:
        raise ValueError('缺少实际训练损失参数marker，拒绝猜测权重。')
    training = json.loads(audit_path.read_text())
    weights = training['args']
    light_path = model/f'static_dl_lighting/iteration_{args.iteration}/lighting.json'
    ply_path = model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    deform_path = model/f'deform/iteration_{args.iteration}/deform.pth'
    lighting = json.loads(light_path.read_text())
    if lighting['iteration'] != args.iteration or lighting['mode'] != config['static_dl_mode']:
        raise AssertionError('lighting迭代或模式不一致。')
    loss_model = lighting['loss_model']
    if loss_model != weights['loss_model']:
        raise AssertionError('lighting与实际训练loss不符。')
    bias = config['static_dl_shadow_bias']
    if bias != weights['shadow_bias'] or bias <= 0:
        raise AssertionError('实际bias不一致或非正。')
    source = Path(config['source_path']).resolve()
    transforms = source/'transforms_train.json'
    records = json.loads(transforms.read_text())['frames']
    expected = {str(i).zfill(4) for i in range(1,151) if i % 10}
    keys = [frame_key(Path(r['file_path']).stem) for r in records]
    if len(keys) != 135 or set(keys) != expected:
        raise AssertionError('只允许原135训练帧，不能包含测试帧或重复帧。')
    state = torch.load(deform_path, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp','gaussian_rotation','gaussian_scaling'):
        for suffix in ('weight','bias'):
            if torch.count_nonzero(state[name+'.'+suffix]).item():
                raise AssertionError('非严格零形变头。')
    del state
    config.update(load2gpu_on_the_fly=True, data_device='cpu')
    pipe = Namespace(**config)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config.get('hyper_dim',1))
    pc.load_ply(str(ply_path))
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if lights.sha256 != lighting['directions_sha256'] or set(lights.directions) != expected:
        raise AssertionError('方向SHA或135训练帧不符。')
    if any(v.shape != (1,3) for v in lights.directions.values()):
        raise AssertionError('当前诊断严格限合并后的单个同向SUN方向。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    lights.requires_grad_(False)
    infos = readCamerasFromTransforms(str(source),'transforms_train.json',config['train_light_folder'],
                                     config['white_background'],no_bg=True)
    cameras = cameraList_from_camInfos(infos,1.,pipe)
    if len(cameras) != 135 or {frame_key(c.image_name_train_light) for c in cameras} != expected:
        raise AssertionError('实际加载相机不符。')
    for path in (static_path,audit_path,light_path,ply_path,deform_path,transforms,Path(config['static_dl_directions'])):
        files[str(path)] = digest(path)
    for info in infos:
        files[str(Path(info.image_path_train_light).resolve())] = digest(info.image_path_train_light)
    for name,value in vars(pc).items():
        if isinstance(value,torch.Tensor) and value.is_leaf:
            value.requires_grad_(name == '_xyz')
    parameters = {name:value.detach().clone() for name,value in vars(pc).items()
                  if isinstance(value,torch.Tensor) and value.is_leaf}
    light_before = lights.log_intensity.detach().clone()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
    count = len(pc.get_xyz)
    background = torch.tensor([1.,1.,1.] if config['white_background'] else [0.,0.,0.], device='cuda')
    saturation_alpha = 1-float(pc.gaussian_tracer.transmittance_min)
    normal = build_rotation(pc.get_rotation).detach()[:,:,2]
    albedo = srgb_to_linear(pc.get_albedo).detach()
    numpy = lambda x:x.detach().cpu().numpy()
    h_camera = np.zeros((count,3,3),dtype=np.float64)
    h_info = np.zeros_like(h_camera)
    totals = {name:np.zeros(count,dtype=np.float64) for name in
              ('camera_mass','foreground_mass','information_weight','information_weight_square',
               'visibility_mass','uncensored_mass','backlit_mass','bright_backlit_mass',
               'positive_occluded_mass','trace_saturated_mass','transition_mass','bias_half_delta_mass',
               'bias_double_delta_mass','receiver_abs_dloss_dv','occluder_xyz_norm')}
    counts = {name:np.zeros(count,dtype=np.int32) for name in
              ('observed_frames','nonzero_camera_frames','informative_frames','positive_face_frames',
               'negative_face_frames','tangent_frames','receiver_grad_frames','occluder_grad_frames')}
    per_frame = []
    (output/'frames').mkdir()
    contract = {'model':str(model),'iteration':args.iteration,'source':str(source),'frames':sorted(expected),
                'point_count':count,'parameter_updates':0,'loss_model':loss_model,'shadow_bias':bias,
                'loss_weights':{key:weights[key] for key in ('shadow_weight','highlight_weight','residual_threshold')},
                'native':provenance,'actual_training_dynamic_libraries_match':runtime_training_match,
                'input_and_code_sha256':files,'camera_contribution_threshold_pixels':.01,
                'foreground_original_alpha_threshold':.99,'gradient_nonzero_threshold':1e-12,
                'incidence_smoothstep_low_high':[.05,.25],'bright_point_mean_srgb_threshold':.1,
                'H_information_weight':'native_foreground_mass * uncensored_channel_fraction * smoothstep(mu) * V^2',
                'H_is_full_fisher_information':False,'bias_multipliers':[.5,1.,2.],
                'native_wrapper_saturation_alpha':saturation_alpha,'train_image_count':135,'test_image_count':0,
                'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'hostname':socket.gethostname(),
                'python':sys.executable,'torch':torch.__version__}
    write(output/'contract.json',contract)
    for index,camera in enumerate(cameras):
        tick = time.monotonic()
        camera.load2device()
        directions,intensity = lights.for_camera(camera)
        target = camera.original_image_train_light.cuda()
        alpha = camera.gt_alpha_mask.cuda()
        if config['white_background']:
            target = alpha*target+(1-alpha)*background[:,None,None]
        fg = (alpha > .99).float()
        with capture_raw_alpha(pc.gaussian_tracer) as captured:
            package = render_split_photometric(camera,pc,pipe,background,directions,intensity,bias,route='occluder')
        if len(captured) != 1:
            raise AssertionError('期望一次真实native trace。')
        raw_alpha = captured[0].reshape(-1)
        visibility = package['light_visibility'].reshape(-1)
        wrapped = torch.where(raw_alpha < saturation_alpha,1-raw_alpha,torch.zeros_like(raw_alpha)).clamp(0,1)
        if not torch.equal(wrapped,visibility):
            raise AssertionError('旁观raw alpha与实际包装V不一致。')
        if index == 0:
            with torch.no_grad():
                control = render_split_photometric(camera,pc,pipe,background,directions,intensity,bias,route='control',rebuild=False)
            forward_difference = float((control['render']-package['render']).abs().max())
            if forward_difference != 0:
                raise AssertionError('occluder诊断改变RGB前向。')
            del control
        colors,linear = package['override_colors'],package['render_linear']
        def color_jacobian(pixel_weights):
            return torch.autograd.grad(linear,colors,grad_outputs=pixel_weights.contiguous(),retain_graph=True)[0].detach()
        mass_rgb = color_jacobian(torch.ones_like(linear))
        fg_mass_rgb = color_jacobian(fg.expand_as(linear))
        unsat_mass_rgb = color_jacobian(fg*(target < 1).float())
        observed_rgb_sum = color_jacobian(fg*target)
        fg_mass = fg_mass_rgb.mean(-1)
        # 原生贡献的两条独立恒等约束，容差预设，不按结果放宽。
        alpha_sum = (package['rend_alpha'].detach()*fg).double().sum()
        mass_error = float((fg_mass.double().sum()-alpha_sum).abs())
        mass_limit = 1e-3+1e-4*float(alpha_sum.abs())
        predicted_integral = (colors.detach().double()*mass_rgb.double()).sum(0)
        predicted_integral += srgb_to_linear(background).double()*(1-package['rend_alpha'].detach().double()).sum()
        actual_integral = linear.detach().double().sum((1,2))
        integral_error = (predicted_integral-actual_integral).abs()
        channel_difference = (mass_rgb-mass_rgb[:,:1]).abs()
        channel_error = float(channel_difference.max())
        if (mass_error > mass_limit or not torch.all(integral_error <= 1e-3+1e-4*actual_integral.abs())
                or float(mass_rgb.min()) < -1e-7
                or not torch.all(channel_difference <= 1e-6+1e-5*mass_rgb[:,:1].abs())):
            write(output/'failed_native_contribution_check.json',{'frame':camera.image_name_train_light,
                 'mass_error':mass_error,'mass_limit':mass_limit,'integral_error':numpy(integral_error).tolist(),
                 'integral':numpy(actual_integral).tolist(),'channel_error':channel_error})
            raise AssertionError('原生颜色Jacobian贡献的守恒／通道一致性检查失败。')
        observed_rgb = observed_rgb_sum/fg_mass_rgb.clamp_min(1e-20)
        uncensored = (unsat_mass_rgb/fg_mass_rgb.clamp_min(1e-20)).mean(-1).clamp(0,1)
        view = F.normalize(camera.camera_center-pc.get_xyz.detach(),dim=-1)
        normal_view = (normal*view).sum(-1)
        facing = torch.where(normal_view >= 0,1.,-1.)
        mu = ((normal*facing[:,None])@directions.T).reshape(-1)
        photo,_ = loss_for_mode(loss_model)(package['render'],target,alpha,
                        weights['shadow_weight'],weights['highlight_weight'],weights['residual_threshold'])
        color_gradient,xyz_gradient = torch.autograd.grad(photo,(colors,pc._xyz))
        receiver_gradient = (color_gradient.detach()*albedo*mu.clamp_min(0)[:,None]*intensity.detach()/math.pi).sum(-1)
        xyz_norm = xyz_gradient.detach().norm(dim=-1)
        with torch.no_grad():
            half = visibility_xyz_only(pc,directions,bias*.5).reshape(-1)
            double = visibility_xyz_only(pc,directions,bias*2).reshape(-1)
        values = {'camera_mass':mass_rgb.mean(-1),'foreground_mass':fg_mass,'uncensored_fraction':uncensored,
                  'observed_rgb':observed_rgb,'mu':mu,'facing':facing,'normal_view_dot':normal_view,
                  'visibility':visibility,'raw_trace_alpha':raw_alpha,'visibility_bias_half':half,
                  'visibility_bias_double':double,'receiver_dloss_dv':receiver_gradient,
                  'occluder_dloss_dxyz':xyz_gradient.detach(),'occluder_xyz_norm':xyz_norm}
        if not torch.isfinite(photo) or any(not torch.isfinite(value).all() for value in values.values()):
            raise FloatingPointError('诊断前向或导数非有限。')
        arrays = {key:numpy(value) for key,value in values.items()}
        del values
        m = arrays['foreground_mass'].astype(np.float64)
        mu_a,v,f = arrays['mu'],arrays['visibility'],arrays['uncensored_fraction']
        x = np.clip((mu_a-.05)/.20,0,1)
        smooth = x*x*(3-2*x)
        a = m*f*smooth*v.astype(np.float64)**2
        observed = m > .01
        backlit = mu_a <= 0
        bright_backlit = backlit & (arrays['observed_rgb'].max(-1) > .1)
        positive_occluded = (mu_a > 0) & (v < .1)
        saturated = arrays['raw_trace_alpha'] >= saturation_alpha
        transition = (mu_a > .05) & (v > 0) & (v < 1)
        direction_np = numpy(directions)[0].astype(np.float64)
        transported = arrays['facing'][:,None]*direction_np[None]
        outer = transported[:,:,None]*transported[:,None,:]
        h_camera += m[:,None,None]*outer
        h_info += a[:,None,None]*outer
        increments = {'camera_mass':arrays['camera_mass'],'foreground_mass':m,'information_weight':a,
                      'information_weight_square':a*a,'visibility_mass':m*v,'uncensored_mass':m*f,
                      'backlit_mass':m*backlit,'bright_backlit_mass':m*bright_backlit,
                      'positive_occluded_mass':m*positive_occluded,'trace_saturated_mass':m*saturated,
                      'transition_mass':m*transition,'bias_half_delta_mass':m*np.abs(arrays['visibility_bias_half']-v),
                      'bias_double_delta_mass':m*np.abs(arrays['visibility_bias_double']-v),
                      'receiver_abs_dloss_dv':np.abs(arrays['receiver_dloss_dv']),
                      'occluder_xyz_norm':arrays['occluder_xyz_norm']}
        for name,value in increments.items():
            totals[name] += value
        for name,value in {'observed_frames':observed,'nonzero_camera_frames':m>1e-12,
             'informative_frames':a>.01,'positive_face_frames':observed&(arrays['facing']>0),
             'negative_face_frames':observed&(arrays['facing']<0),'tangent_frames':observed&(np.abs(arrays['normal_view_dot'])<.05),
             'receiver_grad_frames':np.abs(arrays['receiver_dloss_dv'])>1e-12,
             'occluder_grad_frames':arrays['occluder_xyz_norm']>1e-12}.items():
            counts[name] += value
        key = frame_key(camera.image_name_train_light)
        np.savez_compressed(output/'frames'/f'{key}.npz',**arrays,information_weight=a.astype(np.float32))
        denominator = max(float(m.sum()),1e-20)
        record = {'frame':key,'photo':float(photo.detach()),'seconds':time.monotonic()-tick,
                  'foreground_mass':float(m.sum()),'observed_points':int(observed.sum()),
                  'uncensored_mass_fraction':float((m*f).sum()/denominator),
                  'backlit_mass_fraction':float((m*backlit).sum()/denominator),
                  'bright_backlit_mass_fraction':float((m*bright_backlit).sum()/denominator),
                  'positive_occluded_mass_fraction':float((m*positive_occluded).sum()/denominator),
                  'trace_saturated_mass_fraction':float((m*saturated).sum()/denominator),
                  'transition_mass_fraction':float((m*transition).sum()/denominator),
                  'receiver_grad_nonzero_points':int((np.abs(arrays['receiver_dloss_dv'])>1e-12).sum()),
                  'occluder_xyz_grad_nonzero_points':int((arrays['occluder_xyz_norm']>1e-12).sum()),
                  'receiver_grad_mass_fraction':float(m[np.abs(arrays['receiver_dloss_dv'])>1e-12].sum()/denominator),
                  'occluder_xyz_gradient_norm':float(np.linalg.norm(arrays['occluder_dloss_dxyz'])),
                  'mass_conservation_error':mass_error,'linear_integral_max_error':float(integral_error.max()),
                  'camera_channel_error':channel_error}
        per_frame.append(record)
        with (output/'per_frame.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        log(f'{index+1}/135 帧{key} photo={record["photo"]:.7f} '+
            f'前景有效点={record["observed_points"]} 遮挡XYZ非零点={record["occluder_xyz_grad_nonzero_points"]}')
        camera.load2device('cpu')
        del package,colors,linear,color_gradient,xyz_gradient,photo

    # 只读检查不仅检查checkpoint磁盘，也检查加载后全部模型叶子张量。
    unchanged = all(torch.equal(getattr(pc,name).detach(),value) for name,value in parameters.items())
    unchanged = unchanged and torch.equal(lights.log_intensity.detach(),light_before)
    numpy_state = np.random.get_state()
    rng_unchanged = (random.getstate()==rng[0] and numpy_state[0]==rng[1][0]
        and np.array_equal(numpy_state[1],rng[1][1]) and numpy_state[2:]==rng[1][2:]
        and torch.equal(torch.get_rng_state(),rng[2]) and torch.equal(torch.cuda.get_rng_state(),rng[3]))
    files_unchanged = all(digest(path)==sha for path,sha in files.items())
    if not unchanged or not rng_unchanged or not files_unchanged:
        write(output/'read_only_failure.json',{'parameters':unchanged,'RNG':rng_unchanged,'files':files_unchanged})
        raise AssertionError('只读、RNG或输入SHA检查失败。')
    eig_camera = np.maximum(np.linalg.eigvalsh(h_camera),0)
    eig_info = np.maximum(np.linalg.eigvalsh(h_info),0)
    trace = eig_info.sum(-1)
    ratio = eig_info[:,0]/np.maximum(trace,1e-30)
    condition = np.full(count,np.inf)
    full_rank = (trace>0)&(eig_info[:,0] > 1e-8*trace)
    condition[full_rank] = eig_info[full_rank,-1]/eig_info[full_rank,0]
    effective_frames = totals['information_weight']**2/np.maximum(totals['information_weight_square'],1e-30)
    point_arrays = {**totals,**counts,'H_camera':h_camera,'H_information':h_info,
                    'H_camera_eigenvalues':eig_camera,'H_information_eigenvalues':eig_info,
                    'lambda_min_over_trace':ratio,'condition_fullrank_only_else_inf':condition,
                    'effective_information_frames':effective_frames,'xyz':numpy(pc.get_xyz),'canonical_normal':numpy(normal)}
    np.savez_compressed(output/'points.npz',**point_arrays)
    mass_total = float(totals['foreground_mass'].sum())
    ever_observed = counts['observed_frames']>0
    mixed_faces = (counts['positive_face_frames']>0)&(counts['negative_face_frames']>0)
    summary = {'status':'PASS','scope':'135训练帧零参数更新诊断；不是Goal阈值验收或训练权重有效性证明',
               'frame_count':len(per_frame),'point_count':count,'parameter_updates':0,
               'parameters_exactly_unchanged':unchanged,'RNG_exactly_unchanged':rng_unchanged,
               'inputs_and_sources_sha_unchanged':files_unchanged,'first_frame_control_occluder_forward_max_difference':forward_difference,
               'foreground_contribution_mass':mass_total,'ever_observed_points':int(ever_observed.sum()),
               'mixed_camera_facing_points':int(mixed_faces.sum()),
               'mass_weighted':{key:float(totals[key].sum()/max(mass_total,1e-20)) for key in
                  ('visibility_mass','uncensored_mass','backlit_mass','bright_backlit_mass',
                   'positive_occluded_mass','trace_saturated_mass','transition_mass','bias_half_delta_mass','bias_double_delta_mass')},
               'observed_point_statistics':{key:array_summary(value[ever_observed]) for key,value in
                  [('lambda_min_over_trace',ratio),('condition_fullrank_only',condition),
                   ('effective_information_frames',effective_frames),*counts.items()]},
               'observed_low_direction_rank_fraction':float((~full_rank)[ever_observed].mean()),
               'occluder_xyz_gradient_seen_points':int((counts['occluder_grad_frames']>0).sum()),
               'receiver_dloss_dv_seen_points':int((counts['receiver_grad_frames']>0).sum()),
               'all_135_per_frame_native_contribution_checks':True,
               'max_mass_conservation_error':max(r['mass_conservation_error'] for r in per_frame),
               'max_linear_integral_error':max(r['linear_integral_max_error'] for r in per_frame)}
    write(output/'summary.json',summary)
    mappings = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if '.so' in line and any(word in line.lower() for word in ('raster','tracer','gaussian'))})
    write(output/'actual_shared_libraries.json',{p:digest(p) for p in mappings if Path(p).is_file()})
    log('全部135帧完成；参数、RNG、输入SHA未变；诊断PASS。')
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--model_path',type=Path,required=True)
    parser.add_argument('--iteration',type=int,required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT/'output') or output.exists() or args.iteration <= 0:
        parser.error('必须选择output下全新目录和正迭代；拒绝覆盖历史证据。')
    output.mkdir(parents=True)
    def log(message):
        line = time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+' '+message
        print(line,flush=True)
        with (output/'run.log').open('a') as stream:
            stream.write(line+'\n')
    command = [sys.executable,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),
               'LIHAN.scripts.goal_direction_shadow_diagnostic',*sys.argv[1:]]
    write(output/'command.json',{'argv':command,'shell':shlex.join(command),
                                'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')})
    status = 'FAILED'
    try:
        result = run(args,output,log)
        status = result['status']
    except Exception as exc:
        write(output/'failure.json',{'status':'FAILED','error':str(exc),'traceback':traceback.format_exc(),
                                    'parameter_updates':0,'partial_outputs_preserved':True})
        log('诊断FAILED：'+str(exc))
        raise
    finally:
        (output/'README.md').write_text('# 原135训练帧方向与遮挡诊断\n\n'+
            f'最终结论：**{status}**。这是零参数更新数值诊断，不是完整测试MAE验收。\n\n'+
            f'模型：`{args.model_path.resolve()}`；iteration：{args.iteration}。source及全部输入SHA见`contract.json`。\n\n'+
            '完整执行命令（CUDA_VISIBLE_DEVICES见command.json）：\n\n```bash\n'+shlex.join(command)+'\n```\n\n'+
            '输出为本目录；日志`run.log`，逐帧统计`per_frame.jsonl`，逐点逐帧资产`frames/*.npz`，'+
            '跨135帧汇总`points.npz`、`summary.json`，失败细节`failure.json`。不保存或修改模型。\n\n'+
            '四类可视化及目检：本工具只计算训练帧数值，不生成RGB/Normal/Alpha/separation视频，均不适用；'+
            '原15帧评分与150帧可视化仍保留于模型所属实验evaluation目录，没有修改评价。\n\n'+
            '限制：H只是方向覆盖近似，不是完整Fisher；预测背光不是GT标签；原生相机合成贡献不是物理对应证明；'+
            '相机翻面可能混合两侧，近切线需谨慎。满码通道不进入H等式权重，但原censored loss下界违反仍参与梯度。'+
            '接收点dL/dV与遮挡点XYZ梯度是不同角色，不能逐ID相乘冒充同一覆盖。'+
            '原trace alpha>=.97包装截断、native透射终止/BVH离散命中及0.03 bias限制不被本诊断消除；'+
            '半/双bias差只衡量敏感性，不能判断哪一个为真。\n')


if __name__ == '__main__':
    main()
