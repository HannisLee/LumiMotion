"""TRAIN固定几何的颜色/共享EL梯度审计；不拟合、不改变几何FD结论。"""
import argparse
import math
import os
from pathlib import Path
import shlex
import socket
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts import calibrate_eldl_v3_raster as calibration

FRAMES = (1, 81, 149)
STEPS = (.01, .003, .001, .0003)
RULES = dict(relative_fd_tolerance=.01, consecutive_passing_scales=2,
             derivative_signal_floor=1e-6, forward_tolerance=2e-6,
             gradient_relative_tolerance=1e-4, restoration_loss_tolerance=1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'fit', 'model', 'source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--iteration', type=int, default=16000)
    a = p.parse_args()
    for key in ('cache', 'fit', 'model', 'source', 'output'):
        setattr(a, key, getattr(a, key).resolve())
    if socket.gethostname() != 'ushas' or Path(sys.executable).resolve() != v.PYTHON.resolve():
        p.error('仅ushas/lumimotion-ushas。')
    if not a.output.is_relative_to(ROOT/'output/smoke_test') or os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        p.error('仅协调后的GPU1，输出smoke_test新目录。')
    a.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), Path(calibration.__file__), Path(v.__file__),
                    ROOT/'scene/eldl_v3_photometric.py', ROOT/'scene/static_dl_photometric.py',
                    ROOT/'scene/stage2_photometric.py', ROOT/'gaussian_renderer/__init__.py',
                    ROOT/'scene/gaussian_model.py', ROOT/'utils/rgba16.py', ROOT/'utils/sh_utils.py']
    hashes = {str(x):v.sha(x) for x in source_paths}
    command = shlex.join([sys.executable, str(ROOT/'LIHAN/scripts/bootstrap_1011.py'), str(ROOT),
                         'scripts.audit_eldl_v3_raster_color', *sys.argv[1:]])
    v.write(a.output/'plan.json', dict(frames=FRAMES, steps=STEPS, rules=RULES, source_sha256=hashes,
        gpu=os.environ['CUDA_VISIBLE_DEVICES'], geometry_exact_fd='FAILED_UNCHANGED_NOT_TESTED_HERE',
        internal_validation_passed=False, command=command,
        paths={key:str(getattr(a,key)) for key in ('cache','fit','model','source','output')}))
    (a.output/'README.md').write_text('# 冻结几何raster颜色梯度审计\n\n2026-10-19（Asia/Tokyo启动日）。仅fit1/81/149，不拟合、不读外部GT/TEST、不导出。calibration.load装载135个TRAIN相机容器，但只有上述三张fit用于渲染/损失/梯度，不统计held。\n\n```bash\nCUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 '+command+'\n```\n\nsource/model/iteration与路径见plan.json，输出summary.json、source/input SHA；外层启动日志同目录run.log。冻结完整几何与光向，中心bias仅采用输入原值，无bias扫描。FD检查override RGB、线性rho及log EL；步长/门槛在首次模型读取前固定。实际pc._albedo_dc映射用于生产前向与梯度配对。\n\n本任务不是四类模型或灯光精度验收，不生成完整视频；四类目检不适用。既有几何exact FD仍FAILED，不在本次范围。最终PENDING，internal_validation_passed=false。\n')
    try:
        run(a, hashes)
    except Exception as exc:
        v.write(a.output/'FAILED.json', dict(status='FAILED_AUDIT_EXECUTION', error=str(exc),
            retained=True, internal_validation_passed=False))
        with (a.output/'README.md').open('a') as f:f.write('\n最终FAILED（工程执行异常），全部保留。\n')
        raise


def run(a, hashes):
    from scene.eldl_v3_photometric import render_photometric
    from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
    from scene.static_dl_photometric import StaticVisibility
    from utils.general_utils import build_rotation
    from utils.sh_utils import RGB2SH
    import diff_surfel_rasterization as backend
    if getattr(backend, 'provenance', {}).get('method') != 'repository_lumimotion_cuda_rasterizer':
        raise ValueError('必须原生bootstrap。')
    pc, cameras, pipe, data, state, loaded_hashes = calibration.load(a)
    hashes.update(loaded_hashes)
    geometric_names = ('_xyz', '_scaling', '_rotation', '_opacity')
    geometry = {key:getattr(pc,key).detach().clone() for key in geometric_names}
    original_albedo = pc._albedo_dc
    lower, upper = [float(srgb_to_linear(torch.tensor(x))) for x in (.03,.97)]
    # 工程代表值：与图像、held和真实材质无关；远离生产clamp边界。
    ids = torch.arange(len(pc.get_xyz), device='cuda', dtype=torch.float32)
    rho0 = torch.tensor([.25,.35,.45],device='cuda')[None] + .02*torch.sin(ids[:,None]*.017)
    log_environment0 = torch.tensor([math.pi*.3,math.pi*.5,math.pi*.7],device='cuda').log()
    if not bool(((rho0 > lower+.02) & (rho0 < upper-.02)).all()):
        raise ValueError('代表rho没有严格处于生产线性范围内。')
    normal = build_rotation(pc.get_rotation)[:,:,2].detach()
    visibility = StaticVisibility(bias=float(data['shadow_bias']), update_interval=1000000)
    records = []
    checks = []
    try:
        for frame in FRAMES:
            i = v.TRAIN.index(frame)
            if frame not in v.FIT:raise ValueError('不得使用held代表帧。')
            camera = cameras[i]
            light = state['directions'][i:i+1].cuda().detach()
            vis = visibility(pc, light, 0).detach()
            facing = torch.where((normal*(camera.camera_center-pc.get_xyz)).sum(-1,keepdim=True)>=0,1.,-1.).detach()
            q = (((normal*facing)@light.T).clamp_min(0)*vis).detach()
            rho = rho0.detach().clone().requires_grad_()
            log_environment = log_environment0.detach().clone().requires_grad_()
            def local(r, e):
                return calibration.render(camera,pc,pipe,r*(e.exp()/math.pi+q))['render']
            def production(r, e):
                pc._albedo_dc = RGB2SH(linear_to_srgb(r)).unsqueeze(1)
                lights = SimpleNamespace(ambient_rgb=e.exp(), for_camera=lambda unused:(light,light.new_tensor([math.pi])))
                return render_photometric(camera,pc,pipe,torch.zeros(3,device='cuda'),lights,lambda *unused:vis,0)
            lp = local(rho,log_environment)
            pp = production(rho,log_environment)
            linear_error = float((lp-pp['render_linear']).abs().max())
            srgb_error = float((linear_to_srgb(lp.clamp_min(0))-pp['render']).abs().max())
            loss = calibration.objective(lp,camera)
            prod_loss = calibration.objective(pp['render_linear'],camera)
            grads = torch.autograd.grad(loss,(rho,log_environment),retain_graph=False)
            production_grads = torch.autograd.grad(prod_loss,(rho,log_environment),retain_graph=False)
            gradient_errors = [float((x-y).norm()/torch.maximum(x.norm(),y.norm()).clamp_min(1e-12)) for x,y in zip(grads,production_grads)]
            base_colors = (rho0*(log_environment0.exp()/math.pi+q)).detach().requires_grad_()
            color_loss = calibration.objective(calibration.render(camera,pc,pipe,base_colors)['render'],camera)
            color_grad, = torch.autograd.grad(color_loss,(base_colors,))
            tests=[]
            def finite_difference(name, gradient, direction, fn):
                automatic=float((gradient*direction).sum())
                signal=abs(automatic)>=RULES['derivative_signal_floor']
                results=[]
                for step in STEPS:
                    with torch.no_grad():plus=float(fn(step));minus=float(fn(-step))
                    fd=(plus-minus)/(2*step)
                    relative=abs(fd-automatic)/max(abs(fd),abs(automatic),RULES['derivative_signal_floor'])
                    ok=signal and relative<=RULES['relative_fd_tolerance'] and fd*automatic>0
                    results.append(dict(step=step,automatic=automatic,finite_difference=fd,relative_error=relative,passed=ok))
                consecutive=any(results[k]['passed'] and results[k+1]['passed'] for k in range(len(results)-1))
                tests.append(dict(name=name,informative=signal,passed=consecutive,scales=results))
            for channel in range(3):
                direction=torch.zeros_like(rho0);direction[:,channel]=1
                finite_difference(f'override_color_channel_{channel}',color_grad,direction,
                    lambda h,d=direction:calibration.objective(calibration.render(camera,pc,pipe,base_colors.detach()+h*d)['render'],camera))
                finite_difference(f'linear_rho_channel_{channel}',grads[0],direction,
                    lambda h,d=direction:calibration.objective(local(rho0+h*d,log_environment0),camera))
                edirection=torch.zeros_like(log_environment0);edirection[channel]=1
                finite_difference(f'log_EL_channel_{channel}',grads[1],edirection,
                    lambda h,d=edirection:calibration.objective(local(rho0,log_environment0+h*d),camera))
            # 空间符号方向覆盖不同Gaussian索引，避免只检查全局同色扰动。
            pattern=torch.where((ids.long()%7)<3,1.,-1.)[:,None].expand_as(rho0)
            finite_difference('linear_rho_spatial_pattern',grads[0],pattern,
                lambda h:calibration.objective(local(rho0+h*pattern,log_environment0),camera))
            optimizer=torch.optim.Adam([rho,log_environment],lr=0.)
            optimizer.zero_grad(set_to_none=True)
            calibration.objective(local(rho,log_environment),camera).backward()
            optimizer.step()
            restored=bool(torch.equal(rho.detach(),rho0) and torch.equal(log_environment.detach(),log_environment0))
            with torch.no_grad():restored_loss=float(calibration.objective(local(rho,log_environment),camera))
            loss_error=abs(restored_loss-float(loss.detach()))
            geometry_intact=all(torch.equal(getattr(pc,key),value) and getattr(pc,key).grad is None for key,value in geometry.items())
            row=dict(frame=frame,split='fit',loss=float(loss.detach()),production_linear_max_error=linear_error,
                production_srgb_max_error=srgb_error,production_gradient_relative_errors=gradient_errors,
                full_foreground_pixels=int((camera.gt_alpha_mask>.5).sum()),
                saturated_foreground_channels=int((camera.observed_saturation_mask.cuda() & (camera.gt_alpha_mask.cuda()>.5)).sum()),
                finite_differences=tests,zero_update_restored_exactly=restored,zero_update_loss_error=loss_error,
                geometry_frozen_and_grad_none=geometry_intact)
            row['passed']=bool(linear_error<=RULES['forward_tolerance'] and srgb_error<=RULES['forward_tolerance']
                and max(gradient_errors)<=RULES['gradient_relative_tolerance'] and restored and geometry_intact
                and loss_error<=RULES['restoration_loss_tolerance'] and all(t['passed'] for t in tests))
            records.append(row)
            v.write(a.output/f'fit_{frame:04d}.json',row)
            print(dict(frame=frame,passed=row['passed'],fd_checks=len(tests),max_forward_error=linear_error,
                max_gradient_error=max(gradient_errors)),flush=True)
        v.check_hashes(hashes)
        passed=all(row['passed'] for row in records)
        v.write(a.output/'summary.json',dict(status='PASS_COLOR_EL_ENGINEERING_ONLY' if passed else 'FAILED_COLOR_EL_AUDIT',
            internal_validation_passed=False,gt_read=False,external_test_read=False,geometry_exact_fd='FAILED_UNCHANGED_NOT_TESTED_HERE',
            linear_rho_bounds=[lower,upper],direction_source='冻结fit_v4 selected3 TRAIN预测',geometry_iteration=a.iteration,
            frames=FRAMES,held_used_for_loss_or_selection=False,optimization_steps_with_nonzero_lr=0,
            rules=RULES,records=records,input_sha256=hashes,rasterizer_provenance=backend.provenance))
        with (a.output/'README.md').open('a') as f:f.write('\n最终'+('PASS（仅颜色/EL工程审计）' if passed else 'FAILED（颜色/EL工程门槛）')+'；三帧逐项结果见fit_*.json与summary.json。几何exact FD失败状态不变，无精度/灯资产PASS。\n')
    finally:
        pc._albedo_dc=original_albedo
        if not all(torch.equal(getattr(pc,key),value) for key,value in geometry.items()):
            raise RuntimeError('几何意外变化。')


if __name__=='__main__':
    main()
