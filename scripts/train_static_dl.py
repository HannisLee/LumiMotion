"""静态方向光实验：保留原 Stage1 外观学习，严格关闭时变几何。

检查点继续使用原生 MLP 格式。三个几何输出头的权重和偏置均为零且冻结，
因此原版加载器在任意时间读取检查点时也只能得到零形变。
"""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from random import randint
import sys

import torch
import torch.nn.functional as F

from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts.train_stage1 import Trainer
from utils.general_utils import safe_state
from utils.sh_utils import RGB2SH


def freeze_geometry_heads(network):
    """只冻结时变几何输出；共享编码和时变颜色仍参与原生优化。"""
    names = ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling')
    for name in names:
        layer = getattr(network, name)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
        layer.requires_grad_(False)
    return names


class StaticDLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        backend = sys.modules.get('diff_surfel_rasterization')
        provenance = getattr(backend, 'provenance', None)
        if (not isinstance(provenance, dict) or
                provenance.get('method') != 'repository_lumimotion_cuda_rasterizer' or
                not provenance.get('source_sha256') or
                not provenance.get('extension_sha256')):
            raise RuntimeError('静态DL训练必须通过bootstrap_1011.py绑定仓库原生CUDA光栅；禁止隐式pip实现。')
        super().__init__(*args, **kwargs)
        if self.deform.name != 'mlp':
            raise ValueError('静态DL入口使用原生mlp检查点，必须保持deform_type=mlp。')
        names = freeze_geometry_heads(self.deform.deform)
        # 已建优化器保留参数组结构；requires_grad=False确保无更新，便于原版加载。
        manifest = {
            'method': '静态零几何输出头',
            'static_dl_mode': self.args.static_dl_mode,
            'geometry_heads': list(names),
            'geometry_head_values': '严格全零且requires_grad=False',
            'canonical_geometry_optimized': True,
            'temporal_geometry_optimized': False,
            'time_dependent_color_optimized': True,
            'train_images': len(self.scene.getTrainCameras()),
            'test_images': len(self.scene.getTestCameras()),
            'rasterizer_provenance': provenance,
            'args': vars(self.args),
        }
        path = Path(self.args.model_path) / 'static_dl_training.json'
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        (path.parent / 'cfg_args').write_text(str(Namespace(**vars(self.args))))
        print('[静态DL] 几何输出头严格归零并冻结，保留原生SH与d_color。', flush=True)
        self.physical_initialized = False
        self.physical_lights = None
        if self.args.static_dl_mode != 'sh':
            from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
            if not self.args.static_dl_directions:
                raise ValueError('物理模式必须提供纯方向文件static_dl_directions。')
            self.physical_lights = DirectionOnlyLights(self.args.static_dl_directions)
            self.light_optimizer = torch.optim.Adam(self.physical_lights.parameters(), lr=self.args.static_dl_light_lr)
            self.visibility = StaticVisibility(self.args.static_dl_shadow_bias, self.args.static_dl_bvh_interval)
            # 训练只需要135个TRAIN方向。TEST方向由独立渲染流程读取，不用于训练。
            for camera in self.scene.getTrainCameras():
                self.physical_lights.for_camera(camera)
        self.gradient_audited = set()

    def _begin_physical(self):
        self.physical_initialized = True
        self.deform.deform.requires_grad_(False)
        with torch.no_grad():
            self.gaussians._albedo_dc.fill_(float(RGB2SH(.5)))
            self.gaussians._albedo_rest.zero_()
        self.gaussians.optimizer.state.pop(self.gaussians._albedo_dc, None)
        print('[静态DL] 切换线性Lambert着色；关闭时变颜色，光强仅由训练RGB优化。', flush=True)

    def _save_physical(self):
        self.scene.save(self.iteration)
        self.deform.save_weights(self.args.model_path, self.iteration)
        folder = Path(self.args.model_path) / 'static_dl_lighting' / f'iteration_{self.iteration}'
        folder.mkdir(parents=True, exist_ok=False)
        record = {
            'iteration': self.iteration,
            'mode': self.args.static_dl_mode,
            'directions_sha256': self.physical_lights.sha256,
            'intensity_from_train_rgb': self.physical_lights.log_intensity.exp().detach().cpu().tolist(),
            'visibility_detached': True,
            'temporal_geometry': False,
            'temporal_color': False,
        }
        (folder / 'lighting.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')

    def train_step(self):
        if self.args.static_dl_mode == 'sh' or self.iteration <= self.args.static_dl_physical_start:
            return super().train_step()
        from scene.static_dl_photometric import render_photometric, robust_photometric_loss
        if not self.physical_initialized:
            self._begin_physical()
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device()
        use_visibility = (self.args.static_dl_mode == 'lambertian_visibility' and
                          self.iteration >= self.args.static_dl_visibility_start)
        package = render_photometric(camera, self.gaussians, self.pipe, self.background,
                                     self.physical_lights, self.visibility if use_visibility else None,
                                     self.iteration)
        target = camera.original_image_train_light.cuda()
        gt_alpha = camera.gt_alpha_mask.cuda()
        if self.dataset.white_background:
            target = gt_alpha*target+(1-gt_alpha)*self.background[:, None, None]
        photo, diagnostics = robust_photometric_loss(
            package['render'], target, gt_alpha, self.args.static_dl_shadow_weight,
            self.args.static_dl_highlight_weight, self.args.static_dl_residual_threshold)
        alpha = package['rend_alpha'].clamp(1e-6, 1-1e-6)
        alpha_loss = F.binary_cross_entropy(alpha, gt_alpha)
        loss = photo+self.opt.lambda_alpha_loss*alpha_loss
        normal_loss = loss.new_zeros(())
        dist_loss = loss.new_zeros(())
        if self.iteration > self.opt.start_normal_reg:
            normal = F.normalize(package['rend_normal'], dim=0)
            depth_normal = F.normalize(package['surf_normal'], dim=0)
            mask = (gt_alpha > .99).float()
            mask = -F.max_pool2d(-mask[None], 5, stride=1, padding=2)[0]
            mask = mask*(alpha.detach() > .5)
            normal_loss = ((1-(normal*depth_normal).sum(0, keepdim=True))*mask).sum()/mask.sum().clamp_min(1)
            dist_loss = package['rend_dist'].mean()
            loss = loss+self.args.static_dl_normal_weight*normal_loss+self.opt.lambda_dist*dist_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'静态物理loss非有限，iteration={self.iteration}。')
        loss.backward()
        audit_key = 'visibility' if use_visibility else 'unoccluded'
        if audit_key not in self.gradient_audited:
            audit = {'iteration': self.iteration, 'mode': audit_key, 'gradients': {}}
            for name, param in (('xyz', self.gaussians._xyz), ('rotation', self.gaussians._rotation),
                                ('albedo_dc', self.gaussians._albedo_dc),
                                ('log_intensity', self.physical_lights.log_intensity)):
                grad = param.grad
                if grad is None or not torch.isfinite(grad).all() or grad.norm() <= 0:
                    raise FloatingPointError(f'静态物理{name}梯度缺失、非有限或全零。')
                audit['gradients'][name] = {'norm': float(grad.norm()), 'max_abs': float(grad.abs().max())}
            audit['visibility'] = {key: float(package[key]) for key in
                                   ('mean_visibility', 'visibility_below_point_one', 'visibility_above_point_nine')}
            audit['trace_rays'] = len(self.gaussians.get_xyz)*len(self.physical_lights.log_intensity) if use_visibility else 0
            folder = Path(self.args.model_path)/'static_dl_gradient_audit'
            folder.mkdir(exist_ok=True)
            (folder/f'{audit_key}.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
            self.gradient_audited.add(audit_key)
            print('[静态物理梯度] '+json.dumps(audit, ensure_ascii=False), flush=True)
        with torch.no_grad():
            visibility = package['visibility_filter']
            radii = package['radii']
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility] = torch.maximum(self.gaussians.max_radii2D[visibility], radii[visibility])
            self.ema_loss_for_log = .4*float(loss)+.6*self.ema_loss_for_log
            if self.iteration % 100 == 0:
                values = dict(iteration=self.iteration, photo=float(photo), alpha=float(alpha_loss),
                              normal=float(normal_loss), distortion=float(dist_loss),
                              visibility=float(package['mean_visibility']), points=len(self.gaussians.get_xyz),
                              light=package['light_intensity'].cpu().tolist(),
                              **{k: float(v) for k, v in diagnostics.items()})
                print('[静态物理] '+json.dumps(values, ensure_ascii=False), flush=True)
                with (Path(self.args.model_path)/'static_dl_metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(values, ensure_ascii=False)+'\n')
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({'Loss': f'{self.ema_loss_for_log:.7f}'})
                self.progress_bar.update(10)
            if self.iteration in self.saving_iterations:
                self._save_physical()
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(package['viewspace_points'], visibility)
                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    limit = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.min_opacity,
                                                     self.scene.cameras_extent, limit)
                    self.visibility.invalidate()
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()
                    self.visibility.invalidate()
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.light_optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                # 投影作用于参数本身，防止hard clamp之外的albedo永久丢失梯度。
                self.gaussians._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
                self.physical_lights.log_intensity.clamp_(-5, 5)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.light_optimizer.zero_grad(set_to_none=True)
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device('cpu')
        if self.iteration == self.opt.iterations:
            self.progress_bar.close()
        self.iteration += 1


def main():
    parser = ArgumentParser(description='静态DL：原生SH及颜色调制，禁止时变几何')
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
    args = parser.parse_args()
    if args.deform_type != 'mlp':
        parser.error('该入口仅支持mlp格式，以保持标准评测加载一致。')
    if not args.model_path:
        parser.error('必须显式提供独立的output实验模型路径。')
    if not args.model_path.endswith('_mlp'):
        args.model_path = args.model_path.rstrip('/') + '_mlp'
    if (Path(args.model_path) / 'point_cloud').exists():
        parser.error('只允许从头训练；已有point_cloud目录不能覆盖。')
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    args.test_iterations = sorted(set(args.test_iterations + [args.iterations]))
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = StaticDLTrainer(args=args, dataset=lp.extract(args), opt=op.extract(args),
                              pipe=pp.extract(args), testing_iterations=args.test_iterations,
                              saving_iterations=args.save_iterations)
    trainer.train(args.iterations)
    print('静态DL训练完成。')


if __name__ == '__main__':
    main()
