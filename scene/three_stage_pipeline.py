"""三阶段共同模型、保存重载和原生PBR残差接口。"""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scene import GaussianModel, DeformModel
from scene.static_dl_photometric import StaticVisibility
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
from scene.three_stage_lighting import EffectiveGaussians, SharedLighting, render_stage2, surface_components


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


class ThreeStageModel:
    def __init__(self, parent, iteration, binding, stage, config):
        self.stage, self.config, self.binding = stage, config, binding
        self.parent, self.parent_iteration = Path(parent), iteration
        self.pc = GaussianModel(3, no_binary_separation=False, fea_dim=1)
        self.pc.load_ply(str(self.parent/f'point_cloud/iteration_{iteration}/point_cloud.ply'))
        self.pc.active_sh_degree = 3
        self.deform = DeformModel(deform_type='mlp', is_blender=True, hyper_dim=1, pred_color=True)
        if not self.deform.load_weights(str(self.parent), iteration):
            raise FileNotFoundError('必须完整保留原生MLP，禁止缺失回退。')
        self.deform.deform.eval()
        for p in self.deform.deform.parameters():
            p.requires_grad_(False)
        self.lights = SharedLighting(json.loads(Path(binding['directions']).read_text()), config['direction_lr'])
        self.visibility = StaticVisibility(config['shadow_bias'], 1)
        self.pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False,
            depth_ratio=0., pbr_rasterizer_backend='lumimotion', render_mode='original_ir',
            pbr_light_mode='envmap', diffuse_sample_num=config['pbr_samples'], light_sample_num=0,
            light_t_min=config['shadow_bias'], wo_indirect=False, wo_indirect_relight=False,
            detach_indirect=True, wo_specular=False)
        self.stats = json.loads(Path(binding['stats']).read_text())
        self.env, self.env_initial = None, None
        self.residual_raw = torch.nn.Parameter(torch.tensor(math.log(.01/.24), device='cuda'))
        self.anchor_geometry = {key: getattr(self.pc, key).detach().clone() for key in ('_xyz', '_rotation', '_scaling', '_opacity')}
        marker = self.parent/'three_stage_training.json'
        self.parent_record = json.loads(marker.read_text()) if marker.exists() else None
        if self.parent_record:
            state = torch.load(self.parent/f'three_stage/iteration_{iteration}/state.pt', weights_only=False)
            self.lights.load_state_dict(state['lighting'])
        else:
            # 保留完整native外观在_albedo_dc_stage1与_albedo_rest；仅物理rho初始化转换。
            from utils.sh_utils import RGB2SH
            rho = linear_to_srgb(srgb_to_linear(self.pc.get_albedo)/1.5).clamp(.03, .97)
            with torch.no_grad():
                self.pc._albedo_dc.copy_(RGB2SH(rho[:, None]))
        if stage == 3:
            if not self.parent_record or self.parent_record['stage'] != 2:
                raise ValueError('Stage3必须来自本轮Stage2完整资产。')
            from scene.light import EnvLight
            mean_rho = float(srgb_to_linear(self.pc.get_albedo).detach().mean())
            init = min(self.stats['equal_frame_linear_mean'], self.stats['q_base_value'])/max(mean_rho, 1e-3)
            self.env_initial = max(.05, min(init, 4.))
            self.env = EnvLight(path=None, device='cuda', resolution=[16, 32], max_res=32,
                                activation='exp', init_value=self.env_initial)
            self.lights.raw_direction.requires_grad_(False)
            self.lights.sh.requires_grad_(False)
        self.completed = 0

    def effective(self, camera):
        # 与原生推理一致，固定网络输入canonical点；几何优化不改变网络查询。
        with torch.no_grad():
            values = self.deform.step(self.pc.get_xyz.detach(),
                camera.fid.unsqueeze(0).expand(len(self.pc.get_xyz), -1), iteration=35000,
                feature=self.pc.get_binary_feature(), camera_center=camera.camera_center, is_training=False)
        return EffectiveGaussians(self.pc, values)

    def render(self, camera, step=0, components=False, training=False):
        pc = self.effective(camera)
        if self.stage == 2:
            return render_stage2(camera, pc, self.pipe, self.lights, self.visibility, step,
                                 self.config['lambda_mix'], components)
        from gaussian_renderer.render_ir import render_ir
        # 先按当前有效几何构建全BVH，后续原生PBR半球追踪和残差共用。
        _, direct, vis = surface_components(camera, pc, self.lights, self.visibility, step)
        package = render_ir(camera, pc, self.pipe, pc.get_xyz.new_zeros(3), 0., 0., 0.,
            d_color=pc.values['d_color'], env_light=self.env, training=training,
            opt=SimpleNamespace(train_ray=training, trace_num_rays=self.config['pbr_pixels']*self.config['pbr_samples'],
                                pbr_sampling_alpha_threshold=0., pbr_sample_original_foreground=True))
        from gaussian_renderer import render
        amp = .25*self.residual_raw.sigmoid()
        residual = render(camera, pc, self.pipe, pc.get_xyz.new_zeros(3), 0., 0., 0.,
                          override_color=direct*amp*self.config['lambda_residual'])
        base = package['render_linear']
        contribution = residual['render']
        if training:
            contribution = contribution*package['mask'][None]
        linear = base+contribution
        package.update(base_linear=base, residual_linear=contribution, render_linear=linear,
                       render=linear_to_srgb(linear.clamp_min(0)), visibility_mean=vis.mean().detach(),
                       residual_amplitude=amp, component_add_max_error=(linear-base-contribution).abs().max())
        if training:
            package['loss_support'] = package['mask'][None] | (package['rend_alpha'] <= 0)
        if not torch.equal(package['rend_alpha'], residual['rend_alpha']):
            raise AssertionError('原生PBR和残差Alpha不一致。')
        return package

    def checkpoint(self, folder, step, optimizer=None):
        folder = Path(folder)
        ply = folder/f'point_cloud/iteration_{step}/point_cloud.ply'
        if ply.exists():
            raise FileExistsError('禁止覆盖已有checkpoint。')
        self.pc.save_ply(str(ply))
        self.deform.save_weights(str(folder), step)
        state_dir = folder/f'three_stage/iteration_{step}'
        state_dir.mkdir(parents=True, exist_ok=False)
        torch.save(dict(format_version=1, completed_updates=step, lighting=self.lights.state_dict(),
            stage=self.stage, config=self.config, stats=self.stats, binding=self.binding,
            env=None if self.env is None else self.env.state_dict(), env_initial=self.env_initial,
            residual_raw=self.residual_raw.detach(), optimizer=None if optimizer is None else optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
            python_rng=__import__('random').getstate(), numpy_rng=__import__('numpy').random.get_state()), state_dir/'state.pt')
        write(state_dir/'receipt.json', dict(stage=self.stage, completed_updates=step,
            cumulative_actual_updates=34999+step+(self.parent_iteration if self.stage == 3 else 0),
            parent=str(self.parent), parent_iteration=self.parent_iteration, config=self.config,
            geometry_sha256={k: hashlib.sha256(getattr(self.pc, k).detach().cpu().numpy().tobytes()).hexdigest() for k in self.anchor_geometry},
            checkpoint_sha256=sha(state_dir/'state.pt'), ply_sha256=sha(ply),
            deform_sha256=sha(folder/f'deform/iteration_{step}/deform.pth'),
            direction_max_deviation_deg=float(torch.rad2deg(torch.acos((self.lights.directions()*self.lights.initial).sum(-1).clamp(-1, 1))).max()),
            gt_read=False, test_read=False, semantics='真实更新后保存；相对步数，父Adam未导入'))

    def reload_endpoint(self, folder, step):
        """原地换成磁盘参数，以相同相机验证保存/重载；不读取父路径猜测照明。"""
        folder = Path(folder)
        state = torch.load(folder/f'three_stage/iteration_{step}/state.pt', weights_only=False)
        self.pc.load_ply(str(folder/f'point_cloud/iteration_{step}/point_cloud.ply'))
        if not self.deform.load_weights(str(folder), step):
            raise FileNotFoundError('重载缺少MLP。')
        self.lights.load_state_dict(state['lighting'])
        if self.env is not None:
            self.env.load_state_dict(state['env'])
        with torch.no_grad():
            self.residual_raw.copy_(state['residual_raw'])
        self.visibility.invalidate()

    @classmethod
    def load(cls, folder, step):
        """仅靠该模型目录重建推理对象，不依赖祖先checkpoint或GT灯光。"""
        folder = Path(folder)
        record = json.loads((folder/'three_stage_training.json').read_text())
        receipt = json.loads((folder/f'three_stage/iteration_{step}/receipt.json').read_text())
        state_path = folder/f'three_stage/iteration_{step}/state.pt'
        for path, digest in [(state_path, receipt['checkpoint_sha256']),
                (folder/f'point_cloud/iteration_{step}/point_cloud.ply', receipt['ply_sha256']),
                (folder/f'deform/iteration_{step}/deform.pth', receipt['deform_sha256'])]:
            if sha(path) != digest:
                raise ValueError('checkpoint来源SHA变化：'+str(path))
        state = torch.load(state_path, weights_only=False)
        if state['config'] != record['config'] or state['stage'] != record['stage'] or state['completed_updates'] != step:
            raise ValueError('checkpoint阶段或配置不匹配。')
        obj = cls.__new__(cls)
        obj.stage, obj.config, obj.binding, obj.stats = state['stage'], state['config'], state['binding'], state['stats']
        obj.parent, obj.parent_iteration = Path(record['parent']), record['parent_iteration']
        obj.pc = GaussianModel(3, no_binary_separation=False, fea_dim=1)
        obj.pc.load_ply(str(folder/f'point_cloud/iteration_{step}/point_cloud.ply')); obj.pc.active_sh_degree = 3
        obj.deform = DeformModel(deform_type='mlp', is_blender=True, hyper_dim=1, pred_color=True)
        if not obj.deform.load_weights(str(folder), step):
            raise FileNotFoundError('缺少保存的MLP。')
        obj.deform.deform.eval().requires_grad_(False)
        ids = [n for n in range(1, 151) if n % 10]
        asset = dict(gt_read=False, source_split='TRAIN', frames={f'{n:04d}': dict(light_dir_world=v.tolist())
                    for n, v in zip(ids, state['lighting']['initial'].detach().cpu())})
        obj.lights = SharedLighting(asset, direction_lr=0.)
        obj.lights.load_state_dict(state['lighting'])
        obj.visibility = StaticVisibility(obj.config['shadow_bias'], 1)
        obj.pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False,
            depth_ratio=0., pbr_rasterizer_backend='lumimotion', render_mode='original_ir', pbr_light_mode='envmap',
            diffuse_sample_num=obj.config['pbr_samples'], light_sample_num=0, light_t_min=obj.config['shadow_bias'],
            wo_indirect=False, wo_indirect_relight=False, detach_indirect=True, wo_specular=False)
        obj.env, obj.env_initial = None, state['env_initial']
        if obj.stage == 3:
            from scene.light import EnvLight
            obj.env = EnvLight(device='cuda', resolution=[16, 32], max_res=32, activation='exp', init_value=obj.env_initial)
            obj.env.load_state_dict(state['env'])
        obj.residual_raw = torch.nn.Parameter(state['residual_raw'].cuda())
        obj.completed = step
        return obj
