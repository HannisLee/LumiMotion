"""复用透明度保护及真实几何训练次序的V3固定EL+DL训练器。"""
import hashlib
import json
from pathlib import Path
import torch
from scene.static_unknown_light import UnknownLightTrainer
from scene.eldl_v3_photometric import FixedELDLLights,render_photometric,censored_symmetric_loss


class ELDLTrainer(UnknownLightTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.physical_lights=FixedELDLLights(self.args.eldl_lighting)
        # 旧训练次序保留，参数requires_grad=False且优化器无梯度，不更新灯光。
        self.light_optimizer=torch.optim.Adam(self.physical_lights.parameters(),lr=0.)
        self._frozen_direct=self.physical_lights.log_intensity.detach().clone()
        self._frozen_environment=self.physical_lights.ambient_rgb.detach().clone()
        assert set(self.physical_lights.directions)=={f'{i:04d}' for i in range(1,151) if i%10}
        model=Path(self.args.model_path);root=Path(__file__).resolve().parents[1]
        record=json.loads((model/'unknown_light_training.json').read_text())
        record.update(actual_entrypoint='scripts/train_eldl_v3.py',lighting_frozen=True,
                      exact_gradient_check='FAILED_NATIVE_GEOMETRY_FINITE_DIFFERENCE',
                      gradient_policy='NATIVE_PIECEWISE_APPROXIMATE',
                      environment='空间与时间共享非负RGB，独立于DL可见度',
                      loss_model='逐通道缩小前满码删失，其余对称Charbonnier；无DL黑暗标签',
                      lighting_sha256=self.physical_lights.sha256)
        for relative in ('scene/eldl_v3_trainer.py','scene/eldl_v3_photometric.py','scripts/train_eldl_v3.py',
                         'utils/rgba16.py','utils/camera_utils.py','scene/dataset_readers.py'):
            record['new_sources'][str(root/relative)]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
        (model/'eldl_v3_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))

    def _render_photometric(self,camera,use_visibility):
        return render_photometric(camera,self.gaussians,self.pipe,self.background,self.physical_lights,
                                  self.visibility if use_visibility else None,self.iteration)

    def _photo_loss(self,package,target,gt_alpha,camera):
        if not hasattr(camera,'observed_saturation_mask'):
            raise AssertionError('混合照明必须保留缩小前逐通道饱和信息')
        return censored_symmetric_loss(package['render'],target,gt_alpha,camera.observed_saturation_mask)

    def _save_physical(self):
        assert torch.equal(self._frozen_direct,self.physical_lights.log_intensity)
        assert torch.equal(self._frozen_environment,self.physical_lights.ambient_rgb)
        assert all(p.grad is None for p in self.physical_lights.parameters())
        super()._save_physical()
        path=Path(self.args.model_path)/f'static_dl_lighting/iteration_{self.iteration}/lighting.json'
        value=json.loads(path.read_text())
        value.update(actual_entrypoint='scripts/train_eldl_v3.py',lighting_frozen=True,
                     ambient_rgb=self.physical_lights.ambient_rgb.cpu().tolist(),
                     environment_mode='shared_rgb_constant',lighting_asset=self.physical_lights.asset,
                     frozen_light_check='PASS')
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
