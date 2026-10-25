"""只读三个TRAIN，诊断原生射线起点偏移；不更新模型，不读取TEST。"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from scripts.render_stage2 import Stage2RenderRun, save_image
from scripts.stage2_runtime import supervision, file_sha256
from scene.stage2_photometric import frame_key
from utils.loss_utils import ssim


def main():
    root = Path(__file__).resolve().parents[2]
    # 本模块在冻结快照中执行，结果根路径由固定模型配置追溯到实际output。
    experiment = Path('/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-07-dynamicDL-Stage3估计方向光')
    output = experiment.parent/'1025-02-Stage3-PPT配对总控/train_bias_diagnostic_4000'
    output.mkdir(exist_ok=False)
    run = Stage2RenderRun(experiment/'model_mlp', 20500)
    assert not run.context.test_cameras
    cameras = [run.context.camera_by_key[key] for key in ['0001', '0085', '0149']]
    checksums = dict(run.context.source_files)
    values = []
    with torch.no_grad():
        for bias in [.1, .01, .03, .2]:
            run.args.light_t_min = bias
            directory = output/str(bias)
            directory.mkdir()
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            rows = []
            for camera in cameras:
                package = run.render(camera)
                target, mask = supervision(camera, run.context.background)
                image = package['render'].clamp(0, 1)
                mse = (image-target).square()
                row = dict(frame=frame_key(camera.image_name_train_light), psnr=-10*math.log10(max(float(mse.mean()), 1e-12)),
                           ssim=float(ssim(image, target)), psnr_foreground=-10*math.log10(max(float((mse*mask).sum()/(3*mask.sum())), 1e-12)))
                save_image(directory/(row['frame']+'.png'), image)
                rows.append(row)
                run.context.unload_camera(camera)
            values.append(dict(light_t_min=bias, frames=rows, metrics={k: float(np.mean([r[k] for r in rows])) for k in ['psnr', 'ssim', 'psnr_foreground']}))
            print(json.dumps(values[-1], ensure_ascii=False), flush=True)
    baseline = values[0]['metrics']
    eligible = [value for value in values[1:] if value['metrics']['ssim'] >= baseline['ssim']+.001 and value['metrics']['psnr'] >= baseline['psnr']-.05]
    selected = max(eligible, key=lambda value: value['metrics']['ssim'])['light_t_min'] if eligible else .1
    assert all(file_sha256(Path(p)) == value for p, value in checksums.items())
    result = dict(status='COMPLETED_TRAIN_ONLY', model=str(experiment/'model_mlp'), iteration=20500, values=values,
                  selected_light_t_min=selected, adopt_render_override=selected != .1, train_light_t_min=.1,
                  criterion='三个TRAIN平均SSIM≥默认+0.001且平均PSNR≥默认-0.05dB；否则保留0.1',
                  checkpoint_sha256=checksums, checkpoint_unchanged=True, test_read=False, optimizer_updates=0)
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
