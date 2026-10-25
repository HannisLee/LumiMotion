"""只读 TRAIN TensorBoard，汇总真实新增循环和分段损失，不读取 TEST。"""
import json
import re
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT/'output/1025-02-Stage3-PPT配对总控'
FONT = ROOT/'exports/1024-01-静态ELDL-研究展示素材/NotoSansCJKsc-Regular.otf'


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(), 'axes.unicode_minus': False})
    figure, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    records = []
    experiments = [('静态原生', '1025-03-staticDL-Stage3原生基线', 0), ('静态改进', '1025-04-staticDL-Stage3改进几何', 0),
                   ('动态原生', '1025-05-dynamicDL-Stage3原生基线', 1), ('动态改进', '1025-06-dynamicDL-Stage3改进几何', 1)]
    for name, directory, panel in experiments:
        path = ROOT/'output'/directory
        state = json.loads((path/'state.json').read_text())
        accumulator = EventAccumulator(str(path/'model_mlp'), size_guidance={'scalars': 0})
        accumulator.Reload()
        tags = accumulator.Tags()['scalars']
        series = accumulator.Scalars('train_loss_patches/total_loss') if 'train_loss_patches/total_loss' in tags else []
        x = np.array([point.step-state['source_iteration'] for point in series])
        y = np.array([point.value for point in series])
        assert np.isfinite(y).all()
        record = dict(experiment=name, status=state['status'], source_iteration=state['source_iteration'],
                      last_flushed_train_loop=int(x[-1]) if len(x) else 0, flushed_train_scalars=len(y), gpu=state['gpu'])
        if len(y) >= 200:
            smoothed = np.convolve(y, np.ones(200)/200, mode='valid')
            axes[panel].plot(x[199:], smoothed, label=name, linewidth=1.7)
            record['last_200_mean_loss'] = float(y[-200:].mean())
            record['last_1000_mean_loss'] = float(y[-1000:].mean())
            record['first_1000_full_pbr_mean_loss'] = float(y[(x>2000)&(x<=3000)].mean()) if np.any((x>2000)&(x<=3000)) else None
        records.append(record)
    for axis, title in zip(axes, ['静态 DL-V2', '动态 DL-V2']):
        axis.axvline(2000, color='gray', ls='--', linewidth=1)
        axis.set_title(title+'：TRAIN损失（200循环滑动平均）')
        axis.set_xlabel('Stage3新增循环')
        axis.set_ylabel('原生训练总损失')
        axis.grid(alpha=.2)
        axis.legend()
        axis.set_ylim(bottom=0)
    figure.suptitle('前2000循环仅漫反射；之后为完整PBR。训练loss不等于TEST PSNR或SSIM。', fontsize=12)
    figure.savefig(CONTROL/'TRAIN损失曲线-当前.png', dpi=160)
    plt.close(figure)
    result = dict(time=time.time(), split='TRAIN_ONLY', records=records)
    (CONTROL/'train_progress.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
