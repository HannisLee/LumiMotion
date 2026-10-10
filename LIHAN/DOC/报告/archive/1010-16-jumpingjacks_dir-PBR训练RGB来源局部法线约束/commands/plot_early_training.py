"""固定当前事件字节快照，绘制B09/B10早期训练曲线；不是离线MAE。"""
from pathlib import Path
import json, csv, hashlib
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent/'1010-15-jumpingjacks_dir-PBR概率gate优化'
dest = ROOT/'inspection/early_training'
dest.mkdir(exist_ok=False)
tags = ['train_loss_patches/l1_loss', 'train_loss_patches/total_loss', 'pbr_light/intensity_scale']
runs, provenance = {}, {}
for label, root in [('B09 Direct SUN + probability gate', OLD), ('B10 source-local + TRAIN RGB axis', ROOT)]:
    event = list((root/'model_mlp').glob('events.out.tfevents.*'))
    assert len(event) == 1
    data = event[0].read_bytes()
    snapshot = dest/('B09.tfevents' if root == OLD else 'B10.tfevents')
    snapshot.write_bytes(data)
    provenance[label] = dict(source=str(event[0]), snapshot=str(snapshot),
                             bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    accumulator = EventAccumulator(str(snapshot), size_guidance={'scalars': 0})
    accumulator.Reload()
    runs[label] = {tag: accumulator.Scalars(tag) for tag in tags}
limit = min(runs[label][tags[0]][-1].step for label in runs)-35000
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
titles = ['Photo L1: selected foreground', 'Total training loss', 'RGB-fitted global light scale']
rows = []
for label, values in runs.items():
    for ax, tag, title in zip(axes, tags, titles):
        points = [(v.step-35000, v.value) for v in values[tag] if v.step-35000 <= limit]
        if 'intensity' not in tag:
            bins = {}
            for step, value in points:
                bins.setdefault((step-1)//100, []).append((step, value))
            points = [(np.mean([a for a, _ in batch]), np.mean([b for _, b in batch]))
                      for batch in bins.values()]
        ax.plot([a for a, _ in points], [b for _, b in points], label=label, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Stage2 steps after fixed 35000')
        ax.grid(alpha=.25)
        rows += [dict(run=label, tag=tag, step=float(a), value=float(b)) for a, b in points]
axes[0].legend(fontsize=7)
fig.suptitle('Early training snapshot: 100-step loss averages; not final MAE', fontsize=11)
fig.tight_layout()
fig.savefig(dest/'training_curves.png', dpi=160)
with (dest/'training_curves.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['run', 'tag', 'step', 'value'])
    writer.writeheader()
    writer.writerows(rows)
differences={}
labels=list(runs)
for tag in tags:
    left={v.step-35000:v.value for v in runs[labels[0]][tag]}
    right={v.step-35000:v.value for v in runs[labels[1]][tag]}
    differences[tag]={}
    for phase,selection in [('warmup',lambda step:step<=2000),('geometry',lambda step:2000<step<=limit)]:
        values=np.array([right[step]-left[step] for step in sorted(set(left)&set(right)) if step<=limit and selection(step)])
        differences[tag][phase]=dict(points=len(values),mean_delta=float(values.mean()) if len(values) else None,max_abs_delta=float(np.abs(values).max()) if len(values) else None)
metadata = dict(status='TRAINING_CURVE_SNAPSHOT_ONLY', stage2_steps=limit,
                created_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                snapshots=provenance,pointwise_differences=differences, optimizer_steps=0, gt_labels_read=False,
                loss_selection='B09/B10均all-eligible alpha>.9、相同Direct SUN/native GGX/BVH；B09/B10均概率坐标/[0,1]，B10 source-local真实q与lambda2自身训练RGB轴约束，total包含新增约束故不可仅直接比较total，训练loss不代表法线MAE',
                limitations='训练loss和RGB拟合光强，不是GT材质/法线MAE，不由曲线声明质量或7度通过。')
(dest/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
