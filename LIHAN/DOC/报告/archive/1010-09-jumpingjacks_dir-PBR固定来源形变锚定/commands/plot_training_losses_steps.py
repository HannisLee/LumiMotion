"""只读TensorBoard生成可复核的训练损失图；原始数据CSV保留。"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path(__file__).resolve().parent.parent
CWD=ROOT.parent.parent
BASE=CWD/'output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复'
tags=[('train_loss_patches/l1_loss','Sampled RGB L1'),
      ('pbr_light/intensity_scale','Train RGB fitted irradiance scale'),
      ('pbr_geometry/pbr_normal_depth','Predicted GS/depth cosine consistency'),
      ('pbr_geometry/pbr_plane_normal','Source plane axis consistency'),
      ('pbr_geometry/pbr_plane_position','Source plane position consistency'),
      ('pbr_geometry/pbr_silhouette','Input RGBA alpha L1')]
data={};rows=[]
for name,folder in [('B03 Refine',BASE),('B04 Anchor',ROOT)]:
    event=EventAccumulator(str(folder/'model_mlp'),size_guidance={'scalars':0});event.Reload()
    data[name]={}
    for tag,label in tags:
        values=event.Scalars(tag);assert values
        data[name][tag]=values
        rows.extend(dict(run=name,tag=tag,iteration=v.step,stage2_step=v.step-35000,
                         wall_time=v.wall_time,value=v.value) for v in values)
fig,axs=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
for ax,(tag,label) in zip(axs.ravel(),tags):
    for name,color in [('B03 Refine','#888888'),('B04 Anchor','#1769aa')]:
        values=data[name][tag];x=np.array([v.step-35000 for v in values]);y=np.array([v.value for v in values])
        window=201 if tag=='train_loss_patches/l1_loss' else 1
        if window>1 and len(x)>=window:
            y=np.convolve(y,np.ones(window)/window,mode='valid');x=x[window-1:]
        ax.plot(x,y,color=color,label=name,linewidth=1.4)
    ax.set_title(label);ax.set_xlabel('Stage2 steps (35k source)');ax.set_ylabel('Raw scalar');ax.grid(alpha=.25)
    ax.set_xlim(0,20000)
axs[0,0].legend();fig.suptitle('Native PBR training loss diagnostics; B04 observed prefix')
path=ROOT/'inspection/training_losses_prefix_steps.png';fig.savefig(path,dpi=140);plt.close(fig)
with (ROOT/'inspection/training_losses_prefix_steps.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['run','tag','iteration','stage2_step','wall_time','value']);writer.writeheader();writer.writerows(rows)
report=dict(status='PASS_READ_ONLY_LOSS_PLOT',b04_last_observed_iteration=max(v.step for vs in data['B04 Anchor'].values() for v in vs),
            rgb_l1_display_trailing_window=201,other_display_window=1,raw_csv_rows=len(rows),
            scope='训练内部损失与光强；B04只是当前前缀，不是完整最终结果或GT法线MAE',
            gt_inputs=False,training_modified=False,image=str(path))
(ROOT/'inspection/training_losses_prefix_steps.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
