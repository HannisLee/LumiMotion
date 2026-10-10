"""固定TRAIN事件字节快照与源RGB轴候选；不读GT、不修改训练。"""
from pathlib import Path
import json,hashlib,datetime,sys,csv
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
r=Path(__file__).resolve().parent.parent;c=r.parent.parent;sys.path.insert(0,str(c))
from scene.pbr_geometry import quaternion_normal
out=r/'inspection/rgb_axis_lifecycle';out.mkdir(exist_ok=False)
events=list((r/'model_mlp').glob('events.out.tfevents.*'));assert len(events)==1
p=events[0];snapshot=out/'B10.tfevents';snapshot.write_bytes(p.read_bytes())
a=EventAccumulator(str(snapshot),size_guidance={'scalars':0});a.Reload()
axis=a.Scalars('pbr_geometry/pbr_TRAIN_rgb_axis');assert axis
with np.load(r/'model_mlp/pbr_rgb_normal.npz',allow_pickle=False) as z:
 q=z['source_quaternion'];target=z['canonical_axis'];confidence=z['confidence']>0
 n=quaternion_normal(torch.tensor(q)).numpy();n/=np.linalg.norm(n,axis=-1,keepdims=True)
 target=target/np.linalg.norm(target,axis=-1,keepdims=True)
 angle=np.degrees(np.arccos(np.clip((n*target).sum(-1),-1,1)))
 norm=np.linalg.norm(q,axis=-1)
def stats(x):return dict(count=len(x),mean=float(x.mean()),q0_q50_q90_q99_q100=np.quantile(x,[0,.5,.9,.99,1]).tolist())
source_angle=stats(angle[confidence]);source_norm=stats(norm[confidence]);all_norm=stats(norm)
residual=np.maximum(angle[confidence]-60,0)
rows=[dict(iteration=x.step,Stage2_steps=x.step-35000,loss=x.value,weighted_ramped_loss=2*x.value*min(max((x.step-37000)/1000,0),1)) for x in axis]
report=dict(status='PASS_READ_ONLY_TRAIN_RGB_AXIS_LIFECYCLE',created_at=datetime.datetime.now().astimezone().isoformat(),
 event_source=str(p),event_snapshot=str(snapshot),event_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
 source_asset_sha256=hashlib.sha256((r/'model_mlp/pbr_rgb_normal.npz').read_bytes()).hexdigest(),
 points=len(q),confidence_points=int(confidence.sum()),candidate_initial_angle_deg_nonGT=source_angle,
 candidate_source_raw_q_norm=source_norm,all_source_raw_q_norm=all_norm,
 candidate_angle_gt60_count=int((angle[confidence]>60).sum()),
 canonical_60cap_ideal_axis_loss_lower_bound=float((1-np.cos(np.radians(residual))).mean()),
 first=rows[0],last=rows[-1],optimizer_steps_diagnostic=0,GT_labels_read=False,full_MAE=False,normal_7_goal='NOT_EVALUATED',
 loss_denominator='sum(confidence)=29621; all99827 arrays/normal scoring/photo unchanged',
 limitations='RGB候选角是对source预测axis，不是对GT normal；已实际优化loss下降不证明GT法线7度或外观通过。canonical60度下限是独立轴理想旋转条件，未考虑photo/BVH/Alpha冲突，不能当可达GT误差。norm统计没有充分理由改Adam坐标，本轮保持原raw q。')
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (out/'axis_loss.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
axes[0].plot([x['Stage2_steps'] for x in rows],[x['loss'] for x in rows],label='TRAIN RGB axis loss')
axes[0].set_title('New constraint in actual training');axes[0].set_xlabel('Stage2 steps');axes[0].set_ylabel('Mean 1-dot over confidence points');axes[0].grid(alpha=.25)
x=np.sort(angle[confidence]);axes[1].plot(x,np.arange(1,len(x)+1)/len(x));axes[1].axvline(60,color='gray',linestyle='--')
axes[1].set_title('Source predicted axis vs RGB candidate');axes[1].set_xlabel('Candidate angle (degrees; not GT error)');axes[1].set_ylabel('Confidence point fraction');axes[1].grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'axis_lifecycle.png',dpi=150);plt.close(fig)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
