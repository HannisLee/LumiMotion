"""实际FrozenScene source-local RGB模型只读加载；不优化、不读GT。"""
from pathlib import Path
import sys,json,numpy as np,torch
CWD=Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0,str(CWD))
def audit_rgb_normal(model,step):
 from scripts.render_stage2 import Stage2RenderRun
 run=Stage2RenderRun(model,step);ctx=run.context;cfg=vars(run.args)
 assert cfg['pbr_source_local_rotation'] and cfg['lambda_pbr_rgb_normal_prior']==2.0
 assert len(ctx.pc.get_xyz)==99827 and len(ctx.train_cameras)==135 and len(ctx.test_cameras)==15
 assert ctx.pbr_rgb_normal is not None and int(ctx.pbr_rgb_normal.confidence.sum())==29621
 assert not any(p.requires_grad for p in ctx.deform.deform.parameters())
 record=json.loads((Path(model)/'pbr_rgb_normal.json').read_text())
 assert record['points']==99827 and record['confidence_points']==29621 and record['constraint_weight']==2.0
 for name in ['pbr_rgb_normal.npz','pbr_rgb_normal_input.npz']:
  with np.load(Path(model)/name,allow_pickle=False) as values:
   assert all(np.isfinite(values[k]).all() for k in values.files)
 with torch.no_grad():value=float(ctx.pbr_rgb_normal.loss(ctx.pc))
 assert np.isfinite(value)
 ctx.verify_frozen()
 result=dict(status='PASS_SHARED_RGB_MODEL_LOAD',points=99827,confidence_points=29621,
  constraint_weight=2.0,canonical_RGB_axis_loss=value,source_MLP_frozen=True,
  source_RGB_derived=True,GT_labels_read=False,optimizer_steps=0,source_files=ctx.source_files)
 del ctx,run;torch.cuda.empty_cache()
 return result
