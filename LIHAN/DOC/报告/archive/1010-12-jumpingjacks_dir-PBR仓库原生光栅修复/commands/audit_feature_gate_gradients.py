"""40k只读native gate梯度审计；等待同train检查点，不修改本轮/不读取GT标签。"""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
marker=ROOT/'model_mlp/pbr_directional/iteration_40000/lighting.json'
handles=json.loads((ROOT/'process_handles.json').read_text());train=handles['train']
proc=Path('/proc')/str(train['pid'])
while not marker.exists():
    if not proc.exists() or (proc/'stat').read_text().split()[21]!=train['start_ticks']:
        raise RuntimeError('40k前原训练已退出，禁止自动重启')
    time.sleep(5)
import torch,numpy as np
import torch.nn.functional as F
from plyfile import PlyData
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_deformation import deformation_input
from scene.pbr_geometry import quaternion_normal
from gaussian_renderer.render_ir import render_ir
from utils.loss_utils import l1_loss,ssim

torch.manual_seed(1010)
run=Stage2RenderRun(ROOT/'model_mlp',40000);ctx=run.context;pc=ctx.pc
pc.feature.requires_grad_(True)
assert not any(p.requires_grad for p in ctx.deform.deform.parameters())
source=PlyData.read(ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
xyz=torch.as_tensor(np.column_stack([source[k] for k in ['x','y','z']]),device='cuda')
q=torch.as_tensor(np.column_stack([source[f'rot_{k}'] for k in range(4)]),device='cuda')
source_feature=torch.sigmoid(torch.tensor(source['fea_0'].copy(),device='cuda')*2)
source_alpha=torch.sigmoid(torch.tensor(source['opacity'].copy(),device='cuda'))
with np.load(ROOT/'model_mlp/pbr_plane_projection.npz') as data:
    arrays={k:torch.tensor(data[k].copy(),device='cuda') for k in data.files}
plane=arrays['plane'];offset=arrays['offset']
candidate=((source_feature>=.5)&(source_alpha>.05)&((xyz@plane+offset).abs()<=.2)&
           (arrays['max_plane_distance']<=.2)&((quaternion_normal(q)@plane).abs()>=.5)&
           (arrays['min_axis_cosine']>=.5))
groups={'all':torch.ones(len(xyz),dtype=torch.bool,device='cuda'),
        'projected_plane':arrays['support'],'dynamic_plane_candidates':candidate,
        'other_dynamic':(source_feature>=.5)&~candidate,
        'static_unprojected':(source_feature<.5)&~arrays['support']}
run.args.train_ray=True;rows=[]
for key in ['0001','0075','0149']:
    cam=ctx.camera_by_key[key];ctx.load_camera(cam)
    # NN权重与source XYZ固定；feature只在输出乘法gate中产生梯度，无优化。
    d=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz).detach(),
         cam.fid.unsqueeze(0).expand(len(xyz),-1),iteration=40000,
         feature=pc.get_binary_feature(),camera_center=cam.camera_center)
    assert not d['hidden'].requires_grad
    assert d['d_xyz'].requires_grad and d['d_rotation'].requires_grad
    d=ctx.pbr_plane_projection.apply(pc,d)
    with torch.no_grad():
        reference=ctx.deformation(cam)
        assert all(torch.equal(d[k].detach(),reference[k]) for k in ['d_xyz','d_rotation'])
    if run.bvh:pc.update_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']})
    else:pc.build_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']});run.bvh=True
    pkg=render_ir(cam,pc,run.args,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                  opt=run.args,training=True)
    target=cam.original_image_train_light.cuda()
    photo=F.l1_loss(pkg['render'].permute(1,2,0)[pkg['mask']],target.permute(1,2,0)[pkg['mask']])
    shmask=(pkg['rend_alpha']>.9).float().expand_as(pkg['render_sh'])
    sh=.8*l1_loss(pkg['render_sh']*shmask,target*shmask)+.2*(1-ssim(pkg['render_sh']*shmask,target*shmask))
    n=F.normalize(pkg['rend_normal'],dim=0);z=F.normalize(pkg['surf_normal'],dim=0)
    valid=(pkg['rend_alpha'].detach()>.5)&(pkg['rend_normal'].detach().norm(dim=0,keepdim=True)>1e-6)&(pkg['surf_normal'].detach().norm(dim=0,keepdim=True)>1e-6)
    valid=(F.avg_pool2d(valid.float()[None],3,stride=1,padding=1)[0]>=1).detach()
    nd=((1-(n*z).sum(0,keepdim=True).clamp(-1,1))*valid).sum()/valid.sum().clamp_min(1)*run.args.lambda_pbr_normal_depth
    silhouette=(pkg['rend_alpha']-cam.gt_alpha_mask.to(pkg['rend_alpha'])).abs().mean()*run.args.lambda_pbr_silhouette
    terms={'native_photo':photo+sh,'PBR_photo':photo,'SH_photo':sh,'normal_depth':nd,'input_alpha':silhouette,
           'combined_feature_dependent':photo+sh+nd+silhouette}
    row={'frame':key,'losses':{},'gradient_groups':{}}
    for name,value in terms.items():
        grad=torch.autograd.grad(value,pc.feature,retain_graph=True)[0][:,0]
        assert torch.isfinite(value) and torch.isfinite(grad).all()
        row['losses'][name]=float(value);row['gradient_groups'][name]={}
        for label,mask in groups.items():
            values=grad[mask];nonzero=values.abs()>1e-12
            row['gradient_groups'][name][label]=dict(points=int(mask.sum()),L2=float(values.norm()),
                 nonzero_points=int(nonzero.sum()),positive_fraction_among_nonzero=float((values[nonzero]>0).float().mean()) if nonzero.any() else None,
                 gradient_sum=float(values.sum()))
    rows.append(row);ctx.unload_camera(cam)
    del d,pkg,terms,photo,sh,nd,silhouette,n,z;torch.cuda.empty_cache()
ctx.verify_frozen()
manifest=json.loads((ROOT/'code_manifest.json').read_text())
assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
record=dict(status='PASS_READ_ONLY_FEATURE_GATE_GRADIENT_DIAGNOSTIC',iteration=40000,rows=rows,
    optimizer_steps=0,feature_values_changed=False,mlp_weights_changed=False,gt_labels_read=False,
    new_training_prior_saved=False,current_training_modified=False,final_mae=False,
    source_files=ctx.source_files,limits='只证明在同预测状态开放输出乘法gate的RGB/Alpha/ND梯度；candidate由source预测筛选，不是真实物体标注；不据此宣称训练后质量或7°，后续需完整新训练评测。')
(ROOT/'inspection/feature_gate_gradients.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
