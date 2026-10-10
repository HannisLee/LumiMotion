"""55k只读概率/原raw gate梯度比较：0 optimizer，不保存训练prior，不读GT标签。"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import torch,numpy as np
import torch.nn.functional as F
from plyfile import PlyData
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_deformation import deformation_input
from scene.pbr_geometry import quaternion_normal
from gaussian_renderer.render_ir import render_ir
from utils.loss_utils import l1_loss,ssim

torch.manual_seed(1010)
run=Stage2RenderRun(ROOT/'model_mlp',55000);ctx=run.context;pc=ctx.pc
pc.feature.requires_grad_(True)
assert json.loads((ROOT/'training_runtime.json').read_text())['training_exit_code']==0
probability=pc.get_binary_feature()
source_derivative=2*probability.detach()*(1-probability.detach())
def stat(value):
    value=value.detach().double().reshape(-1)
    return dict(mean=float(value.mean()),median=float(value.median()),p99=float(torch.quantile(value,.99)),maximum=float(value.max()))
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
         cam.fid.unsqueeze(0).expand(len(xyz),-1),iteration=55000,
         feature=probability,camera_center=cam.camera_center)
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
        raw_gradient,probability_gradient=torch.autograd.grad(value,[pc.feature,probability],retain_graph=True)
        assert torch.allclose(raw_gradient,probability_gradient*source_derivative,rtol=2e-5,atol=1e-9)
        grad=raw_gradient[:,0]
        grad_prob=probability_gradient[:,0]
        assert torch.isfinite(grad_prob).all()
        assert torch.isfinite(value) and torch.isfinite(grad).all()
        row['losses'][name]=float(value);row['gradient_groups'][name]={}
        for label,mask in groups.items():
            values=grad[mask];nonzero=values.abs()>1e-12
            row['gradient_groups'][name][label]=dict(points=int(mask.sum()),L2=float(values.norm()),
                 nonzero_points=int(nonzero.sum()),positive_fraction_among_nonzero=float((values[nonzero]>0).float().mean()) if nonzero.any() else None,
                 gradient_sum=float(values.sum()),probability_gradient_L2=float(grad_prob[mask].norm()),
                 sigmoid_derivative=stat(source_derivative[:,0][mask]),
                 probability_gradient_nonzero=int((grad_prob[mask].abs()>1e-12).sum()),
                 raw_first_Adam_probability_delta=stat((torch.sigmoid(2*(pc.feature[:,0].detach()-.001*grad/(grad.abs()+1e-8)))-probability[:,0].detach()).abs()[mask]),
                 direct_first_Adam_probability_delta=stat(((probability[:,0].detach()-.001*grad_prob/(grad_prob.abs()+1e-8)).clamp(0,1)-probability[:,0].detach()).abs()[mask]))
    rows.append(row);ctx.unload_camera(cam)
    del d,pkg,terms,photo,sh,nd,silhouette,n,z;torch.cuda.empty_cache()
ctx.verify_frozen()
manifest=json.loads((ROOT/'code_manifest.json').read_text())
assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
assert not (ROOT/'inspection/probability_gate_gradients_55000.json').exists()
record=dict(status='PASS_READ_ONLY_PROBABILITY_GATE_GRADIENT_DIAGNOSTIC',iteration=55000,rows=rows,
    optimizer_steps=0,feature_values_changed=False,mlp_weights_changed=False,gt_labels_read=False,
    new_training_prior_saved=False,current_training_modified=False,final_mae=False,
    source_files=ctx.source_files,hypothetical_first_Adam_lr=.001,
    hypothetical_optimizer_step=False,probability_gradient_chain_rule_verified=True,
    limits='RAM中只估算无动量首Adam步，当前实际gate/任何权重不更新；不能当真实训练或7°指标。只证明在同预测状态开放输出乘法gate的RGB/Alpha/ND梯度；candidate由source预测筛选，不是真实物体标注；不据此宣称训练后质量或7°，后续需完整新训练评测。')
(ROOT/'inspection/probability_gate_gradients_55000.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
