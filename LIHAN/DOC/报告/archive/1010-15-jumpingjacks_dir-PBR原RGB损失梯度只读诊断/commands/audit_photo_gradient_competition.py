"""同40k checkpoint原RGB各损失几何梯度关系，0优化，无GT标签/prior。"""
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import torch
import torch.nn.functional as F
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_deformation import deformation_input
from gaussian_renderer.render_ir import render_ir
from utils.loss_utils import l1_loss,ssim

def pair_stats(a,b,mask):
    a=a[mask].double().reshape(int(mask.sum()),-1);b=b[mask].double().reshape(int(mask.sum()),-1)
    an=a.norm(dim=1);bn=b.norm(dim=1);both=(an>1e-12)&(bn>1e-12)
    dot=(a*b).sum(1);cos=dot/(an*bn).clamp_min(1e-30)
    return dict(points=int(mask.sum()),a_L2=float(a.norm()),b_L2=float(b.norm()),
        aggregate_cosine=float(dot.sum()/(a.norm()*b.norm()).clamp_min(1e-30)),
        both_nonzero_points=int(both.sum()),negative_dot_points=int(((dot<0)&both).sum()),
        negative_fraction=float((dot[both]<0).double().mean()) if both.any() else None,
        median_point_cosine=float(cos[both].median()) if both.any() else None,
        b_to_a_L2_ratio=float(b.norm()/a.norm().clamp_min(1e-30)),
        a_squared_gradient_fraction_where_negative=float(an[dot<0].square().sum()/an.square().sum().clamp_min(1e-30)))

def main():
    torch.manual_seed(1010)
    run=Stage2RenderRun(ROOT/'model_mlp',40000);ctx=run.context;pc=ctx.pc
    prob=torch.nn.Parameter(pc.get_binary_feature().detach().clone())
    parameters=[pc._xyz,pc._rotation,prob]
    for p in parameters:p.requires_grad_(True)
    assert not any(p.requires_grad for p in ctx.deform.deform.parameters())
    source_feature=ctx.pbr_deformation_xyz
    plane=ctx.pbr_plane_projection.support
    # 分组只表示既有source projection支持，不是GT物体或训练筛选。
    groups={'all':torch.ones(len(pc.get_xyz),dtype=torch.bool,device='cuda'),
            'source_projected':plane,'source_unprojected':~plane}
    run.args.train_ray=True;rows=[]
    for key in ['0001','0075','0149']:
        cam=ctx.camera_by_key[key];ctx.load_camera(cam)
        gate=pc.get_binary_feature().detach()+(prob-prob.detach())
        d=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),
            cam.fid.unsqueeze(0).expand(len(pc.get_xyz),-1),iteration=40000,
            feature=gate,camera_center=cam.camera_center)
        d=ctx.pbr_plane_projection.apply(pc,d)
        with torch.no_grad():
            reference=ctx.deformation(cam)
            assert all(torch.equal(d[k].detach(),reference[k]) for k in ['d_xyz','d_rotation'])
        if run.bvh:pc.update_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']})
        else:pc.build_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']});run.bvh=True
        pkg=render_ir(cam,pc,run.args,ctx.background,
            **{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
            opt=run.args,training=True)
        target=cam.original_image_train_light.cuda()
        photo=F.l1_loss(pkg['render'].permute(1,2,0)[pkg['mask']],target.permute(1,2,0)[pkg['mask']])
        shmask=(pkg['rend_alpha']>.9).float().expand_as(pkg['render_sh'])
        sh=.8*l1_loss(pkg['render_sh']*shmask,target*shmask)+.2*(1-ssim(pkg['render_sh']*shmask,target*shmask))
        n=F.normalize(pkg['rend_normal'],dim=0);z=F.normalize(pkg['surf_normal'],dim=0)
        valid=(pkg['rend_alpha'].detach()>.5)&(pkg['rend_normal'].detach().norm(dim=0,keepdim=True)>1e-6)&(pkg['surf_normal'].detach().norm(dim=0,keepdim=True)>1e-6)
        valid=(F.avg_pool2d(valid.float()[None],3,stride=1,padding=1)[0]>=1).detach()
        nd=((1-(n*z).sum(0,keepdim=True).clamp(-1,1))*valid).sum()/valid.sum().clamp_min(1)*run.args.lambda_pbr_normal_depth
        alpha=(pkg['rend_alpha']-cam.gt_alpha_mask.to(pkg['rend_alpha'])).abs().mean()*run.args.lambda_pbr_silhouette
        terms={'PBR_photo':photo,'SH_photo':sh,'normal_depth':nd,'own_alpha':alpha}
        grads={}
        for name,value in terms.items():
            grads[name]=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
            grads[name]=[torch.zeros_like(p) if g is None else g for g,p in zip(grads[name],parameters)]
            assert torch.isfinite(value) and all(torch.isfinite(g).all() for g in grads[name])
        row=dict(frame=key,eligible_pixels=int(pkg['mask'].sum()),losses={k:float(v) for k,v in terms.items()},pairs={})
        for name in ['SH_photo','normal_depth','own_alpha']:
            row['pairs']['PBR_photo_vs_'+name]={label:{group:pair_stats(a,b,mask) for group,mask in groups.items()}
                for label,a,b in zip(['canonical_XYZ','canonical_quaternion','native_gate_probability'],grads['PBR_photo'],grads[name])}
        rows.append(row);ctx.unload_camera(cam)
        del d,pkg,grads,terms,photo,sh,nd,alpha,n,z;torch.cuda.empty_cache()
    ctx.verify_frozen()
    manifest=json.loads((ROOT/'code_manifest.json').read_text());assert len(manifest)==474
    assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
    record=dict(status='PASS_READ_ONLY_LOSS_GRADIENT_RELATION',iteration=40000,rows=rows,
        optimizer_steps=0,GT_labels_read=False,training_changed=False,training_prior_saved=False,
        source_files=ctx.source_files,created_at=datetime.datetime.now().astimezone().isoformat(),
        limitations='3 TRAIN原RGB/自身Alpha固定40k的局部梯度；不是优化轨迹、全15正式MAE或GT normal结论。source plane支持仅预测分组，全99827点保留。负dot和范数比不能单独证明损失有害；只读不改loss/Preset。')
    p=ROOT/'inspection/photo_gradient_competition.json';assert not p.exists()
    p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
