"""对失败的纯几何正则做零更新分支有限差分；不读GT。"""
import argparse,json,math,shlex,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from utils.general_utils import build_rotation
from scripts.probe_dynamic_dl_selfcal import read_config,digest_tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failed-run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
    failed=Path(args.failed_run);runtime=json.loads((failed/'runtime.json').read_text());cfg=read_config(Path(runtime['args']['model'])/'cfg_args')
    assert runtime['args']['control']=='regularizer' and runtime['gt_read'] is False
    cfg.resolution=4;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    pc.load_ply(str(Path(runtime['args']['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
    deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
    assert deform.load_weights(runtime['args']['model'],35000);deform.deform.eval().requires_grad_(False)
    for v in vars(pc).values():
        if isinstance(v,torch.Tensor):v.requires_grad_(False)
    infos=readCamerasFromTransforms(runtime['args']['source'],'transforms_train.json',cfg.train_light_folder,False)
    cam=loadCam(cfg,0,infos[0],1.);cam.load2device()
    state=torch.load(failed/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['geometry_updates']==0 and state['material_updates']==600
    with torch.no_grad():
        ds=deform.step(pc.get_xyz,cam.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=cam.camera_center)
        ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
        direction=torch.tensor(json.loads((failed/'used_light.json').read_text())['directions'][0],device='cuda')
        n=build_rotation(pc.get_rotation_bias(ds['d_rotation']))[:,:,2]
        sign=torch.where((n*(cam.camera_center-pc.get_xyz-ds['d_xyz'])).sum(-1,keepdim=True)>=0,1.,-1.)
        energy=F.softplus(state['energy'].cuda());energy/=energy.mean()
        colors=F.softplus(state['albedo'].cuda())*((n*sign)@direction).clamp_min(0)[:,None]*energy[0]
    mask=cam.gt_alpha_mask.cuda();interior=-F.max_pool2d(-(mask>.99).float()[None],5,stride=1,padding=2)[0]
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda')
    xyz=pc._xyz;before=digest_tensor(xyz);xyz.requires_grad_(True)
    def maps(color=colors):return render(cam,pc,pipe,bg,**ds,override_color=color)
    package=maps();base_rn=F.normalize(package['rend_normal'],dim=0).detach();base_dn=F.normalize(package['surf_normal'],dim=0).detach()
    def losses():
        package=maps();rn=F.normalize(package['rend_normal'],dim=0);dn=F.normalize(package['surf_normal'],dim=0)
        def normal_loss(a,b):return .05*((1-(a.double()*b.double()).sum(0,keepdim=True))*interior.double()).sum()/interior.sum().double()
        normal=normal_loss(rn,dn)
        alpha=.01*F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6,1-1e-6).double(),mask.double())
        return dict(total=normal+alpha,normal=normal,alpha=alpha,rend_branch=normal_loss(rn,base_dn),depth_branch=normal_loss(base_rn,dn))
    values=losses();gradients={k:torch.autograd.grad(v,xyz,retain_graph=True)[0].detach() for k,v in values.items()}
    original_index=json.loads((failed/'gradient.json').read_text())['coordinate']
    coordinates=sorted(set([original_index]+gradients['total'].abs().flatten().topk(3).indices.tolist()+[int(gradients[k].abs().argmax()) for k in ['normal','alpha','rend_branch','depth_branch']]))
    records=[]
    with torch.no_grad():
        zero=maps(torch.zeros_like(colors));actual=maps()
        color_map_error=max(float((zero[k]-actual[k]).abs().max()) for k in ['rend_alpha','rend_normal','surf_depth','surf_normal'])
        flat=xyz.flatten()
        for coordinate in coordinates:
            value=float(flat[coordinate]);sweep=[]
            for epsilon in [1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6]:
                flat[coordinate]=value+epsilon;plus_coordinate=float(flat[coordinate]);plus={k:float(v) for k,v in losses().items()}
                flat[coordinate]=value-epsilon;minus_coordinate=float(flat[coordinate]);minus={k:float(v) for k,v in losses().items()}
                flat[coordinate]=value
                row={'epsilon':epsilon,'actual_span':plus_coordinate-minus_coordinate,'branches':{}}
                for key in gradients:
                    analytic=float(gradients[key].flatten()[coordinate]);numeric=(plus[key]-minus[key])/(plus_coordinate-minus_coordinate)
                    row['branches'][key]=dict(analytic=analytic,numeric=numeric,relative_error=abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-10))
                sweep.append(row)
            records.append(dict(coordinate=coordinate,sweep=sweep))
            print(json.dumps(dict(coordinate=coordinate,total_errors=[r['branches']['total']['relative_error'] for r in sweep])),flush=True)
    assert digest_tensor(xyz)==before
    record=dict(source_runtime=runtime,failed_coordinate=original_index,zero_updates=True,xyz_hash=before,
                losses={k:float(v) for k,v in values.items()},color_map_max_error=color_map_error,
                branch_sum_max_error=float((gradients['normal']-gradients['rend_branch']-gradients['depth_branch']).abs().max()),records=records)
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));torch.save(gradients,out/'gradients.pt')


if __name__=='__main__':main()
