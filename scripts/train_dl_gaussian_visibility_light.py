"""固定Stage1的正交光源Gaussian覆盖代理；接收平面解析材质对照，无GT。"""
import argparse, hashlib, json
from pathlib import Path
import torch
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from utils.general_utils import build_rotation
from scripts.probe_dynamic_dl_selfcal import read_config

import math
from scripts.audit_dl_gaussian_visibility import visibility
from scripts.fit_dl_shadow_tracks import interpolate

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['run','audit','reference','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();torch.set_num_threads(4);out=Path(a.output);out.mkdir(exist_ok=False)
    rt=json.loads((Path(a.run)/'runtime.json').read_text());args=rt['args'];assert not args['gt_direction_reference']
    meta=json.loads((Path(a.audit)/'result.json').read_text());assert not meta['gt_read']
    sources=meta['source_hashes']
    for path,sha in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    for f in [Path(args['model'])/'cfg_args',Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply',Path(args['model'])/'deform/iteration_35000/deform.pth',Path(args['source'])/'transforms_train.json']:
        assert hashlib.sha256(f.read_bytes()).hexdigest()==rt['inputs'][str(f)]
    cache=torch.load(args['cache'],map_location='cpu',weights_only=False)
    state=torch.load(next(x for x in sources if 'T05' in x and x.endswith('checkpoint.pt')),map_location='cpu',weights_only=False)
    light=json.loads(Path(next(x for x in sources if 'T05' in x and x.endswith('result.json'))).read_text())
    d=torch.load(Path(a.audit)/'predictions.pt',map_location='cpu',weights_only=False)
    energy=torch.load(next(x for x in sources if 'T57' in x and x.endswith('checkpoint.pt')),map_location='cpu',weights_only=False)['energy'].double()
    ambient=json.loads(Path(next(x for x in sources if 'T57' in x and x.endswith('diagnostics.pt'))).with_name('result.json').read_text())['ambient']
    cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=4;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
    deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
    infos=readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False);cams=[loadCam(cfg,i,c,1.) for i,c in enumerate(infos)]
    ids=cache['ids'].cuda();assert torch.equal(pc.get_xyz[ids].cpu(),cache['canonical_xyz'])
    origin,normal,basis=[state[k].cuda() for k in ['plane_origin','plane_normal','basis']]
    full=((pc.get_xyz-origin)@normal)>light['scene_extent']*.015
    sampled=torch.zeros(len(full),dtype=torch.bool,device='cuda');sampled[ids[state['occluder']]]=True
    assert torch.equal(full[ids],state['occluder'].cuda())
    rid=torch.where(d['receiver'])[0]; gaussian_ids=ids[rid]; masks={'sampled':sampled,'full':full}
    refroot=Path(a.reference);refmeta=json.loads((refroot/'result.json').read_text());assert not refmeta['gt_read']
    ref=torch.load(refroot/'checkpoint.pt',map_location='cpu',weights_only=False);assert torch.equal(ref['receiver_ids'],rid)
    albedo=ref['groups']['sampled']['albedo'].cuda();mask=d['groups']['all_unsaturated']['mask'][:,rid]
    old_dirs=state['smooth'].cuda();new_dirs=old_dirs.clone();saved={};rows=[];fd=[];world_cache=[]
    # 先缓存固定模型；优化期间不再查询或修改几何。
    for i,c in enumerate(cams):
        c.load2device();ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(full),1),feature=pc.get_binary_feature(),camera_center=c.camera_center)
        world=pc.get_xyz+ds['d_xyz'];assert (world[ids]-cache['positions'][i].cuda()).abs().max()<1e-5
        rot=build_rotation(pc.get_rotation_bias(ds['d_rotation']));scale=pc.get_scaling+ds['d_scaling'];op=pc.get_opacity.squeeze(-1)
        if ds.get('d_opacity') is not None:op=op+ds['d_opacity'].squeeze(-1)
        world_cache.append(((world-origin).detach(),rot[:,:,:2].detach(),scale.detach(),op[sampled].detach(),((world[gaussian_ids]-origin)@basis.T).detach()))
        c.load2device('cpu')
    def predict(i,l):
        pos,rot,scale,op,q=world_cache[i];projection=torch.eye(3,device='cuda')-l[:,None]*normal[None]/(l@normal)
        center=pos@projection.T@basis.T;projected=(basis[None]@projection[None]@rot)*scale[:,None,:]
        v,_=visibility.__wrapped__(q,center[sampled],projected[sampled],op)
        cosine=(cache['normals'][i,rid].cuda().double()*l.double()).sum(-1).clamp_min(0)
        return albedo*(energy[i].cuda()*(v.double()*cosine+ambient))[:,None]
    def loss(i,l):return (predict(i,l)-d['rgb'][i,rid].cuda())[mask[i].cuda()].square().mean()
    max_reload=0.;initial=[]
    for i in range(len(cams)):
        pred=predict(i,old_dirs[i]);initial.append(pred.cpu());max_reload=max(max_reload,float((pred.cpu()-ref['groups']['sampled']['prediction'][i]).abs().max()))
    assert max_reload<1e-5, max_reload
    for i in d['train']:
        l=old_dirs[i].clone();start=float(loss(i,l));accepted=0
        for step in range(5):
            with torch.enable_grad():
                variable=l.detach().requires_grad_(True);value=loss(i,variable);grad,=torch.autograd.grad(value,variable)
            tangent=grad-l*(grad@l);norm=tangent.norm();assert torch.isfinite(norm)
            if norm<1e-12:break
            direction=-tangent/norm
            if i in [0,66,134] and step==0:
                for eps in [1e-3,1e-4,1e-5,1e-6]:
                    numerical=float((loss(i,torch.nn.functional.normalize(l+eps*direction,dim=0))-loss(i,torch.nn.functional.normalize(l-eps*direction,dim=0)))/(2*eps))
                    analytical=float(grad@direction);fd.append(dict(frame=i,epsilon=eps,numerical=numerical,analytical=analytical,relative_error=abs(numerical-analytical)/max(abs(numerical),abs(analytical),1e-12)))
            for backtrack in range(12):
                angle=math.radians(.25)*(.5**backtrack);candidate=l*math.cos(angle)+direction*math.sin(angle);candidate=torch.nn.functional.normalize(candidate,dim=0)
                score=float(loss(i,candidate))
                if math.isfinite(score) and score<float(value)-1e-12:
                    l=candidate;accepted+=1;break
            else:break
        new_dirs[i]=l;angle=float(torch.rad2deg(torch.acos((l@old_dirs[i]).clamp(-1,1))));assert angle<1.26
        row=dict(frame=i,before=start,after=float(loss(i,l)),accepted=accepted,angle_degrees=angle);rows.append(row);print(json.dumps(row),flush=True)
    times=cache['times'].cuda().reshape(-1);train=state['train'].cuda();held=~train
    interpolated=interpolate(times,times[train],new_dirs[train]);new_dirs[held]=interpolated[held]
    final=torch.stack([predict(i,new_dirs[i]).cpu() for i in range(len(cams))]);initial=torch.stack(initial);y=d['rgb'][:,rid];stats={}
    for name,pred in [('initial',initial),('updated',final)]:
        stats[name]={}
        for split,ts in [('train',d['train']),('held',d['held'])]:
            for label,extra in [('all',torch.ones_like(mask)),('bright',state['negative'][:,rid]),('dark',state['positive'][:,rid]),('strict',d['groups']['strict_bright']['mask'][:,rid])]:
                m=(mask&extra)[ts];stats[name][split+'_'+label]=dict(count=int(m.sum()),mse=float((pred-y)[ts][m].square().mean()))
    torch.save(dict(directions=new_dirs.cpu(),initial_directions=old_dirs.cpu(),prediction=final,initial_prediction=initial,receiver_ids=rid,albedo=albedo.cpu(),geometry=[tuple(x.cpu() for x in g) for g in world_cache],rows=rows,fd=fd),out/'checkpoint.pt')
    result=dict(status='PASS',gt_read=False,geometry_updates=0,material_updates=0,accepted_light_updates=sum(x['accepted'] for x in rows),training_frames=len(rows),reference_prediction_max_error=max_reload,statistics=stats,finite_difference=fd,scope='固定采样Gaussian透射及共享材质；120独立训练节点各最多5次0.25度实际下降，held仅时间插值。硬3sigma边界导数未包含，FD结果需独立判断，不自动证明梯度正确或动态几何改善。',source_hashes={str(refroot/'checkpoint.pt'):hashlib.sha256((refroot/'checkpoint.pt').read_bytes()).hexdigest(),__file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},inputs=rt['inputs'])
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
