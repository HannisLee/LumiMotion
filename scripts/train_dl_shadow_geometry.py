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

from scripts.audit_dl_visibility_precision import stable_visibility
@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['run','audit','reference','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--active-region',choices=['all','dynamic'],default='all')
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
    reference=Path(a.reference);ck=torch.load(reference/'train/checkpoint.pt',map_location='cpu',weights_only=False)
    occ=torch.where(sampled)[0];assert not torch.isin(occ,gaussian_ids).any()
    partition_path=Path(next(x for x in sources if 'T57' in x and x.endswith('diagnostics.pt')))
    partition=torch.load(partition_path,map_location='cpu',weights_only=False)
    dynamic_ids=ids[partition['dynamic'].cuda()]
    active=occ if a.active_region=='all' else occ[torch.isin(occ,dynamic_ids)]
    assert len(active)>0
    xyz0=pc.get_xyz.detach().clone();base_rotation=pc._rotation.detach();base_scale=pc.get_scaling.detach();base_opacity=pc.get_opacity.detach().squeeze(-1);feature=pc.get_binary_feature().detach()
    albedo=ck['albedo'].cuda();mask=d['groups']['all_unsaturated']['mask'][:,rid].cuda();y=d['rgb'][:,rid].cuda();normal_receiver=cache['normals'][:,rid].double().cuda();directions=state['smooth'].double().cuda();train=d['train'];held=d['held'];count=int(mask[train].sum());metric_path=Path(a.run)/'xyz_metric.pt';metric_state=torch.load(metric_path,map_location='cpu',weights_only=False);assert metric_state['indices']==train;metric=metric_state['diagonal'][occ.cpu()].cuda()
    metric=metric[torch.isin(occ,active)]
    delta=torch.zeros((len(active),3),device='cuda');initial_world=[v[0].cuda()+origin for v in ck['geometry']]
    def forward(i,change,with_rgb=True):
        camera=cams[i];camera.load2device();xyz=xyz0.index_copy(0,active,xyz0[active]+change)
        ds=deform.step(xyz,camera.fid.reshape(1,1).expand(len(xyz),1),feature=feature,camera_center=camera.camera_center)
        world=xyz+ds['d_xyz']
        if not with_rgb:return world,None
        rotation=build_rotation(torch.nn.functional.normalize(base_rotation+ds['d_rotation'],dim=-1));scale=base_scale+ds['d_scaling'];opacity=base_opacity
        if ds.get('d_opacity') is not None:opacity=opacity+ds['d_opacity'].squeeze(-1)
        l=directions[i];nn=normal.double();bb=basis.double();projection=torch.eye(3,device='cuda',dtype=torch.float64)-l[:,None]*nn[None]/(l@nn)
        center=(world[occ]-origin).double()@projection.T@bb.T;tangent=bb[None]@projection[None]@(rotation[occ,:,:2].double()*scale[occ,None,:].double());query=((world[gaussian_ids]-origin)@basis.T).double()
        v,_=stable_visibility(query,center,tangent,opacity[occ].double());cos=(normal_receiver[i]*l).sum(-1).clamp_min(0);pred=albedo*(energy[i].cuda()*(v*cos+ambient))[:,None]
        return world,pred
    def frame_loss(i,pred):return (pred-y[i])[mask[i]].square().sum()/(3*count)
    def scan(change,previous=None):
        total=0.;motion=[];step_motion=[]
        for i in range(135):
            world,pred=forward(i,change,i in train);motion.append(float((world-initial_world[i]).norm(dim=-1).max()))
            if previous is not None:step_motion.append(float((world-previous[i]).norm(dim=-1).max()))
            if i in train:total+=float(frame_loss(i,pred))
            cams[i].load2device('cpu')
        return total,motion,step_motion
    initial_loss,initial_motion,_=scan(delta);assert max(initial_motion)<1e-6
    history=[];fd=[];accepted=0
    for step in range(3):
        variable=delta.detach().requires_grad_(True);gradient=torch.zeros_like(delta);objective=0.
        for i in train:
            with torch.enable_grad():
                _,pred=forward(i,variable);value=frame_loss(i,pred);g,=torch.autograd.grad(value,variable)
            gradient+=g;objective+=float(value);cams[i].load2device('cpu')
        assert torch.isfinite(gradient).all();direction=-gradient/(1+metric);direction/=direction.norm(dim=-1).max().clamp_min(1e-20)
        if step==0:
            for i in [0,66,134]:
                with torch.enable_grad():
                    variable=delta.detach().requires_grad_(True);_,pred=forward(i,variable);value=frame_loss(i,pred);g,=torch.autograd.grad(value,variable)
                analytic=float((g*direction).sum())
                for eps in [1e-4,1e-5,1e-6]:
                    _,plus=forward(i,delta+eps*direction);_,minus=forward(i,delta-eps*direction);num=float((frame_loss(i,plus)-frame_loss(i,minus))/(2*eps));fd.append(dict(frame=i,epsilon=eps,analytic=analytic,numeric=num,relative_error=abs(num-analytic)/max(abs(num),abs(analytic),1e-12)))
                cams[i].load2device('cpu')
        previous=[]
        for i in range(135):
            world,_=forward(i,delta,False);previous.append(world.detach());cams[i].load2device('cpu')
        trials=[]
        for k in range(12):
            amount=1e-5*.5**k;candidate=delta+amount*direction;loss,motion,step_motion=scan(candidate,previous);ok=max(motion)<=.00100002 and max(step_motion)<=.00010002 and loss<objective-1e-12
            trials.append(dict(amount=amount,loss=loss,max_total=max(motion),max_step=max(step_motion),accepted=ok))
            if ok:delta=candidate;accepted+=1;break
        history.append(dict(step=step,before=objective,trials=trials));print(json.dumps(history[-1]),flush=True)
        if not trials[-1]['accepted']:break
    final_loss,motion,_=scan(delta);predictions=[]
    for i in range(135):
        _,pred=forward(i,delta);predictions.append(pred.cpu());cams[i].load2device('cpu')
    final_pred=torch.stack(predictions);metrics={}
    for name,inds in [('train',train),('held',held)]:metrics[name]=float((final_pred-d['rgb'][:,rid])[inds][mask.cpu()[inds]].square().mean())
    final_xyz=xyz0.index_copy(0,active,xyz0[active]+delta);inputs=dict(rt['inputs']);inputs[str(metric_path.resolve())]=hashlib.sha256(metric_path.read_bytes()).hexdigest();inputs[str(partition_path.resolve())]=hashlib.sha256(partition_path.read_bytes()).hexdigest()
    torch.save(dict(protocol='shadow_canonical_v1',geometry={'_xyz':final_xyz.cpu()},deform_state_dict={k:v.cpu() for k,v in deform.deform.state_dict().items()},args=args,active_region=a.active_region,active_ids=active.cpu(),occluder_ids=occ.cpu(),delta=delta.cpu(),prediction=final_pred,receiver_ids=rid,geometry_updates=accepted,material_updates=0),out/'checkpoint.pt')
    (out/'runtime.json').write_text(json.dumps(dict(inputs=inputs,gt_read=False,protocol='shadow_canonical_v1'),indent=2))
    result=dict(status='PASS' if accepted else 'FAILED',gt_read=False,geometry_updates=accepted,initial_training_mse=initial_loss,metrics=metrics,canonical_max_displacement=float(delta.norm(dim=-1).max()),max_world_displacement=max(motion),motion=motion,history=history,finite_difference=fd,scope='固定未知T05灯向/材质/光强，对1817个canonical XYZ经完整形变优化投影阴影；位置引起的旋转/尺度输出随完整查询变化，尚未独立几何验收')
    result.update(active_region=a.active_region,active_points=len(active),occluder_points=len(occ),scope='固定未知光及完整1817点遮挡集合，只改变指定canonical XYZ，经完整形变重新查询；未完成独立几何验收')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
