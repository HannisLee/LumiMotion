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

@torch.no_grad()
def visibility(query, center, tangent, opacity):
    # 平面坐标上的投影椭圆；非透视光源，未添加像素低通或形态学扩张。
    cov=tangent.double()@tangent.double().transpose(-1,-2)
    aa,bb,cc=cov[:,0,0],cov[:,0,1],cov[:,1,1]
    determinant=aa*cc-bb.square()
    largest=.5*(aa+cc+torch.sqrt((aa-cc).square()+4*bb.square()))
    keep=(determinant/largest.clamp_min(1e-30))>1e-12
    inv=torch.stack((cc,-bb,-bb,aa),-1).view(-1,2,2)/determinant.clamp_min(1e-30)[:,None,None]
    center,inv,opacity=center[keep],inv[keep].float(),opacity[keep]
    values=[]
    for q in query.split(64):
        delta=q[:,None]-center[None]
        distance=torch.einsum('qni,nij,qnj->qn',delta,inv,delta).clamp_min(0)
        alpha=opacity[None]*torch.exp(-.5*distance)
        alpha=torch.where(distance<=9,alpha,0.).clamp(0,.99)
        values.append(torch.exp(torch.log1p(-alpha).sum(-1)))
    return torch.cat(values),int((~keep).sum())

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['run','audit','output']:p.add_argument('--'+k,required=True)
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
    outputs={k:[] for k in masks}; checks=[]
    # 单Gaussian中心及1sigma、3sigma外部的解析值校验。
    test,_=visibility(torch.tensor([[0.,0.],[1.,0.],[4.,0.]],device='cuda'),torch.zeros(1,2,device='cuda'),torch.eye(2,device='cuda')[None],torch.tensor([.5],device='cuda'))
    assert torch.allclose(test,torch.tensor([.5,1-.5*torch.exp(torch.tensor(-.5)),1.],device='cuda'),atol=1e-6)
    for i,c in enumerate(cams):
        c.load2device();ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(full),1),feature=pc.get_binary_feature(),camera_center=c.camera_center)
        world=pc.get_xyz+ds['d_xyz'];direction=state['smooth'][i].cuda();den=direction@normal;assert den>.001
        projection=torch.eye(3,device='cuda')-direction[:,None]*normal[None]/den
        center=(world-origin)@projection.T@basis.T
        rotation=build_rotation(pc.get_rotation_bias(ds['d_rotation']))
        scales=pc.get_scaling+ds['d_scaling'];tangent=basis[None]@projection[None]@rotation[:,:,:2]*scales[:,None,:]
        opacity=pc.get_opacity.squeeze(-1)
        if ds.get('d_opacity') is not None:opacity=opacity+ds['d_opacity'].squeeze(-1)
        assert torch.isfinite(tangent).all() and torch.isfinite(opacity).all()
        err=float((world[ids]-cache['positions'][i].cuda()).abs().max());assert err<1e-5
        query=(world[gaussian_ids]-origin)@basis.T
        row={'frame':i,'cached_position_max_error':err}
        for name,mask in masks.items():
            v,removed=visibility(query,center[mask],tangent[mask],opacity[mask]);outputs[name].append(v.cpu());row[name+'_singular_removed']=removed
        checks.append(row);c.load2device('cpu');print(json.dumps(row),flush=True)
    train,held=d['train'],d['held'];y=d['rgb'][:,rid];mask=d['groups']['all_unsaturated']['mask'][:,rid]
    cosine=(cache['normals'][:,rid].double()*state['smooth'].double()[:,None]).sum(-1).clamp_min(0)
    stats={};saved={}
    for name,items in outputs.items():
        v=torch.stack(items).double();s=energy[:,None]*(v*cosine+ambient)
        albedo=(y[train]*s[train,:,None]*mask[train,:,None]).sum(0)/(s[train].square()*mask[train]).sum(0).clamp_min(1e-12)[:,None]
        pred=s[...,None]*albedo[None];rows={}
        for split,ts in [('train',train),('held',held)]:
            for group,extra in [('all',torch.ones_like(mask)),('bright',state['negative'][:,rid]),('dark',state['positive'][:,rid]),('strict',d['groups']['strict_bright']['mask'][:,rid])]:
                m=(mask&extra)[ts];rows[split+'_'+group]={'count':int(m.sum()),'mse':float((pred-y)[ts][m].square().mean()),'mean_visibility':float(v[ts][m].mean())}
        stats[name]=rows;saved[name]={'visibility':v,'prediction':pred,'albedo':albedo}
    torch.save({'groups':saved,'receiver_ids':rid,'checks':checks},out/'checkpoint.pt')
    result=dict(status='PASS',gt_read=False,parameter_updates=0,analytic_fits=2,statistics=stats,occluders={k:int(v.sum()) for k,v in masks.items()},scope='固定默认旋转头及尺度的正交投影3sigma椭圆透射代理；不等于物理真值或完整ray tracing；无几何更新',inputs=rt['inputs'],source_hashes=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
