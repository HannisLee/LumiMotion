"""隔离GT方向参考：固定粗几何与能量的阴影诊断，禁止回灌未知光主线。"""
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
    for k in ['run','audit','reference','candidate','gt','output']:p.add_argument('--'+k,required=True)
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
    # 独立参考组：只有此脚本读取GT方向，所有输出显式标记oracle。
    gt=json.loads(Path(a.gt).read_text())['frames'];oracle=[]
    for name in light['frames']:
        key=str(int(name.rsplit('_',1)[-1])).zfill(4);ls=gt[key]['lights'];dirs=[]
        for lamp in ls:
            assert lamp['type']=='directional' and 'surface toward light' in lamp['direction_convention']
            direction=torch.tensor(lamp['light_dir_world'],dtype=torch.float64);dirs.append(direction/direction.norm())
        assert all(torch.allclose(v,dirs[0],atol=1e-5) for v in dirs);oracle.append(dirs[0])
    oracle=torch.stack(oracle).cuda();unknown=state['smooth'].double().cuda()
    reference=Path(a.reference);ref=torch.load(reference/'train/checkpoint.pt',map_location='cpu',weights_only=False)
    candidate=Path(a.candidate);cand=torch.load(candidate/'train/checkpoint.pt',map_location='cpu',weights_only=False);cm=json.loads((candidate/'train/result.json').read_text());assert not cm['gt_read'] and cand['protocol']=='shadow_canonical_v1';assert torch.equal(cand['receiver_ids'],rid)
    occ=torch.where(sampled)[0];assert torch.equal(cand['occluder_ids'],occ.cpu());feature=pc.get_binary_feature().detach();xyz0=pc.get_xyz.detach().clone();base_rotation=pc._rotation.detach();base_scale=pc.get_scaling.detach();base_opacity=pc.get_opacity.detach().squeeze(-1)
    assert all(torch.equal(v.cpu(),cand['deform_state_dict'][k]) for k,v in deform.deform.state_dict().items())
    n=normal.double();b=basis.double();visibility={};shadings={};counts={};max_reproduction=0.
    for shape in ['initial','candidate']:
        values={name:[] for name in ['unknown','oracle']}
        for i,c in enumerate(cams):
            if shape=='initial':
                pos,rot,scale,opacity,query=ref['geometry'][i];pos=pos[occ.cpu()].double().cuda();tan=(rot[occ.cpu()].double()*scale[occ.cpu(),None,:].double()).cuda();opacity=opacity.double().cuda();query=query.double().cuda()
            else:
                c.load2device();xyz=cand['geometry']['_xyz'].cuda();ds=deform.step(xyz,c.fid.reshape(1,1).expand(len(xyz),1),feature=feature,camera_center=c.camera_center);world=xyz+ds['d_xyz'];rot=build_rotation(torch.nn.functional.normalize(base_rotation+ds['d_rotation'],dim=-1));scale=base_scale+ds['d_scaling'];opacity=base_opacity
                if ds.get('d_opacity') is not None:opacity=opacity+ds['d_opacity'].squeeze(-1)
                pos=(world[occ]-origin).double();tan=rot[occ,:,:2].double()*scale[occ,None,:].double();opacity=opacity[occ].double();query=((world[gaussian_ids]-origin)@basis.T).double();c.load2device('cpu')
            for name,dirs in [('unknown',unknown),('oracle',oracle)]:
                l=dirs[i];assert l@n>.001;proj=torch.eye(3,dtype=torch.float64,device='cuda')-l[:,None]*n[None]/(l@n);v,removed=stable_visibility(query,pos@proj.T@b.T,b[None]@proj[None]@tan,opacity);values[name].append(v.cpu());counts[shape+'/'+name]=max(counts.get(shape+'/'+name,0),removed)
            if i%30==0:print(json.dumps(dict(shape=shape,frame=i)),flush=True)
        for name,parts in values.items():
            visibility[shape+'/'+name]=torch.stack(parts);dirs=unknown.cpu() if name=='unknown' else oracle.cpu();cos=(cache['normals'][:,rid].double()*dirs[:,None]).sum(-1).clamp_min(0);shadings[shape+'/'+name]=energy[:,None]*(visibility[shape+'/'+name]*cos+ambient)
    y=d['rgb'][:,rid];mask=d['groups']['all_unsaturated']['mask'][:,rid];train=d['train'];held=d['held'];statistics={};saved={}
    for name in ['unknown','oracle']:
        shading=shadings['initial/'+name];refit=(y[train]*shading[train,:,None]*mask[train,:,None]).sum(0)/(shading[train].square()*mask[train]).sum(0).clamp_min(1e-12)[:,None]
        altered=y.clone();altered[held]=.123;check=(altered[train]*shading[train,:,None]*mask[train,:,None]).sum(0)/(shading[train].square()*mask[train]).sum(0).clamp_min(1e-12)[:,None];assert torch.equal(check,refit)
        for material,albedo in [('frozen_T61',ref['albedo']),('initial_refit',refit)]:
            for shape in ['initial','candidate']:
                key=name+'/'+material+'/'+shape;pred=albedo[None]*shadings[shape+'/'+name][...,None];rows={}
                for split,ts in [('train',train),('held',held)]:
                    for label,extra in [('all',torch.ones_like(mask)),('bright',state['negative'][:,rid]),('dark',state['positive'][:,rid])]:
                        selected=(mask&extra)[ts];rows[split+'_'+label]=dict(count=int(selected.sum()),mse=float((pred-y)[ts][selected].square().mean()))
                statistics[key]=rows;saved[key]=pred
                if key=='unknown/frozen_T61/candidate':
                    max_reproduction=float((pred-cand['prediction']).abs().max());assert max_reproduction<1e-8,max_reproduction
    torch.save(dict(oracle_directions=oracle.cpu(),unknown_directions=unknown.cpu(),visibility=visibility,predictions=saved,receiver_ids=rid,gt_read=True,reference_only=True),out/'ORACLE_diagnostics.pt')
    result=dict(status='PASS',gt_read=True,reference_only=True,geometry_updates=0,analytic_material_fits=2,light_energy='固定未知光主线T57估计，未使用GT能量',material_protocol='初始几何训练RGB解析拟合一次，candidate评测冻结该材质，不重拟合',candidate_prediction_reproduction=max_reproduction,max_singular_removed=counts,statistics=statistics,scope='隔离GT方向参考，未使用GT几何或能量；不是未知光恢复、不是严格upper bound，不回灌主线',source_hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in [a.gt,str(candidate/'train/checkpoint.pt'),str(reference/'train/checkpoint.pt'),__file__]})
    (out/'ORACLE_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
