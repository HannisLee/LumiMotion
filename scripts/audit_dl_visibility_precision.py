"""固定T62缓存，对照投影运算顺序、精度和局部坐标解；不更新参数。"""
import argparse,json,hashlib
from pathlib import Path
import torch
from scripts.audit_dl_gaussian_visibility import visibility

def stable_visibility(query,center,tangent,opacity):
    # 直接解投影切平面坐标，避免协方差求逆后二次型消减。
    a,b,c,d=tangent[:,0,0],tangent[:,0,1],tangent[:,1,0],tangent[:,1,1]
    det=a*d-b*c;trace=tangent.square().sum((-1,-2))
    largest=.5*(trace+torch.sqrt((trace.square()-4*det.square()).clamp_min(0)))
    keep=det.square()/largest.clamp_min(1e-30)>1e-12
    a,b,c,d,det,center,opacity=[x[keep] for x in (a,b,c,d,det,center,opacity)]
    values=[]
    for q in query.split(64):
        delta=q[:,None]-center[None];dx,dy=delta[...,0],delta[...,1]
        u=(d*dx-b*dy)/det;v=(-c*dx+a*dy)/det;distance=u.square()+v.square()
        alpha=torch.where(distance<=9,opacity[None]*torch.exp(-.5*distance),0.).clamp(0,.99)
        values.append(torch.exp(torch.log1p(-alpha).sum(-1)))
    return torch.cat(values),int((~keep).sum())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4);root=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False)
    ck=torch.load(root/'train/checkpoint.pt',map_location='cpu',weights_only=False);meta=json.loads((root/'train/result.json').read_text());assert not meta['gt_read']
    cache=torch.load('output/1015-07-LH-dynamic-DL-T02表面轨迹初始化/cache/tracks.pt',weights_only=False);shadow=torch.load('output/1015-15-LH-dynamic-DL-T05复核/seed_1015/checkpoint.pt',weights_only=False)
    d=torch.load('output/smoke_test/1015-50-LH-dynamic-DL-T58接收平面可见性/audit/predictions.pt',weights_only=False);energy=torch.load('output/smoke_test/1015-49-LH-dynamic-DL-T57底项约定修正配对/supported/checkpoint.pt',weights_only=False)['energy'];ambient=json.loads(Path('output/smoke_test/1015-49-LH-dynamic-DL-T57底项约定修正配对/audit/result.json').read_text())['ambient']
    n=shadow['plane_normal'].cuda();basis=shadow['basis'].cuda();rid=ck['receiver_ids'];mask=torch.zeros(len(ck['geometry'][0][0]),dtype=torch.bool,device='cuda');mask[cache['ids'][shadow['occluder']]]=True
    modes=['native','center_subset','scale_first','both','double_native','double_both'];predictions={k:[] for k in modes};rows=[];fd=[]
    for i in range(135):
        pos,rot,scale,op,q=[x.cuda() for x in ck['geometry'][i]];l=ck['initial_directions'][i].cuda();alb=ck['albedo'].cuda()
        def pred(direction,mode):
            double=mode.startswith('double');dtype=torch.float64 if double else torch.float32
            ll=direction.to(dtype);nn=n.to(dtype);bb=basis.to(dtype);pp=pos.to(dtype);rr=rot.to(dtype);ss=scale.to(dtype)
            proj=torch.eye(3,device='cuda',dtype=dtype)-ll[:,None]*nn[None]/(ll@nn)
            subset=mode in ['center_subset','both','double_both'];early=mode in ['scale_first','both','double_both']
            center=(pp[mask] if subset else pp)@proj.T@bb.T
            if not subset:center=center[mask]
            tangent=(bb[None]@proj[None]@(rr*ss[:,None,:])) if early else (bb[None]@proj[None]@rr)*ss[:,None,:]
            v,_=stable_visibility(q.double(),center,tangent[mask],op.double()) if double else visibility.__wrapped__(q,center,tangent[mask],op)
            cos=(cache['normals'][i,rid].cuda().double()*ll.double()).sum(-1).clamp_min(0)
            return alb*(energy[i].cuda()*(v.double()*cos+ambient))[:,None]
        with torch.no_grad():
            current={mode:pred(l,mode).cpu() for mode in modes}
            assert (current['native']-ck['initial_prediction'][i]).abs().max()<1e-12
            row={'frame':i}
            support=d['groups']['all_unsaturated']['mask'][i,rid]
            for mode in modes:
                predictions[mode].append(current[mode]);delta=(current[mode]-current['native']).abs();row[mode]={'max_delta':float(delta.max()),'supported_max_delta':float(delta[support].max()),'observations_above_1e_3':int((delta.max(-1).values[support]>.001).sum())}
            row['double_reorder_max']=float((current['double_native']-current['double_both']).abs().max());rows.append(row);print(json.dumps(row),flush=True)
        if i in [0,66,134]:
            ll=l.double().detach().requires_grad_(True)
            def loss(x):return (pred(x,'double_native')-d['rgb'][i,rid].cuda())[support.cuda()].square().mean()
            value=loss(ll);grad,=torch.autograd.grad(value,ll);direction=-(grad-ll*(grad@ll));direction/=direction.norm()
            for eps in [1e-3,1e-4,1e-5,1e-6,1e-7]:
                with torch.no_grad():numeric=float((loss(torch.nn.functional.normalize(ll+eps*direction,dim=0))-loss(torch.nn.functional.normalize(ll-eps*direction,dim=0)))/(2*eps))
                analytic=float(grad@direction);fd.append(dict(frame=i,epsilon=eps,numeric=numeric,analytic=analytic,relative_error=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-12)))
    mask_support=d['groups']['all_unsaturated']['mask'][:,rid];stats={};saved={}
    for mode,parts in predictions.items():
        prediction=torch.stack(parts);saved[mode]=prediction;stats[mode]={}
        for split,ts in [('train',d['train']),('held',d['held'])]:
            stats[mode][split]=float((prediction-d['rgb'][:,rid])[ts][mask_support[ts]].square().mean())
    torch.save(dict(predictions=saved,rows=rows,fd=fd),out/'checkpoint.pt')
    result=dict(status='PASS',gt_read=False,updates=0,statistics=stats,finite_difference=fd,max_double_reorder=max(x['double_reorder_max'] for x in rows),max_native_reorder={m:max(x[m]['max_delta'] for x in rows) for m in modes},scope='固定T62初始灯向/材质，不拟合；双精度同时改为投影切平面直接坐标解，收益不代表几何改善',source_hashes={str(root/'train/checkpoint.pt'):hashlib.sha256((root/'train/checkpoint.pt').read_bytes()).hexdigest(),__file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
