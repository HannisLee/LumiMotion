"""固定已登记灯向，检验默认动态几何的遮挡是否改善光度模型。"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from utils.general_utils import build_scaling_rotation
from utils.dl_selfcal import SurfaceLightFit, observation_loss
from scripts.probe_dynamic_dl_selfcal import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--unknown-result', required=True)
    parser.add_argument('--reference-result', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=600)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'visibility.pt').exists():
        raise FileExistsError('禁止覆盖已有遮挡诊断。')
    started = time.monotonic()
    data = torch.load(args.cache, map_location='cpu', weights_only=False)
    old = data['config']
    cfg = read_config(Path(old['model'])/'cfg_args')
    cfg.resolution, cfg.load2gpu_on_the_fly = 8, True
    pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
    pc.load_ply(str(Path(old['model'])/f"point_cloud/iteration_{old['iteration']}/point_cloud.ply"))
    deform = DeformModel(deform_type=cfg.deform_type, is_blender=cfg.is_blender,
                         hyper_dim=cfg.hyper_dim, pred_color=cfg.pred_color)
    assert deform.load_weights(old['model'], old['iteration'])
    deform.deform.eval()
    results = [json.loads(Path(path).read_text()) for path in [args.unknown_result, args.reference_result]]
    cache_hash = hashlib.sha256(Path(args.cache).read_bytes()).hexdigest()
    for r in results:
        assert r['cache_sha256'] == cache_hash
        assert r['frames'] == [f['name'] for f in data['frames']]
    if results[0].get('gt_direction_reference', False) or not results[1]['gt_direction_reference']:
        raise ValueError('必须分别传入未知光结果与明确GT参考结果。')
    directions = torch.tensor([r['directions'] for r in results], device='cuda')
    infos = readCamerasFromTransforms(old['source'], 'transforms_train.json', cfg.train_light_folder, False)
    ids = data['ids'].cuda()
    visibility = []
    diagnostics = []
    with torch.no_grad():
        for index, info in enumerate(infos):
            camera = loadCam(cfg, index, info, 1.)
            camera.load2device()
            ds = deform.step(pc.get_xyz, camera.fid.reshape(1, 1).expand(len(pc.get_xyz), 1),
                             feature=pc.get_binary_feature(), camera_center=camera.camera_center)
            xyz = pc.get_xyz + ds['d_xyz']
            scales = pc.get_scaling + ds['d_scaling']
            rotation = pc.get_rotation_bias(ds['d_rotation'])
            opacity = pc.get_opacity if ds['d_opacity'] is None else pc.get_opacity + ds['d_opacity']
            if not torch.isfinite(xyz).all() or not torch.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError('变形几何含非有限／非正scale。')
            if (opacity < 0).any() or (opacity > 1).any():
                raise ValueError('变形opacity越界，不静默更改。')
            position_error = float((xyz[ids]-data['positions'][index].cuda()).abs().max())
            if position_error > 1e-5:
                raise ValueError('实际几何与缓存表面位置不一致。')
            scale3 = torch.cat([scales, torch.full_like(scales[:, :1], 1e-6)], -1)
            transform = build_scaling_rotation(scale3, rotation)
            radius = (2*(opacity.clamp_min(1e-12)/pc.alpha_min).log().clamp_min(0)).sqrt()[:, None]
            vertices = radius*(pc.unit_icosahedron_vertices[None]@transform.transpose(-1,-2))+xyz[:,None]
            faces = pc.unit_icosahedron_faces[None]+torch.arange(len(xyz),device='cuda')[:,None,None]*12
            gaussian_ids = torch.arange(len(xyz),device='cuda')[:,None].expand(-1,faces.shape[1])
            pc.gaussian_tracer.build_bvh(vertices.reshape(-1,3),faces.reshape(-1,3),gaussian_ids.reshape(-1))
            incident = directions[:,index][None].expand(len(ids),-1,-1).contiguous()
            traced = pc.trace(xyz[ids,None]+incident*.03,incident,
                              features=torch.zeros(len(xyz),4,device='cuda'),xyz=xyz,
                              scales=scales,rotation=rotation,opacity=opacity,use_zeros=True)
            value = (1-traced['alpha']).clamp(0,1).T
            if not torch.isfinite(value).all():
                raise ValueError('可见度非有限。')
            visibility.append(value.cpu())
            valid = data['valid'][index].cuda()
            record = dict(frame=info.image_name_train_light, position_error=position_error,
                          valid_visibility_means=value[:,valid].mean(-1).tolist(),
                          valid_occluded_fraction=(value[:,valid]<.1).float().mean(-1).tolist())
            diagnostics.append(record)
            print(json.dumps(record),flush=True)
    visibility = torch.stack(visibility,1)
    torch.save(dict(visibility=visibility,directions=directions.cpu(),diagnostics=diagnostics,
                    cache_sha256=cache_hash,bias=.03,labels=['unknown_fixed','gt_reference_fixed']),out/'visibility.pt')
    times,normals,rgb,saturated,valid = [data[k].cuda() for k in ['times','normals','rgb','saturated','valid']]
    train = torch.arange(len(times),device='cuda')%9!=4
    support = (valid[train].sum(0)>=3)&((valid&~saturated.any(-1))[train].sum(0)>=1)
    valid = valid & support[None]
    fits={}
    for source_index,label in enumerate(['unknown_fixed','gt_reference_fixed']):
        for use_visibility in [False,True]:
            name=label+('_visibility' if use_visibility else '_no_visibility')
            model=SurfaceLightFit(times[train],directions[source_index,train],len(ids)).cuda()
            model.raw_directions.requires_grad_(False)
            factor=visibility[source_index].cuda() if use_visibility else torch.ones_like(valid,dtype=torch.float32)
            optimizer=torch.optim.Adam([{'params':[model.log_energy],'lr':.01},{'params':[model.raw_albedo],'lr':.02}])
            def prediction():
                # 全部时间严格复用旧结果中的方向；不再次插值已经插值过的保留方向。
                _,energy=model.lighting(times)
                cosine=(normals*directions[source_index,:,None]).sum(-1).clamp_min(0)
                return torch.nn.functional.softplus(model.raw_albedo)[None]*cosine[...,None]*energy[:,None,None]*factor[...,None]
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                pred=prediction()
                loss=observation_loss(pred[train],rgb[train],saturated[train],valid[train])+.001*model.regularizer()
                if not torch.isfinite(loss):raise RuntimeError('非有限拟合目标。')
                loss.backward();optimizer.step()
            with torch.no_grad():
                pred=prediction()
                metrics={}
                for split,mask in [('fit',train),('held_time',~train)]:
                    metrics[split]=float(observation_loss(pred[mask],rgb[mask],saturated[mask],valid[mask]))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),directions=directions[source_index],
                            visibility=factor,cache_sha256=cache_hash,updates=args.steps),out/(name+'.pt'))
            fits[name]=metrics
            print(json.dumps(dict(name=name,**metrics)),flush=True)
    result=dict(config=vars(args),fits=fits,diagnostics=diagnostics,actual_updates=4*args.steps,
                seconds=time.monotonic()-started,scope='固定灯向诊断，不是未知光重新优化',
                cache_sha256=cache_hash,source_result_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.unknown_result,args.reference_result]},
                scientific_status='待分析；几何与灯向均未更新')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
