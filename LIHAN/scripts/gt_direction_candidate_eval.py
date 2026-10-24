"""TRAIN方向例外资产冻结与独立已知光照DEV/TEST评测适配，不修改训练状态。"""
import argparse, importlib, json
from pathlib import Path
import numpy as np
from LIHAN.scripts import dynamic_v3_evaluate as e
from LIHAN.scripts import gt_direction_evaluate as g


def freeze(a):
    from scripts.eval_geometry_gt import read_config
    bundle=e.source_bundle(a.source);model=a.model.resolve();config,cfg=read_config(model)
    marker_path=model/'dynamic_v3_stage2.json';marker=e.read(marker_path)
    assert marker['train_frames']==e.TRAIN and marker['external_test_read'] is False
    assert marker['gt_supervision']==['TRAIN_direction_only'] and marker['gt_geometry_supervision'] is False
    assert marker['config']['gt_direction_only'] is True
    assert marker['parent_iteration']==35000
    receipt_path=model/f'dynamic_v3/iteration_{a.iteration}/receipt.json';receipt=e.read(receipt_path)
    assert receipt['completed_updates']==a.iteration and receipt['gt_fields']==['TRAIN_direction_only']
    for rel,digest in receipt['checkpoint_sha256'].items():assert e.sha(model/rel)==digest
    lighting=model.parent/'lighting_train.json'
    if not lighting.exists():lighting=Path(marker['lighting'])
    if not lighting.exists():
        matches=list(model.parent.rglob('lighting_train.json'));assert len(matches)==1;lighting=matches[0]
    light=e.read(lighting)
    assert light['train_frames']==e.TRAIN and light['gt_direction_only'] is True and light['forbidden_gt_fields_included']==[]
    assert len(light['directions'])==117 and light['direction_audit']['heldout_directions_exported'] is False
    assert light['times']==[(n-1)/149 for n in e.TRAIN]
    paths=[cfg,receipt_path,lighting]+[model/r for r in receipt['checkpoint_sha256']]
    config.update(marker['config'])
    code=[e.ROOT/x for x in ['scripts/eval_geometry_gt.py','utils/predicted_mesh.py','gaussian_renderer/__init__.py','scene/gaussian_model.py','scene/deform_model.py','scene/cameras.py','LIHAN/scripts/dynamic_v3_radiometry.py','utils/time_utils.py','scene/pbr_rasterizer.py','scene/dataset_readers.py','utils/graphics_utils.py','utils/general_utils.py','utils/point_utils.py','utils/sh_utils.py','scene/stage2_photometric.py']]
    code += [Path(e.__file__),Path(g.__file__),Path(__file__),Path(importlib.import_module(a.stage2_module).__file__),e.ROOT/'utils/geometry_eval_utils.py',e.ROOT/'LIHAN/scripts/eldl_v3_eval.py']
    out=e.fresh(a.output)
    e.write(out/'marker_snapshot.json',marker)
    paths.append(out/'marker_snapshot.json')
    e.write(out/'freeze.json',dict(evaluation_allowed=True,selection_uses_test_or_gt=False,
        source=str(a.source.resolve()),source_sha256=e.sha(a.source),model=str(model),iteration=a.iteration,
        stage='stage2',stage2_module=a.stage2_module,source_files={str(p.resolve()):e.sha(p) for p in paths},
        code_sha256={str(p.resolve()):e.sha(p) for p in code},marker_snapshot=marker,config=config,
        mesh=dict(static_scene=False,voxel_size=.02,sdf_trunc=.08,depth_trunc=20.,alpha_threshold=.5),
        evaluation_direction_source='heldout_GT_direction_evaluation_only',
        lighting_gt_source=str(Path(light['direction_audit']['source']).resolve()),
        lighting_gt_sha256=light['direction_audit']['sha256'],dev_selection_allowed=True))


def run(a):
    import torch
    from scene.stage2_photometric import DirectionalLightTable
    frozen=e.read(a.freeze);assert frozen['evaluation_direction_source']=='heldout_GT_direction_evaluation_only'
    lights_path=Path(frozen['lighting_gt_source']);assert e.sha(lights_path)==frozen['lighting_gt_sha256']
    raw=e.read(lights_path);frames=e.DEV if a.split=='dev' else list(range(10,151,10))
    module=importlib.import_module(frozen['stage2_module']);cls=module.DynamicSurfaceModel;load=cls.load.__func__
    heldout={};captured={}
    for n in frames:
        r=raw[f'{n:04d}'];d=np.asarray(r['light_dir_world'],float);d/=np.linalg.norm(d)
        axis=np.asarray(r['lights'][r['primary_light']]['matrix_world'],float)[:3,2];axis/=np.linalg.norm(axis)
        assert np.allclose(d,axis,atol=1e-7,rtol=0);heldout[n]=d
    def loaded(klass,folder,step):
        obj=load(klass,folder,step)
        obj.radiance.lights=DirectionalLightTable(torch.tensor([(n-1)/149 for n in frames],device='cuda',dtype=torch.float32),torch.tensor(np.stack(list(heldout.values())),device='cuda',dtype=torch.float32),learned=False)
        captured['photo']=obj
        return obj
    cls.load=classmethod(loaded)
    original=e.runtime
    def runtime(f):
        model,predict=original(f)
        def wrapped(cam,c2w,k,with_rgb=True,with_separation=False):
            result=predict(cam,c2w,k,with_rgb=with_rgb,with_separation=with_separation)
            if with_rgb:
                from gaussian_renderer import render
                pred,parity=result;photo=captured['photo'];geometry=photo.effective(cam)
                pkg=render(cam,geometry,photo.pipe,photo.pc.get_xyz.new_zeros(3),0.,0.,0.,override_color=photo.radiance.rho)
                alpha=pkg['rend_alpha'].permute(1,2,0).detach().cpu().numpy()
                rho=pkg['render'].permute(1,2,0).detach().cpu().numpy()/np.maximum(alpha,1e-6)
                pred['albedo_rho_linear']=rho
            return result
        return model,wrapped
    e.runtime=runtime
    original_write=e.write
    def write(path,value):
        if Path(path).name=='protocol.json':
            value['lighting']='对应DEV/TEST GT单位方向，独立已知光照评测，无参数更新'
            value['albedo']='relative Lambertian rho，linear RGB median gauge固定，非绝对物理可辨识；完整统计稍后写入'
        original_write(path,value)
    e.write=write
    class FastComplete(BaseException):pass
    if a.fast_only:
        def no_mesh(*unused):raise FastComplete()
        e.build_meshes=no_mesh
    try:g.run(a)
    except FastComplete:pass
    e.TEST=list(range(10,151,10))
    out=a.output
    receipt=dict(direction_source=str(lights_path),source_sha256=e.sha(lights_path),frames=frames,
        directions={str(n):d.tolist() for n,d in heldout.items()},training_updated=False,known_lighting_evaluation=True,
        note='仅评测帧GT单位方向；无GT光强/环境/材质；geometry TSDF不受灯表影响')
    e.write(out/'heldout_lighting_receipt.json',receipt)
    p=out/('fast_metrics.json' if a.fast_only else 'metrics.json');metrics=e.read(p);metrics['protocol']['lighting']='对应DEV/TEST GT单位方向，独立已知光照评测，无参数更新'
    # GT albedo PNG为sRGB unlit base color；AREA到400，linear RGB，固定GT前景缺失预测罚1/通道。
    bundle=e.source_bundle(frozen['source']);rows=[]
    for n in frames:
        with np.load(out/'arrays'/f'prediction_{n:04d}.npz') as z:
            pred=z['albedo_rho_linear'];alpha=z['alpha'];gt_alpha=z['source_alpha']
        path=Path(bundle['raw'])/f'albedo/albedo_{n:04d}.png';rgba=g.cv2.imread(str(path),g.cv2.IMREAD_UNCHANGED)
        assert rgba is not None and rgba.dtype==np.uint16 and rgba.shape==(800,800,4)
        rgb=rgba[...,[2,1,0]].astype(float)/65535.;gt=g.cv2.resize(e.linearize(rgb),(400,400),interpolation=g.cv2.INTER_AREA)
        eligible=gt_alpha>=.5;valid=eligible&(alpha>=.5)&np.isfinite(pred).all(-1)
        err=np.abs(pred-gt).mean(-1);count=int(eligible.sum());v=int(valid.sum())
        rows.append(dict(frame=n,gt_pixels=count,valid_pixels=v,coverage=v/count,conditional_mae=float(err[valid].mean()),fixed_missing1_mae=float((err[valid].sum()+count-v)/count),source=str(path),source_sha256=e.sha(path)))
    metrics['albedo']=dict(status='AVAILABLE_RELATIVE_RHO',meaning='逐Gaussian Lambertian rho光栅除预测Alpha，linear RGB；模型median gauge固定，非绝对可辨识物理反照率；无GT拟合尺度',
        gt='Blender unlit base-color PNG按sRGB逆变换后AREA400；alpha>=.5固定前景，缺失罚1/通道',
        conditional_frame_mean=float(np.mean([r['conditional_mae'] for r in rows])),fixed_missing1_frame_mean=float(np.mean([r['fixed_missing1_mae'] for r in rows])),rows=rows)
    if a.fast_only:
        for value in metrics['summary'].values():
            value['normal'].pop('mesh',None)
            value['depth'].pop('mesh_depth',None)
            value.pop('common',None)
        for row in metrics['rows']:
            for region in row['regions'].values():
                region['normal'].pop('mesh',None)
                region['depth'].pop('mesh_depth',None)
                region.pop('common',None)
        metrics['mesh_status']='PENDING_NOT_EVALUATED'
        e.write(out/'FAST_COMPLETED.json',dict(status='GS_DEPTH_RGB_ALBEDO_ONLY',mesh='PENDING_NOT_EVALUATED',frames=frames))
    e.write(p,metrics)
    with (out/'README.md').open('a') as s:s.write('\n光照：对应帧GT方向的已知光照独立评测，未更新训练模型。证据heldout_lighting_receipt.json。Albedo为relative Lambertian rho、linear RGB，median gauge非绝对可辨识；完整统计metrics.json。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['freeze','run'],required=True);p.add_argument('--source',type=Path);p.add_argument('--model',type=Path);p.add_argument('--iteration',type=int);p.add_argument('--stage2-module',default='LIHAN.scripts.dynamic_v3_surface');p.add_argument('--fast-only',action='store_true');p.add_argument('--freeze',type=Path);p.add_argument('--split',choices=['dev','test']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.action=='freeze':freeze(a)
    else:run(a)
if __name__=='__main__':main()
