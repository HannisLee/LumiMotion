#!/usr/bin/env python3
"""新独立进程比较实际TRAIN前向11maps与精确保存；不复用旧终点门禁。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal4_appearance_eval_common import ROOT,BASE,MAPS,audit,destination,checked,unchanged,frozen_hashes
from goal_eval_protocol import CONTROL,read,write,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model_path',required=True);p.add_argument('--load_iter',type=int,required=True);p.add_argument('--source_path',required=True);p.add_argument('--output_dir',type=Path,required=True);a=p.parse_args()
    out=destination(a.output_dir);e=audit(a.model_path,a.load_iter);marker=e['marker'];sources=e['sources'];cfg=marker['inherited_config']
    if Path(a.source_path).resolve()!=Path(e['protocol']['source_path']):raise AssertionError('fresh须原eval source。')
    import numpy as np,torch
    from argparse import Namespace
    from scripts.eval_geometry_gt import GeometryRun
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos
    from scene.static_dl_photometric import DirectionOnlyLights,StaticVisibility
    from scene.static_dl_ambient import render_photometric
    from goal4_appearance_eval_render import physical_renderer
    import diff_surfel_rasterization as native
    backend=getattr(native,'provenance',None);trained=read(Path(a.model_path)/'static_dl_training.json')['rasterizer_provenance']
    if not backend or backend['method']!='repository_lumimotion_cuda_rasterizer':raise AssertionError('必须bootstrap原生光栅。')
    for k in ('method','source_sha256','extension_sha256','source_files'):
        if trained[k]!=backend[k]:raise AssertionError('训练与真实fresh native不同：'+k)
    checked(backend['extension_path'],sources,trained['extension_sha256'])
    # TRAIN相机使用原真实读取器与参数，不用GeometryRun中用于离线GT的空RGB相机。
    pipe=Namespace(**cfg);infos=readCamerasFromTransforms(cfg['source_path'],'transforms_train.json',cfg['train_light_folder'],False,no_bg=True)
    cameras=cameraList_from_camInfos(infos,1.,pipe);camera=cameras[0];camera.load2device()
    for label in ('initial','iteration_'+str(a.load_iter)):
        meta=read(Path(a.model_path)/f'appearance_forward/{label}.json')
        if meta['camera']!=camera.image_name_train_light:raise AssertionError('真实first TRAIN相机不同。')
    run=GeometryRun(a.model_path,a.load_iter,a.source_path);render,prov=physical_renderer(run,a.load_iter,CONTROL/'eval_directions_only_single.json')
    with torch.no_grad():package=render(camera)
    comparisons={};arrays={}
    def compare(label,package,reference):
        values={}
        for k in MAPS:
            actual=package[k].detach().cpu().numpy();expected=reference[k]
            shape=actual.shape==expected.shape;finite=np.isfinite(actual).all() and np.isfinite(expected).all()
            error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))) if shape and actual.size else 0.
            # 同GPU/native下允许不可重现舍入，实际max逐项保存；不以该tol放宽法线科学门槛。
            passed=shape and actual.dtype==expected.dtype and finite and error<=2e-6
            values[k]=dict(passed=bool(passed),max_abs=error,dtype=str(actual.dtype),shape=list(actual.shape))
            arrays[label+'_'+k]=actual
        comparisons[label]=dict(status='PASS' if all(x['passed'] for x in values.values()) else 'FAILED',checks=values)
    with np.load(Path(a.model_path)/f'appearance_forward/iteration_{a.load_iter}.npz') as old:compare('checkpoint_training_to_fresh',package,old)
    with torch.no_grad():saved_ply_package=render.saved_ply_renderer(camera)
    compare('saved_ply_to_rebuilt_state',saved_ply_package,{k:package[k].detach().cpu().numpy() for k in MAPS})
    # 新独立进程实际重载T06，并在同TRAIN相机复现最初11maps；resume的initial由精确恢复state检查。
    if marker['appearance_updates_before']==0:
        base=GeometryRun(BASE,16000,a.source_path)
        from goal2_eval_reset_audit_render import physical_renderer as base_renderer
        original,_=base_renderer(base,16000,CONTROL/'eval_directions_only_single.json')
        with torch.no_grad():first=original(camera)
        with np.load(Path(a.model_path)/'appearance_forward/initial.npz') as old:compare('t06_to_actual_initial',first,old)
        sources.update(base.files)
    else:
        state_path=Path(marker['args']['resume_state']).resolve();state=torch.load(state_path,map_location='cpu',weights_only=True)
        with torch.no_grad():run.pc._albedo_dc.copy_(state['albedo_dc'].to('cuda'));render.eval_lights.log_intensity.copy_(state['log_intensity'].to('cuda'));first=render(camera)
        with np.load(Path(a.model_path)/'appearance_forward/initial.npz') as old:compare('resume_to_actual_initial',first,old)
        checked(state_path,sources)
    sources.update(run.files);checked(Path(__file__),sources)
    unchanged(sources);out.mkdir(parents=True)
    np.savez_compressed(out/'fresh_forward.npz',**arrays)
    status='PASS' if all(x['status']=='PASS' for x in comparisons.values()) else 'FAILED'
    report=dict(status=status,iteration=a.load_iter,point_count=len(run.pc.get_xyz),comparisons=comparisons,parameter_updates=0,
        frozen_sha256=frozen_hashes(run.pc),sources=sources,native=backend,physical_renderer=prov,
        scope='真实初始及保存重载前向；不等于GPU反向、PSNR提升或完整脚形成功。',argv=sys.argv)
    write(out/'parity.json',report)
    (out/'README.md').write_text('# 外观阶段独立fresh\n\n真实TRAIN相机，11maps、起点T06/恢复state和已保存checkpoint比较。来源与几何SHA前后不变；逐map max差与native见parity.json。最终'+status+'。本结果不代表科学目标完成。\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','comparisons','iteration')},ensure_ascii=False,indent=2))
    if status!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
