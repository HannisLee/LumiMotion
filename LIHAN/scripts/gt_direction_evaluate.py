"""GT方向续训独立评测；DEV可选模，TEST仅冻结后验收。"""
from pathlib import Path
import argparse, json, os, shlex, sys, traceback
import cv2
import numpy as np
from LIHAN.scripts import dynamic_v3_evaluate as e


def rgb_extra(pred, target, alpha):
    # Wang Gaussian SSIM：11x11 sigma1.5，population covariance，裁去5像素边缘。
    x=target.astype(np.float64);y=pred.astype(np.float64)
    blur=lambda v:cv2.GaussianBlur(v,(11,11),1.5,borderType=cv2.BORDER_REFLECT)
    ux,uy=blur(x),blur(y)
    vx,vy=blur(x*x)-ux*ux,blur(y*y)-uy*uy
    cov=blur(x*y)-ux*uy
    score=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return dict(ssim_linear_full=float(score[5:-5,5:-5].mean()))



def visuals(out, n, pred, target, gt, regions):
    folder=out/'four_classes';folder.mkdir(exist_ok=True)
    visible=pred['alpha']>=.5
    normal=lambda x,mask:np.where(mask[...,None],x*.5+.5,0)
    gs=normal(pred['normal'],visible);gn=normal(gt['gt_exr_normal'],gt['gt_exr_valid'])
    err,_=e.normal_errors(pred['normal'],gt['gt_exr_normal'])
    error=e.scalar_display(err['angular_deg'],visible,60.)
    missing=(gt['alpha']>=.5)&gt['valid']&~visible;error[missing]=[1.,0.,1.]
    panels={
        'rgb':[e.panel('GT linear -> sRGB',e.display_rgb(target)),e.panel('Prediction',e.display_rgb(pred['rgb_linear'])),e.panel('Alpha',np.repeat(pred['alpha'][...,None],3,2))],
        'normal':[e.panel('GT',gn),e.panel('GS',gs),e.panel('Depth',normal(pred['depth_normal'],visible)),e.panel('GS error 0..60; missing magenta',error)],
        'depth_mesh':[e.panel('Expected Z 0..10',e.scalar_display(pred['expected'],visible,10.)),e.panel('Mesh Z 0..10',e.scalar_display(pred['mesh_depth'],pred['mesh_valid'],10.)),e.panel('Mesh normal',normal(pred['mesh_normal'],pred['mesh_valid']))],
        'separation':[e.panel('Red dynamic / green static; config gate',pred['separation'])]}
    if 'environment_linear' in pred:
        panels['components']=[e.panel('Environment',e.display_rgb(pred['environment_linear'])),e.panel('Direct',e.display_rgb(pred['direct_linear']))]
    for name,items in panels.items():e.save_visual(folder/f'{name}_{n:04d}.png',np.concatenate(items,axis=1))
    crop=out/'review_crops';crop.mkdir(exist_ok=True)
    e.local_review_crops(crop,n,pred,target,gt,normal(pred['mesh_normal'],pred['mesh_valid']),e.scalar_display(pred['mesh_depth'],pred['mesh_valid'],10.),error)


def run(args):
    import torch
    from utils.predicted_mesh import PredictedMesh
    frozen=e.read(args.freeze);e.verify(frozen['source_files']);e.verify(frozen['code_sha256'])
    bundle=e.source_bundle(frozen['source']);split=args.split
    frames=e.DEV if split=='dev' else list(range(10,151,10))
    # base代码全局TEST只用于mesh目标时刻；source验证在此修改前完成。
    e.TEST=frames
    out=e.fresh(args.output)
    command=shlex.join(['env','CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES','')]+list(getattr(sys,'orig_argv',[sys.executable]+sys.argv)))
    (out/'README.md').write_text('独立'+split.upper()+'评测\n\n完整渲染命令：\n```bash\n'+command+'\n```\nsource：'+frozen['source']+'\nmodel：'+frozen['model']+'\niteration：'+str(frozen['iteration'])+'\n输出目录：'+str(out)+'\n日志：evaluation.log\n统计：metrics.json，fast_metrics.json\n代表图：four_classes；局部：review_crops\nRGB/法线/Depth-Mesh/配置门控目检：PENDING。动态门控不是训练静动态分离效果。\n最终结论：PENDING\n')
    e.write(out/'freeze_snapshot.json',frozen)
    protocol=dict(split=split,frames=frames,resolution=[400,400],alpha_threshold=.5,signed_world_normal=True,
        condition_primary='GS有效像素角度总和/有效像素数，像素加权',
        fixed_primary='每帧固定GT前景缺失180度MAE后等帧平均',
        supplementary_aggregation='同时输出条件等帧、固定分母像素加权及覆盖率',
        gt_foreground='原Alpha AREA>=.5且GT depth valid正值且GT EXR normal有效',
        rgb='历史linear RGB：TRAIN背景透射扣除，400 AREA；PSNR全图/前景等帧及像素',
        ssim='linear RGB full image, OpenCV Gaussian sigma1.5 window11 data_range1 population covariance border crop5, no clip',
        albedo='Stage1无物理albedo；Stage2仅模型明确提供可解释albedo时评估，不以SH色冒充albedo',
        lighting='Stage1 SH无显式光照；Stage2仅117 TRAIN方向表单位向量插值，无DEV/TEST GT方向',
        training_feedback_allowed=split=='dev',historical_test_previously_seen=True)
    e.write(out/'protocol.json',protocol)
    log=(out/'evaluation.log').open('x');old=sys.stdout,sys.stderr
    sys.stdout,sys.stderr=e.LogStream(sys.stdout,log),e.LogStream(sys.stderr,log)
    rows=[];files={};(out/'arrays').mkdir()
    try:
        model,predict=e.runtime(frozen)
        with torch.no_grad():
            for n in frames:
                r=bundle['frames'][str(n)];cam=e.make_camera(r);c2w=np.asarray(r['transform_matrix']);k=np.asarray(r['intrinsics'])
                pred,parity=predict(cam,c2w,k,with_separation=True)
                pred.update(mesh_normal=np.zeros((400,400,3)),mesh_depth=np.zeros((400,400)),mesh_valid=np.zeros((400,400),bool))
                target,alpha,straight,sat,paths=e.target_frame(bundle['raw'],n,bundle['radiometry']['background_linear'])
                gt,sources=e.load_gt(bundle,n,alpha);files.update(sources);files.update({str(p):e.sha(p) for p in paths})
                regions,status,base=e.geometry_metrics(pred,gt)
                rows.append(dict(source_frame=n,time=r['time'],regions=regions,region_status=status,geometry_parity=parity,
                    rgb=e.rgb_metrics(pred['rgb_linear'],target,alpha,straight,sat,bundle['thresholds']),**rgb_extra(pred['rgb_linear'],target,alpha)))
                np.savez_compressed(out/'arrays'/f'prediction_{n:04d}.npz',**pred,target_rgb_linear=target,source_alpha=alpha,normal_eligible=base,camera_to_world=c2w,intrinsics=k)
                print(f'{split} GS/Depth {n:04d}',flush=True)
            fast=dict(status='GS_DEPTH_COMPLETE_MESH_PENDING',split=split,frames=frames,summary=e.aggregate(rows),rgb=e.aggregate_rgb(rows),ssim_linear_full_frame_mean=float(np.mean([r['ssim_linear_full'] for r in rows])),rows=rows,protocol=protocol)
            e.write(out/'fast_metrics.json',fast)
            print('FAST_METRICS_READY '+str(out/'fast_metrics.json'),flush=True)
            meshes=e.build_meshes(bundle,frozen,out,predict)
            for row in rows:
                n=row['source_frame'];entry=meshes['frames'][str(n)]
                with np.load(out/'arrays'/f'prediction_{n:04d}.npz') as z:pred={k:z[k] for k in z.files}
                if entry['available']:pred.update(PredictedMesh(out/'predicted_meshes'/entry['mesh']).render(pred['camera_to_world'],pred['intrinsics'],400,400))
                gt,sources=e.load_gt(bundle,n,pred['source_alpha']);files.update(sources)
                regions,status,base=e.geometry_metrics(pred,gt);row['regions']=regions
                visuals(out,n,pred,pred['target_rgb_linear'],gt,regions)
                np.savez_compressed(out/'arrays'/f'prediction_{n:04d}.npz',**pred)
            fast.update(status='PASS_EXECUTION_ONLY',summary=e.aggregate(rows),rows=rows,source_files=files,
                albedo=dict(status='NOT_AVAILABLE_STAGE1_NO_PHYSICAL_ALBEDO' if frozen['stage']=='stage1' else 'NOT_AVAILABLE_RENDERER_NO_ALBEDO_OUTPUT'),manual_review='PENDING')
            e.write(out/'metrics.json',fast);e.verify(files);e.verify(frozen['code_sha256'])
            print('METRICS_READY '+str(out/'metrics.json'),flush=True)
    except Exception as exc:
        traceback.print_exc();e.write(out/'FAILED.json',dict(error=str(exc),assets_retained=True));raise
    finally:sys.stdout,sys.stderr=old;log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True)
    a=p.parse_args();run(a)
if __name__=='__main__':main()
