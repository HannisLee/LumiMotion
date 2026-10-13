#!/usr/bin/env python3
"""固定完整15测试帧float RGB评测；全图主指标与前景辅助分开。"""
import argparse,csv,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal4_appearance_eval_common import ROOT,BASE,audit,destination,checked,unchanged
from goal_eval_protocol import CONTROL,FRAMES,read,write,verify

def target_image(path,size):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:rgba=np.asarray(image.convert('RGBA'))
    # 逐字保留dataset_readers的float64运算次序；代数重排可在uint8截断处差1LSB。
    normalized=rgba/255.;rgb_view=normalized[...,:3]
    normalized[...,:3]=normalized[...,3:4]*normalized[...,:3]
    rgb=(rgb_view*255.).astype(np.uint8)
    rgb=np.asarray(Image.fromarray(rgb).resize(size),dtype=np.float32)/255.
    alpha=np.asarray(Image.fromarray(rgba[...,3]).resize(size),dtype=np.float32)/255.
    return rgb,alpha

def image_metrics(pred,target,alpha):
    import numpy as np
    fg=alpha>.5;raw=pred.astype(np.float64);gt=target.astype(np.float64);clipped=np.clip(raw,0,1)
    if not np.isfinite(raw).all() or not fg.any():raise AssertionError('RGB非有限或GT前景为空。')
    def score(image,mask=None):
        error=(image-gt)**2;value=float(error.mean() if mask is None else error[mask].mean())
        return value,-10*math.log10(max(value,1e-30))
    mse,psnr=score(clipped);raw_mse,raw_psnr=score(raw);fg_mse,fg_psnr=score(clipped,fg);raw_fg_mse,raw_fg_psnr=score(raw,fg)
    saturated=(gt==1)&fg[...,None];sat_count=int(saturated.sum())
    return dict(full_clipped_mse=mse,full_clipped_psnr_db=psnr,full_raw_mse=raw_mse,full_raw_psnr_db=raw_psnr,
        foreground_clipped_mse=fg_mse,foreground_clipped_psnr_aux_db=fg_psnr,foreground_raw_mse=raw_fg_mse,
        foreground_raw_psnr_aux_db=raw_fg_psnr,foreground_pixels=int(fg.sum()),target_saturated_foreground_channels=sat_count,
        target_saturated_foreground_channel_fraction=float(sat_count/(fg.sum()*3)),
        prediction_above_one_foreground_channel_fraction=float((raw[fg]>1).mean()),prediction_below_zero_full_channel_fraction=float((raw<0).mean()),
        saturated_target_underprediction_mean=float(np.maximum(1-raw[saturated],0).mean()) if sat_count else 0.)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model_path',required=True);p.add_argument('--source_path',required=True);p.add_argument('--load_iter',type=int,required=True)
    p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--baseline',action='store_true');p.add_argument('--reference',type=Path);a=p.parse_args()
    out=destination(a.output_dir);protocol=verify(read(CONTROL/'protocol.json'));sources={};checked(CONTROL/'protocol.json',sources)
    if Path(a.source_path).resolve()!=Path(protocol['source_path']):raise AssertionError('必须原eval source。')
    if a.baseline and (Path(a.model_path).resolve()!=BASE or a.load_iter!=16000):raise AssertionError('baseline仅固定G2-T06@16000。')
    if not a.baseline:sources.update(audit(a.model_path,a.load_iter)['sources'])
    import numpy as np,torch
    from PIL import Image
    from scripts.eval_geometry_gt import GeometryRun
    import diff_surfel_rasterization as native
    backend=getattr(native,'provenance',None)
    if not backend or backend['method']!='repository_lumimotion_cuda_rasterizer':raise AssertionError('必须bootstrap原生光栅。')
    original_native=read(BASE/'static_dl_training.json')['rasterizer_provenance']
    for key in ('method','source_sha256','source_files','extension_sha256'):
        if backend[key]!=original_native[key]:raise AssertionError('RGB实际native与T06不符：'+key)
    checked(backend['extension_path'],sources,original_native['extension_sha256'])
    run=GeometryRun(a.model_path,a.load_iter,a.source_path)
    if a.baseline:
        from goal2_eval_reset_audit_render import physical_renderer
    else:
        from goal4_appearance_eval_render import physical_renderer
    render,provenance=physical_renderer(run,a.load_iter,CONTROL/'eval_directions_only_single.json')
    metadata=read(run.source/'transforms_test.json');records=metadata['frames']
    if [r['source_frame'] for r in records]!=FRAMES:raise AssertionError('必须原完整15测试帧。')
    checked(run.source/'transforms_test.json',sources);checked(Path(__file__),sources)
    out.mkdir(parents=True);(out/'arrays').mkdir();(out/'rgb').mkdir();rows=[]
    with torch.no_grad():
        for record in records:
            n=record['source_frame'];camera,alpha,intrinsics,image_path=run.camera(metadata,record,2)
            package=render(camera);pred=package['render'].permute(1,2,0).cpu().numpy()
            if pred.shape!=(400,400,3):raise AssertionError('原RGB协议必须400×400×3。')
            target,original_alpha=target_image(image_path,(pred.shape[1],pred.shape[0]))
            if not np.array_equal(alpha,original_alpha):raise AssertionError('RGB辅助Alpha必须与原GT自身Alpha同值。')
            row=dict(source_frame=n,**image_metrics(pred,target,alpha));rows.append(row)
            np.savez_compressed(out/'arrays'/f'{n:04d}.npz',prediction_raw_srgb=pred,target_srgb=target,target_alpha=alpha)
            Image.fromarray((np.clip(pred,0,1)*255).round().astype(np.uint8)).save(out/'rgb'/f'{n:04d}.png')
            checked(image_path,sources);print(f"RGB {n:04d} clipped={row['full_clipped_psnr_db']:.8f} raw={row['full_raw_psnr_db']:.8f}",flush=True)
            del camera,package
    primary=float(np.mean([r['full_clipped_psnr_db'] for r in rows]));raw=float(np.mean([r['full_raw_psnr_db'] for r in rows]));fg=float(np.mean([r['foreground_clipped_psnr_aux_db'] for r in rows]))
    summary=dict(status='PASS_RGB_COMPLETENESS',model=str(run.model),iteration=a.load_iter,frames=FRAMES,baseline=a.baseline,
        full_clipped_psnr_mean_db=primary,full_raw_psnr_mean_aux_db=raw,foreground_clipped_psnr_mean_aux_db=fg,
        foreground_raw_psnr_mean_aux_db=float(np.mean([r['foreground_raw_psnr_aux_db'] for r in rows])),
        full_clipped_pooled_mse_psnr_aux_db=-10*math.log10(float(np.mean([r['full_clipped_mse'] for r in rows]))),
        target_saturated_foreground_channel_fraction=sum(r['target_saturated_foreground_channels'] for r in rows)/(3*sum(r['foreground_pixels'] for r in rows)),
        prediction_above_one_foreground_channel_fraction=float(np.average([r['prediction_above_one_foreground_channel_fraction'] for r in rows],weights=[r['foreground_pixels'] for r in rows])),
        primary_protocol='float sRGB clamp[0,1]，完整15测试帧，每帧全图三通道MSE→PSNR，15个PSNR算术均值；不按GT Alpha替换预测背景。',
        target_protocol='原eval_view RGBA原分辨率RGB×Alpha黑底合成截为uint8；RGB/Alpha独立PIL BICUBIC缩为400×400，float32/255。',
        auxiliary_protocol='raw未钳位float、GT自身Alpha>0.5前景、饱和与超1比例；不替代全图主PSNR。',
        physical_renderer=provenance,native=backend,argv=sys.argv,manual_review='PENDING',sources=sources)
    if a.reference:
        reference=read(a.reference);checked(a.reference,sources)
        for key in ('frames','primary_protocol','target_protocol','auxiliary_protocol'):
            if summary[key]!=reference[key]:raise AssertionError('起点和候选RGB协议不同：'+key)
        if not reference['baseline'] or Path(reference['model'])!=BASE or reference['iteration']!=16000:raise AssertionError('必须同法实际T06 float起点。')
        delta=primary-reference['full_clipped_psnr_mean_db'];fg_delta=fg-reference['foreground_clipped_psnr_mean_aux_db']
        summary['reference_comparison']=dict(reference=str(a.reference),full_clipped_delta_db=delta,foreground_clipped_delta_aux_db=fg_delta,
            full_raw_delta_aux_db=raw-reference['full_raw_psnr_mean_aux_db'],psnr_gain_threshold_db=.05,foreground_min_delta_aux_db=-.05,
            psnr_gain_pass=delta>.05,foreground_guardrail_pass=fg_delta>=-.05,visual_pass='PENDING')
    sources.update(run.files);unchanged(sources);write(out/'rgb_summary.json',summary)
    with (out/'rgb_per_frame.csv').open('x') as stream:w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'README.md').write_text('# 固定15帧float RGB评测\n\n完整命令见rgb_summary.json argv；source '+str(run.source)+'；model '+str(run.model)+'；iteration '+str(a.load_iter)+'。原float数组arrays/NNNN.npz，显示图rgb/NNNN.png。主全图clipped PSNR '+str(primary)+' dB；raw '+str(raw)+' dB；前景辅助 '+str(fg)+' dB。流水线PASS；四类与足部目检尚待独立验收，科学结论PENDING。不得用PNG量化值代替float统计。\n')
    print(json.dumps({k:v for k,v in summary.items() if k.endswith('_db') or k=='reference_comparison'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
