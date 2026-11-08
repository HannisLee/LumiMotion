"""1108-02 T01 冻结后一次性独立评测；复用 E4 的 GT 与角度数学路径。"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import static_rgb_light_evaluate_r5b as old

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑'
BASE = ROOT / 'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角/fitting'
EXPECTED = 'f7b0f9df47192a82fce9dcacd05a7827224afd31fb28b558543b8b4ccaca0f92'
NAMES = ['full','A','B','boot103108','boot103109','hi','lo','repeat_full']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def dump(p,v):
    with Path(p).open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def summary(x):
    return dict(min=float(x.min()),p1=float(np.percentile(x,1)),p10=float(np.percentile(x,10)),p50=float(np.percentile(x,50)),p90=float(np.percentile(x,90)),p99=float(np.percentile(x,99)),max=float(x.max()),std=float(x.std()))
def main():
    m=json.loads((RUN/'freeze_manifest.json').read_text())
    assert sha(RUN/'freeze_manifest.json')==EXPECTED
    assert m['gt_read'] is False and m['selected']=='C01/full'
    checked={}
    for section in ('sha256','input_and_code_sha256'):
        for name,want in m[section].items():
            p=ROOT/name
            assert p.name!='lights.json'
            assert sha(p)==want,name
            checked[name]=want
    sel=json.loads((RUN/'fitting/selection.json').read_text())
    assert sel['selected']=='C01/full' and sel['status']=='VALID'
    preds={}
    with np.load(BASE/'predictions.npz',allow_pickle=False) as d:
        assert np.array_equal(d['frames'],np.arange(1,151))
        preds['E4']=d['E4'].astype(np.float64)
    with np.load(RUN/'fitting/predictions.npz',allow_pickle=False) as d:
        assert np.array_equal(d['frames'],np.arange(1,151))
        assert np.array_equal(d['reference_E4'],preds['E4'])
        for name in NAMES:
            preds['C01/'+name]=d['C01_'+name].astype(np.float64)
            assert np.array_equal(d['C01_'+name],np.load(RUN/f'fitting/runs/{name}/direction.npy'))
    maxnorm={}
    for name,v in preds.items():
        assert v.shape==(150,3) and v.dtype==np.float64 and np.isfinite(v).all()
        norm=np.linalg.norm(v,axis=1);maxnorm[name]=float(abs(norm-1).max());assert maxnorm[name]<=1e-10
        preds[name]=v/norm[:,None] # 与原 E4 validate 一致，评测前单位化
    old.self_test()
    out=RUN/'evaluation';assert not out.exists(),'禁止二次评测或覆盖'
    out.mkdir()
    dump(out/'integrity_validation.json',dict(status='PASS',checked_count=len(checked),checked_sha256=checked,max_direction_norm_error=maxnorm,freeze_manifest_sha256=EXPECTED,gt_read=False,checked_utc=now()))
    dump(RUN/'gt_evaluation_receipt.json',dict(status='STARTED',utc=now(),gt_file_open_count_planned=1,freeze_manifest_sha256=EXPECTED,code_sha256=sha(__file__),dependencies={str(Path(old.__file__).relative_to(ROOT)):sha(old.__file__),str(Path(old.base.__file__).relative_to(ROOT)):sha(old.base.__file__)},gt_used_for_decisions=False))
    gt,gt_hash=old.read_gt_once() # 唯一 GT 打开；原 E4 校验/约定/单位化路径
    errors=old.base.evaluate(preds,gt)
    metrics={};rows=[]
    for name,pred in preds.items():
        el,az=old.decompose(pred,gt);err=errors[name];elev=np.degrees(np.arcsin(np.clip(pred[:,2],-1,1)))
        metrics[name]=dict(mean_deg=float(err.mean()),p50_deg=float(np.percentile(err,50)),p90_deg=float(np.percentile(err,90)),within_5deg_count=int((err<=5).sum()),frame_count=150,
            azimuth=dict(signed_bias_deg=float(az.mean()),mae_deg=float(abs(az).mean()),p50_abs_deg=float(np.percentile(abs(az),50)),p90_abs_deg=float(np.percentile(abs(az),90))),
            elevation=dict(signed_bias_deg=float(el.mean()),mae_deg=float(abs(el).mean()),p50_abs_deg=float(np.percentile(abs(el),50)),p90_abs_deg=float(np.percentile(abs(el),90))),predicted_elevation_deg=summary(elev),nominal_band_fraction=float((abs(elev-45)>=4.75).mean()))
        for i in range(150):rows.append([i+1,name,*pred[i],*gt[i],float(err[i]),float(az[i]),float(el[i])])
    with (out/'per_frame.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['frame','method','pred_x','pred_y','pred_z','gt_x','gt_y','gt_z','angle_deg','azimuth_pred_minus_gt_deg','elevation_pred_minus_gt_deg']);w.writerows(rows)
    certificate=json.loads((RUN/'fitting/runs/full/certificate.json').read_text())
    converged=certificate['converged']
    result=dict(status='PASS' if converged and metrics['C01/full']['mean_deg']<metrics['E4']['mean_deg'] else 'FAILED',integrity=True,full_converged=converged,strict_mean_improvement=metrics['C01/full']['mean_deg']<metrics['E4']['mean_deg'],methods=metrics,gt_elevation_deg=summary(np.degrees(np.arcsin(gt[:,2]))),gt_sha256=gt_hash,gt_file_open_count=1,freeze_manifest_sha256=EXPECTED,evaluated_utc=now(),selected='C01/full',rule='完整性通过、full收敛、full Mean严格小于历史E4全精度Mean；否则FAILED',gt_feedback_parameter_tuning=False)
    dump(out/'results.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    frames=np.arange(1,151)
    for name in ('E4','C01/full'):
        axes[0].plot(frames,errors[name],label=name)
        axes[1].plot(frames,np.degrees(np.arcsin(preds[name][:,2])),label=name)
        axes[2].plot(frames,old.decompose(preds[name],gt)[0],label=name)
    axes[1].plot(frames,np.degrees(np.arcsin(gt[:,2])),color='black',label='GT')
    axes[0].set_ylabel('3D error (deg)');axes[1].set_ylabel('Elevation (deg)');axes[2].set_ylabel('Elevation bias (deg)');axes[2].set_xlabel('Frame')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'angular_elevation_posthoc.png',dpi=160);plt.close(fig)
    dump(out/'evaluation_complete.json',dict(status='COMPLETE',result=result['status'],utc=now(),gt_file_open_count=1,artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
