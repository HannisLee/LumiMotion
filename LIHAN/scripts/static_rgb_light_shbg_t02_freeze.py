"""T02 完整性冻结；仅在四类图人工目检后调用。"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest


def freeze(root):
    root=Path(root).resolve(); manifest=root/'freeze_manifest.json'
    if manifest.exists(): raise FileExistsError('冻结清单已存在')
    images=['01_sh_bg_observation_residual.png','02_direction_azimuth_elevation_split.png','03_gates_and_holdout.png','04_initialization_bootstrap_convergence.png']
    required=['baseline_scoring/score_masks.npz','baseline_scoring/baseline_scores.json','input_lock.json','extraction/extraction.json','extraction/sh_values.npz','fit/fit_results.json','selection/direction.npy','selection/candidate_table.json','README.md','inspection.json']+['figures/'+x for x in images]
    for name in required:
        if not (root/name).is_file(): raise FileNotFoundError('冻结缺少 '+name)
    choice=json.loads((root/'selection/candidate_table.json').read_text()); runs=json.loads((root/'fit/fit_results.json').read_text()); review=json.loads((root/'inspection.json').read_text())
    if set(runs)!={'C00L','C01','C02','C03','C04'} or len(choice['candidates'])!=4 or choice['status'] not in ('SELECTED_SH','FALLBACK_E4'): raise ValueError('候选或选择未完成')
    keys=('full','A','B','boot103108','boot103109','hi','lo')
    for candidate,result in runs.items():
        if result.get('failure') in ('SH_DEGENERATE','SH_CLIPPING'): continue
        for key in keys:
            if key not in result: raise ValueError('缺少流程状态: '+candidate+'/'+key)
            run=root/'fit/runs'/candidate/key
            for file in ('certificate.json','history.json','checkpoint.pt'):
                if not (run/file).is_file(): raise FileNotFoundError('缺少流程证据: '+str(run/file))
            if result[key].get('converged') and not all((run/f).is_file() for f in ('direction.npy','parameters.npz')): raise FileNotFoundError('缺少已收敛流程预测')
    if set(review.get('figures',{}))!=set(images) or not review.get('inspected_at_utc') or not all(review['figures'].values()): raise ValueError('目检记录不完整')
    if datetime.strptime(review['inspected_at_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)>datetime.now(timezone.utc): raise ValueError('目检时间无效')
    readme=(root/'README.md').read_text()
    for term in ('source','model','iteration','命令','日志','目检','新渲染不适用','PASS','FAILED'):
        if term not in readme: raise ValueError('README 缺少 '+term)
    inputs=json.loads((root/'input_lock.json').read_text())
    for path,entry in inputs['files'].items():
        p=Path(path)
        if not p.is_file() or digest(p)!=entry['sha256'] or p.stat().st_size!=entry['size']: raise ValueError('输入发生变化: '+path)
    exit_status=root/'freeze.exit_code'
    if exit_status.exists(): raise FileExistsError('冻结退出状态已存在')
    exit_status.write_text('0\n')
    files=[]
    for directory,dirs,names in os.walk(root,topdown=True,followlinks=False):
        dirs[:]=[name for name in dirs if name!='evaluation']
        for name in names:
            path=Path(directory)/name
            if path!=manifest and path.is_file(): files.append(path)
    files.sort()
    result=dict(status='FROZEN_AWAITING_INDEPENDENT_EVALUATION',frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selected=choice['selected'],selection_status=choice['status'],external_inputs=inputs['files'],gt_read=False,files={str(p.relative_to(root)):dict(sha256=digest(p),size=p.stat().st_size,mtime_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(p.stat().st_mtime))) for p in files})
    try: manifest.write_text(json.dumps(result,indent=2))
    except Exception:
        exit_status.unlink(missing_ok=True)
        raise
    return result
