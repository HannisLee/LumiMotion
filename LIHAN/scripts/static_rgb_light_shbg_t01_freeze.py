"""冻结 T01 非 GT 产物；只在完整自检与真实目检后使用。"""
import json
from pathlib import Path
import time
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest


def freeze(root):
    root=Path(root).resolve(); manifest=root/'freeze_manifest.json'
    if manifest.exists(): raise FileExistsError('冻结清单已存在，禁止覆盖')
    required=['sh_templates.npz','baseline_scoring/score_masks.npz','baseline_scoring/baseline_scores.json','input_lock.json','extraction/extraction.json','extraction/sh_values.npz','fit/fit_results.json','selection/score_masks.npz','selection/predictions.npz','selection/candidate_table.json','README.md','inspection.json']
    images=['01_sh_bg_observation_residual.png','02_direction_azimuth_elevation_split.png','03_gates_and_holdout.png','04_initialization_bootstrap_convergence.png']
    required += ['figures/'+f for f in images]
    for name in required:
        if not (root/name).is_file(): raise FileNotFoundError(f'冻结缺少 {name}')
    choice=json.loads((root/'selection/candidate_table.json').read_text()); runs=json.loads((root/'fit/fit_results.json').read_text()); review=json.loads((root/'inspection.json').read_text())
    if set(runs)!=set(('C01','C02','C03','C04')) or len(choice['candidates'])!=4: raise ValueError('四候选未结束')
    if choice['status'] not in ('FALLBACK_E4','SELECTED_SH'): raise ValueError('选择状态错误')
    if set(review.get('figures',{}))!=set(images) or not review.get('inspected_at_utc') or not all(review['figures'].values()): raise ValueError('四类人工目检记录不完整')
    from datetime import datetime,timezone
    inspected=datetime.strptime(review['inspected_at_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    if inspected>datetime.now(timezone.utc): raise ValueError('目检时间晚于当前时间')
    readme=(root/'README.md').read_text()
    for required_text in ('source','model','iteration','命令','日志','目检','新渲染不适用','PASS','FAILED'):
        if required_text not in readme: raise ValueError(f'README 缺少 {required_text}')
    files=sorted(p for p in root.rglob('*') if p.is_file() and 'evaluation' not in p.relative_to(root).parts and p!=manifest)
    input_lock=json.loads((root/'input_lock.json').read_text())
    result={'external_inputs':input_lock.get('files',{}),'input_lock_utc':input_lock.get('utc'),'status':'FROZEN_AWAITING_INDEPENDENT_EVALUATION','frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selected':choice['selected'],'selection_status':choice['status'],'gt_read':False,'files':{str(p.relative_to(root)):{'sha256':digest(p),'size':p.stat().st_size} for p in files}}
    manifest.write_text(json.dumps(result,indent=2)); return result
