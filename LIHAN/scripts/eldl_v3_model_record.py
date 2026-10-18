"""仅登记已审计的固定EL+DL训练入口；学习率适配不改变评测前向。"""
import hashlib
import json
from pathlib import Path

OLD_ENTRY='scripts/train_eldl_v3.py'
LATE_ENTRY='scripts/train_eldl_v3_late_anneal.py'


def fixed_eldl_record_files(model,record):
    """返回附加来源文件；None表示不是已登记混合入口。"""
    entry=record.get('actual_entrypoint')
    if entry==OLD_ENTRY:return []
    if entry=='scripts/train_eldl_v3_geometry_warmstart.py':
        from LIHAN.scripts.eldl_v3_geometry_record import fixed_geometry_record_files
        return fixed_geometry_record_files(model,record)
    if entry!=LATE_ENTRY:return None
    path=Path(model)/'late_lr_training.json'
    late=json.loads(path.read_text())
    expected_boundaries=(8,16) if record['args'].get('smoke_schedule') else (16000,20000)
    if (late!=record.get('late_learning_rate') or late['actual_entrypoint']!=LATE_ENTRY
            or late['case'] not in ('control','anneal') or late['case']!=record['args'].get('late_lr_case')
            or (late['start'],late['end'])!=expected_boundaries
            or late['final_multiplier']!=(.1 if late['case']=='anneal' else 1.)
            or late['active_groups']!=['xyz','albedo_dc','opacity','scaling','rotation']
            or late['gt_supervision'] is not False or late['adam_moments_changed'] is not False
            or not all(late[k] is True for k in ('lighting_unchanged','loss_unchanged','topology_policy_unchanged'))):
        raise ValueError('晚期学习率模型来源或固定物理语义不一致。')
    names={Path(p).name for p in late['sources']}
    if names!={'train_eldl_v3_late_anneal.py','eldl_v3_late_lr.py'} or len(late['sources'])!=2:
        raise ValueError('晚期入口源码集合不同。')
    for source,expected in late['sources'].items():
        file=Path(source)
        if file.suffix!='.py' or record['new_sources'].get(source)!=expected:
            raise ValueError('晚期入口未绑定训练源码。')
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError('晚期训练源码SHA变化：'+source)
    return [path,Path(__file__).resolve(),*[Path(p) for p in late['sources']]]
