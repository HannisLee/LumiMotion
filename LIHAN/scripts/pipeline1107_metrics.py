"""T01(1107-03) I4：训练循环的只加日志钩子（不改任何计算）。

由 pipeline1107_prepare.py 在 snapshot 的 train_stage2.py 内插入一行调用。
每 100 循环写 <model_path>/metrics.jsonl；任何异常只写 metrics_error.log，绝不中断训练。
"""
import json
import math
import time
import traceback
from pathlib import Path

INTERVAL = 100


def _number(value):
    try:
        value = float(value.detach().float().item()) if hasattr(value, 'detach') else float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def build_record(trainer, loss, result, l1):
    """构造一行日志；全部取自已有训练状态，不触发新的前向。"""
    import torch
    record = dict(iteration=int(trainer.iteration), wall_time=time.time(),
                  total_loss=_number(loss), rgb_l1=_number(l1))
    terms = getattr(result, 'terms', None) or {}
    for name, value in terms.items():
        record['loss_' + str(name)] = _number(value)
    for name, value in (getattr(trainer, '_pipeline_geometry_terms', None) or {}).items():
        record['geo_' + str(name)] = _number(value)
    try:
        record['step_ms'] = float(trainer.iter_start.elapsed_time(trainer.iter_end))
    except Exception:
        record['step_ms'] = None
    env = getattr(trainer, 'env_light', None)
    if env is not None:
        with torch.no_grad():
            record['env_mean'] = _number(env.activation(env.base).mean())
            if hasattr(env, 'mixed_log_intensity'):
                record['sun_intensity'] = _number(env.mixed_log_intensity.exp())
    if torch.cuda.is_available():
        record['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 2**20
        record['cuda_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 2**20
    record['gaussians'] = int(trainer.gaussians.get_xyz.shape[0])
    return record


def log_step(trainer, loss, result, l1):
    if trainer.iteration % INTERVAL:
        return
    root = Path(trainer.args.model_path)
    try:
        record = build_record(trainer, loss, result, l1)
        with (root / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        with (root / 'metrics_error.log').open('a') as handle:
            handle.write(f'iteration {trainer.iteration}\n{traceback.format_exc()}\n')
