#!/usr/bin/env python3
"""足部恢复Goal原口径数值门槛，几何目检独立且不得提前通过。"""
import argparse
from pathlib import Path
from goal_eval_protocol import ROOT, read, write, digest

BASELINE = ROOT / 'LIHAN/DOC/报告/assets/1012-03-足部恢复冻结基线复核/基线复核.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    evaluation, output = args.evaluation.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        p.error('结果必须保存output内。')
    baseline, measured = read(BASELINE), read(evaluation / 'goal_mae_summary.json')
    if measured['completeness'] != 'PASS' or measured['test_frames'] != 15:
        raise AssertionError('必须完整原15帧。')
    if measured['protocol_sha256'] != baseline['protocol_sha256']:
        raise AssertionError('不得修改原测量协议。')
    for path, expected in baseline['checkpoint_sha256'].items():
        if digest(path) != expected:
            raise AssertionError('冻结基线已变化：' + path)
    marker = read(Path(measured['model']) / 'foot_resume_training.json')
    if marker['actual_entrypoint'] != 'scripts/train_static_dl_foot_resume.py':
        raise AssertionError('必须实际新训练入口。')
    limits = baseline['new_goal_max_mae']
    checks = {k: measured['values'][k] <= v for k, v in limits.items()}
    output.mkdir(parents=True, exist_ok=False)
    status = 'DIAGNOSTIC_ONLY' if marker['clock_shift_diagnostic'] or marker['smoke'] else ('PASS' if all(checks.values()) else 'FAILED')
    write(output / 'goal3_acceptance.json', dict(
        numeric_status=status, model=measured['model'], iteration=measured['iteration'],
        values=measured['values'], baseline_values=baseline['values'], max_mae=limits,
        checks=checks, delta={k: measured['values'][k]-baseline['values'][k] for k in limits},
        reset_policy=marker['reset_policy'], smoke=marker['smoke'], clock_shift_diagnostic=marker['clock_shift_diagnostic'],
        original_protocol_sha256=measured['protocol_sha256'], baseline_sha256=digest(BASELINE),
        coverage={k: measured['details'][k]['coverage'] for k in limits},
        geometry_status='PENDING', overall_status='PENDING',
        note='工程容忍上限，无统计显著性声明；正式整体验收须有效性和多视角腿足完整。smoke及前移clock仅诊断，不得作正式效果。'))


if __name__ == '__main__':
    main()
