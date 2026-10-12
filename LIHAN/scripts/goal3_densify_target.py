#!/usr/bin/env python3
"""足部恢复Goal原口径数值门槛，几何目检独立且不得提前通过。"""
import argparse
from decimal import Decimal
from pathlib import Path
from goal_eval_protocol import ROOT, read, write, digest

BASELINE = ROOT / 'LIHAN/DOC/报告/assets/1012-06-足部恢复新轮最佳法线基线复核/基线复核.json'
BASELINE_SHA256 = '08f26bcfcaad545c257af301c9e5e04e80f05487a5339ce5c9af6fd7aa7b413b'


def numeric_checks(values, limits):
    """十进制比较冻结边界，避免显示舍入放宽门槛。"""
    return {k: Decimal(str(values[k])) <= Decimal(v) for k,v in limits.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    evaluation, output = args.evaluation.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        p.error('结果必须保存output内。')
    if digest(BASELINE) != BASELINE_SHA256:
        raise AssertionError('新轮冻结JSON发生变化。')
    baseline, measured = read(BASELINE), read(evaluation / 'goal_mae_summary.json')
    if measured['completeness'] != 'PASS' or measured['test_frames'] != 15:
        raise AssertionError('必须完整原15帧。')
    if measured['protocol_sha256'] != baseline['protocol_sha256']:
        raise AssertionError('不得修改原测量协议。')
    for path, expected in baseline['checkpoint_sha256'].items():
        if digest(path) != expected:
            raise AssertionError('冻结基线已变化：' + path)
    from goal3_densify_render import actual_sources
    marker, sources = actual_sources(Path(measured['model']))
    limits, historical = baseline['new_round_max_mae_decimal_deg'], baseline['historical_max_mae_decimal_deg']
    checks, historical_checks = numeric_checks(measured['values'], limits), numeric_checks(measured['values'], historical)
    output.mkdir(parents=True, exist_ok=False)
    status = 'DIAGNOSTIC_ONLY' if marker['smoke'] else ('PASS' if all(checks.values()) and all(historical_checks.values()) else 'FAILED')
    write(output / 'goal3_acceptance.json', dict(
        numeric_status=status, model=measured['model'], iteration=measured['iteration'],
        new_round_numeric_status='PASS' if all(checks.values()) else 'FAILED',
        historical_numeric_status='PASS' if all(historical_checks.values()) else 'FAILED',
        values=measured['values'], baseline_values=baseline['baseline_mae_deg'], max_mae_decimal_deg=limits,
        historical_max_mae_decimal_deg=historical, historical_checks=historical_checks,
        checks=checks, delta={k: measured['values'][k]-baseline['baseline_mae_deg'][k] for k in limits},
        densification_route=marker['densification_route'],
        reset_policy=marker['reset_policy'], smoke=marker['smoke'], clock_shift_diagnostic=marker['clock_shift_diagnostic'],
        original_protocol_sha256=measured['protocol_sha256'], baseline_sha256=digest(BASELINE),
        coverage={k: measured['details'][k]['coverage'] for k in limits},
        actual_source_sha256=sources,
        geometry_status='PENDING', overall_status='PENDING',
        note='同模型同时满足新更严格上限与历史上限，原MAE不变；多视角腿足完整性独立否决。smoke仅诊断。'))


if __name__ == '__main__':
    main()
