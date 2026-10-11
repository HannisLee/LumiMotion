#!/usr/bin/env python3
"""独立G2目标元数据判定；原六文件、原15帧measurement与旧结果不变。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_eval_protocol import CONTROL, ROOT, read, write, digest
from goal_eval_audit import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--target_metadata', type=Path, default=ROOT / 'output/1012-01-静态DL-两度目标总控/baseline_and_acceptance.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geometry_review', type=Path, help='独立人工/诊断整体几何验收JSON，status=PASS或FAILED；缺失时不声称Goal完成。')
    args = p.parse_args()
    evaluation, output = args.evaluation.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise AssertionError('输出必须位于output。')
    target = read(args.target_metadata)
    protocol = CONTROL / 'protocol.json'
    if target['original_protocol_sha256'] != digest(protocol) or target['original_test_frames'] != 15:
        raise AssertionError('目标元数据必须引用原完整15帧measurement协议。')
    old = read(evaluation / 'goal_mae_summary.json')
    output.mkdir(parents=True, exist_ok=False)
    # 调用已有逐字原AST复算，输出单独目录；不覆盖旧goal_mae_summary及CSV。
    measurement = audit(evaluation, old['iteration'], protocol, output / 'original_measurement_reaudit.json')
    if measurement['completeness'] != 'PASS':
        raise AssertionError('G2整体PASS必须建立在完整15原measurement通过之上。')
    if measurement['model'] != old['model'] or measurement['protocol_sha256'] != target['original_protocol_sha256']:
        raise AssertionError('同checkpoint／原协议要求不符。')
    for key in ('values', 'details', 'missing_penalty'):
        if measurement[key] != old[key]:
            raise AssertionError('原指标复算与已保存摘要不同：' + key)
    checks = {}
    required = {'gs_render': ('<', 2.), 'depth_normal': ('<=', 5.887606052822573), 'mesh': ('<=', 4.7507435937717695)}
    for key, (operator, threshold) in required.items():
        definition = target['acceptance'][key]
        if definition != {'operator': operator, 'value': threshold}:
            raise AssertionError('G2目标元数据不同于预注册目标。')
        value = measurement['values'][key]
        checks[key] = dict(value=value, operator=operator, threshold=threshold, passed=value < threshold if operator == '<' else value <= threshold,
                           delta_from_baseline=value-target['metrics_deg'][key])
    numeric = all(row['passed'] for row in checks.values())
    review = read(args.geometry_review) if args.geometry_review else None
    if review is not None:
        if review['status'] not in ('PASS', 'FAILED') or Path(review['model']).resolve() != Path(old['model']).resolve() or review['iteration'] != old['iteration']:
            raise AssertionError('整体几何验收必须对应同一个checkpoint。')
    status = 'FAILED' if not numeric or review is not None and review['status'] == 'FAILED' else ('PASS' if review else 'PENDING_GEOMETRY_REVIEW')
    report = dict(status=status, numeric_status='PASS' if numeric else 'FAILED', completeness=measurement['completeness'],
                  geometry_review_status=review['status'] if review else 'PENDING', geometry_review=review,
                  checks=checks, model=old['model'], source=old['source'], iteration=old['iteration'],
                  original_measurement=dict(values=measurement['values'], details=measurement['details'], missing_penalty=measurement['missing_penalty']),
                  target_metadata=str(args.target_metadata.resolve()), target_metadata_sha256=digest(args.target_metadata),
                  protocol_sha256=digest(protocol), original_summary_sha256=digest(evaluation / 'goal_mae_summary.json'),
                  helper_sha256=digest(Path(__file__)), geometry_review_sha256=digest(args.geometry_review) if args.geometry_review else None,
                  scope='仅目标元数据判定，原有效像素加权法线MAE/Alpha/TSDF/失败像素处理完全不变；GT区域诊断不进入训练。')
    write(output / 'goal2_acceptance.json', report)
    print('G2目标判定：' + status + '；' + str(output / 'goal2_acceptance.json'))


if __name__ == '__main__':
    main()
