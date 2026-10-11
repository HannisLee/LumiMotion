#!/usr/bin/env python3
"""不改原数学，复用原独立geometry复算，生成Goal三项完整性与阈值报告。"""
import argparse
import ast
import csv
import json
from pathlib import Path
import sys

from goal_eval_protocol import CONTROL, ROOT, FRAMES, read, write, verify, digest


def audit(evaluation, iteration, protocol_path, output=None):
    protocol = verify(read(protocol_path))
    # 原脚本顶层还强制Albedo/Roughness；Goal只验收geometry。
    # 仅执行原import、check_close、unit、geometry函数AST，函数体逐字原样。
    path = ROOT / 'LIHAN/scripts/audit_1011_mae.py'
    source = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in source.body if isinstance(n, (ast.Import, ast.ImportFrom)) or
             isinstance(n, ast.FunctionDef) and n.name in {'check_close', 'unit', 'geometry'}]
    namespace = dict(ROOT=Path(evaluation), ITERATION=iteration, expected_ids=FRAMES,
                     results={}, errors=[], details={})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['geometry']()
    metrics = read(Path(evaluation) / 'normal_depth/metrics.json')
    mesh = read(Path(evaluation) / 'predicted_meshes/manifest.json')
    command = metrics['command']
    for relative, expected_hash in metrics['code_files'].items():
        if protocol['code_sha256'].get(str(ROOT / relative)) != expected_hash:
            raise AssertionError('指标文件记录的评测代码不符：' + relative)
    expected = dict(split='test', resolution=2, alpha_threshold=.5, interior_radius=2,
                    source_path=protocol['source_path'], gt_root=protocol['gt_root'],
                    gt_depth_dir=protocol['gt_depth_dir'], save_arrays=True)
    if any(command.get(k) != v for k, v in expected.items()):
        raise AssertionError('几何命令与固定协议不同。')
    settings = mesh['settings']
    if any(settings.get(k) != v for k, v in protocol['tsdf'].items()):
        raise AssertionError('TSDF参数改变。')
    if settings['resolution'] != 2 or settings['split'] != 'test':
        raise AssertionError('TSDF分辨率／划分改变。')
    if set(map(int, mesh['frames'])) != set(FRAMES):
        raise AssertionError('TSDF必须完整覆盖15测试时刻。')
    for rel in ['normal_depth/EVAL_COMPLETED', 'predicted_meshes/MESH_COMPLETED']:
        if not (Path(evaluation) / rel).is_file():
            raise AssertionError('缺少完成标记：' + rel)
    values = namespace['results']
    passed = all(values[k] < threshold for k, threshold in protocol['thresholds'].items())
    report = dict(status='PASS' if passed else 'FAILED',
        completeness='PASS', threshold_pass=passed, iteration=iteration, test_frames=15,
        source=protocol['source_path'], model=metrics['model_path'], values=values,
        thresholds=protocol['thresholds'], strict_less_than=True, details=namespace['details'],
        protocol_sha256=digest(protocol_path), metrics_sha256=digest(Path(evaluation) / 'normal_depth/metrics.json'),
        missing_penalty={k: sum(r['regions']['foreground']['normal_metrics'][k]['angular_with_missing_deg']['sum']
                               for r in metrics['frames']) /
                           sum(r['regions']['foreground']['normal_metrics'][k]['angular_with_missing_deg']['count']
                               for r in metrics['frames'])
                         for k in protocol['thresholds']},
        scope='Goal三类法线；完整性与数值阈值，不声称已人工目检。')
    output = Path(output or Path(evaluation) / 'goal_mae_summary.json')
    if output.exists():
        raise FileExistsError('总结禁止覆盖：' + str(output))
    write(output, report)
    with (output.parent / 'goal_mae_per_frame.csv').open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['source_frame'] + [k + '_mae_deg' for k in protocol['thresholds']] +
                        [k + '_coverage' for k in protocol['thresholds']])
        for row in metrics['frames']:
            m = row['regions']['foreground']['normal_metrics']
            writer.writerow([row['source_frame']] + [m[k]['angular_deg']['mean'] for k in protocol['thresholds']] +
                            [m[k]['coverage'] for k in protocol['thresholds']])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation', type=Path)
    p.add_argument('iteration', type=int)
    p.add_argument('--protocol', type=Path, default=CONTROL / 'protocol.json')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = audit(args.evaluation, args.iteration, args.protocol, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 阈值未达标仍是有效评测，不阻止保留视觉产物。


if __name__ == '__main__':
    main()
