#!/usr/bin/env python3
"""汇总已完成的同口径完整15帧评测，不替代原指标或筛选帧。"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

from goal_eval_protocol import ROOT, FRAMES, read, write, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--intermediate', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'output') or output.exists():
        p.error('汇总须在output内新建，禁止覆盖。')
    rows = []
    protocols = set()
    for directory in args.evaluations:
        directory = directory.resolve()
        r = read(directory / 'goal_mae_summary.json')
        completed = read(directory / 'PIPELINE_COMPLETED.json')
        if r['completeness'] != 'PASS' or r['test_frames'] != 15 or completed['pipeline'] != 'PASS':
            raise AssertionError('仅汇总完整成功执行的15帧评测：' + str(directory))
        with (directory / 'goal_mae_per_frame.csv').open() as f:
            frames = [int(x['source_frame']) for x in csv.DictReader(f)]
        if sorted(frames) != list(FRAMES):
            raise AssertionError('逐帧CSV缺失／重复：' + str(directory))
        protocols.add(r['protocol_sha256'])
        rows.append(dict(experiment=Path(r['model']).parent.name, model=r['model'],
            iteration=r['iteration'], evaluation=str(directory), source=r['source'],
            values={k: r['values'][k] for k in r['thresholds']}, thresholds=r['thresholds'],
            coverage={k: r['details'][k]['coverage'] for k in r['thresholds']},
            missing_penalty=r['missing_penalty'], completeness=r['completeness'],
            numeric_status=r['status'], visual_review=completed['visual_review'],
            summary_sha256=digest(directory / 'goal_mae_summary.json')))
    if len(protocols) != 1:
        raise AssertionError('协议哈希不同，禁止合并为同口径比较。')
    output.mkdir(parents=True)
    batch_numeric = 'PASS' if any(r['numeric_status'] == 'PASS' for r in rows) else 'FAILED'
    write(output / 'summary.json', dict(protocol_sha256=protocols.pop(), numeric_status=batch_numeric,
        intermediate=args.intermediate, experiments=rows,
        scope='原完整15帧结果汇总；不筛选测试像素／帧，不推断单因素因果。'))
    with (output / 'summary.csv').open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'iteration', 'gs_mae_deg', 'depth_mae_deg', 'mesh_mae_deg',
                         'gs_coverage', 'depth_coverage', 'mesh_coverage', 'numeric_status', 'evaluation'])
        for r in rows:
            keys = ['gs_render', 'depth_normal', 'mesh']
            writer.writerow([r['experiment'], r['iteration']] + [r['values'][k] for k in keys] +
                            [r['coverage'][k] for k in keys] + [r['numeric_status'], r['evaluation']])
    lines = ['# 静态DL完整测试评测汇总', '',
        '固定15测试帧10、20、…、150；原前景有效像素加权MAE，原掩膜、TSDF及缺失处理不变。',
        '严格阈值GS<3.51°、Depth<7.35°、Mesh<5.46°。', '',
        '本表为中间诊断，固定35000步训练继续；中间结果不能替代最终验收。' if args.intermediate else
        '数值与四类目检结论独立记录；数值通过不代替完整可视化验收。', '',
        '| 实验 | 迭代 | GS° | Depth° | Mesh° | 数值 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        v = r['values']
        lines.append(f"| {r['experiment']} | {r['iteration']} | {v['gs_render']:.8f} | {v['depth_normal']:.8f} | {v['mesh']:.8f} | {r['numeric_status']} |")
    lines += ['', '## 可复现命令与原始证据', '', '```bash',
              shlex.join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]), '```', '']
    for r in rows:
        lines += [f"- [{r['experiment']}完整命令／目检／日志]({r['evaluation']}/README.md)；"
                  f"source=`{r['source']}`，model=`{r['model']}`；原始数组与逐帧CSV保存在该评测目录。"]
    lines += ['', f'本批次至少一个实验同时满足三阈值的数值判定：**{batch_numeric}**。',
              '各实验全部检查点、渲染、数组、统计和日志保留，不覆盖失败结果。',
              '四类RGB／Alpha／Normal／分离的实际状态见各实验README；汇总不虚构目检PASS。',
              'B/C只在训练公平性证据成立时可作可见性消融；A/B联合改动不作单因素因果归因。', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print('完整测试汇总已保存：' + str(output))


if __name__ == '__main__':
    main()
