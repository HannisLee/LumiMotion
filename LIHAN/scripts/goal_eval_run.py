#!/usr/bin/env python3
"""为独立实验建立原协议评测计划，显式指定GPU后执行且禁止覆盖任何结果。"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from goal_eval_protocol import ROOT, PYTHON, CONTROL, FRAMES, read, write, verify, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=CONTROL / 'protocol.json')
    p.add_argument('--gpu', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--plan_destination', type=Path,
                   help='仅生成计划时将计划保存到此目录，保持实际output尚未创建。')
    args = p.parse_args()
    if subprocess.check_output(['hostname']).decode().strip() != 'ushas':
        p.error('当前执行器只允许ushas。')
    if args.execute and args.gpu is None:
        p.error('执行前须与trainer协调GPU，并显式提供--gpu。')
    if args.execute and args.plan_destination is not None:
        p.error('--plan_destination仅供不执行的计划。')
    args.output = args.output.resolve()
    args.model = args.model.resolve()
    if not args.output.is_relative_to(ROOT / 'output'):
        p.error('所有评测输出必须在output内。')
    if args.output.exists():
        p.error('输出禁止覆盖；失败重试请另建新目录。')
    protocol = verify(read(args.protocol))
    shared = ['--model_path', str(args.model), '--source_path', protocol['source_path'],
              '--load_iter', str(args.iteration)]
    boot = [PYTHON, str(ROOT / 'LIHAN/scripts/bootstrap_1011.py'), str(ROOT)]
    frame_args = ['--frames'] + list(map(str, FRAMES))
    commands = [
        ('predicted_meshes', boot + ['scripts.extract_geometry_mesh_dynamic'] + shared + frame_args +
            ['--split', 'test', '--resolution', '2', '--voxel_size', '.02', '--sdf_trunc', '.08',
             '--depth_trunc', '20', '--alpha_threshold', '.5',
             '--output_dir', str(args.output / 'predicted_meshes')]),
        ('normal_depth', boot + ['scripts.eval_geometry_gt'] + shared + frame_args +
            ['--gt_root', protocol['gt_root'], '--gt_depth_dir', protocol['gt_depth_dir'],
             '--split', 'test', '--resolution', '2', '--alpha_threshold', '.5', '--interior_radius', '2',
             '--output_dir', str(args.output / 'normal_depth'),
             '--pred_mesh_manifest', str(args.output / 'predicted_meshes/manifest.json'), '--save_arrays']),
        ('audit', [PYTHON, str(ROOT / 'LIHAN/scripts/goal_eval_audit.py'), str(args.output),
                   str(args.iteration), '--protocol', str(args.protocol.resolve())]),
        ('four_classes', boot + ['LIHAN.scripts.goal_eval_visuals'] + shared +
            ['--output_dir', str(args.output / 'four_classes')]),
    ]
    plan_directory = args.plan_destination.resolve() if args.plan_destination is not None else args.output
    if not plan_directory.is_relative_to(ROOT / 'output'):
        p.error('计划也必须保存在output内。')
    plan_directory.mkdir(parents=True, exist_ok=False)
    (plan_directory / 'logs').mkdir()
    write(plan_directory / 'protocol.json', protocol)
    plan = dict(model=str(args.model), iteration=args.iteration, gpu=args.gpu,
                command={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                steps=[dict(name=n, command=c, shell=shlex.join(c)) for n, c in commands],
                wrapper_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                    sorted((ROOT / 'LIHAN/scripts').glob('goal_eval*.py'))})
    write(plan_directory / 'plan.json', plan)
    lines = ['# 静态DL Goal评测记录', '', f'模型：`{args.model}`；迭代：{args.iteration}。',
             f"source：`{protocol['source_path']}`；固定15测试帧：10、20、…、150。",
             '原协议：前景有效像素加权MAE；阈值GS<3.51°、Depth<7.35°、Mesh<5.46°。',
             f'评测／渲染输出：`{args.output}`；日志：`{args.output / "logs"}`。', '', '## 完整命令', '']
    for name, command in commands:
        lines += [f'### {name}', '', '```bash', shlex.join(command), '```', '']
    lines += ['## 定量与目检', '', '定量指标：待执行；`goal_mae_summary.json`保存独立复算及三阈值结论。',
              '原始数组：`normal_depth/prediction_0010.npz`至`prediction_0150.npz`；逐帧CSV保留。',
              '代表图：`four_classes/four_classes_10_80_150.png`；四类全150帧PNG及视频。',
              'RGB：PENDING；Alpha：PENDING；Normal：PENDING；分离：PENDING。',
              '静态几何分离语义：零几何输出头冻结，红／绿显示原binary feature门控，不能声称学到运动。',
              '最终结论：PENDING。数值未达标、异常或验收失败均保留完整结果，禁止覆盖或删除。', '']
    readme = plan_directory / 'README.md'
    readme.write_text('\n'.join(lines))
    if not args.execute:
        print('评测计划已保存：' + str(plan_directory / 'plan.json'))
        return
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', TORCH_CUDA_ARCH_LIST='8.6')
    libraries = ROOT / 'output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu'
    environment['LD_LIBRARY_PATH'] = str(libraries) + ':' + environment.get('LD_LIBRARY_PATH', '')
    outcomes = []
    for name, command in commands:
        verify(protocol)
        log_path = args.output / 'logs' / (name + '.log')
        with log_path.open('x') as log:
            log.write(shlex.join(command) + '\n'); log.flush()
            code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        outcomes.append(dict(step=name, exit_code=code, log=str(log_path)))
        write(args.output / 'executed.json', outcomes)
        print(f'{name}退出码{code}；日志{log_path}', flush=True)
        if code:
            write(args.output / 'FAILED.json', dict(step=name, exit_code=code, retained=True))
            with readme.open('a') as f:
                f.write(f'\n执行失败：{name}退出码{code}。最终结论：**FAILED**；完整失败现场保留。\n')
            sys.exit(code)
    verify(protocol)
    report = read(args.output / 'goal_mae_summary.json')
    with readme.open('a') as f:
        f.write('\n## 执行结果\n\n')
        for key in protocol['thresholds']:
            f.write(f'- {key} MAE：{report["values"][key]:.8f}°；coverage={report["details"][key]["coverage"]:.8f}。\n')
        f.write(f'\n数值验收：**{report["status"]}**。四类完整性通过；实际目检结论仍须人工回填。\n')
        if report['status'] == 'FAILED':
            f.write('\n最终结论：**FAILED**，未同时达到三个法线阈值。\n')
    write(args.output / 'PIPELINE_COMPLETED.json', dict(pipeline='PASS', goal_numeric=report['status'],
                                                     visual_review='PENDING'))


if __name__ == '__main__':
    main()
