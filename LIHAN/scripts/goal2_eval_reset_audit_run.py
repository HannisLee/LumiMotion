#!/usr/bin/env python3
"""reset-audit独立全流程：原15测量、同8对象离线诊断和真实a0四类。"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_eval_protocol import ROOT, PYTHON, CONTROL, FRAMES, read, write, verify, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--geometry_only', action='store_true')
    p.add_argument('--plan_destination', type=Path)
    args = p.parse_args()
    if subprocess.check_output(['hostname']).decode().strip() != 'ushas':
        p.error('必须ushas/lumimotion-ushas。')
    if args.execute and (args.gpu is None or args.plan_destination):
        p.error('执行须已协调显式GPU，不能同时plan_destination。')
    model, output = args.model.resolve(), args.output.resolve()
    destination = args.plan_destination.resolve() if args.plan_destination else output
    if not output.is_relative_to(ROOT / 'output') or not destination.is_relative_to(ROOT / 'output'):
        p.error('输出及计划必须output内。')
    if output.exists() or destination.exists():
        p.error('结果/计划均禁止覆盖，失败重试另建目录。')
    protocol_path = CONTROL / 'protocol.json'
    protocol = verify(read(protocol_path))
    shared = ['--model_path', str(model), '--source_path', protocol['source_path'], '--load_iter', str(args.iteration)]
    boot = [PYTHON, str(ROOT / 'LIHAN/scripts/bootstrap_1011.py'), str(ROOT)]
    frames = ['--frames'] + list(map(str, FRAMES))
    commands = [
        ('predicted_meshes', boot + ['scripts.extract_geometry_mesh_dynamic'] + shared + frames +
         ['--split', 'test', '--resolution', '2', '--voxel_size', '.02', '--sdf_trunc', '.08', '--depth_trunc', '20',
          '--alpha_threshold', '.5', '--output_dir', str(output / 'predicted_meshes')]),
        ('normal_depth', boot + ['scripts.eval_geometry_gt'] + shared + frames +
         ['--gt_root', protocol['gt_root'], '--gt_depth_dir', protocol['gt_depth_dir'], '--split', 'test', '--resolution', '2',
          '--alpha_threshold', '.5', '--interior_radius', '2', '--output_dir', str(output / 'normal_depth'),
          '--pred_mesh_manifest', str(output / 'predicted_meshes/manifest.json'), '--save_arrays']),
        ('audit', [PYTHON, str(ROOT / 'LIHAN/scripts/goal_eval_audit.py'), str(output), str(args.iteration), '--protocol', str(protocol_path)]),
        ('goal2_numeric', [PYTHON, str(ROOT / 'LIHAN/scripts/goal2_eval_target.py'), '--evaluation', str(output),
                          '--output', str(output / 'goal2_numeric')]),
        ('spatial_diagnostic', [PYTHON, str(ROOT / 'LIHAN/scripts/goal2_error_spatial_diagnostic.py'), '--evaluation', str(output),
                               '--gt_depth_dir', protocol['gt_depth_dir'], '--output', str(output / 'spatial_diagnostic'), '--figures']),
    ]
    if not args.geometry_only:
        commands.append(('four_classes', boot + ['LIHAN.scripts.goal2_eval_reset_audit_visuals'] + shared + ['--output_dir', str(output / 'four_classes')]))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'logs').mkdir()
    dependencies = ('goal2_eval_reset_audit_run.py', 'goal2_eval_target.py', 'goal2_error_spatial_diagnostic.py',
                    'goal2_eval_reset_audit_render.py', 'goal2_eval_reset_audit_visuals.py', 'goal_eval_audit.py',
                    'goal_eval_visuals.py', 'goal_eval_render.py', 'goal_eval_protocol.py')
    helper_sha = {str(ROOT / 'LIHAN/scripts' / name): digest(ROOT / 'LIHAN/scripts' / name) for name in dependencies}
    plan = dict(model=str(model), iteration=args.iteration, source=protocol['source_path'], gpu=args.gpu,
                measurement_protocol_sha256=digest(protocol_path), code_sha256=protocol['code_sha256'],
                helper_sha256=helper_sha, steps=[dict(name=n, command=c, shell=shlex.join(c)) for n,c in commands])
    write(destination / 'plan.json', plan)
    write(destination / 'protocol.json', protocol)
    lines = ['# G2原协议评测与实际ambient完整渲染', '', f'模型：`{model}`；iteration={args.iteration}；source：`{protocol["source_path"]}`。',
             '完整15固定TEST；原6源码/掩膜/TSDF/失败像素数学不改。G2阈值仅目标元数据：GS<2°、Depth/Mesh<=T13精确基线。',
             f'输出：`{output}`；日志：`{output / "logs"}`。', '', '## 完整实际命令', '']
    for name, command in commands:
        lines += [f'### {name}', '', '```bash', shlex.join(command), '```', '']
    lines += ['定量、四类目检和最终状态待实际执行后追加。新RGB必须恢复真实ambient_fraction，原static feature分离不能解读为运动。', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    if not args.execute:
        print('计划已保存：' + str(destination / 'plan.json'))
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', TORCH_CUDA_ARCH_LIST='8.6')
    libraries = ROOT / 'output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu'
    env['LD_LIBRARY_PATH'] = str(libraries)+':'+env.get('LD_LIBRARY_PATH','')
    outcomes = []
    for name, command in commands:
        verify(protocol)
        # 每步记录真实helperSHA；活跃流程中改变来源则明确失败，不能用计划SHA冒充。
        actual_sha = {path: digest(path) for path in helper_sha}
        if actual_sha != helper_sha:
            raise AssertionError('活跃评测helper发生变化，须保留本结果并显式重试。')
        log = output / 'logs' / f'{name}.log'
        with log.open('x') as stream:
            stream.write(shlex.join(command)+'\n')
            stream.flush()
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        outcomes.append(dict(step=name, exit_code=code, log=str(log), actual_helper_sha256=actual_sha))
        write(output / 'executed.json', outcomes)
        print(name + ' exit=' + str(code), flush=True)
        if code:
            write(output / 'FAILED.json', dict(step=name, exit_code=code, retained=True))
            raise SystemExit(code)
        if name == 'audit':
            report = read(output / 'goal_mae_summary.json')
            print('完整15实际原MAE/coverage：' + str(report['values']) + '；' + str(report['details']), flush=True)
    target = read(output / 'goal2_numeric/goal2_acceptance.json')
    write(output / 'PIPELINE_COMPLETED.json', dict(pipeline='PASS', original_measurement_complete=True,
        goal2_numeric=target['numeric_status'], geometry_review='PENDING', visual_review='DEFERRED_INTERMEDIATE' if args.geometry_only else 'PENDING'))


if __name__ == '__main__':
    main()
