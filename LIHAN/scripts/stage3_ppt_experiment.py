"""Stage3 展示配对实验：四臂隔离、冒烟、固定预算训练和统一评价。"""
import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
CONTROL = ROOT / 'output/1025-02-Stage3-PPT配对总控'
ARMS = {
    'static_native': ('1025-03-staticDL-Stage3原生基线', '1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段', 35000, 0),
    'static_geometry': ('1025-04-staticDL-Stage3改进几何', '1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k', 16500, 1),
    'dynamic_native': ('1025-05-dynamicDL-Stage3原生基线', '1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段', 35000, 2),
    'dynamic_geometry': ('1025-06-dynamicDL-Stage3改进几何', '1021-06-LH-dynamic-jumpingjacks_DL_V2-完整路线复现', 16500, 3),
}


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    CONTROL.mkdir(exist_ok=False)
    sources = {
        'static': ROOT / 'output/1024-02-静态ELDL-展示数值补全总控/source_DL',
        'dynamic': ROOT / 'output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view',
    }
    dynamic_frames = json.loads((ROOT / 'output/1021-02-LH-dynamic-jumpingjacks_DL_V2-阴影投影H0Stage2复现重试/data_train_v1/transforms_train.json').read_text())['frames']
    dynamic_ids = {int(f['source_frame']) for f in dynamic_frames}
    manifest = {'hostname': socket.gethostname(), 'python': PYTHON, 'sources': {}, 'created': time.time()}
    for kind, source in sources.items():
        train = json.loads((source / 'transforms_train.json').read_text())
        if kind == 'dynamic':
            train['frames'] = [f for f in train['frames'] if int(f['source_frame']) in dynamic_ids]
            assert len(train['frames']) == 117
        for mode in ['train', 'eval']:
            target = CONTROL / f'{kind}_{mode}_source'
            target.mkdir()
            for p in source.iterdir():
                if p.name.startswith('transforms_'):
                    continue
                (target / p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
            write(target / 'transforms_train.json', train)
            if mode == 'eval':
                shutil.copy2(source / 'transforms_test.json', target / 'transforms_test.json')
        manifest['sources'][kind] = {'original': str(source), 'train_count': len(train['frames']), 'test_count': 15,
                                   'train_sha256': digest(CONTROL / f'{kind}_train_source/transforms_train.json'),
                                   'test_sha256': digest(CONTROL / f'{kind}_eval_source/transforms_test.json')}
    snapshot = CONTROL / 'snapshot'
    snapshot.mkdir()
    for name in ['arguments', 'scene', 'scripts', 'utils', 'gaussian_renderer', 'lpipsPyTorch', 'LIHAN/scripts']:
        shutil.copytree(ROOT / name, snapshot / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['submodules', 'assets', 'example_envmaps']:
        (snapshot / name).symlink_to((ROOT / name).resolve(), target_is_directory=True)
    manifest['code_sha256'] = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py') if not p.is_symlink()}
    write(CONTROL / 'manifest.json', manifest)


def run(arm, smoke, retry=''):
    name, parent_name, start, gpu = ARMS[arm]
    kind = arm.split('_')[0]
    output = ROOT / 'output' / ('smoke_test' if smoke else '') / (name + ('-冒烟' if smoke else '') + retry)
    output.mkdir(parents=True, exist_ok=False)
    parent = ROOT / 'output' / parent_name / 'model_mlp'
    end = start + (24 if smoke else 20000)
    snapshot = CONTROL / ('snapshot_v2' if (CONTROL / 'snapshot_v2').exists() else 'snapshot')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       PYTHONUNBUFFERED='1', TORCH_EXTENSIONS_DIR=str(CONTROL / 'native_cache'),
                       CUDA_HOME=str(Path(PYTHON).parents[1]), TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='4',
                       PYTHONPATH=str(ROOT / 'output/archive/1011-07-六组训练总控/tools/nvdiffrast-sm86/extension'))
    boot = [PYTHON, str(snapshot / 'LIHAN/scripts/bootstrap_1011.py'), str(snapshot)]
    command = boot + ['LIHAN.scripts.stage3_ppt_native_entry', '--source_path', str(CONTROL / f'{kind}_train_source'),
                     '--model_path', str(output / 'model_mlp'), '--stage1_model_path', str(parent), '--load_iter', str(start),
                     '--iterations', str(end), '--test_iterations', str(end), '--save_iterations',
                     *[str(start + n) for n in ([24] if smoke else [4000, 12000, 20000])],
                     '--is_blender', '--eval', '--load2gpu_on_the_fly', '--resolution', '2', '--train_light_folder', 'images',
                     '--test_light_folder', 'images', '--gt_alpha_mask_as_scene_mask', '--diffuse_sample_num', '512',
                     '--depth_ratio', '1', '--pbr_rasterizer_backend', 'lumimotion']
    state = dict(arm=arm, smoke=smoke, status='RUNNING', source=str(CONTROL / f'{kind}_train_source'),
                 parent=str(parent), source_iteration=start, iteration=end, gpu=gpu, started=time.time(),
                 command=command, environment={k: environment[k] for k in ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'PYTHONPATH', 'TORCH_EXTENSIONS_DIR']},
                 parent_sha256={str(parent / p): digest(parent / p) for p in [f'point_cloud/iteration_{start}/point_cloud.ply', f'deform/iteration_{start}/deform.pth']})
    state['optimizer_updates'] = sum(i % 4 == 0 for i in range(start + 1, end))
    write(output / 'state.json', state)
    readme = output / 'README.md'
    readme.write_text('# Stage3 原生 PBR 配对实验\n\n逻辑 Stage3 对应原生 train_stage2。原生损失、每4循环更新规则保持；默认混合深度 depth_ratio=1。\n\n'
                      + f'输入：{state["source"]}；父：{parent}@{start}；输出：model_mlp@{end}；新增循环{end-start}，真实更新{state["optimizer_updates"]}。GPU{gpu}。\n\n'
                      + '完整训练命令：\n```bash\n' + shlex.join(command) + '\n```\n\n环境：' + json.dumps(state['environment'], ensure_ascii=False)
                      + '\n\n训练日志：train.log；当前未读取 TEST。渲染及指标待完成。四类 RGB/Alpha/Normal/分离目检待完成；材质另列。最终结论 PENDING。\n')

    def execute(label, argv):
        state[label + '_command'] = argv
        with readme.open('a') as f:
            f.write('\n完整命令（' + label + '）：\n```bash\n' + shlex.join(argv) + '\n```\n日志：' + label + '.log\n')
        with (output / (label + '.log')).open('x') as log:
            process = subprocess.Popen(argv, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state['pid'] = process.pid
            write(output / 'state.json', state)
            result = process.wait()
        state[label + '_exit_code'] = result
        write(output / 'state.json', state)
        if result:
            raise RuntimeError(label + ' 失败：' + str(result))

    try:
        execute('train', command)
        state['status'] = 'TRAIN_COMPLETED'
        # 冒烟仅用 TRAIN；正式模型训练全部结束后才接入隔离 TEST。
        evaluate = boot + ['LIHAN.scripts.stage3_ppt_evaluate', '--model', str(output / 'model_mlp'), '--iteration', str(end),
                           '--source', str(CONTROL / f'{kind}_eval_source'), '--output', str(output / 'evaluation'),
                           '--split', 'train' if smoke else 'test']
        execute('evaluation', evaluate)
        state['status'] = 'SMOKE_READY_FOR_REVIEW' if smoke else 'READY_FOR_REVIEW'
    except Exception as exc:
        state['status'] = 'FAILED'
        state['error'] = str(exc)
        with readme.open('a') as f:
            f.write('\n最终结论 FAILED；失败现场完整保留。' + str(exc) + '\n')
    state['finished'] = time.time()
    write(output / 'state.json', state)
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state['status'] != 'FAILED' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--retry', default='')
    args = p.parse_args()
    assert socket.gethostname() == 'ushas' and Path(sys.executable).resolve() == Path(PYTHON).resolve()
    if args.prepare:
        prepare()
    else:
        sys.exit(run(args.arm, args.smoke, args.retry))
