"""固定几何材质预适配的双父预算与来源审计；不使用GT择优。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FRAMES = [i for i in range(1, 151) if i % 10]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(ROOT / 'output') and not out.exists(), '必须新审计目录。')
    expected_steps = [3, 20] if args.smoke else [1000, 16000]
    steps = 20 if args.smoke else 4050
    saves = [0, 10, 20] if args.smoke else [0, 1350, 2700, 3375, 4050]
    rows, hashes, orders = [], {}, []
    for experiment, parent_step in zip(args.experiments, expected_steps):
        experiment = experiment.resolve()
        require(experiment.is_relative_to(ROOT / 'output') and ('smoke_test' in experiment.parts) == args.smoke, '正式/冒烟输出混用。')
        fit = experiment / 'material_fit'
        paths = [experiment / 'plan.json', experiment / 'state.json', fit / 'material_manifest.json', fit / 'configuration.json', fit / 'training_finished.json', fit / 'training.jsonl']
        state, manifest, cfg, finish = map(read, paths[1:5])
        require(state['exit_code'] == 0 and finish['actual_updates'] == steps, '训练没有完成登记预算。')
        require(manifest['actual_entrypoint'] == cfg['actual_entrypoint'] == 'scripts/preadapt_eldl_v3_material.py', '未知预适配入口。')
        require(manifest['smoke'] == cfg['smoke'] == args.smoke and manifest['parent_geometry_step'] == cfg['parent_geometry_step'] == parent_step, '父几何/正式身份不同。')
        require(manifest['source'] == cfg['source'] and manifest['initial_model'] == cfg['initial_model'] and manifest['geometry_binding'] == cfg['geometry_binding'], '来源绑定不同。')
        require(manifest['configuration_sha256'] == sha(fit / 'configuration.json') and manifest['input_sha256'] == cfg['input_sha256'], '配置或输入清单不符。')
        require(manifest['gt_supervision'] is False and manifest['external_test_read'] is False and cfg['train_frames'] == FRAMES, '无GT/TRAIN边界非法。')
        require(manifest['only_tensor'] == cfg['only_trainable_tensor'] == '_albedo_dc' and manifest['non_dc_bitexact'] and manifest['all_geometry_render_maps_bitexact'] and manifest['no_optimizer_rng_transfer'], '仅DC冻结声明不全。')
        require(manifest['formal_use_permitted'] is False and manifest['pair_eligible'] is False, '单父不得自行授权。')
        require(cfg['steps'] == manifest['completed_optimizer_steps'] == steps and cfg['optimizer'] == 'Adam' and cfg['eps'] == 1e-15 and cfg['learning_rates'] == [.01, .001, .0001] and cfg['no_visibility'] is True, '预适配预算/目标不同。')
        for name, expected in cfg['input_sha256'].items():
            require(sha(Path(name)) == expected, '输入SHA变更：' + name)
            hashes[name] = expected
        history = list(map(json.loads, (fit / 'training.jsonl').read_text().splitlines()))
        require(len(history) == steps, '实际更新记录缺失。')
        generator = np.random.default_rng(0)
        expected_order = [FRAMES[int(i)] for _ in range(1 if args.smoke else 30) for i in generator.permutation(135)][:steps]
        orders.append([r['frame'] for r in history])
        require(orders[-1] == expected_order, '相机顺序不符合登记seed0。')
        for i, row in enumerate(history, 1):
            phase = min((i - 1) // (7 if args.smoke else 1350), 2)
            require(row['step'] == i and row['epoch'] == (i - 1) // 135 + 1 and row['lr'] == [.01, .001, .0001][phase], '实际学习率/epoch不同。')
            require(math.isfinite(row['photo']) and math.isfinite(row['dc_gradient_norm']), '存在非有限。')
        require(any(r['dc_gradient_norm'] > 0 for r in history), 'DC无有效梯度。')
        base_maps, scores = None, {}
        terminal_dc = None
        for step in saves:
            score_path = fit / f'fit_step_{step}.json'
            dc_path = fit / f'dc_step_{step}.pt'
            opt_path = fit / f'optimizer_step_{step}.pt'
            paths += [score_path, dc_path, opt_path]
            score = read(score_path)
            require(score['step'] == step and [r['frame'] for r in score['per_frame']] == FRAMES and score['geometry_maps_bitexact'], '回放缺帧/错步。')
            actual = sum(r['photo'] for r in score['per_frame']) / 135
            require(math.isfinite(actual) and abs(actual - score['frame_mean']) < 1e-14 and actual == manifest['scores'][str(step)], '完整回放汇总不符。')
            scores[str(step)] = actual
            maps = [r['geometry_map_sha256'] for r in score['per_frame']]
            if base_maps is None:
                base_maps = maps
            require(maps == base_maps, '保存点几何渲染不是逐位不变。')
            dc = torch.load(dc_path, map_location='cpu', weights_only=True)
            require(set(dc) == {'_albedo_dc'} and dc['_albedo_dc'].shape == (manifest['points'], 1, 3) and torch.isfinite(dc['_albedo_dc']).all(), 'DC导出schema不符。')
            terminal_dc = dc['_albedo_dc']
            opt = torch.load(opt_path, map_location='cpu', weights_only=False)
            require(opt['step'] == step and len(opt['optimizer']['param_groups']) == 1, '优化器不是单一DC。')
            states = opt['optimizer']['state']
            require((not states) if step == 0 else (len(states) == 1 and int(next(iter(states.values()))['step']) == step), 'Adam真实步数不符。')
        zero = torch.load(fit / 'dc_step_0.pt', map_location='cpu', weights_only=True)['_albedo_dc']
        require(not torch.equal(zero, terminal_dc), '终点DC未改变。')
        c0 = .28209479177387814
        require(float(terminal_dc.min()) >= (.031 - .5) / c0 - 1e-6 and float(terminal_dc.max()) <= (.969 - .5) / c0 + 1e-6, '终点DC越界。')
        require(manifest['material'] == str(fit / f'dc_step_{steps}.pt') and manifest['material_sha256'] == sha(Path(manifest['material'])), '终点材质路径或SHA不符。')
        change = None if args.smoke else abs(scores['4050'] - scores['3375']) / max(scores['3375'], 1e-12)
        eligible = not args.smoke and scores['4050'] < scores['0'] and change <= .01
        require(manifest['budget_eligible'] == finish['budget_eligible'] == eligible and manifest['terminal_absolute_relative_change'] == change, '预算门槛计算不符。')
        if args.smoke:
            reload_path = fit / 'material_reload_audit.json'
            reload = read(reload_path)
            require(reload['status'] == 'PASS' and [r['frame'] for r in reload['rows']] == [1, 81, 149] and reload['extra_forward_calls'] == 6, '缺少真实材质重载。')
            require(all(math.isfinite(v) and v < 1e-5 for r in reload['rows'] for v in r['max_abs'].values()), '生产重载不一致。')
            paths.append(reload_path)
        for path in paths:
            hashes[str(path)] = sha(path)
        rows.append(dict(experiment=str(experiment), fit=str(fit), manifest=str(fit / 'material_manifest.json'), manifest_sha256=sha(fit / 'material_manifest.json'), material=manifest['material'], material_sha256=manifest['material_sha256'], parent_geometry_step=parent_step, geometry_sha256=manifest['geometry_binding']['geometry_sha256'], scores=scores, terminal_absolute_relative_change=change, budget_eligible=eligible))
    require(orders[0] == orders[1], '两个父起点实际相机顺序不同。')
    eligible = all(r['budget_eligible'] for r in rows)
    status = 'PASS_MATERIAL_PREADAPT_SMOKE_ENGINEERING' if args.smoke else ('PASS_MATERIAL_PREADAPT_PAIR_BUDGET' if eligible else 'FAILED_MATERIAL_PREADAPT_PAIR_BUDGET')
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    out.mkdir()
    result = dict(status=status, smoke=args.smoke, formal_use_permitted=eligible, pair_eligible=eligible, scientific_accuracy_claim=False, gt_read=False, external_test_read=False, allowed_geometry_steps=expected_steps, entries=rows, common_camera_order=True, input_sha256=hashes)
    (out / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=status, formal_use_permitted=eligible, entries=rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
