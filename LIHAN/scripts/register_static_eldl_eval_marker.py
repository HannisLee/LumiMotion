"""为1024静态ELDL模型写入评测入口所需 eldl_v3_training.json marker（不改checkpoint）。

S0（scripts/train_static_eldl_stage1.py）→ baseline=static_sh（零几何头；pred_color=False，d_color=None，原生SH着色）；
A0/G（scripts/train_static_eldl_physical.py）→ baseline=static_eldl_mixed（由 eldl_1024_eval_static_gt.py 包装器登记）。
args 由训练实际命令按原argparse重新解析得到带类型的完整参数，不手写猜测。
"""
import argparse
import hashlib
import json
import shlex
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIT = [n for n in range(1, 151) if n % 10 and not 61 <= n <= 79]
HELD = [n for n in range(61, 80) if n != 70]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def typed_native_args(command_file, snapshot):
    sys.path.insert(0, str(snapshot))
    from arguments import ModelParams, OptimizationParams, PipelineParams
    tokens = shlex.split(Path(command_file).read_text())
    tokens = tokens[tokens.index('scripts.train_static_eldl_stage1') + 1:]
    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    parser.add_argument('--split'); parser.add_argument('--save_iterations', type=int, nargs='+')
    parser.add_argument('--deform-type', dest='deform_type'); parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = vars(parser.parse_args(tokens))
    args['pred_color'] = False
    return args


def main(a):
    model = a.model.resolve()
    marker = model / 'eldl_v3_training.json'
    if marker.exists():
        raise SystemExit('marker已存在，禁止覆盖：' + str(marker))
    if a.kind == 'S0':
        run = model.parent
        rec = json.loads((model / 'training_finished.json').read_text())
        assert rec['stage'] == 'S0静态原生Stage1' and rec['status'] == 'TRAINING_FINISHED_PENDING_EVALUATION'
        assert rec['geometry_heads_final_all_zero'] and rec['deform_state_unchanged'] and rec['train_frames'] == FIT
        args = typed_native_args(run / 'train_command.sh', run / 'snapshot_code')
        assert args['source_path'] == rec['args']['source_path'] and args['iterations'] == 35000
        out = dict(status=rec['status'], baseline='static_sh', args=args, gt_supervision=False, init_weights_loaded=False,
                   train_frames=FIT, internal_held_frames=HELD, external_test_read=False,
                   input_manifest_sha256=rec['input_manifest_sha256'], rasterizer_provenance=rec['rasterizer_provenance'],
                   source_code_root=str(run / 'snapshot_code'), actual_entrypoint='scripts/train_static_eldl_stage1.py',
                   actual_optimizer_updates=rec['actual_optimizer_updates'],
                   static_config=rec['static_config'],
                   registration='1024评测adapter登记：训练后补写marker（训练记录为static_eldl_stage1.json/training_finished.json）。'
                                'static_sh语义：几何头严格零；本模型另有pred_color=False，形变step返回d_color=None，原生SH着色无时变颜色。',
                   training_record_sha256=sha(model / 'training_finished.json'))
    else:
        fin = json.loads((model / 'physical_finished.json').read_text())
        plan = json.loads((model / 'physical_plan.json').read_text())
        assert fin['status'] == 'TRAINING_FINISHED' and not fin['smoke'] and fin['stage'] in ('A0', 'G', 'F', 'R', 'H')
        args = dict(sh_degree=3, source_path=plan['args']['source'], model_path=str(model), images='images',
                    train_light_folder='images', test_light_folder='images', resolution=2, white_background=False,
                    data_device='cpu', eval=True, load2gpu_on_the_fly=True, is_blender=True, deform_type='mlp', hyper_dim=1,
                    pred_color=False, no_binary_separation=False, convert_SHs_python=False, compute_cov3D_python=False,
                    debug=False, depth_ratio=1.0, warm_up=1000, oneupSHdegree_step=1000, binarization_warm_up=1000,
                    smoke=False, stage=fin['stage'], route=fin['route'], directions=plan['args']['directions'],
                    shadow_bias=plan['frozen']['shadow_bias'], geometry_lr_multiplier=plan.get('geometry_lr_multiplier', 1.0),
                    photo_partial_alpha=plan.get('photo_partial_alpha', 'keep'),
                    opacity_lr_multiplier=plan.get('opacity_lr_multiplier', 1.0),
                    normal_weight=plan.get('normal_weight', 0.1), composite=plan.get('composite', 'linear'),
                    rotation_lr_multiplier=plan.get('rotation_lr_multiplier'), photo_threshold=plan.get('photo_threshold', 0.08),
                    seed=plan.get('seed', 1012), continue_from_g=plan.get('continue_from_g', False))
        out = dict(status=fin['status'], baseline='static_eldl_mixed', args=args,
                   gt_supervision='TRAIN_GT_direction_only', init_weights_loaded='S0谱系父几何（GOAL阶段B允许）',
                   train_frames=FIT, internal_held_frames=HELD, external_test_read=False,
                   rasterizer_provenance=plan['rasterizer'], source_code_root=a.snapshot.resolve().as_posix(),
                   actual_entrypoint={'F': 'scripts/train_static_eldl_topology.py', 'R': 'scripts/train_static_eldl_reset.py',
                        'H': 'scripts/train_static_eldl_lineage_h.py'}.get(
                       fin['stage'], 'scripts/train_static_eldl_physical.py'), variant=fin.get('variant'),
                   stage=fin['stage'], route=fin['route'], densify=fin.get('densify'),
                   final_iteration_relative_total=fin['final_iteration_relative_total'],
                   lineage_updates=fin['lineage_updates'], physical_plan_sha256=sha(model / 'physical_plan.json'),
                   physical_finished_sha256=sha(model / 'physical_finished.json'),
                   registration='1024评测adapter登记：GT方向仅TRAIN训练使用；评测DEV/TEST使用对应帧GT方向（评测专用），由'
                                'LIHAN/scripts/eldl_1024_eval_static_gt.py包装器渲染混合光RGB；几何法线与static_sh相同语义。')
    marker.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(marker, sha(marker))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['S0', 'physical'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, help='physical：训练所用源码快照')
    main(p.parse_args())
