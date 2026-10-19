"""从真实EL_V2资产与原始预测NPZ复核固定分母法线指标，不参与训练。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.geometry_eval_utils import evaluate_region, unit_normal


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/'output')
    out.mkdir(parents=True, exist_ok=False)
    base = ROOT/'output/1011-04-LH-dynamic-jumpingjacks_EL_V2-原生LumiMotion两阶段'
    metrics_path = base/'evaluation_35000/normal_depth/metrics.json'
    historical = json.loads(metrics_path.read_text())
    rows, hashes = [], {str(metrics_path): sha(metrics_path)}
    for i in range(10, 151, 10):
        path = base/f'evaluation_35000/normal_depth/prediction_{i:04d}.npz'
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            prediction = {k: data[k] for k in ('alpha','normal','expected','median','depth_normal','mesh_normal','mesh_valid','mesh_depth')}
            normal, valid = unit_normal(data['gt_normal'])
            report = evaluate_region(prediction, normal, valid, data['gt_depth'], data['gt_foreground'], .5)
        assert report['gt_pixels'] > 0
        row = dict(frame=i, gt_pixels=report['gt_pixels'], methods={})
        for key, item in report['normal_metrics'].items():
            assert item['status'] == 'evaluated'
            assert item['angular_with_missing_deg']['count'] == report['gt_pixels']
            row['methods'][key] = dict(conditional_mae=item['angular_deg']['mean'],
                fixed_denominator_mae=item['angular_with_missing_deg']['mean'],
                valid_pixels=item['valid_pixels'], coverage=item['coverage'])
        rows.append(row)
    summary = {}
    for method in ('gs_render','depth_normal','mesh'):
        summary[method] = dict(
            conditional_mae=float(np.mean([r['methods'][method]['conditional_mae'] for r in rows])),
            fixed_denominator_mae=float(np.mean([r['methods'][method]['fixed_denominator_mae'] for r in rows])),
            pixel_weighted_coverage=sum(r['methods'][method]['valid_pixels'] for r in rows)/sum(r['gt_pixels'] for r in rows))
    # 与历史原件三单项复算吻合才接受这个补充口径。
    for method, expected in [('gs_render',5.661060810285603),('depth_normal',10.552236907360648),('mesh',7.550293044074406)]:
        assert abs(summary[method]['conditional_mae']-expected) < 1e-8
    assets = [base/'stage1_frozen/cfg_args',
              base/'stage1_frozen/point_cloud/iteration_35000/point_cloud.ply',
              base/'stage1_frozen/deform/iteration_35000/deform.pth',
              base/'planned_commands.json']
    for path in assets:
        assert path.is_file()
        hashes[str(path)] = sha(path)
    # 模型评测引用原model_mlp，必须证明35000权重与冻结副本完全一致。
    for relative in ('point_cloud/iteration_35000/point_cloud.ply','deform/iteration_35000/deform.pth'):
        live, frozen = base/'model_mlp'/relative, base/'stage1_frozen'/relative
        assert sha(live) == sha(frozen), '历史评测与冻结Stage1不是相同资产'
        hashes[str(live)] = sha(live)
    new = ROOT/'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3'
    old = ROOT/'data/LH-data/danamic/LH-dynamic-jumpingjacks_EL_V2'
    a, b = [json.loads((p/'camera.json').read_text()) for p in (new,old)]
    pose_a, pose_b = [json.loads((p/'object_pose.json').read_text()) for p in (new,old)]
    camera_error = max(float(np.max(np.abs(np.asarray(a['frames'][k]['extrinsics']['camera_to_world'])-
                                          np.asarray(b['frames'][k]['extrinsics']['camera_to_world'])))) for k in a['frames'])
    object_errors, bone_errors, missing_bone_frames = [], [], []
    for key in a['frames']:
        x, y = pose_a['frames'][key], pose_b['frames'][key]
        for name in x.keys() & y.keys():
            object_errors.append(float(np.max(np.abs(np.asarray(x[name]['world_pose']['matrix_world'])-
                                                     np.asarray(y[name]['world_pose']['matrix_world'])))))
        bx = x.get('Character_Armature', {}).get('bones')
        by = y.get('Character_Armature', {}).get('bones')
        if not bx or not by:
            missing_bone_frames.append(key)
            continue
        for name in bx.keys() & by.keys():
            bone_errors.append(float(np.max(np.abs(np.asarray(bx[name]['matrix_world'])-np.asarray(by[name]['matrix_world'])))))
    correspondence = dict(camera_c2w_max_abs_difference=camera_error,
        common_object_matrix_max_abs_difference=max(object_errors),
        common_bone_matrix_max_abs_difference=max(bone_errors) if bone_errors else None,
        frames_missing_comparable_bone_metadata=missing_bone_frames,
        geometry_vertex_topology_equivalence_verified=False, evaluation_region_equivalence_verified=False,
        old_time_rule='frame/150', new_time_rule='(frame-1)/149',
        old_stage1_train_frames=135, new_stage1_train_frames=117,
        conclusion='相机可对应；物体矩阵不能证明骨骼蒙皮或网格逐顶点相同。EL_V2保留指定数值目标，不把跨数据差异全部归因于方法。')
    for p in (new,old):
        for name in ('camera.json','object_pose.json'):
            hashes[str(p/name)] = sha(p/name)
    result = dict(status='PASS_HISTORICAL_ASSET_AND_METRIC_RECOMPUTATION',
        model=str(base/'stage1_frozen'), iteration=35000, gaussian_count=historical['gaussian_count'],
        methods=summary,
        normal_composite_fixed_denominator_mae=float(np.mean([v['fixed_denominator_mae'] for v in summary.values()])),
        normal_composite_conditional_mae=float(np.mean([v['conditional_mae'] for v in summary.values()])),
        protocol=dict(frames=list(range(10,151,10)),resolution=[400,400],alpha_threshold=.5,
            region='原NPZ的gt_foreground且GT法线有效，所有法线方法同一分母；Depth有效性保留原1像素邻域语义',
            missing_normal_penalty_degrees=180,frame_aggregation='equal_frame_mean',method_aggregation='equal_method_mean',
            scope='从历史NPZ重算；不是用新训练运行时重新渲染，统一当前renderer的复核仍待完成'),
        per_frame=rows, dataset_correspondence=correspondence, source_sha256=hashes)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(methods=summary,normal_composite_fixed_denominator_mae=result['normal_composite_fixed_denominator_mae'],correspondence=correspondence),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
