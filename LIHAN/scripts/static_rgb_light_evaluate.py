"""冻结后的独立GT灯向评测；禁止将评测结果回传估计或参数选择。"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = tuple(f'E{i}' for i in range(5))
FRAMES = np.arange(1, 151)
DEFAULT = ROOT / 'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_freeze(predictions, manifest):
    """在任何GT读取前校验预测、方法代码和冻结基线。"""
    frozen = json.loads(manifest.read_text())
    require(bool(frozen.get('frozen_at')), '缺少冻结时间 frozen_at')
    require(frozen.get('GT_read_before_freeze') is False, '必须声明冻结前未读取GT')
    require(frozen.get('final_method') in METHODS, '必须在冻结前选定 final_method=E0..E4')
    require(frozen.get('coordinate_system') == 'world', '预测必须声明世界坐标')
    require(frozen.get('direction_convention') == 'surface_to_light', '预测必须声明surface_to_light')
    require(frozen.get('frames') == FRAMES.tolist(), '冻结清单必须含完整1..150帧')
    require(sha(predictions) == frozen.get('predictions_sha256'), '预测SHA256与冻结清单不一致')
    files = frozen.get('files', {})
    require(isinstance(files, dict) and files, '缺少冻结代码/模型files映射')
    paths = [resolve(path) for path in files]
    require(any(path.suffix == '.py' for path in paths), '必须冻结方法代码SHA256')
    require(any(path.suffix in {'.pth', '.pt', '.ply'} for path in paths), '必须冻结Stage1模型SHA256')
    for path, expected in files.items():
        actual = resolve(path)
        require(actual.name != 'lights.json', '冻结清单不应包含GT灯向文件')
        require(sha(actual) == expected, f'冻结文件SHA256不一致：{path}')
    with np.load(predictions, allow_pickle=False) as saved:
        require('frames' in saved and np.array_equal(saved['frames'], FRAMES), '预测必须完整且顺序为1..150')
        values = {method: np.asarray(saved[method], dtype=np.float64) for method in METHODS}
    for method, value in values.items():
        require(value.shape == (150, 3), f'{method}方向必须为150×3')
        require(np.isfinite(value).all(), f'{method}存在非有限方向')
        norms = np.linalg.norm(value, axis=1)
        require(np.all(np.abs(norms - 1) < 1e-4), f'{method}必须预先保存单位方向')
        values[method] = value / norms[:, None]
    return frozen, values


def read_gt(source):
    """只按导出约定读取顶层SUN方向，不搜索符号或变换。"""
    path = source / 'lights.json'
    records = json.loads(path.read_text())
    require(set(records) == {f'{i:04d}' for i in FRAMES}, 'GT帧必须恰好为0001..0150')
    directions = []
    for frame in FRAMES:
        item = records[f'{frame:04d}']
        require(item.get('source_type') == 'SUN', f'{frame}顶层主灯不是SUN')
        if 'direction_convention' in item:
            require(item['direction_convention'] == 'Unit vector from surface toward sun; max(dot(normal, light_dir_world), 0)',
                    f'{frame}GT方向定义与导出代码不同')
        direction = np.asarray(item['light_dir_world'], dtype=np.float64)
        require(direction.shape == (3,) and np.isfinite(direction).all(), f'{frame}GT方向无效')
        norm = np.linalg.norm(direction)
        require(norm > 1e-12, f'{frame}GT方向为零')
        directions.append(direction / norm)
    return np.stack(directions), sha(path)


def evaluate(predictions, gt):
    return {method: np.degrees(np.arccos(np.clip(np.sum(value * gt, axis=1), -1., 1.)))
            for method, value in predictions.items()}


def write_outputs(output, predictions, gt, errors, final_method):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (output / 'per_frame.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'method', 'pred_x', 'pred_y', 'pred_z', 'gt_x', 'gt_y', 'gt_z', 'angular_error_deg', 'within_5deg'])
        for method in METHODS:
            for idx, frame in enumerate(FRAMES):
                writer.writerow([int(frame), method, *predictions[method][idx], *gt[idx], float(errors[method][idx]), bool(errors[method][idx] <= 5)])
    metrics = {}
    for method, value in errors.items():
        metrics[method] = dict(mean_deg=float(value.mean()), median_deg=float(np.median(value)),
                               p90_deg=float(np.percentile(value, 90)), within_5deg_count=int((value <= 5).sum()),
                               within_5deg_ratio=float((value <= 5).mean()), frame_count=len(value))
    fig, ax = plt.subplots(figsize=(12, 5))
    for method in METHODS:
        ax.plot(FRAMES, errors[method], label=method, lw=1.3)
    ax.axhline(5, color='black', ls='--', label='5 degrees')
    ax.set(xlabel='Frame', ylabel='Angular error (degrees)', xlim=(1, 150))
    ax.legend(ncol=6)
    fig.tight_layout(); fig.savefig(output / 'angular_error_curve.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in METHODS:
        ax.plot(np.sort(errors[method]), np.arange(1, 151) / 150, label=method)
    ax.axvline(5, color='black', ls='--')
    ax.set(xlabel='Angular error (degrees)', ylabel='Cumulative frame fraction', ylim=(0, 1))
    ax.legend()
    fig.tight_layout(); fig.savefig(output / 'angular_error_cdf.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for axis, label in enumerate('xyz'):
        axes[axis].plot(FRAMES, gt[:, axis], color='black', label='GT')
        axes[axis].plot(FRAMES, predictions[final_method][:, axis], label=final_method)
        axes[axis].set_ylabel(label); axes[axis].legend()
    axes[-1].set_xlabel('Frame')
    fig.tight_layout(); fig.savefig(output / 'direction_components.png', dpi=160); plt.close(fig)
    worst = np.argsort(-errors[final_method], kind='stable')[:12]
    cases = [dict(frame=int(FRAMES[i]), error_deg=float(errors[final_method][i]),
                  prediction=predictions[final_method][i].tolist(), gt=gt[i].tolist()) for i in worst]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(item['frame']) for item in cases], [item['error_deg'] for item in cases])
    ax.axhline(5, color='black', ls='--'); ax.set(xlabel='Frame (worst first)', ylabel='Angular error (degrees)', title=final_method)
    fig.tight_layout(); fig.savefig(output / 'failure_cases.png', dpi=160); plt.close(fig)
    (output / 'failure_cases.json').write_text(json.dumps(dict(method=final_method, cases=cases,
        interpretation='按冻结最终方法误差排序的12帧；仅定位失败，不据此修改预测。误差本身不能区分法线偏差、阴影、材质混淆或不可辨识性。'), ensure_ascii=False, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=DEFAULT / 'predictions.npz')
    parser.add_argument('--freeze-manifest', type=Path, default=DEFAULT / 'freeze_manifest.json')
    parser.add_argument('--source', type=Path, default=ROOT / 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3')
    parser.add_argument('--output', type=Path, default=DEFAULT / 'evaluation')
    parser.add_argument('--validate-freeze-only', action='store_true', help='仅核验冻结内容，绝不读取GT')
    parser.add_argument('--allow-gt-evaluation', action='store_true', help='收到冻结信号后才能明确启用GT读取')
    args = parser.parse_args()
    frozen, predictions = validate_freeze(args.predictions, args.freeze_manifest)
    if args.validate_freeze_only:
        print('冻结核验通过，GT未读取。')
        return
    require(args.allow_gt_evaluation, '必须在冻结完成后明确传入--allow-gt-evaluation')
    require(args.output.resolve().is_relative_to((ROOT / 'output').resolve()), '评测输出必须位于output目录')
    require(not args.output.exists(), '禁止覆盖已有评测（包括失败结果）')
    receipt = args.freeze_manifest.resolve().parent / 'gt_evaluation_receipt.json'
    require(receipt.parent.is_relative_to((ROOT / 'output').resolve()), '冻结清单必须位于output实验目录')
    # exclusive创建：更换输出路径也不能重复读取同一冻结实验的GT。
    with receipt.open('x') as handle:
        json.dump(dict(status='GT_EVALUATION_STARTED', started_at=datetime.now(timezone.utc).isoformat(),
                       freeze_manifest_sha256=sha(args.freeze_manifest), final_method=frozen['final_method'],
                       predictions_sha256=sha(args.predictions), output=str(args.output.resolve()),
                       failure_policy='任何失败均保留本收据与已有产物，禁止自动重开GT评测'),
                  handle, ensure_ascii=False, indent=2)
    args.output.mkdir(parents=True, exist_ok=False)
    # 任何GT打开操作均在上述全部冻结核验及显式评测开关之后。
    gt, gt_hash = read_gt(args.source)
    errors = evaluate(predictions, gt)
    metrics = write_outputs(args.output, predictions, gt, errors, frozen['final_method'])
    final = frozen['final_method']
    result = dict(status='PASS' if metrics[final]['mean_deg'] <= 5 else 'FAILED', final_method=final,
                  target_mean_deg=5, methods=metrics, evaluated_at=datetime.now(timezone.utc).isoformat(),
                  freeze_manifest_sha256=sha(args.freeze_manifest), predictions_sha256=sha(args.predictions),
                  frozen_files=frozen['files'], gt_sha256=gt_hash, evaluation_code_sha256=sha(__file__),
                  coordinate_system='world', direction_convention='surface_to_light', sign_search_performed=False,
                  gt_used_for_estimation=False, gt_feedback_parameter_tuning=False,
                  gt_definition_evidence=['_export/skill_exporter_original.py:490-491：世界变换乘局部+Z并归一化',
                                          '_export/run_export.py:151,157：surface向SUN，max(dot(normal,light_dir_world),0)'])
    (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
