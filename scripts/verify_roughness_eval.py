"""不导入渲染器或指标实现，使用保存的浮点图在 CPU 独立复算 roughness 评测。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def close(actual, expected):
    if expected is None:
        assert actual is None, (actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def verify(directory):
    report = json.loads((directory / 'metrics.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    if report.get('schema_version', 1) >= 2:
        return verify_foreground(directory, report, provenance)
    palette = provenance['material_manifest']['materials']
    names = ['foreground'] + [m['name'] for m in palette]
    sums = {name: [0, 0, 0.] for name in names}
    total_core = 0
    assert [r['frame'] for r in report['frames']] == [f'{i:04d}' for i in range(8, 121, 8)]
    rows = list(csv.DictReader((directory / 'frames.csv').open()))
    assert len(rows) == 15
    for rec, row in zip(report['frames'], rows):
        assert row['frame'] == rec['frame']
        with np.load(directory / 'float' / (rec['frame'] + '.npz')) as arrays:
            for value in arrays.values():
                assert value.shape == (360, 640) and np.isfinite(value).all()
            alpha = arrays['alpha'].astype(np.float64)
            accumulated = arrays['accumulated'].astype(np.float64)
            pred = np.where(alpha > 1e-6, accumulated / np.maximum(alpha, 1e-6), 0.)
            np.testing.assert_allclose(pred, arrays['roughness'], atol=1e-12, rtol=1e-12)
            labels, core = arrays['material_labels'], arrays['gt_core']
            assert np.isin(labels, [0, 1, 2, 3, 4]).all()
            np.testing.assert_array_equal(arrays['gt_mask'], labels > 0)
            assert not ((labels > 0) & ~core).any()
            np.testing.assert_array_equal(arrays['valid_mask'], (labels > 0) & (alpha >= 0.9))
            gt = np.zeros_like(pred)
            for i, material in enumerate(palette, 1):
                gt[labels == i] = material['roughness']
            np.testing.assert_array_equal(gt, arrays['gt_roughness'])
            total_core += int(core.sum())
            close(rec['gt_core_pixels'], int(core.sum()))
            close(rec['gt_scorable_pct'], 100 * int((labels > 0).sum()) / int(core.sum()))
            for i, name in enumerate(names):
                mask = labels > 0 if i == 0 else labels == i
                valid = mask & (alpha >= 0.9)
                n, k = int(mask.sum()), int(valid.sum())
                error = float(np.abs(pred[valid] - gt[valid]).sum(dtype=np.float64))
                expected = {'gt_pixels': n, 'valid_pixels': k, 'absolute_error_sum': error,
                            'mae': error / k if k else None, 'coverage_pct': 100 * k / n if n else None}
                original = rec['foreground'] if i == 0 else rec['materials'][name]
                for key, value in expected.items():
                    close(original[key], value)
                for key in ('mae', 'coverage_pct'):
                    value = row[key if i == 0 else name + '_' + key]
                    close(float(value) if value else None, expected[key])
                sums[name][0] += n
                sums[name][1] += k
                sums[name][2] += error
    material_maes = []
    for name, (n, k, error) in sums.items():
        expected = {'gt_pixels': n, 'valid_pixels': k, 'absolute_error_sum': error,
                    'mae': error / k if k else None, 'coverage_pct': 100 * k / n if n else None}
        original = report['summaries']['foreground'] if name == 'foreground' else report['summaries']['materials'][name]
        for key, value in expected.items():
            close(original[key], value)
        if name != 'foreground':
            material_maes.append(expected['mae'])
    close(report['summaries']['balanced_material_mae'],
          sum(material_maes) / 4 if all(m is not None for m in material_maes) else None)
    close(report['summaries']['gt_scorable_pct'], 100 * sums['foreground'][0] / total_core)
    for path, sha in provenance['input_hashes'].items():
        assert digest(path) == sha, '来源哈希改变：' + path
    assert report['inputs_unchanged'] and (directory / 'COMPLETED').exists()
    assert not (directory / 'FAILED').exists()
    result = {'passed': True, 'frames': len(report['frames']), 'recomputed_from': 'accumulated/alpha/labels; independent CPU formulas',
              'input_hashes_verified': len(provenance['input_hashes']),
              'metrics_sha256': digest(directory / 'metrics.json'),
              'float_sha256': {p.name: digest(p) for p in sorted((directory / 'float').glob('*.npz'))}}
    return result


def verify_foreground(directory, report, provenance):
    """新协议：任意帧号、分辨率，仅前景，不使用任何标签或材质清单。"""
    from scipy.ndimage import binary_erosion
    protocol = report['protocol']
    rows = list(csv.DictReader((directory / 'frames.csv').open()))
    assert len(rows) == len(report['frames']) and len(rows) > 0
    assert [r['frame'] for r in report['frames']] == protocol['frames']
    n_total, k_total, error_total = 0, 0, 0.
    for record, row in zip(report['frames'], rows):
        assert record['frame'] == row['frame']
        with np.load(directory / 'float' / (record['frame'] + '.npz')) as data:
            shape = (record['height'], record['width'])
            for value in data.values():
                assert value.shape == shape and np.isfinite(value).all()
            alpha = data['alpha'].astype(np.float64)
            pred = np.where(alpha > 1e-6, data['accumulated'].astype(np.float64) / np.maximum(alpha, 1e-6), 0.)
            np.testing.assert_allclose(pred, data['roughness'], atol=1e-12, rtol=1e-12)
            core = data['gt_alpha'] >= protocol['gt_alpha_threshold']
            radius = protocol['gt_erosion_pixels']
            if radius:
                core = binary_erosion(core, structure=np.ones((3, 3)), iterations=radius)
            np.testing.assert_array_equal(core, data['gt_mask'])
            np.testing.assert_array_equal(core, data['gt_core'])
            valid = core & (alpha >= protocol['pred_alpha_threshold'])
            np.testing.assert_array_equal(valid, data['valid_mask'])
            n, k = int(core.sum()), int(valid.sum())
            error = float(np.abs(pred[valid] - data['gt_roughness'][valid]).sum(dtype=np.float64))
            expected = {'gt_pixels': n, 'valid_pixels': k, 'absolute_error_sum': error,
                        'mae': error / k if k else None, 'coverage_pct': 100 * k / n if n else None}
            for key, value in expected.items():
                close(record['foreground'][key], value)
            for key in ('mae', 'coverage_pct', 'gt_pixels', 'valid_pixels'):
                close(float(row[key]) if row[key] else None, expected[key])
            assert 'materials' not in record and 'material_labels' not in data
            n_total += n
            k_total += k
            error_total += error
    expected = {'gt_pixels': n_total, 'valid_pixels': k_total, 'absolute_error_sum': error_total,
                'mae': error_total / k_total if k_total else None,
                'coverage_pct': 100 * k_total / n_total if n_total else None}
    assert set(report['summaries']) == {'foreground'}
    for key, value in expected.items():
        close(report['summaries']['foreground'][key], value)
    for path, sha in provenance['input_hashes'].items():
        assert digest(path) == sha, '来源哈希改变：' + path
    assert report['inputs_unchanged'] and (directory / 'COMPLETED').exists()
    assert not (directory / 'FAILED').exists()
    return {'passed': True, 'frames': len(rows), 'protocol': 'generic_foreground',
            'input_hashes_verified': len(provenance['input_hashes']),
            'metrics_sha256': digest(directory / 'metrics.json'),
            'float_sha256': {p.name: digest(p) for p in sorted((directory / 'float').glob('*.npz'))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    content = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(content)
    print(content)


if __name__ == '__main__':
    main()
