"""核验B00来源，重算等帧TRAIN亮度分布并冻结三阶段共同输入。"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    control = root/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控'
    source = control/'data_train_v2'
    parent = root/'output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp'
    lighting = control/'data_train_eldl_raster_v1/estimated_eldl_train.json'
    a.output = a.output.resolve()
    assert a.output.is_relative_to(root/'output')
    a.output.mkdir(parents=True, exist_ok=False)
    record = json.loads((parent/'eldl_v3_training.json').read_text())
    manifest = json.loads((source/'no_gt_manifest.json').read_text())
    assert record['baseline'] == 'native' and record['actual_optimizer_updates'] == 34999
    assert record['gt_supervision'] is False and record['external_test_read'] is False
    assert record['args']['source_path'] == str(source) and record['train_frames'] == manifest['train_frames']
    assert record['train_frames'] == [i for i in range(1, 151) if i % 10]
    assert record['input_manifest_sha256'] == sha(source/'no_gt_manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(source/name) == digest, name
    values, weights, frames = [], [], []
    transforms = json.loads((source/'transforms_train.json').read_text())
    assert transforms['rgba16'] is True
    for f in transforms['frames']:
        path = source/'images'/f['file_path']
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert rgba.dtype == np.uint16 and rgba.shape == (800, 800, 4)
        alpha = rgba[..., 3].astype(np.float32)/65535
        rgb = rgba[..., 2::-1].astype(np.float32)/65535
        # 与FloatRGBA和原生loadCam相同：sRGB黑底合成后AREA缩小，再线性化。
        rgb = cv2.resize(rgb*alpha[..., None], (400, 400), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (400, 400), interpolation=cv2.INTER_AREA)
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        fg = alpha > .5
        y = linear@np.array([.2126, .7152, .0722])
        data = y[fg].astype(np.float64)
        values.append(data)
        weights.append(np.full(data.shape, 1/(135*len(data)), dtype=np.float64))
        frames.append(dict(frame=f['source_frame'], foreground_pixels=int(fg.sum()),
                           linear_mean=float(data.mean()), saturated_channel_fraction=float((rgb[fg] >= 1).mean())))
    y, w = np.concatenate(values), np.concatenate(weights)
    order = np.argsort(y, kind='stable'); cumulative = np.cumsum(w[order])
    qs = [.01, .1, .25, .5, .75, .8, .9, .95, .99, 1.]
    quantiles = {str(q): float(y[order[np.searchsorted(cumulative, q, side='left').clip(0, len(y)-1)]]) for q in qs}
    stats = dict(source=str(source), manifest_sha256=sha(source/'no_gt_manifest.json'),
        color_domain='sRGB黑底合成及AREA到400后转linear RGB', alpha='原始Alpha AREA到400，固定alpha>0.5',
        denominator='每帧原Alpha前景内均匀采样，135帧等权；不是去重表面分布',
        train_frames=manifest['train_frames'], equal_frame_linear_mean=float(np.dot(y, w)),
        foreground_pixel_count=len(y), quantiles=quantiles, q_base=.8, q_bright=.95,
        q_base_value=quantiles['0.8'], q_bright_value=quantiles['0.95'], q50=quantiles['0.5'],
        transition_width=.05, frames=frames, gt_read=False, test_read=False,
        saturation_note='高分位在1时不可区分更高辐射度；保持预登记软权重与逐通道删失，不再调分位。')
    write(a.output/'train_brightness.json', stats)
    asset = json.loads(lighting.read_text())
    assert asset['gt_read'] is False and asset['source_split'] == 'TRAIN'
    assert sorted(map(int, asset['frames'])) == manifest['train_frames']
    write(a.output/'train_directions.json', asset)
    files = [parent/'eldl_v3_training.json', parent/'cfg_args', parent/'training_finished.json',
             parent/'point_cloud/iteration_35000/point_cloud.ply', parent/'deform/iteration_35000/deform.pth',
             source/'no_gt_manifest.json', lighting]
    source_root = Path(record['source_code_root'])
    core = ['scene/gaussian_model.py', 'scene/deform_model.py', 'utils/time_utils.py',
            'gaussian_renderer/__init__.py', 'scene/cameras.py', 'utils/camera_utils.py', 'utils/rgba16.py']
    comparison = {name: sha(source_root/name) == sha(root/name) for name in core}
    assert all(comparison.values()), comparison
    write(a.output/'asset_binding.json', dict(status='PASS_CPU_SOURCE_AND_ASSET_BINDING',
        parent=str(parent), parent_iteration=35000, parent_actual_updates=34999, source=str(source),
        directions=str(a.output/'train_directions.json'), stats=str(a.output/'train_brightness.json'),
        input_sha256={str(path): sha(path) for path in files}, core_matches=comparison,
        source_manifest_sha256=sha(source/'no_gt_manifest.json'),
        alpha_provenance=manifest['alpha_source'], gt_read=False, test_read=False,
        geometry_network='完整保留并冻结原生MLP，逐帧有效几何，未烘焙为静态四张量',
        stage1_color='原生sRGB视角SH及d_color仅留档和PBR间接光；Stage2改用共享法线SH乘反照率',
        stage2_material_initialization='B00 DC转linear除以1.5再转sRGB，范围[.03,.97]；不加载父Adam或日程'))
    print(json.dumps(dict(status='PASS', mean=stats['equal_frame_linear_mean'], quantiles=quantiles), ensure_ascii=False))


if __name__ == '__main__':
    main()
