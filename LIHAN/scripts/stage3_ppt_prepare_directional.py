"""只承接父TRAIN估光，原始RGBA生成可审计的动态方向PBR输入。"""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT/'output/1025-02-Stage3-PPT配对总控'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parent = ROOT/'output/1021-06-LH-dynamic-jumpingjacks_DL_V2-完整路线复现/model_mlp'
    state_file = parent/'dynamic_full_route/iteration_16500/state.pt'
    state = torch.load(state_file, map_location='cpu', weights_only=False)
    times = state['lighting']['table.times']
    directions = F.normalize(state['lighting']['table.raw_directions'], dim=-1)
    intensity = float(state['lighting']['log_intensity'].exp())
    frames = {}
    for key in range(1, 151):
        time = torch.tensor(key/150, dtype=torch.float32)
        right = torch.searchsorted(times, time).clamp(1, len(times)-1)
        left = right-1
        weight = ((time-times[left])/(times[right]-times[left])).clamp(0, 1)
        direction = F.normalize((1-weight)*directions[left]+weight*directions[right], dim=0)
        frames[f'{key:04d}'] = dict(light_dir_world=direction.tolist(), intensity=intensity, type='directional', source_type='SUN')
    write(CONTROL/'dynamic_estimated_directions.json', dict(coordinate_space='world', frames=frames,
          gt_directions_used=False, direction_source='父模型117 TRAIN阴影估光方向，单位向量线性插值；CPU float32转换',
          intensity_source='父模型TRAIN光度拟合强度', parent_state=str(state_file), parent_state_sha256=sha(state_file)))
    reference = CONTROL/'dynamic_eval_source'
    train = json.loads((reference/'transforms_train.json').read_text())
    test = json.loads((reference/'transforms_test.json').read_text())
    image_root = CONTROL/'dynamic_original_rgba_images'
    image_root.mkdir(exist_ok=False)
    raw_root = ROOT/'data/LH-data/danamic/LH-dynamic-jumpingjacks_DL_V2/image'
    records, differences = {}, []
    for split, metadata in [('train', train), ('test', test)]:
        for frame in metadata['frames']:
            key = f'{int(frame["source_frame"]):04d}'
            source = raw_root/f'image_{key}.png'
            with Image.open(source) as image:
                assert image.mode == 'RGBA'
                rgba = np.array(image, dtype=np.uint8)
            rgba[rgba[..., 3] == 0, :3] = 0
            target = image_root/frame['file_path']
            Image.fromarray(rgba).save(target)
            old = np.array(Image.open(reference/'images'/frame['file_path']).convert('RGBA'))
            differences.append(dict(frame=key, split=split, alpha_byte_difference_pixels=int(np.count_nonzero(rgba[..., 3] != old[..., 3])),
                                    threshold_mask_disagreement=int(np.count_nonzero((rgba[..., 3]>127)!=(old[..., 3]>127))),
                                    rgb_difference_channels=int(np.count_nonzero(rgba[..., :3] != old[..., :3]))))
            records[key] = dict(source_image=str(source), source_image_sha256=sha(source), image_sha256=sha(target), split=split)
    for mode in ['train', 'eval']:
        source = CONTROL/f'dynamic_directional_{mode}_source'
        source.mkdir(exist_ok=False)
        (source/'images').symlink_to(image_root.resolve(), target_is_directory=True)
        if (reference/'points3d.ply').exists():
            (source/'points3d.ply').symlink_to((reference/'points3d.ply').resolve())
        write(source/'transforms_train.json', train)
        write(source/'transforms_test.json', test if mode == 'eval' else {**test, 'frames': []})
        write(source/'dataset_manifest.json', dict(summary=dict(mask_source='original-image-alpha'),
              frames={k: v for k, v in records.items() if mode == 'eval' or v['split'] == 'train'},
              reference=str(reference), camera_policy='保留本轮原生矩阵与time，未读GT材质或GT几何'))
    write(CONTROL/'directional_input_comparison.json', dict(records=differences, compared_frames=len(differences),
          total_alpha_byte_difference_pixels=sum(x['alpha_byte_difference_pixels'] for x in differences),
          total_threshold_mask_disagreement=sum(x['threshold_mask_disagreement'] for x in differences),
          total_rgb_difference_channels=sum(x['rgb_difference_channels'] for x in differences)))
    # 静态已知方向条件只继承既有TRAIN拟合光强；统一改用本轮混合追踪深度1。
    shutil.copy2(ROOT/'output/1024-02-静态ELDL-展示数值补全总控/known_directions_fitted_intensity.json', CONTROL/'static_known_directions.json')
    static_reference = CONTROL/'static_eval_source'
    for mode in ['train', 'eval']:
        source = CONTROL/f'static_directional_{mode}_source'
        source.mkdir(exist_ok=False)
        for p in static_reference.iterdir():
            if p.name not in ['transforms_train.json', 'transforms_test.json', 'dataset_manifest.json']:
                (source/p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
        metadata = json.loads((static_reference/'transforms_train.json').read_text())
        write(source/'transforms_train.json', metadata)
        test_metadata = json.loads((static_reference/'transforms_test.json').read_text())
        write(source/'transforms_test.json', test_metadata if mode == 'eval' else {**test_metadata, 'frames': []})
        manifest = json.loads((static_reference/'dataset_manifest.json').read_text())
        if mode == 'train':
            manifest['frames'] = {k: v for k, v in manifest['frames'].items() if v['split'] == 'train'}
        write(source/'dataset_manifest.json', manifest)
    print(dict(frames=len(records), intensity=intensity, alpha_difference=sum(x['alpha_byte_difference_pixels'] for x in differences)))


if __name__ == '__main__':
    main()
