"""动态V3输入预审与隔离：只复制TRAIN观测，初始化不读取历史资产。"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    assert source.name in ('LH-dynamic-jumpingjacks-ELDL-V3', 'LH-dynamic-jumpingjacks_DL_V2')
    dl_v2 = source.name == 'LH-dynamic-jumpingjacks_DL_V2'
    assert out.is_relative_to(root/'output')
    camera = json.loads((source/'camera.json').read_text())
    ids = sorted(map(int, camera['frames']))
    assert ids == list(range(1, 151))
    test = [i for i in ids if i % 10 == 0]
    dev = [i for i in ids if i % 10 and 61 <= i <= 79]
    train = [i for i in ids if i not in test + dev]
    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    audit = out.parent/'input_audit_v1'
    audit.mkdir(exist_ok=False)
    records, stats, tiles, original = [], [], [], {}
    # 固定每帧等量抽样组成亮度分布；不足时重复，不按前景面积加权。
    samples = []
    rng_stats = np.random.default_rng(1019)
    for i in train:
        key = f'{i:04d}'
        image = source/'image'/f'image_{key}.png'
        alpha_path = source/'albedo'/f'albedo_{key}.png'
        raw = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        alpha_raw = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
        assert raw.dtype == alpha_raw.dtype == np.uint16
        assert raw.shape == alpha_raw.shape == (800, 800, 4)
        # image的Alpha是全满码背景；原始物体透明度仅在albedo的Alpha内。
        # albedo RGB不参与任何训练统计、初始化或下游写入。
        alpha = alpha_raw[..., 3].copy()
        merged = raw.copy()
        merged[..., 3] = alpha
        merged[alpha == 0, :3] = 0
        assert cv2.imwrite(str(out/'images'/image.name), merged)
        fg = alpha > 32767
        assert fg.any() and (~fg).any()
        rgb = raw[..., [2, 1, 0]].astype(np.float64)/65535
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        lum = linear @ np.array([.2126, .7152, .0722])
        foreground = lum[fg]
        samples.append(foreground[rng_stats.integers(0, len(foreground), 8192)])
        stats.append(dict(frame=i, foreground_pixels=int(fg.sum()),
                          mean_linear_luminance=float(foreground.mean()),
                          saturated_fraction=float((raw[..., :3][fg] == 65535).any(-1).mean()),
                          original_image_alpha_all_one=bool((raw[..., 3] == 65535).all()),
                          alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest()))
        original[str(image)] = sha(image)
        original[str(alpha_path)] = dict(file_sha256=sha(alpha_path), used_channels=['Alpha'])
        frame = camera['frames'][key]
        intr = frame['intrinsics']
        c2w = np.array(frame['extrinsics']['camera_to_world'])
        w2c = np.array(frame['extrinsics']['world_to_camera'])
        assert np.max(np.abs(c2w@w2c-np.eye(4))) < 1e-5
        assert abs(intr['cx']-400) < 1e-6 and abs(intr['cy']-400) < 1e-6
        assert abs(intr['fx']-intr['fy']) < 1e-6
        records.append(dict(file_path=image.name, source_frame=i, time=i/150 if dl_v2 else (i-1)/149,
                            camera_angle_x=intr['fov_x_rad'], camera_angle_y=intr['fov_y_rad'],
                            fl_x=intr['fx'], fl_y=intr['fy'], cx=intr['cx'], cy=intr['cy'],
                            transform_matrix=c2w.tolist()))
        if i in (1, 21, 41, 81, 101, 121, 141, 149):
            preview = np.rint(rgb*(alpha[..., None]/65535)*255).astype(np.uint8)
            tile = Image.fromarray(preview).resize((300, 300))
            ImageDraw.Draw(tile).text((8, 8), f'TRAIN {i:04d}', fill='white')
            tiles.append(tile)
    common = {k: records[0][k] for k in ('camera_angle_x', 'camera_angle_y', 'fl_x', 'fl_y', 'cx', 'cy')}
    common.update(w=800, h=800, rgba16=True, per_frame_intrinsics=True)
    write(out/'transforms_train.json', dict(**common, frames=records))
    # 与原生随机点云构造一致，独立固定种子；既无SfM/GT几何也无模型权重。
    rng = np.random.RandomState(0)
    xyz = rng.random((100000, 3))*2.6-1.3
    colors = ((rng.random((100000, 3))/255*.28209479177387814+.5)*255).astype(np.uint8)
    vertices = np.empty(100000, dtype=[(k, 'f4') for k in ('x','y','z','nx','ny','nz')]+[(k,'u1') for k in ('red','green','blue')])
    for j, k in enumerate(('x', 'y', 'z')):
        vertices[k] = xyz[:, j]
    for k in ('nx', 'ny', 'nz'):
        vertices[k] = 0
    for j, k in enumerate(('red', 'green', 'blue')):
        vertices[k] = colors[:, j]
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(out/'points3d.ply')
    manifest = dict(dataset=source.name, source=str(source), gt_read=False, init_weights_loaded=False,
        train_frames=train, dev_frames=dev, test_frames=test,
        time_rule=('source_frame/150' if dl_v2 else '(source_frame-1)/149')+'，沿用各自原生Stage1时间，禁止按子集重新编号',
        alpha_source='原albedo的Alpha通道；image的Alpha全满码，albedo RGB禁止进入训练',
        color='16位Standard/sRGB码值；Stage1沿原生码值损失，Stage2统一sRGB逆变换至线性后光栅合成',
        initialization=dict(method='原生随机100k立方体', seed=0),
        original_train_inputs=original, camera_sha256=sha(source/'camera.json'),
        source_code_sha256=sha(__file__),
        files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    write(out/'no_gt_manifest.json', manifest)
    distribution = np.concatenate(samples)
    write(audit/'train_statistics.json', dict(status='PASS_INPUT_PREPARATION', train_count=len(train),
        dev_count=len(dev), test_count=len(test), per_frame=stats,
        equal_frame_samples=8192, sample_seed=1019,
        q_base_probability=.5, q_bright_probability=.9,
        q_base=float(np.quantile(distribution, .5)), q_bright=float(np.quantile(distribution, .9)),
        mean_linear_luminance=float(np.mean([s['mean_linear_luminance'] for s in stats])),
        scope='仅117 TRAIN；未读取DEV/TEST图像、GT法线/几何/灯向'))
    grid = Image.new('RGB', (1200, 600))
    for j, tile in enumerate(tiles):
        grid.paste(tile, ((j%4)*300, (j//4)*300))
    grid.save(audit/'train_representatives.jpg')
    print(json.dumps(dict(status='PASS_INPUT_PREPARATION', output=str(out), train=len(train), dev=len(dev), test=len(test)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
