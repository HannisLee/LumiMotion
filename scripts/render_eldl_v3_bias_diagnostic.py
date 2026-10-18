"""仅CPU补绘固定TRAIN视角的bias透射率、尾部与接触风险代理。"""
import argparse
import json
from pathlib import Path
import shlex
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_eldl_v3 as v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnostic, output = args.diagnostic.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise ValueError('仅允许output目录。')
    plan = v.read(diagnostic / 'plan.json')
    cache = Path(plan['paths']['cache'])
    reference = Path(plan['paths']['reference'])
    output.mkdir(parents=True, exist_ok=False)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([cache, reference, diagnostic, output])
    hashes = {}
    def load(path):
        hashes[str(path)] = v.sha(path)
        return torch.load(path, map_location='cpu', weights_only=False)
    data = load(cache)
    proxy = load(diagnostic / 'contact_proxy.pt')
    old, new = {}, {}
    ref_summary = v.read(reference / 'summary.json')
    for key in ('0p5', '1', '2'):
        path = reference / 'arrays' / f'exact_bias_{key}.pt'
        value = load(path)
        if hashes[str(path)] != ref_summary['output_sha256'][str(path)]:
            raise ValueError('旧exact数组SHA不一致。')
        if not torch.equal(value['valid'], data['valid']):
            raise ValueError('旧exact支持变化。')
        old[key] = value['visibility'].numpy()
        value = load(diagnostic / 'arrays' / f'candidate_bias_{key}.pt')
        if not torch.equal(value['valid'], data['valid']):
            raise ValueError('新候选支持变化。')
        new[key] = value['visibility'].numpy()
    valid = data['valid'].numpy()
    near = proxy['receiver_near_low_caster'].numpy()
    def tile(value, mask, label, kind='heat'):
        if kind == 'heat':
            panel = cv2.applyColorMap((np.clip(value, 0, 1).reshape(64, 64) * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        elif kind == 'overlay':
            panel = np.repeat((np.clip(value, 0, 1).reshape(64, 64, 1) * 180).astype(np.uint8), 3, -1)
            panel[near.reshape(64, 64) & mask.reshape(64, 64)] = (panel[near.reshape(64, 64) & mask.reshape(64, 64)] * .45 + np.array([0, 210, 230]) * .55).astype(np.uint8)
        else:
            panel = np.full((64, 64, 3), 40, np.uint8)
            panel[(value.reshape(64, 64) > .5)] = [235, 190, 0]
            panel[(value.reshape(64, 64) > .9)] = [255, 30, 30]
        panel[~mask.reshape(64, 64)] = [80, 80, 80]
        result = np.full((224, 256, 3), 24, np.uint8)
        result[32:] = cv2.resize(panel, (256, 192), interpolation=cv2.INTER_NEAREST)
        cv2.putText(result, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        return result
    metrics = {}
    for frame in (1, 81, 149):
        index = v.TRAIN.index(frame)
        if not bool(data['fit'][index]):
            raise ValueError('代表帧必须是fit。')
        mask = valid[index]
        a, b = old['1'][index], new['1'][index]
        columns = [a, new['0p5'][index], b, new['2'][index]]
        names = ['old center', 'new half', 'new center', 'new double']
        sensitivity = [abs(old['0p5'][index]-a), abs(new['0p5'][index]-b), abs(old['2'][index]-a), abs(new['2'][index]-b)]
        snames = ['old half-center', 'new half-center', 'old double-center', 'new double-center']
        rows = [np.concatenate([tile(x, mask, name+' V') for x,name in zip(columns,names)], 1),
                np.concatenate([tile(abs(x-a), mask, name+' |V-old|') for x,name in zip(columns,names)], 1),
                np.concatenate([tile(x, mask, name+' |dV|') for x,name in zip(sensitivity,snames)], 1)]
        overlays = []
        per_frame = {}
        for x, name in zip(columns, names):
            panel = tile(x, mask, name+' contact/open', 'overlay')
            opened = mask & (a < .5) & (x > .9)
            # 红色：旧中心暗到当前亮；青色：固定低高度caster近邻代理。
            red = np.zeros((64,64,3),np.uint8)
            red[opened.reshape(64,64)] = [255,30,30]
            red = cv2.resize(red,(256,192),interpolation=cv2.INTER_NEAREST)
            red_mask = cv2.resize(opened.reshape(64,64).astype(np.uint8),(256,192),interpolation=cv2.INTER_NEAREST).astype(bool)
            panel[32:][red_mask] = red[red_mask]
            overlays.append(panel)
            per_frame[name] = dict(support=int(mask.sum()), near_support=int((mask & near).sum()),
                old_dark_count=int((mask & (a < .5)).sum()), dark_to_open=int(opened.sum()),
                near_dark_to_open=int((opened & near).sum()), mean_V=float(x[mask].mean()))
        rows.append(np.concatenate(overlays, 1))
        rows.append(np.concatenate([tile(x, mask, name+' tail .5/.9', 'tail') for x,name in zip(sensitivity,snames)],1))
        path = output / f'fit_{frame:04d}_visibility_contact_tail.png'
        if not cv2.imwrite(str(path), np.concatenate(rows,0)[...,::-1]):
            raise IOError('写图失败。')
        metrics[str(frame)] = dict(visibility=per_frame, sensitivity={name:dict(count_gt_p9=int((mask & (x>.9)).sum()), near_count_gt_p9=int((mask & near & (x>.9)).sum())) for x,name in zip(sensitivity,snames)})
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    hashes[str(Path(__file__).resolve())] = v.sha(Path(__file__).resolve())
    v.check_hashes(hashes)
    v.write(output / 'summary.json', dict(status='SPATIAL_DIAGNOSTIC_PENDING_ACTUAL_REVIEW', internal_validation_passed=False,
        gt_read=False, external_test_read=False, fit_representatives=[1,81,149], input_sha256=hashes,
        image_sha256={str(p):v.sha(p) for p in output.glob('*.png')}, statistics=metrics, command=command))
    (output/'README.md').write_text('# 固定fit透射率空间诊断\n\n2026-10-19（Asia/Tokyo）。仅CPU重绘已完成诊断；固定fit1/81/149，无held选择，无拟合。\n\n命令：\n\n```bash\n'+command+'\n```\n\n来源/model/iteration继承上级plan.json与command.json，输出本目录，日志run.log，统计summary.json。\n\n五行依次：①旧中心/新半/新中/新双V，统一turbo0—1；②各列相对旧中心的绝对ΔV；③旧半、候选半、旧双、候选双相对各自中心的绝对ΔV；④灰度V上青色表示距低高度caster XY≤原格点步长，红色表示旧中心V<0.5而当前V>0.9；⑤与第三行相同ΔV的尾部位置，黄>0.5、红>0.9。灰色无原valid。图格点保持原64×64行列，显示纵横比为192×256；这是接收平面，不是相机RGB图。\n\n低高度caster代理无鞋语义，不可据此声称接触阴影正确。阈值只用于空间诊断，无验收门槛。四类完整模型验收不适用，本次仅V/ΔV/支持与接触代理。最终PENDING实际目检，internal_validation_passed=false，不导出。\n')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
