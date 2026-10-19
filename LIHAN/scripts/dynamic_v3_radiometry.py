"""从TRAIN空背景核验固定合成颜色，只移除Alpha透射的已知背景项。"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def linearize(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def black_observation(rgb,alpha,background_linear):
    """I=alpha*C+(1-alpha)*B；仅移除背景B，物体上的环境照明保留。"""
    observed=linearize(np.asarray(rgb,dtype=np.float64))
    return np.maximum(0.,observed-(1-np.asarray(alpha))[...,None]*np.asarray(background_linear))


def estimate_background(raw,frames):
    values=[];records=[];files={}
    for n in frames:
        image=Path(raw)/f'image/image_{n:04d}.png'
        alpha_path=Path(raw)/f'albedo/albedo_{n:04d}.png'
        rgba=cv2.imread(str(image),cv2.IMREAD_UNCHANGED)
        original_alpha=cv2.imread(str(alpha_path),cv2.IMREAD_UNCHANGED)
        assert rgba.dtype==original_alpha.dtype==np.uint16 and rgba.shape==original_alpha.shape==(800,800,4)
        background=rgba[...,[2,1,0]][original_alpha[...,3]==0]
        assert len(background)>100,'TRAIN缺少完全透明背景支持'
        low,high=background.min(0),background.max(0)
        assert np.array_equal(low,high),'背景非均匀，不能静默使用固定合成背景'
        values.append(low)
        rgb=rgba[...,[2,1,0]].astype(np.float64)/65535.
        alpha=original_alpha[...,3].astype(np.float64)/65535.
        proposed=linearize(rgb)-(1-alpha)[...,None]*linearize(low.astype(np.float64)/65535.)
        partial=(alpha>0)&(alpha<1)
        records.append(dict(frame=n,background_pixels=len(background),rgb16=low.tolist(),
            partial_alpha_pixels=int(partial.sum()),preclamp_negative_partial_channels=int((proposed[partial]<0).sum()),
            preclamp_minimum_partial=float(proposed[partial].min()) if partial.any() else None))
        for p in (image,alpha_path):files[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    colors=np.stack(values)
    assert np.array_equal(colors.min(0),colors.max(0)),'背景跨时刻变化，不能当固定B'
    srgb=colors[0].astype(np.float64)/65535.
    return dict(format_version=1,status='PASS_TRAIN_FIXED_BACKGROUND',gt_read=False,source_split='TRAIN',
        source=str(Path(raw).resolve()),frames=frames,background_rgb16=colors[0].tolist(),
        background_srgb=srgb.tolist(),background_linear=linearize(srgb).tolist(),
        per_frame=records,source_files=files,
        target_rule='max(0,inverse_sRGB(original_RGB)-(1-original_alpha)*B_linear)，之后AREA',
        exact_inverse_compositing_proven=False,
        limitation='albedo单独渲染的Alpha与原RGB边界可能不完全一致；保留负值诊断，固定max0截断不能称无误差反演',
        scope='已知原始图像合成背景；不扣除物体环境照明，不提供逐帧曝光/颜色参数')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    assert not manifest['gt_read']
    frames=manifest['train_frames'];assert len(frames)==117
    assert not set(frames)&set(manifest['dev_frames']+manifest['test_frames'])
    assert not a.output.exists()
    result=estimate_background(a.source,frames)
    result['input_manifest_sha256']=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','background_rgb16','background_linear','target_rule')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
