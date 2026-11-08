"""为真实第0006视角生成参数预览，保留原始数值及来源校验。"""
from pathlib import Path
import hashlib
import json
import shutil
import OpenEXR
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/LH-data/static/LH-static-jumpingjacks_EL_V2'
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'LIHAN/export/1018-01'
OUT.mkdir(exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def exr(p, channels):
    f = OpenEXR.InputFile(str(p))
    win = f.header()['dataWindow']
    shape = (win.max.y-win.min.y+1, win.max.x-win.min.x+1)
    return np.stack([np.frombuffer(f.channel(c), np.float32).reshape(shape) for c in channels], -1)

copies = {
    SRC/'image/image_0006.png': 'rgb_0006.png',
    SRC/'albedo/albedo_0006.png': 'albedo_0006.png',
    SRC/'normal_exr/normal_0006.exr': 'normal_world_0006.exr',
    RUN/'roughness/roughness_0006.exr': 'roughness_0006.exr',
}
for src, name in copies.items():
    shutil.copy2(src, OUT/name)
    assert sha(src) == sha(OUT/name)

n = exr(OUT/'normal_world_0006.exr', ['Normal.X','Normal.Y','Normal.Z'])
r = exr(OUT/'roughness_0006.exr', ['R','G','B','A'])
assert n.shape == (800,800,3) and r.shape == (800,800,4)
assert np.isfinite(n).all() and np.isfinite(r).all()
valid = np.linalg.norm(n,axis=-1) > 0.5
alpha = r[...,3]
mask = alpha > 1e-6
rough = np.divide(r[...,0], alpha, out=np.zeros_like(alpha), where=mask)
assert rough[mask].min() > -1e-4 and rough[mask].max() < 1.0001
# 法线是方向数值的线性RGB映射，不应用sRGB传递函数；背景透明。
normal_rgb = np.clip((n+1)*.5,0,1)
normal_rgba = np.concatenate([normal_rgb, valid[...,None]],-1)
Image.fromarray(np.round(normal_rgba*255).astype('uint8'),'RGBA').save(OUT/'normal_world_0006.png')
# 粗糙度PNG直接将r映射到灰度，黑=0，白=1；EXR保留原始预乘值。
rough_rgba = np.concatenate([np.repeat(np.clip(rough,0,1)[...,None],3,-1),alpha[...,None]],-1)
Image.fromarray(np.round(rough_rgba*255).astype('uint8'),'RGBA').save(OUT/'roughness_0006.png')
Image.fromarray(np.round(np.clip(rough,0,1)*65535).astype('uint16')).save(OUT/'roughness_linear16_0006.png')

overview=Image.new('RGB',(1600,1680),'#e8edf2')
font=ImageFont.truetype('/tmp/lumimotion-ppt-1017/NotoSansCJKsc-Regular.otf',26)
draw=ImageDraw.Draw(overview)
for i,(name,label) in enumerate([
    ('rgb_0006.png','原始 RGB / 第 0006 视角'),
    ('normal_world_0006.png','法线 / 世界坐标'),
    ('albedo_0006.png','Albedo / 无光照基础色'),
    ('roughness_0006.png','Roughness / 线性灰度 0—1'),
]):
    x=(i%2)*800;y=(i//2)*840
    draw.text((x+24,y+6),label,fill='#152d46',font=font)
    im=Image.open(OUT/name).convert('RGBA')
    bg=Image.new('RGBA',im.size,'#505050');bg.alpha_composite(im)
    overview.paste(bg.convert('RGB'),(x,y+40))
overview.save(OUT/'overview_0006.jpg',quality=95)
stats={
    '状态':'PASS', '视角':6,'分辨率':[800,800],
    '人物姿态来源帧':100,'有效法线像素':int(valid.sum()),
    '法线单位长度误差中位数':float(np.median(abs(np.linalg.norm(n[valid],axis=-1)-1))),
    '粗糙度有效像素':int(mask.sum()),
    '粗糙度范围':[float(rough[mask].min()),float(rough[mask].max())],
    '粗糙度RGB通道最大差':float(abs(r[...,0]-r[...,1]).max()),
    '来源文件':[{'路径':str(p),'SHA256':sha(p),'输出':name} for p,name in copies.items()],
    '场景SHA256':sha(SRC/'scene_snapshot.blend'),
    '说明':'RGB、Albedo、世界法线直接复用数据集真实导出；Roughness从同场景真实Principled输入补渲染。透明层为透射率与Alpha加权有效粗糙度。',
}
(OUT/'verification.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(stats,indent=2,ensure_ascii=False))
