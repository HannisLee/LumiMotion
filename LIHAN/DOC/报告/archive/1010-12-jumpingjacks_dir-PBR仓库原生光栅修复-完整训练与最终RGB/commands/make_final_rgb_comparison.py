"""55k最终RGB实际产物齐备后生成GT/B05/B06首中尾拼图，不判质量、不改统计。"""
from pathlib import Path
from PIL import Image,ImageDraw
import json
ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'1010-10-jumpingjacks_dir-PBR来源平面几何投影'
CURRENT=ROOT/'model_mlp/renders_stage2/ours_55000/eval';PREVIOUS=OLD/'model_mlp/renders_stage2/ours_55000/eval'
metrics=json.loads((CURRENT/'metrics.json').read_text());assert len(metrics['frames'])==15
frames=['0010','0080','0150'];cols=['GT RGB','B05 installed CUDA 55000','B06 native CUDA 55000']
canvas=Image.new('RGB',(1200,1260),(32,32,32));draw=ImageDraw.Draw(canvas)
for col,label in enumerate(cols):draw.text((col*400+12,8),label,fill='white')
for row,frame in enumerate(frames):
 paths=[CURRENT/f'{frame}_gt.png',PREVIOUS/f'{frame}_render.png',CURRENT/f'{frame}_render.png']
 for col,path in enumerate(paths):
  im=Image.open(path).convert('RGB');assert im.size==(400,400);canvas.paste(im,(col*400,row*410+30));draw.text((col*400+8,row*410+30),frame,fill='white')
p=ROOT/'inspection/rgb_final_55000_comparison.png';assert not p.exists();canvas.save(p);print(p)
