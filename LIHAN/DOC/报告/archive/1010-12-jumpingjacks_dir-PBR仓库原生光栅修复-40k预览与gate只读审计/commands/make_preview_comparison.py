from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'1010-10-jumpingjacks_dir-PBR来源平面几何投影'
frames=['0010','0080','0150']; cols=['GT RGB','B05 old CUDA 40000','B06 native CUDA 40000']
canvas=Image.new('RGB',(1200,1260),(32,32,32));draw=ImageDraw.Draw(canvas)
for col,label in enumerate(cols): draw.text((col*400+12,8),label,fill='white')
for row,frame in enumerate(frames):
 paths=[ROOT/f'preview_40000/eval/{frame}_gt.png',OLD/f'preview_40000/eval/{frame}_render.png',ROOT/f'preview_40000/eval/{frame}_render.png']
 for col,path in enumerate(paths):
  im=Image.open(path).convert('RGB');assert im.size==(400,400)
  canvas.paste(im,(col*400,row*410+30))
  draw.text((col*400+8,row*410+30),frame,fill='white')
out=ROOT/'inspection/preview_40000_comparison.png';assert not out.exists();canvas.save(out);print(out)
