"""全量HDR+转灯完成后生成代表图，无GT精度或质量自动结论。"""
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent.parent;render=ROOT/'model_mlp/renders_stage2/ours_55000'
assert len(list((render/'hdr').glob('*.png')))==150 and len(list((render/'directional').glob('*.png')))==72
rows=[('hdr',f'{i:04d}') for i in [1,76,150]]+[('directional',f'{i:04d}') for i in [0,36,71]]
canvas=Image.new('RGB',(1200,840),(32,32,32));draw=ImageDraw.Draw(canvas)
for index,(category,key) in enumerate(rows):
 col=index%3;row=index//3;im=Image.open(render/category/f'{key}.png').convert('RGB');assert im.size==(400,400)
 canvas.paste(im,(col*400,row*420+20));draw.text((col*400+10,row*420+5),category+' '+key+' native 55000',fill='white')
p=ROOT/'inspection/relight_final_55000.png';assert not p.exists();canvas.save(p);print(p)
