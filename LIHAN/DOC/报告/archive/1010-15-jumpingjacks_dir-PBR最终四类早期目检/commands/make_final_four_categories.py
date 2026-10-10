from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent.parent;render=ROOT/'model_mlp/renders_stage2/ours_55000/insights'
categories=['rgb','alpha','normal','albedo'];allfiles={k:sorted((render/k).glob('*.png')) for k in categories}
assert all(len(files)==150 for files in allfiles.values());assert all([p.stem for p in allfiles[k]]==[p.stem for p in allfiles['rgb']] for k in categories)
canvas=Image.new('RGB',(1600,1260),(32,32,32));draw=ImageDraw.Draw(canvas)
for col,label in enumerate(categories):draw.text((col*400+12,8),label+' native 55000',fill='white')
for row,index in enumerate([0,74,149]):
 for col,label in enumerate(categories):
  path=allfiles[label][index];im=Image.open(path).convert('RGB');assert im.size==(400,400);canvas.paste(im,(col*400,row*410+30));draw.text((col*400+8,row*410+30),path.stem,fill='white')
p=ROOT/'inspection/four_categories_final_55000.png';assert not p.exists();canvas.save(p);print(p)
