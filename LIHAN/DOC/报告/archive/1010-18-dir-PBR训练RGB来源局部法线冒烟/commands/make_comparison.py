from pathlib import Path
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parent.parent

def sheet(name,rows):
 canvas=Image.new('RGB',(400*len(rows[0]),425*len(rows)),'white');draw=ImageDraw.Draw(canvas)
 for i,row in enumerate(rows):
  for j,(label,path) in enumerate(row):
   im=Image.open(path).convert('RGB');assert im.size==(400,400)
   canvas.paste(im,(j*400,i*425+25));draw.text((j*400+4,i*425+5),label,fill='black')
 canvas.save(r/'inspection'/name)
sheet('four_categories.png',[[(c+' '+f,r/'insights/insights'/c/(f+'.png')) for c in ['rgb','alpha','normal','albedo']] for f in ['0001','0075','0150']])
sheet('relight.png',[[(c+' '+f,r/'inspection/relight_images'/(f+'_'+c+'.png')) for c in ['directional','hdr']] for f in ['0001','0075','0150']])
print('comparison images written')
