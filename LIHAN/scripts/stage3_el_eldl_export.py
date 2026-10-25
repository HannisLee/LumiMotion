"""导出四组PSNR/SSIM和同视角RGB；不含几何、材质、固定相机页面。"""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'LIHAN/exports/1025-02-静动态EL与GT方向光ELDL-Stage3外观对比'
FONT=ROOT/'output/1025-33-静动态EL与GT方向光ELDL-Stage3总控/assets/NotoSansCJKsc-Regular.otf'
def font(n):return ImageFont.truetype(str(FONT),n)
def main():
 from LIHAN.scripts.stage3_el_eldl_four import CONTROL
 selection=json.loads((CONTROL/'final_selection.json').read_text())
 DEST.mkdir(parents=True,exist_ok=False)
 rows=[];reports={};provenance={}
 labels={'static_EL':'静态 EL','static_ELDL':'静态 GT方向光 ELDL','dynamic_EL':'动态 EL','dynamic_ELDL':'动态 GT方向光 ELDL'}
 for arm,entry in selection.items():
  path=Path(entry['evaluation']);r=json.loads((path/'metrics.json').read_text());reports[arm]=r
  assert len(r['frames'])==15 and sorted(int(x['frame']) for x in r['frames'])==list(range(10,151,10))
  row={'实验':labels[arm],'PSNR_dB':r['metrics']['psnr'],'SSIM':r['metrics']['ssim'],'iteration':r['iteration']};rows.append(row)
  for name in ['rgb','gt']:shutil.copytree(path/name,DEST/arm/name)
  provenance[arm]={'evaluation':str(path),'model':r['model'],'iteration':r['iteration'],'source':r['source'],'directions':r.get('direction_source'),'metrics_sha256':hashlib.sha256((path/'metrics.json').read_bytes()).hexdigest()}
 with (DEST/'1025-02-四组Stage3指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (DEST/'1025-02-逐视角指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['实验','视角','PSNR_dB','SSIM']);w.writeheader()
  for arm,r in reports.items():
   for v in sorted(r['frames'],key=lambda x:int(x['frame'])):w.writerow({'实验':labels[arm],'视角':v['frame'],'PSNR_dB':v['psnr'],'SSIM':v['ssim']})
 assert {r['samples'] for r in reports.values()}=={512}
 pages=[];page=Image.new('RGB',(1920,1080),'#f5f6f8');draw=ImageDraw.Draw(page)
 draw.text((70,65),'静动态 EL 与 GT方向光 ELDL',font=font(56),fill='#192b42')
 draw.text((70,150),'Stage3 外观对比 · 15 个固定 TEST 视角',font=font(34),fill='#536278')
 for x,t in [(90,'实验'),(1170,'PSNR ↑ (dB)'),(1530,'SSIM ↑')]:draw.text((x,290),t,font=font(32),fill='#192b42')
 for i,row in enumerate(rows):
  y=385+i*110
  draw.text((90,y),row['实验'],font=font(34),fill='#192b42')
  draw.text((1170,y),f"{row['PSNR_dB']:.6f}",font=font(34),fill='#192b42')
  draw.text((1530,y),f"{row['SSIM']:.6f}",font=font(34),fill='#192b42')
 conclusions=[]
 for kind,label in [('static','静态'),('dynamic','动态')]:
  baseline=reports[kind+'_EL']['metrics'];candidate=reports[kind+'_ELDL']['metrics']
  better=all(candidate[k]>baseline[k] for k in ['psnr','ssim'])
  conclusions.append(f"{label}：ΔPSNR {candidate['psnr']-baseline['psnr']:+.4f} dB，ΔSSIM {candidate['ssim']-baseline['ssim']:+.6f}；"+('两项均提升' if better else '未达到两项均提升'))
 for i,line in enumerate(conclusions):draw.text((90,865+i*55),line,font=font(27),fill='#225b48' if '两项均提升' in line and '未达到' not in line else '#9b392c')
 page.save(DEST/'1025-02-四组Stage3指标.png');pages.append(page)
 for kind,label in [('static','静态'),('dynamic','动态')]:
  folder=DEST/f'{label}同视角';folder.mkdir()
  for frame in range(10,151,10):
   canvas=Image.new('RGB',(1920,760),'#f5f6f8');d=ImageDraw.Draw(canvas)
   d.text((65,35),f'{label}外观对比 · 视角 {frame:04d}',font=font(42),fill='#192b42')
   columns=[(kind+'_EL','gt','EL 数据 GT'),(kind+'_EL','rgb','EL · Stage3'),(kind+'_ELDL','gt','ELDL 数据 GT'),(kind+'_ELDL','rgb','GT方向光 ELDL · Stage3')]
   for i,(arm,sub,title) in enumerate(columns):
    x=45+i*470;d.text((x,130),title,font=font(26),fill='#192b42')
    im=Image.open(DEST/arm/sub/f'{frame:04d}.png').convert('RGB');assert im.size==(400,400)
    canvas.paste(im.resize((440,440),Image.Resampling.LANCZOS),(x,190))
    if sub=='rgb':
     r=next(v for v in reports[arm]['frames'] if int(v['frame'])==frame)
     d.text((x,655),f"PSNR {r['psnr']:.3f} · SSIM {r['ssim']:.5f}",font=font(24),fill='#536278')
   d.text((45,710),'相同采集视角；EL 与 ELDL 照明不同，分别对照自身 GT。',font=font(24),fill='#536278')
   canvas.save(folder/f'{frame:04d}.png')
   if frame in [10,80,150]:pages.append(canvas)
 pages[0].save(DEST/'1025-02-Stage3指标与同视角外观对比.pdf',save_all=True,append_images=pages[1:],resolution=150)
 (DEST/'1025-02-来源清单.json').write_text(json.dumps({'sources':provenance,'selection':selection},ensure_ascii=False,indent=2)+'\n')
 desc='# 静动态EL与GT方向光ELDL的Stage3外观对比\n\n只包含PSNR、SSIM与同视角RGB；不含几何覆盖、材质诊断或固定相机视频。\n\n'+'\n\n'.join(conclusions)+'\n\n统一指标：400×400、黑底完整浮点sRGB，逐帧PSNR/SSIM后均值，15 TEST；SSIM为11×11、sigma1.5。分数由原始浮点数组计算，展示PNG只用于观看。\n\nEL与ELDL照明、输入和父训练历史不同，属于跨数据集完整方法比较，不能当作只修改照明的单因素结论。两侧GT分别显示。历史TEST已被查看，不称新盲测。训练与参数选择只用TRAIN/DEV；最终选择和来源详见JSON。\n\nPDF为指标页及静/动态各三个固定同视角10、80、150。全部15视角PNG和原图分别保留，CSV可编辑。未达到两指标均提升时据实标明，不删失败结果。\n'
 (DEST/'1025-02-使用与结果说明.md').write_text(desc)
 table=''.join(f"<tr><td>{html.escape(r['实验'])}</td><td>{r['PSNR_dB']:.6f}</td><td>{r['SSIM']:.6f}</td></tr>" for r in rows)
 body=f'<h1>静动态 EL 与 GT方向光 ELDL</h1><p>Stage3 · PSNR / SSIM · 同视角外观</p><table><tr><th>实验</th><th>PSNR ↑</th><th>SSIM ↑</th></tr>{table}</table>'
 for line in conclusions:body+=f'<p>{line}</p>'
 for kind in ['静态','动态']:
  body+=f'<h2>{kind}同视角对比</h2>'
  for i in range(10,151,10):body+=f'<img loading="lazy" src="{kind}同视角/{i:04d}.png">'
 (DEST/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage3四组外观对比</title><style>body{font:18px sans-serif;margin:40px auto;max-width:1400px;color:#192b42;background:#f5f6f8}img{width:100%;margin:16px 0}td,th{padding:15px 35px;text-align:left}table{border-collapse:collapse;background:white}tr{border-bottom:1px solid #ddd}</style>'+body+'</html>')
 files={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file()}
 (DEST/'1025-02-文件校验.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n')
 print(DEST)
if __name__=='__main__':main()
