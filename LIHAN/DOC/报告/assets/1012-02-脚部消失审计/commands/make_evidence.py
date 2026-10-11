"""从原数组和PNG生成审计配图，并统计保存检查点的高斯高度分布。"""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parents[1]
from utils.geometry_eval_utils import camera_rays

def folder(prefix,step=10000):return next((ROOT/'output').glob(prefix+'*/evaluation_'+str(step)))
source=next((ROOT/'output').glob('1011-01*/eval_view'))
cams=json.loads((source/'transforms_test.json').read_text())
records={r['source_frame']:r for r in cams['frames']}
t13=folder('1011-27')

def data(folder,n):return dict(np.load(folder/f'prediction_{n:04d}.npz'))
def setup(a,n):
    r=records[n];rays=camera_rays(r['transform_matrix'],[r['fl_x']/2,r['fl_y']/2,200,200],400,400)
    gt_z=rays[...,2]+rays[...,5]*a['gt_depth'];pred_z=rays[...,2]+rays[...,5]*a['expected']
    feet=a['gt_foreground']&(gt_z>.05)&(gt_z<.35)
    y,x=np.where(feet);box=(max(0,x.min()-13),max(0,y.min()-13),min(400,x.max()+14),min(400,y.max()+14))
    return gt_z,pred_z,feet,box
def normals(a,key):
    out=(a[key]+1)/2
    if key=='gt_normal':out[~a['gt_foreground']]=0
    else:out*=a['alpha'][...,None]
    return np.clip(out,0,1)
def panel(ax,pixels,title,box=None,mask=None,cmap=None,vmin=None,vmax=None):
    if box:
        x1,y1,x2,y2=box;pixels=pixels[y1:y2,x1:x2]
        if mask is not None:mask=mask[y1:y2,x1:x2]
    ax.imshow(pixels,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
    if mask is not None:ax.contour(mask.astype(float),levels=[.5],colors=['#ff433f'],linewidths=.65)
    ax.set_title(title,fontsize=10);ax.axis('off')

fig,axes=plt.subplots(3,7,figsize=(16,7.5),layout='constrained')
for i,n in enumerate((10,80,150)):
    a=data(t13/'normal_depth',n);gz,pz,feet,box=setup(a,n)
    gt=np.asarray(Image.open(source/'images'/records[n]['file_path']).convert('RGB').resize((400,400)))/255
    rgb=np.asarray(Image.open(t13/'four_classes/rgb'/f'{n:04d}.png'))/255
    fields=[(gt,f'GT RGB / {n}'),(rgb,'T13 RGB'),(normals(a,'gt_normal'),'GT Normal'),(normals(a,'normal'),'T13 GS Normal'),(gz,'GT Z-world [0, 0.6 m]'),(pz,'T13 Z-world [0, 0.6 m]'),(a['alpha'],'T13 Alpha [0, 1]')]
    for j,(pixels,title) in enumerate(fields):
        panel(axes[i,j],pixels,title,box,feet if j in (1,3,5) else None,'viridis' if j in (4,5) else 'gray',0,.6 if j in (4,5) else 1)
fig.savefig(OUT/'figures/01-T13脚部RGB法线高度与Alpha.png',dpi=160);plt.close(fig)

comparisons=[('GT',None),('DL native / 35k',folder('1011-01',35000)),('EL native / 35k',folder('1011-02',35000)),('T01 SH / 35k',folder('1011-13',35000)),('T03 physical / 35k',folder('1011-15',35000)),('T13 / 10k',t13),('T14 / 10k',folder('1011-28'))]
fig,axes=plt.subplots(2,len(comparisons),figsize=(16,5.5),layout='constrained')
a=data(t13/'normal_depth',80);gz,pz,feet,box=setup(a,80)
for j,(title,path) in enumerate(comparisons):
    b=a if path is None else data(path/'normal_depth',80)
    pixels=normals(b,'gt_normal' if path is None else 'normal')
    panel(axes[0,j],pixels,title)
    panel(axes[1,j],pixels,'Foot region / frame 80',box,feet)
fig.savefig(OUT/'figures/02-历史训练与最新训练脚部对比.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(2,5,figsize=(12,5.5),layout='constrained')
for i,tag in enumerate(('T13','T14')):
    for j,step in enumerate((999,5000,8000,9000,10000)):
        path=(t13 if tag=='T13' else folder('1011-28'))/'normal_depth' if step==10000 else OUT/'checkpoint_renders'/f'{tag}_{step}'
        b=data(path,80);panel(axes[i,j],normals(b,'normal'),f'{tag} / {step}',box,feet)
fig.savefig(OUT/'figures/03-中间检查点脚部变化.png',dpi=180);plt.close(fig)

counts=[];training={}
for prefix in ('1011-13','1011-14','1011-15','1011-27','1011-28'):
    m=next((ROOT/'output').glob(prefix+'*/model_mlp'))
    for p in sorted(m.glob('point_cloud/iteration_*/point_cloud.ply')):
        v=PlyData.read(p)['vertex'];z=np.asarray(v['z']);alpha=1/(1+np.exp(-np.asarray(v['opacity'])))
        row=dict(model=str(m),iteration=int(p.parent.name.split('_')[-1]),points=len(v),foot_height_points=int(((z>.05)&(z<.35)).sum()),foot_opacity_gt_point1=int(((z>.05)&(z<.35)&(alpha>.1)).sum()))
        for hi in (.25,.30,.40,.45):row[f'height_005_{hi:.2f}']=int(((z>.05)&(z<hi)).sum())
        counts.append(row)
    log=m/'early_geometry_metrics.jsonl'
    if log.exists():training[prefix]=[json.loads(s) for s in log.read_text().splitlines() if json.loads(s)['iteration']>=8700]
(OUT/'checkpoint_point_counts.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n')
(OUT/'reset_window_training_metrics.json').write_text(json.dumps(training,ensure_ascii=False,indent=2)+'\n')
print('三张证据图、点数及9000步窗口训练日志已生成。')
