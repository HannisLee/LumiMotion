"""原生PBR真实权重/全视频/前后景数值审计；人工质量PASS另回填。"""
from pathlib import Path
import json,hashlib,math
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from plyfile import PlyData
ROOT=Path(__file__).resolve().parent.parent
model=ROOT/'model_mlp';render=model/'renders_stage2/ours_55000';out=ROOT/'inspection';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensors(obj):
 if torch.is_tensor(obj):return [obj]
 if isinstance(obj,dict):return [t for v in obj.values() for t in tensors(v)]
 if isinstance(obj,(tuple,list)):return [t for v in obj for t in tensors(v)]
 return []
config=json.loads((model/'stage2_config.json').read_text());source=json.loads((model/'stage1_source.json').read_text())
for p,h in source.items():assert sha(p)==h
assert config['pbr_light_mode']=='gt_directional' and config['loss_preset']=='pbr_gt_directional_geometry'
assert 'GT评测专用' not in json.dumps(config,ensure_ascii=False)
initial=PlyData.read(model/'point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
checkpoints={}
for step in [40000,50000,55000]:
 ply=model/f'point_cloud/iteration_{step}/point_cloud.ply';v=PlyData.read(ply)['vertex'].data
 assert len(v)==len(initial) and all(np.isfinite(v[f]).all() for f in v.dtype.names)
 frozen=[f for f in v.dtype.names if f in ['opacity','fea_0'] or f.startswith('scale_')]
 assert all(np.array_equal(v[f],initial[f]) for f in frozen), 'opacity/scale/feature应冻结'
 displacement=np.linalg.norm(np.column_stack([v[f]-initial[f] for f in ['x','y','z']]),axis=1)
 assert displacement.max()<=config['pbr_geometry_max_displacement']+1e-5
 q=np.column_stack([v[f] for f in ['rot_0','rot_1','rot_2','rot_3']]);q0=np.column_stack([initial[f] for f in ['rot_0','rot_1','rot_2','rot_3']])
 q/=np.linalg.norm(q,axis=1,keepdims=True);q0/=np.linalg.norm(q0,axis=1,keepdims=True)
 angle=2*np.degrees(np.arccos(np.clip(np.abs((q*q0).sum(1)),0,1)))
 assert angle.max()<=config['pbr_geometry_max_angle']+.01
 assert displacement.max()>0 and angle.max()>0
 rough=1/(1+np.exp(-v['roughness']))
 assert np.abs(v['roughness']-initial['roughness']).max()>0, 'roughness没有真实训练'
 pth=model/f'deform/iteration_{step}/deform.pth';state=torch.load(pth,map_location='cpu');ts=tensors(state)
 assert ts and all(torch.isfinite(t).all() for t in ts)
 light=json.loads((model/f'pbr_directional/iteration_{step}/lighting.json').read_text())
 assert light['iteration']==step and light['lights_sha256']==sha(model/'pbr_gt_lights.json')
 assert light['intensity_fitted_from_train_rgb'] is True and .01<=light['effective_intensity_scale']<=100.00001
 checkpoints[str(step)]={'gaussians':len(v),'finite':True,'geometry_unchanged':False,'max_displacement':float(displacement.max()),'max_rotation_deg':float(angle.max()),'opacity_scale_feature_unchanged':True,'roughness_range':[float(rough.min()),float(rough.max())],'ply_sha256':sha(ply),'deform_sha256':sha(pth),'deform_finite_tensors':len(ts)}
assert (render/'RENDER_COMPLETED').read_text().strip()=='all'
metrics=json.loads((render/'eval/metrics.json').read_text());assert len(metrics['frames'])==15
assert all(math.isfinite(v) for v in metrics['metrics'].values())
imgs={};representatives={}
for category in ['rgb','alpha','normal','albedo']:
 files=sorted((render/'insights'/category).glob('*.png'));assert len(files)==150
 for f in files:
  with Image.open(f) as im:assert im.size==(400,400);im.verify()
 imgs[category]=150;representatives[category]=[str(files[i]) for i in [0,74,149]]
for cat,frames in [('directional',72),('hdr',150)]:
 files=sorted((render/cat).glob('*.png'));assert len(files)==frames
 for f in files:
  with Image.open(f) as im:assert im.size==(400,400);im.verify()
 imgs[cat]=frames;representatives[cat]=[str(files[i]) for i in [0,frames//2,frames-1]]
video={};videos=sorted(render.glob('**/*.mp4'));assert len(videos)==6
for f in videos:
 cap=cv2.VideoCapture(str(f));assert cap.isOpened();count=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape[:2]==(400,400);count+=1
 cap.release();assert count==(72 if 'directional' in f.parts else 150);video[str(f)]=count
bg=[]
for row in metrics['frames']:
 k=row['frame'];im=np.asarray(Image.open(render/'eval'/(k+'_render.png')).convert('RGB'),dtype=float)/255
 mask=np.asarray(Image.open(render/'eval'/(k+'_mask.png')).convert('L'))>127
 # 内缩背景，避免抗锯齿/轮廓差；全背景均值也报告，未调整模型。
 core=cv2.erode((~mask).astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
 bg.append({'frame':k,'mean_rgb_background_all':float(im[~mask].mean()),'mean_rgb_background_core':float(im[core].mean()),'p99_background_core':float(np.quantile(im[core],.99))})
def sheet(path,rows):
 cols=len(rows[0]);canvas=Image.new('RGB',(cols*320,len(rows)*344),'white');draw=ImageDraw.Draw(canvas)
 for i,row in enumerate(rows):
  for j,(label,p) in enumerate(row):
   im=Image.open(p).convert('RGB');im.thumbnail((320,320));canvas.paste(im,(j*320,i*344+24));draw.text((j*320+5,i*344+6),label,fill='black')
 canvas.save(path)
for title,cats in [('four_categories.png',['rgb','alpha','normal','albedo']),('relight.png',['hdr','directional'])]:
 sheet(out/title,[[(c+' '+Path(representatives[c][i]).stem,representatives[c][i]) for c in cats] for i in range(3)])
sheet(out/'rgb_comparison.png',[[(r['frame']+' '+suffix,render/'eval'/(r['frame']+'_'+suffix+'.png')) for suffix in ['render','gt']] for r in [metrics['frames'][i] for i in [0,7,14]]])
report=dict(automatic_status='PASS',scope='实际权重有限性、几何信赖域和opacity/scale/feature冻结、roughness已变化、全量图片与6视频解码、15帧背景数值；人工审阅待完成',manual_review='PENDING',checkpoint=checkpoints,rgb=metrics['metrics'],images=imgs,videos=video,representatives=representatives,background=bg,source_files=source)
(out/'artifact_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'automatic_status':'PASS','rgb':metrics['metrics'],'background_mean_core':float(np.mean([r['mean_rgb_background_core'] for r in bg]))},ensure_ascii=False))
