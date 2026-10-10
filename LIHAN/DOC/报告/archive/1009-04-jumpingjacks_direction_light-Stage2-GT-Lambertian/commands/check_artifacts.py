from pathlib import Path
import sys,json,math
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
run=Path(sys.argv[1]);it=int(sys.argv[2]);smoke=it==35200;model=run/'model_mlp';render=model/'renders_stage2'/('ours_'+str(it));ins=run/'inspection';ins.mkdir(exist_ok=True)
assert (model/'TRAIN_COMPLETED').read_text().strip()==str(it)
assert (render/'RENDER_COMPLETED').read_text().strip()=='all'
report={'automatic_status':'PASS','scope':'流程与产物检查；人工质量审阅待完成','iteration':it,'manual_review':'PENDING','images':{},'videos':{},'checkpoint':{}}
def tensors(obj):
    if torch.is_tensor(obj):return [obj]
    if isinstance(obj,dict):return [t for x in obj.values() for t in tensors(x)]
    if isinstance(obj,(tuple,list)):return [t for x in obj for t in tensors(x)]
    return []
initial=torch.load(model/'stage2_photometric/iteration_35000/checkpoint.pth',map_location='cpu')
for step in ([35000,35200] if smoke else [35000,40000,50000,55000]):
    file=model/'stage2_photometric'/('iteration_'+str(step))/'checkpoint.pth';s=torch.load(file,map_location='cpu')
    assert s['iteration']==step and s['source_iteration']==35000
    ts=tensors(s);assert ts and all(torch.isfinite(t).all() for t in ts)
    assert initial['source_files']==s['source_files'] and initial['frozen_fingerprint']==s['frozen_fingerprint']
    assert torch.equal(initial['light']['raw_directions'],s['light']['raw_directions'])
    assert len(s['optimizer']['param_groups'])==1 and s['optimizer']['param_groups'][0]['name']=='albedo'
    updates=[int(v['step']) for v in s['optimizer']['state'].values()];assert updates==([] if step==35000 else [step-35000])
    report['checkpoint'][str(step)]={'finite_tensors':len(ts),'adam_steps':updates,'gaussian_count':s['material']['raw_albedo'].shape[0],'source_unchanged':True,'light_unchanged':True}
    report['frozen_fingerprint']=s['frozen_fingerprint']
import hashlib
for file,digest in initial['source_files'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
metrics=json.loads((render/'eval/metrics.json').read_text());assert len(metrics['frames'])==15
assert all(math.isfinite(v) for v in metrics['metrics'].values())
report['metrics']=metrics['metrics'];report['protocol']=metrics['protocol'];report['source_files']=initial['source_files']
cal=json.loads((model/'calibration.json').read_text());assert cal['intensity']==initial['config']['photometric_light_intensity'] and len(cal['train_keys'])==135
report['calibration']=cal;representatives={}
for cat in ['rgb','alpha','normal','albedo']:
    files=sorted((render/'insights'/cat).glob('*.png'));assert len(files)==150
    for file in files:
        with Image.open(file) as im:assert im.size==(400,400);im.verify()
    representatives[cat]=[str(files[i]) for i in [0,74,149]];report['images'][cat]=150
for folder,count in [('hdr',150),('directional',72)]:
    files=sorted((render/folder).glob('*.png'));assert len(files)==count
    for file in files:
        with Image.open(file) as im:assert im.size==(400,400);im.verify()
    representatives[folder]=[str(files[i]) for i in [0,count//2,count-1]];report['images'][folder]=count
videos=sorted(render.glob('**/*.mp4'));assert len(videos)==6
for file in videos:
    cap=cv2.VideoCapture(str(file));assert cap.isOpened();count=0
    while True:
        ok,img=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==(72 if 'directional' in file.parts else 150)
    report['videos'][str(file)]=count
def sheet_images(path,columns,rows):
    canvas=Image.new('RGB',(320*len(columns),344*len(rows)),'white');draw=ImageDraw.Draw(canvas)
    for row,entries in enumerate(rows):
        for col,(label,file) in enumerate(entries):
            image=Image.open(file).convert('RGB');image.thumbnail((320,320))
            canvas.paste(image,(col*320,row*344+24));draw.text((col*320+4,row*344+5),label,fill='black')
    canvas.save(path)
for name,categories in [('four_categories.png',['rgb','alpha','normal','albedo']),('relight.png',['hdr','directional'])]:
    rows=[[(cat+' '+Path(representatives[cat][i]).stem,representatives[cat][i]) for cat in categories] for i in range(3)]
    sheet_images(ins/name,categories,rows)
rows=[metrics['frames'][i] for i in [0,7,14]]
sheet_images(ins/'rgb_comparison.png',['render','gt'],[[(x['frame']+' '+suffix,render/'eval'/(x['frame']+'_'+suffix+'.png')) for suffix in ['render','gt']] for x in rows])
report['representatives']=representatives
(ins/'artifact_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'run':str(run),'metrics':report['metrics'],'calibration':cal['intensity'],'automatic_status':'PASS'},ensure_ascii=False))
