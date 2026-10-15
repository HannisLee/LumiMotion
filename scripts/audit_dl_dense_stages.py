"""独立稠密对应的离面区域分阶段诊断；不以默认目标投影筛选。"""
import argparse,hashlib,json,shlex,sys
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from scipy.spatial import cKDTree

def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--output',required=True);a=p.parse_args();probe=Path(a.probe);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
 cache=Path('output/1015-07-LH-dynamic-DL-T02表面轨迹初始化/cache/tracks.pt');plane=Path('output/1015-15-LH-dynamic-DL-T05复核/seed_1015/checkpoint.pt');src=Path('output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view');t=torch.load(cache,map_location='cpu',weights_only=False);pl=torch.load(plane,map_location='cpu',weights_only=False);frames=json.loads((src/'transforms_train.json').read_text())['frames'];uv=(t['pixels'].numpy()+.5)*2-.5;valid=t['valid'].numpy();n=pl['plane_normal'].numpy();o=pl['plane_origin'].numpy();rows=[];points=[];inputs=[cache,plane,src/'transforms_train.json',Path(__file__),probe/'result.json']
 for ordinal,path in enumerate(sorted(probe.glob('stages_*.npz'))):
  d=np.load(path);aa,b,c=map(int,d['indices']);pa=d['pa'];idx=np.where(valid[aa])[0];dist,near=cKDTree(uv[aa,idx]).query(pa);ids=idx[near];associated=d['base']&(dist<=4);height=np.abs((t['positions'][aa].numpy()[ids]-o)@n)
  flags=dict(source=d['base'],source_unsaturated=d['base']&d['source_unsat'],pair_support=d['pair_support'],pair_fb=d['pair_support']&(d['fb']<2),pair_fb_unsat=d['pair_support']&(d['fb']<2)&d['pair_unsat'],triple_support=d['support'],triple_fb=d['support']&(d['fb']<2)&(d['fc']<2)&(d['fbc']<2),triple_cycle=d['support']&(d['fb']<2)&(d['fc']<2)&(d['fbc']<2)&(d['cycle']<2),triple_final=d['support']&(d['fb']<2)&(d['fc']<2)&(d['fbc']<2)&(d['cycle']<2)&d['unsat'])
  groups=dict(all=d['base'],unassociated=d['base']&~associated,plane=associated&(height<=.05),border=associated&(height>.05)&(height<=.1),offplane=associated&(height>.1))
  row=dict(indices=[aa,b,c],pair_source_span=frames[b]['source_frame']-frames[aa]['source_frame'],triple_source_span=frames[c]['source_frame']-frames[aa]['source_frame'],counts={g:{k:int((mask&f).sum()) for k,f in flags.items()} for g,mask in groups.items()});rows.append(row)
  off=groups['offplane'];chosen=np.where(off)[0]
  for k in chosen:points.append(dict(indices=[aa,b,c],pa=pa[k].tolist(),pb=d['pb'][k].tolist(),gaussian_id=int(t['ids'][ids[k]]),height=float(height[k]),fb=float(d['fb'][k]),cycle=float(d['cycle'][k]),flags={name:bool(f[k]) for name,f in flags.items()}))
  imgs=[]
  for i in [aa,b]:
   ip=src/'images'/Path(frames[i]['file_path']).name;inputs.append(ip);imgs.append(cv2.resize(np.array(Image.open(ip).convert('RGB')),(400,400),interpolation=cv2.INTER_AREA))
  canvas=np.concatenate(imgs,axis=1)
  for k in chosen:
   x=tuple(np.round(pa[k]).astype(int));cv2.circle(canvas,x,1,(255,220,0),-1)
   if flags['pair_fb'][k]:
    y=tuple(np.round(d['pb'][k]+[400,0]).astype(int));color=(30,255,80) if flags['pair_fb_unsat'][k] else (255,30,30);cv2.circle(canvas,x,2,color,-1);cv2.circle(canvas,y,2,color,-1)
  Image.fromarray(canvas).save(out/f'offplane_{ordinal:02d}.png');inputs.append(path)
 totals={g:{k:sum(r['counts'][g][k] for r in rows) for k in rows[0]['counts'][g]} for g in rows[0]['counts']}
 result=dict(gt_read=False,optimizer_updates=0,rows=rows,totals=totals,scope='源粗几何离面分区不是GT语义，不按目标投影筛选；颜色黄=源离面、红=双向通过但饱和、绿=双向且未饱和',inputs={str(ip.resolve()):hashlib.sha256(ip.read_bytes()).hexdigest() for ip in inputs});(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'offplane_points.json').write_text(json.dumps(points,ensure_ascii=False,indent=2));print(json.dumps(totals,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
