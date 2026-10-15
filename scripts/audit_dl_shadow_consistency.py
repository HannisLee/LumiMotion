"""保存光向对历史训练阴影证据的一致性；不读GT、不更新参数。"""
import argparse,json,hashlib
from pathlib import Path
import torch
from scripts.fit_dl_shadow_tracks import project_shadow,sample_shadow

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 root=Path('output/1015-15-LH-dynamic-DL-T05复核/seed_1015');r=json.loads((root/'result.json').read_text());cp=root/'checkpoint.pt';c=torch.load(cp,map_location='cpu',weights_only=False);cache=Path(r['config']['cache']);assert hashlib.sha256(cache.read_bytes()).hexdigest()==c['cache_sha256'];d=torch.load(cache,map_location='cpu',weights_only=False)
 xyz=d['canonical_xyz'].cuda();positions=d['positions'].cuda();valid=d['valid'].cuda();train=c['train'].cuda();o=c['plane_origin'].cuda();n=c['plane_normal'].cuda();basis=c['basis'].cuda();extent=r['scene_extent']
 # 恢复亮度筛选之前的平面范围，不能直接使用最终receiver推算范围。
 support=valid[train].sum(0)>=10;motion=(positions[train]-positions[train].mean(0)[None]).square().sum(-1).mean(0).sqrt();receiver=support&(((xyz-o)@n).abs()<extent*.01)&(motion<extent*.01)
 uv=(xyz[receiver]-o)@basis.T;lower=uv.amin(0);span=(uv.amax(0)-lower).clamp_min(.01);lower=lower-.05*span;span=span*1.1
 pos=c['positive'].cuda();neg=c['negative'].cuda();usable=c['usable'].cuda();occ=c['occluder'].cuda()
 fits={g:Path('output/smoke_test/1015-87-LH-dynamic-DL-T96稳健对应光度')/g/'checkpoint.pt' for g in ['mse','robust']};states={g:torch.load(f,map_location='cpu',weights_only=False) for g,f in fits.items()}
 current=states['mse'];assert torch.equal(current['times'],d['times']);groups={'T05':c['smooth'].cuda()}|{g:v['directions'].cuda() for g,v in states.items()};scores={};predictions={}
 for name,directions in groups.items():
  pred=[]
  for i in range(len(directions)):
   grid=project_shadow(positions[i,occ],directions[i:i+1],o,n,basis,lower,span);v,inside=sample_shadow(grid,positions[i],o,basis,lower,span);assert not ((pos[i]|neg[i])&~inside).any();pred.append(v[0])
  pred=torch.stack(pred);predictions[name]=pred.cpu()
  if name=='T05':assert torch.equal(pred.cpu(),c['predicted']['smooth'])
  per=.5*((1-pred)*pos).sum(-1)/pos.sum(-1).clamp_min(1)+.5*(pred*neg).sum(-1)/neg.sum(-1).clamp_min(1)
  def stats(indices):
   indices=torch.as_tensor(indices,device='cuda');u=indices[usable[indices]];return dict(count=len(u),mean=float(per[u].mean()))
  scores[name]=dict(historical_train=stats(torch.where(train)[0]),historical_held=stats(torch.where(~train)[0]),current_train=stats(current['train_frames']),current_held=stats(current['held_frames']),per_frame=per.cpu().tolist())
 assert abs(scores['T05']['historical_train']['mean']-r['metrics']['smooth']['fit'])<1e-8
 result=dict(gt_read=False,updates=0,exact_T05_prediction=True,scores=scores,current_held_in_historical_train=[i for i in current['held_frames'] if bool(c['train'][i])],scope='历史阴影标签一致性审计；标签来源含当前全部保留时间，不是严格泛化评测',inputs={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in [cp,root/'result.json',cache,Path(__file__),Path('scripts/fit_dl_shadow_tracks.py')]+list(fits.values())})
 torch.save(dict(predictions=predictions,positive=pos.cpu(),negative=neg.cpu(),usable=usable.cpu()),out/'predictions.pt');(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_frame'} for k,v in scores.items()},indent=2))
if __name__=='__main__':main()
