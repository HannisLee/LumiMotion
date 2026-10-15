"""零更新分解已接受终点的世界位移；不复原被拒候选，不读取GT。"""
import argparse,json,hashlib
from pathlib import Path
import torch
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
 results={}
 for path in map(Path,a.checkpoints):
  ck=torch.load(path,map_location='cpu',weights_only=False);args=ck['args'];runtime=json.loads((path.parent/'runtime.json').read_text());old=json.loads((path.parent/'result.json').read_text())
  assert not args['gt_direction_reference'] and args['geometry_groups']=='canonical'
  for f in [Path(args['model'])/'cfg_args',Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply',Path(args['model'])/'deform/iteration_35000/deform.pth',Path(args['source'])/'transforms_train.json',Path(args['cache'])]:
   assert hashlib.sha256(f.read_bytes()).hexdigest()==runtime['inputs'][str(f)]
  cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=4;cfg.load2gpu_on_the_fly=True
  pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
  deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
  initial={k:v.clone() for k,v in deform.deform.state_dict().items()};final={k:v.cuda() for k,v in ck['deform_state_dict'].items()}
  assert all(torch.equal(v,final[k]) for k,v in initial.items() if not k.startswith('gaussian_warp.'))
  x0=pc.get_xyz.detach().clone();x1=ck['geometry']['_xyz'].cuda();gate=pc.get_binary_feature().detach();n=len(x0)
  infos=readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False);cameras=[loadCam(cfg,i,info,1.) for i,info in enumerate(infos)]
  maxima={k:torch.zeros(n,device='cuda') for k in ['full','xyz_only','head_only','interaction']};argtime=torch.zeros(n,dtype=torch.long,device='cuda');frames=[]
  def query(x,c,state):
   deform.deform.load_state_dict(state,strict=True)
   return x+deform.step(x,c.fid.reshape(1,1).expand(n,1),feature=gate,camera_center=c.camera_center)['d_xyz']
  for i,c in enumerate(cameras):
   c.load2device();w0=query(x0,c,initial);wx=query(x1,c,initial);wh=query(x0,c,final);wf=query(x1,c,final)
   values={'full':(wf-w0).norm(dim=-1),'xyz_only':(wx-w0).norm(dim=-1),'head_only':(wh-w0).norm(dim=-1),'interaction':(wf-wx-wh+w0).norm(dim=-1)}
   argtime=torch.where(values['full']>maxima['full'],i,argtime)
   for k,v in values.items():maxima[k]=torch.maximum(maxima[k],v)
   value,idx=values['full'].max(0);frames.append(dict(index=i,max=float(value),id=int(idx),held=i%9==4));c.load2device('cpu')
   if i%30==0:print(json.dumps(dict(group=path.parent.name,frame=i)),flush=True)
  error=max(abs(f['max']-old['motion'][str(f['index'])]) for f in frames);assert error<1e-7
  delta=(x1-x0).norm(dim=-1);ids=torch.topk(maxima['full'],100).indices
  cache=torch.load(args['cache'],map_location='cpu',weights_only=False);train=runtime['selected'];lookup={int(v):j for j,v in enumerate(cache['ids'])};count=cache['valid'][train].sum(0)
  top=[]
  for idx in ids.tolist():
   j=lookup.get(idx);top.append(dict(id=idx,time=int(argtime[idx]),dynamic_gate=float(gate[idx]),canonical_delta=float(delta[idx]),**{k:float(v[idx]) for k,v in maxima.items()},sampled=j is not None,valid_train_count=int(count[j]) if j is not None else None))
  def quant(v):return torch.quantile(v,torch.tensor([0.,.5,.9,.99,1.],device='cuda')).tolist()
  summary=dict(status='PASS',gt_read=False,updates=0,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reload_max_error=error,quantiles={k:quant(v) for k,v in maxima.items()},top100_dynamic=sum(v['dynamic_gate']>.5 for v in top),top100_sampled=sum(v['sampled'] for v in top),unique_frame_maximizers=len(set(f['id'] for f in frames)),frames=frames,top100=top,scope='已接受终点分解，非被拒候选活跃约束；未采样不等于没有渲染梯度支持')
  torch.save(dict(maxima={k:v.cpu() for k,v in maxima.items()},argtime=argtime.cpu(),canonical_delta=delta.cpu(),gate=gate.cpu()),out/(path.parent.name+'.pt'))
  results[path.parent.name]=summary
 (out/'result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps({k:{f:v[f] for f in ['status','quantiles','top100_dynamic','top100_sampled','unique_frame_maximizers']} for k,v in results.items()},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
