"""只用训练时刻检查实际活跃点的默认形变Jacobian；零更新，无GT。"""
import argparse,json,hashlib
from pathlib import Path
import torch
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False)
 runtime=json.loads((run/'runtime.json').read_text());args=runtime['args'];history=json.loads((run/'result.json').read_text())['history'];assert not args['gt_direction_reference']
 active=sorted({t['candidate_audit']['point_id'] for e in history for t in e['attempts']});ids=active+[93905] if 93905 not in active else active
 for f in [Path(args['model'])/'cfg_args',Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply',Path(args['model'])/'deform/iteration_35000/deform.pth',Path(args['source'])/'transforms_train.json']:
  assert hashlib.sha256(f.read_bytes()).hexdigest()==runtime['inputs'][str(f)]
 cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=4;cfg.load2gpu_on_the_fly=True
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
 deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
 infos=readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False);cameras=[loadCam(cfg,i,info,1.) for i,info in enumerate(infos)];train=runtime['selected'];saved=torch.load(run/'xyz_metric.pt',map_location='cpu',weights_only=False);sample=saved['indices'];assert set(sample)<=set(train)
 x=pc.get_xyz[ids].detach().clone().requires_grad_(True);gate=pc.get_binary_feature()[ids].detach();jac=[]
 for i in train:
  c=cameras[i];c.load2device();world=x+deform.step(x,c.fid.reshape(1,1).expand(len(ids),1),feature=gate,camera_center=c.camera_center)['d_xyz'];rows=[torch.autograd.grad(world[:,k].sum(),x,retain_graph=k<2)[0].detach() for k in range(3)];jac.append(torch.stack(rows,dim=1).cpu());c.load2device('cpu')
 J=torch.stack(jac).double();assert torch.isfinite(J).all();gram=J.transpose(-1,-2)@J;diag=gram.diagonal(dim1=-2,dim2=-1);d9=diag[[train.index(i) for i in sample]].amax(0);dall=diag.amax(0);ref=saved['diagonal'][ids].double();relative=float(((d9-ref).abs()/(1+ref)).max());assert relative<.005
 results=[]
 for n,pid in enumerate(ids):
  rec=dict(id=pid,active_candidate=pid in active,diag9=d9[n].tolist(),diag_all_train=dall[n].tolist(),max_diagonal_ratio=float(((1+dall[n])/(1+d9[n])).max()),negative_det_frames=int((torch.linalg.det(J[:,n])<0).sum()))
  for name,d in [('nine',d9[n]),('all_train',dall[n])]:
   inv=(1+d).rsqrt();white=gram[:,n]*inv[None,:,None]*inv[None,None,:];e=torch.linalg.eigvalsh(white)[:,-1];index=int(e.argmax());rec[name]=dict(max_whitened_eigenvalue=float(e[index]),time=train[index])
  sv=torch.linalg.svdvals(J[:,n]);rec['max_singular_value']=float(sv[:,0].max());rec['median_condition_number']=float((sv[:,0]/sv[:,-1].clamp_min(1e-12)).median());rec['trigger_times']=[]
  for entry in history:
   for t in entry['attempts']:
    v=t['candidate_audit']
    if v['point_id']==pid and v['index'] in train:
     k=train.index(v['index']);rec['trigger_times'].append(dict(block=entry['block'],time=v['index'],accepted=t['accepted'],diagonal_ratio=float(((1+diag[k,n])/(1+d9[n])).max()),singular_values=sv[k].tolist()))
  results.append(rec)
 torch.save(dict(ids=ids,train=train,jacobian=J,diag9=d9,diag_all_train=dall),out/'jacobian.pt')
 data=dict(status='PASS',gt_read=False,updates=0,frames=len(train),ids=ids,source_run=str(run),source_checkpoint_sha256=hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest(),saved_metric_relative_error=relative,results=results,scope='默认位置/默认形变的局部线性检查；block2状态及有限候选非线性未复原；白化特征值是二次型度量比，不是实际优化器步幅保证')
 (out/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print(json.dumps(data,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
