"""独立重载canonical位置检查点，逐时刻复算实际世界位移；不读GT。"""
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(exist_ok=False);path=Path(a.checkpoint);ck=torch.load(path,map_location='cpu',weights_only=False);args=ck['args'];runtime=json.loads((path.parent/'runtime.json').read_text());result=json.loads((path.parent/'result.json').read_text())
 assert not args['gt_direction_reference'] and args['geometry_groups']=='canonical'
 for f in [Path(args['model'])/'cfg_args',Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply',Path(args['model'])/'deform/iteration_35000/deform.pth',Path(args['source'])/'transforms_train.json']:
  assert hashlib.sha256(f.read_bytes()).hexdigest()==runtime['inputs'][str(f)]
 cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=4;cfg.load2gpu_on_the_fly=True
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
 deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
 infos=readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False);cameras=[loadCam(cfg,i,info,1.) for i,info in enumerate(infos)]
 light=json.loads(Path(args['lights']).read_text());assert [c.image_name_train_light for c in cameras]==light['frames']
 def positions(c):
  c.load2device();ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center);pos=(pc.get_xyz+ds['d_xyz']).clone();c.load2device('cpu');return pos
 original=[positions(c) for c in cameras];initial_xyz=pc.get_xyz.clone()
 for k,v in ck['geometry'].items():getattr(pc,k).copy_(v.cuda())
 deform.deform.load_state_dict(ck['deform_state_dict'],strict=True)
 measured={str(i):float((positions(c)-original[i]).norm(dim=-1).max()) for i,c in enumerate(cameras)}
 error=max(abs(v-result['motion'][i]) for i,v in measured.items());xyz=float((pc.get_xyz-initial_xyz).norm(dim=-1).max());passed=error<1e-7 and max(measured.values())<=.00101 and xyz>0
 data=dict(status='PASS' if passed else 'FAILED',gt_read=False,frames=len(cameras),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),canonical_xyz_delta=xyz,max_world_displacement=max(measured.values()),max_difference_to_training=error,per_time=measured)
 (out/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in data.items() if k!='per_time'},ensure_ascii=False));assert passed
if __name__=='__main__':main()
