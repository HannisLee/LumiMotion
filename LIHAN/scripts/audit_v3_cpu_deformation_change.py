"""CPU父/候选canonical与共享MLP变化分解；固定TRAIN采样，不使用GT语义。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from utils.time_utils import DeformNetwork
FRAMES=[1,31,59,81,111,149]

def geometry(path):
    v=PlyData.read(str(path))['vertex'];return dict(xyz=np.stack([v[k] for k in ['x','y','z']],-1),q=np.stack([v['rot_'+str(i)] for i in range(4)],-1),gate=1/(1+np.exp(np.clip(-2*np.asarray(v['fea_0']),-80,80))))
def model(path):
    state=torch.load(path,map_location='cpu',weights_only=False);m=DeformNetwork(is_blender=True,pred_color=True,hyper_dim=1);m.load_state_dict(state);m.pred_color=False;m.eval().requires_grad_(False);return m,state

def summarize(x):
    return dict(mean=float(np.mean(x)),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(np.max(x))) if len(x) else None

def qangle(a,b):
    a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True);return np.degrees(2*np.arccos(np.clip(np.abs((a*b).sum(-1)),0,1)))
@torch.no_grad()
def predict(m,xyz,gate,frame):
    x=torch.from_numpy(xyz.copy());d=m(x,torch.full((len(x),1),(frame-1)/149),feature=torch.from_numpy(gate[:,None].copy()),iteration=35000)
    return {k:d[k].numpy() for k in ['d_xyz','d_rotation']}
@torch.no_grad()
def main(a):
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True);assert not (a.output/'audit.json').exists()
    p=geometry(a.model/'point_cloud/iteration_0/point_cloud.ply');parent,ps=model(a.model/'deform/iteration_0/deform.pth');data=np.load(a.support);n=data['normal'];origin=data['origin'];extent=float(data['extent'])
    indices=np.arange(0,len(p['xyz']),17);xp=p['xyz'][indices];gp=p['gate'][indices];q0=p['q'][indices]
    baseline={f:predict(parent,xp,gp,f) for f in FRAMES};near=np.min(np.stack([np.abs((xp+baseline[f]['d_xyz']-origin)@n) for f in FRAMES]),axis=0)<.1*extent
    masks={'static_gate_lt01':gp<.1,'dynamic_gate_gt09':gp>.9,'transition':(gp>=.1)&(gp<=.9),'dynamic_near_predicted_plane':(gp>.9)&near}
    fullmasks={'static_gate_lt01':p['gate']<.1,'dynamic_gate_gt09':p['gate']>.9,'transition':(p['gate']>=.1)&(p['gate']<=.9)};rows=[]
    for step in [1500,3000]:
        c=geometry(a.model/f'point_cloud/iteration_{step}/point_cloud.ply');assert np.array_equal(c['gate'],p['gate']);m,cs=model(a.model/f'deform/iteration_{step}/deform.pth')
        changed={k:dict(l2=float((cs[k]-v).norm()),relative_l2=float((cs[k]-v).norm()/v.norm().clamp_min(1e-20))) for k,v in ps.items() if not torch.equal(v,cs[k])}
        row=dict(iteration=step,changed_deform_tensors=changed,canonical={k:dict(xyz_change=summarize(np.linalg.norm(c['xyz'][mask]-p['xyz'][mask],axis=-1)),quaternion_rotation_change_deg=summarize(qangle(c['q'][mask],p['q'][mask]))) for k,mask in fullmasks.items()},frames=[])
        xc=c['xyz'][indices];qc=c['q'][indices]
        for frame in FRAMES:
            old=baseline[frame];frozen=predict(parent,xc,gp,frame);new=predict(m,xc,gp,frame)
            positions=[xp+old['d_xyz'],xc+frozen['d_xyz'],xc+new['d_xyz']];rotations=[q0+old['d_rotation'],qc+frozen['d_rotation'],qc+new['d_rotation']]
            values={}
            for label,i,j in [('canonical_and_input',0,1),('deform_parameter',1,2),('total',0,2)]:
                values[label]={k:dict(position_world=summarize(np.linalg.norm(positions[j][mask]-positions[i][mask],axis=-1)),effective_rotation_deg=summarize(qangle(rotations[j][mask],rotations[i][mask]))) for k,mask in masks.items()}
            row['frames'].append(dict(frame=frame,changes=values));print('CPU审计',step,frame,flush=True)
        rows.append(row)
    result=dict(gt_geometry_read=False,heldout_read=False,gpu_used=False,optimizer_updates=0,frames=FRAMES,sampled_indices_stride=17,sampled_count=len(indices),full_gate_counts={k:int(v.sum()) for k,v in fullmasks.items()},sample_group_counts={k:int(v.sum()) for k,v in masks.items()},rows=rows,definitions='canonical_and_input=候选canonical+父MLP(候选xyz)减父canonical+父MLP(父xyz)；deform_parameter=候选canonical+候选MLP减候选canonical+父MLP；total=完整候选减父。分量范数非线性不可相加。rotation为归一化raw quaternion加delta后的最小SO3角，不是GT法线误差。near-plane只按6个TRAIN父预测到既有面距离<.1extent固定，不是鞋足语义。')
    (a.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['model','support','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
