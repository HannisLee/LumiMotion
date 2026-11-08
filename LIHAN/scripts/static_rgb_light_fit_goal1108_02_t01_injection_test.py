import argparse, json, sys
from pathlib import Path
import numpy as np
from LIHAN.scripts import static_rgb_light_fit_goal1108_02_t01 as m
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);root=parser.parse_args().output;root.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20); G=3331; target={(False,True):1450,(False,False):1233,(True,True):648}; cells=[]
for x in range(1,100000):
 h=np.int64(x)*73856093 ^ np.int64(3)*83492791; held=h%5==0; split=(h//5)%2==0; key=(bool(held),bool(split))
 if held:key=(True,True)
 if target.get(key,0)>0:cells.append((x,h));target[key]-=1
 if not any(target.values()):break
assert len(cells)==G
xyz=np.array([[(x+.1)/20,0,.155] for x,h in cells]);n=m.unit(rng.normal(size=(G,3)));n[:,2]=np.clip(n[:,2],-.8,.8);n=m.unit(n);n[:,2]=np.clip(n[:,2],-.84,.84);n[:,:2]=m.unit(n[:,:2])*np.sqrt(1-n[:,2,None]**2)
h=np.array([h for x,h in cells],dtype=np.int64);held=h%5==0;fi=np.flatnonzero(~held);vi=np.flatnonzero(held);A=fi[(h[fi]//5)%2==0];B=fi[(h[fi]//5)%2!=0];assert (len(fi),len(vi),len(A),len(B))==(2683,648,1450,1233)
t=np.linspace(0,7,150);e=np.deg2rad(45);l=np.column_stack((np.cos(t)*np.cos(e),np.sin(t)*np.cos(e),np.full(150,np.sin(e)))); y=np.maximum(l@n.T,0)*.7+.1+rng.normal(0,.01,(150,G));w=np.ones_like(y);ids=np.arange(G)
obs=root/'synthetic_observations.npz';np.savez_compressed(obs,normals=n,intensity=y,weights=w,xyz=xyz,gaussian_ids=ids,frames=np.arange(1,151));base=root/'synthetic_baseline';base.mkdir()
support={'body_fit':fi,'body_val':vi,'all_val':vi,'split_A':A,'split_B':B,'gaussian_ids':ids,'hval':h,'body_mask':np.ones(G,dtype=bool)};np.savez_compressed(base/'support.npz',**support)
np.savez_compressed(base/'predictions.npz',E1=l,E2=l,E3=l,E4=l)
for seed in m.BOOT:np.save(base/f'bootstrap_{seed}_sample.npy',np.random.default_rng(seed).choice(fi,len(fi),replace=True))
for name,count in [('E3',len(fi)),('stage1_A',len(A)),('stage1_B',len(B)),('stage1_boot103108',len(fi)),('stage1_boot103109',len(fi))]:
 dest=base/'runs'/name;dest.mkdir(parents=True);np.save(dest/'direction.npy',l);(dest/'config.json').write_text(json.dumps(dict(support_count=count,stage2=False,geo_smooth=0,balanced=False,steps=250,delta=.03,dtype='float64')))
m.BASE=base;m.r3.OBS=obs;m.r3.OBS_HASH='synthetic';m.EXPECTED={str(p.relative_to(base)):'synthetic' for p in base.rglob('*') if p.is_file()};expected={m.PLAN:'ad122b5d98d442391d866d3fe0e8b94f8e591d43eba5f2182c9369760d799854',Path(m.r3.__file__):'0359461d5c4436698d4eee4d417a564ca273edf0091bfa87153f1ef0eb834517',Path('LIHAN/scripts/static_rgb_light_fit_r5b.py'):'ea6a90084d61d88ee87dbcd5230793de65f56e8b1f562b2125a68d90dd6d12e2',obs:'synthetic',**{base/k:v for k,v in m.EXPECTED.items()}}
m.sha=lambda p:expected.get(Path(p),'synthetic')
for case in ['invalid_full','invalid_aux','valid_unconverged']:
 def fake_fit(n,y,w,init,a0,out,label,**kwargs):
  dest=out/'runs'/label;dest.mkdir(parents=True)
  if (case=='invalid_full' and label=='full') or (case=='invalid_aux' and label=='hi'):
   np.savez_compressed(dest/'checkpoint_latest.npz',phi=t,a=np.zeros(150));raise FloatingPointError('非有限loss(合成注入)')
  ll=init.copy()
  if label=='lo':ll[:,:2]*=-1
  (dest/'history.json').write_text(json.dumps({'final_losses':[1.,.9]}));return ll,{'converged':False,'certificate':{'converged':False}}
 m.fit_traj=fake_fit; out=root/case;sys.argv=['test','--output',str(out)];m.main()
 sel=json.loads((out/'selection.json').read_text());result=json.loads((out/'fit_complete.json').read_text());diag=json.loads((out/'diagnostics.json').read_text())
 assert all((out/f).exists() for f in ['01_support_residual.png','02_trajectories.png','03_coefficients.png','04_uncertainty_convergence.png'])
 if case=='invalid_full':assert sel['selected'] is None and result['status']=='NO_VALID_NOREG_PREDICTION' and 'C01_full' not in np.load(out/'predictions.npz') and (out/'runs/full/failure.json').exists() and len(diag['runs'])==8
 if case=='invalid_aux':assert sel['selected']=='C01/full' and result['status']=='INCOMPLETE_DIAGNOSTICS' and 'hi' in sel['diagnostic_incomplete']
 if case=='valid_unconverged':assert sel['selected']=='C01/full' and diag['runs']['full']['status']=='NOT_CONVERGED' and diag['differences']['lo_vs_full']['large_disagreement']
 print(case,'PASS',flush=True)
(root/'injection_results.json').write_text(json.dumps({'status':'PASS','cases':['invalid_full','invalid_aux','valid_unconverged'],'gt_read':False},indent=2))
