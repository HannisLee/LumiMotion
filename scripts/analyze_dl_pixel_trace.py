"""解析两像素CUDA轨迹并核对深度重建；不修改训练或实验结果。"""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');args=p.parse_args();root=Path(args.root)
runs={};active=None
for line in (root/'run.log').read_text().splitlines():
 if line.startswith('TRACE_BEGIN '):
  active=line.split()[1];runs[active]={}
 elif line.startswith('TRACE_END '):active=None
 elif active and line.startswith('DLTRACE '):
  v=line.split();x,y,order,gid=map(int,v[1:5]);rho3,rho2,depth,alpha,T,testT=map(float,v[5:])
  row=dict(order=order,id=gid,rho3=rho3,rho2=rho2,depth=depth,alpha=alpha,T=T,testT=testT,branch='splat' if rho3<=rho2 else 'filter',accepted=alpha>=1/255 and testT>=.0001)
  runs[active].setdefault(f'{y},{x}',[]).append(row)
result=json.loads((root/'diagnostic/result.json').read_text());summaries=[]
for run in result['rows']:
 for pixel in run['pixels']:
  key=f"{pixel['y']},{pixel['x']}";rows=runs[run['label']][key];accepted=[v for v in rows if v['accepted']]
  accum=sum(v['alpha']*v['T'] for v in accepted);D=sum(v['depth']*v['alpha']*v['T'] for v in accepted)
  error=abs(D/accum-pixel['depth']);assert error<1e-5,(run['label'],key,error)
  summaries.append(dict(label=run['label'],pixel=key,candidates=len(rows),accepted=len(accepted),depth_reconstructed=D/accum,depth_error=error,alpha_error=abs(accum-pixel['alpha'])))
comparisons=[]
for key in runs['base']:
 for positive,negative in [('plus','minus'),('small_plus','small_minus')]:
  a={v['id']:v for v in runs[positive][key]};b={v['id']:v for v in runs[negative][key]}
  changes=[]
  for gid in a.keys()&b.keys():
   x,y=a[gid],b[gid]
   if x['branch']!=y['branch'] or x['accepted']!=y['accepted'] or abs(x['depth']-y['depth'])>.001:
    changes.append(dict(id=gid,plus=x,minus=y))
  aa=[v['id'] for v in runs[positive][key] if v['accepted']];bb=[v['id'] for v in runs[negative][key] if v['accepted']]
  comparisons.append(dict(pixel=key,pair=[positive,negative],only_plus=list(a.keys()-b.keys()),only_minus=list(b.keys()-a.keys()),accepted_order_equal=aa==bb,changes=changes))
(root/'parsed_trace.json').write_text(json.dumps(dict(runs=runs,summaries=summaries,comparisons=comparisons),indent=2))
print(json.dumps(dict(summaries=summaries,comparisons=comparisons),indent=2))
