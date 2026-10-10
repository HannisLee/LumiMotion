"""核对逐帧重叠白色SUN，将新版多灯格式转换为现有GT方向表。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();original=a.source.read_bytes();data=json.loads(original);table={}
for key,record in data['frames'].items():
 lights=record['lights'];assert len(lights)==2
 directions=np.asarray([l['light_dir_world'] for l in lights],float)
 assert np.isfinite(directions).all() and np.allclose(directions[0],directions[1],atol=1e-7,rtol=0)
 assert np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-6,rtol=0)
 assert all(l['source_type']=='SUN' and l['type']=='directional' and l['direction_convention']=='unit vector from surface toward light, for max(n dot l, 0)' and np.allclose(l['light_rgb'],[1,1,1]) and l['sun_angular_size_radians']==0 for l in lights)
 energy=sum(l['intensity'] for l in lights);assert energy==6
 table[key]={'type':'directional','light_dir_world':lights[0]['light_dir_world'],'direction_convention':lights[0]['direction_convention'],'source_type':'SUN','intensity':float(energy),'source_light_names':[l['name'] for l in lights],'intensity_unit':'sum of Blender SUN parameters; trainer uses calibrated scalar'}
assert set(table)=={f'{i:04d}' for i in range(1,151)}
if a.output.exists():assert json.loads(a.output.read_text())==table
else:a.output.write_text(json.dumps(table,ensure_ascii=False,indent=2)+'\n')
assert a.source.read_bytes()==original
print(json.dumps({'status':'PASS','frames':len(table),'coincident_suns':2,'source_sha256':hashlib.sha256(original).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2))
