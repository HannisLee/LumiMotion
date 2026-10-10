"""正式B06与B05唯一改动/来源/实际CUDA构建核验，不改变运行中模型。"""
import json,hashlib,sys
import numpy as np
from pathlib import Path
from argparse import Namespace
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent;sys.path.insert(0,str(CWD))
from scene.pbr_rasterizer import validate_backend_model
m=ROOT/'model_mlp'
a=json.loads((CWD/'output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp/stage2_config.json').read_text())
b=json.loads((m/'stage2_config.json').read_text())
changes={k:dict(B05=a.get(k),B06=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
assert set(changes)=={'loss_preset','pbr_rasterizer_backend','source_path','model_path','pbr_gt_lights_path'},changes
assert b['loss_preset']=='pbr_gt_directional_plane_native' and b['pbr_rasterizer_backend']=='lumimotion'
native=validate_backend_model(m,Namespace(**b))
plane=json.loads((m/'pbr_plane_projection.json').read_text())
assert plane['support_points']==15187 and plane['training_cameras']==135
with np.load(m/'pbr_plane_projection.npz') as new,np.load(CWD/'output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp/pbr_plane_projection.npz') as old:
    assert set(new.files)==set(old.files) and all(np.array_equal(new[k],old[k]) for k in new.files)
source=json.loads((m/'stage1_source.json').read_text())
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in source.items())
for rel,h in json.loads((ROOT/'code_manifest.json').read_text()).items():
    assert hashlib.sha256((CWD/rel).read_bytes()).hexdigest()==h
record=dict(status='PASS_LAUNCH_ONLY',changes=changes,source_derived_plane_arrays_exactly_B05=True,
    supported_points=15187,native_source_sha256=native['source_sha256'],native_extension_sha256=native['extension_sha256'],
    stage1_source_unchanged=source,code_manifest_matches=True,final_mae=False,
    initial_inline_runner_failure='首次临时inline审计误用默认python vllm，import scene缺smplx，未改训练；改正确venus lumimotion执行此归档脚本')
(ROOT/'inspection/launch_configuration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
