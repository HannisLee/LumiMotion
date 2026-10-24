"""只导出TRAIN世界坐标表面朝向SUN的单位方向；不复制光强、环境、材质或几何GT。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    source=args.source.resolve();out=args.output.resolve();parent=args.parent.resolve()
    assert not out.exists(),'禁止覆盖净化光向表'
    manifest=json.loads((source/'no_gt_manifest.json').read_text())
    transforms=json.loads((source/'transforms_train.json').read_text())
    frames=manifest['train_frames'];assert len(frames)==117
    assert frames==[x['source_frame'] for x in transforms['frames']]
    records=json.loads(args.lights.read_text());directions=[];times=[];errors=[]
    for frame,transform in zip(frames,transforms['frames']):
        item=records[f'{frame:04d}']
        assert item['source_type']=='SUN' and item['primary_light']=='Orbit_Directional_Sun'
        direction=np.array(item['light_dir_world'],dtype=float)
        axis=np.asarray(item['lights'][item['primary_light']]['matrix_world'],dtype=float)[:3,2]
        direction/=np.linalg.norm(direction);axis/=np.linalg.norm(axis)
        assert np.isfinite(direction).all() and np.allclose(direction,axis,atol=1e-7,rtol=0)
        time=float(transform['time']);assert abs(time-(frame-1)/149)<1e-7
        directions.append(direction.tolist());times.append(time);errors.append(float(np.abs(direction-axis).max()))
    result=dict(format_version=1,source_split='TRAIN',gt_direction_only=True,forbidden_gt_fields_included=[],
        evaluation_direction_source='heldout_GT_direction_evaluation_only',parent=str(parent),parent_iteration=35000,
        train_frames=frames,times=times,directions=directions,intensity=math.pi,
        source_sha256={str(source/'no_gt_manifest.json'):sha(source/'no_gt_manifest.json'),
                       str(source/'transforms_train.json'):sha(source/'transforms_train.json')},
        direction_audit=dict(source=str(args.lights.resolve()),sha256=sha(args.lights),
             convention='世界坐标，表面指向SUN，等于Blender SUN局部+Z；光传播沿-Z；不翻转不对齐',
             max_axis_absolute_error=max(errors),direction_count=len(frames),time_formula='(frame-1)/149',
             intensity_initialization='pi常数，与GT光强无关',heldout_directions_exported=False))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),count=len(frames),max_axis_error=max(errors)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','parent','lights','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
