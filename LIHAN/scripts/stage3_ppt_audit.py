"""独立只读核验原生 Stage3 几何冻结、MLP冻结和真实材质更新。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    path = args.experiment
    state = json.loads((path/'state.json').read_text())
    old = PlyData.read(Path(state['parent'])/'point_cloud'/f'iteration_{state["source_iteration"]}'/'point_cloud.ply')['vertex'].data
    new = PlyData.read(path/'model_mlp/point_cloud'/f'iteration_{state["iteration"]}'/'point_cloud.ply')['vertex'].data
    assert len(old) == len(new)
    frozen_names = [k for k in old.dtype.names if k in ['x', 'y', 'z', 'nx', 'ny', 'nz'] or k.startswith(('scale_', 'rot_', 'f_feature_', 'fea_'))]
    changed = {k: int(np.count_nonzero(old[k] != new[k])) for k in old.dtype.names}
    assert all(changed[k] == 0 for k in frozen_names)
    source_deform = Path(state['parent'])/'deform'/f'iteration_{state["source_iteration"]}'/'deform.pth'
    target_deform = path/'model_mlp/deform'/f'iteration_{state["iteration"]}'/'deform.pth'
    a = torch.load(source_deform, map_location='cpu', weights_only=False)
    b = torch.load(target_deform, map_location='cpu', weights_only=False)
    assert a.keys() == b.keys()
    mlp_equal = {k: torch.equal(a[k], b[k]) for k in a}
    assert all(mlp_equal.values())
    assert any(count > 0 for k, count in changed.items() if k not in frozen_names)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in state['parent_sha256'].items())
    metrics = json.loads((path/'evaluation/metrics.json').read_text())
    assert metrics['checkpoint_unchanged']
    result = dict(status='PASS', scope='只验证工程冻结和参数变化，不代表画质或科学收益通过',
                  source_iteration=state['source_iteration'], iteration=state['iteration'],
                  gaussian_count=len(old), frozen_fields=frozen_names, changed_count_by_field=changed,
                  all_deformation_tensors_equal=mlp_equal, parent_sha_unchanged=True,
                  optimizer_updates=state['optimizer_updates'], split=metrics['split'], actual_frames=[r['frame'] for r in metrics['frames']])
    (path/'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', gaussian_count=len(old), mlp_tensors=len(a), frozen_fields=frozen_names), ensure_ascii=False))


if __name__ == '__main__':
    main()
