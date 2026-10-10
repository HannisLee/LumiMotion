"""真实启动配置与来源审计；只读，无GT监督/优化。"""
from pathlib import Path
import json, hashlib, os
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CWD = ROOT.parent.parent
OLD = ROOT.parent/'1010-12-jumpingjacks_dir-PBR仓库原生光栅修复'
model = ROOT/'model_mlp'
config = json.loads((model/'stage2_config.json').read_text())
old_config = json.loads((OLD/'model_mlp/stage2_config.json').read_text())
changed = {k: [old_config.get(k), v] for k, v in config.items() if old_config.get(k) != v}
assert set(changed) == {'source_path', 'model_path', 'loss_preset', 'pbr_gt_lights_path',
                        'pbr_directional_actual_ray_budget', 'wo_indirect', 'wo_indirect_relight'}
assert config['loss_preset'] == 'pbr_gt_directional_plane_native_direct'
assert config['wo_indirect'] and config['wo_indirect_relight'] and config['pbr_directional_actual_ray_budget']
assert config['load_iter'] == 35000 and config['iterations'] == 55000
assert config['pbr_rasterizer_backend'] == 'lumimotion'
manifest = json.loads((ROOT/'code_manifest.json').read_text())
assert len(manifest) == 471
for relative, sha in manifest.items():
    for path in [CWD/relative, ROOT/'commands/code_snapshot'/relative,
                 ROOT/'commands/code_snapshot_train_launch'/relative]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
with np.load(model/'pbr_plane_projection.npz', allow_pickle=False) as current:
    with np.load(OLD/'model_mlp/pbr_plane_projection.npz', allow_pickle=False) as previous:
        assert set(current.files) == set(previous.files)
        assert all(np.array_equal(current[k], previous[k]) for k in current.files)
        support = int(current['support'].sum())
assert support == 15187
assert (model/'pbr_gt_lights.json').read_bytes() == (OLD/'model_mlp/pbr_gt_lights.json').read_bytes()
lights = json.loads((model/'pbr_gt_lights.json').read_text())
# 表结构交由真实训练输入loader验证；此处核对完整快照逐byte一致。
provenance = json.loads((model/'rgb_alpha_provenance.json').read_text())
assert provenance['status'] == 'PASS' and provenance['frames'] == 150
source = json.loads((model/'stage1_source.json').read_text())['files']
for path, sha in source.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
native = json.loads((model/'pbr_rasterizer.json').read_text())
assert native['source_sha256'] == '82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0'
runtime = json.loads((ROOT/'training_runtime.json').read_text())
proc = Path('/proc')/str(runtime['training_pid'])
assert (proc/'stat').read_text().split()[21] == runtime['process_start_ticks']
assert os.readlink(proc/'cwd') == str(CWD)
args = (proc/'cmdline').read_bytes().split(b'\0')
assert b'scripts.train_stage2' in args and str(ROOT/'model').encode() in args
result = dict(status='PASS_LIVE_LAUNCH_AUDIT', training_pid=runtime['training_pid'],
              source_iteration=35000, target_iteration=55000, code_files=len(manifest),
              stage1_source_files=source, plane_support_points=support,
              plane_arrays_bit_identical_B06=True, light_table_bit_identical_B06=True,
              configuration_differences=changed, provenance_frames=150,
              native_source_sha=native['source_sha256'], optimizer_steps=0,
              gt_material_geometry_normal_photometry_read=False, final_mae=False)
out = ROOT/'inspection/launch_audit.json'
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
