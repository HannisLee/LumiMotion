"""兼容、重载和实际射线起点证明通过后启动同父法线起点对照。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def main():
    if socket.gethostname() != 'ushas':
        raise ValueError('本轮只登记ushas/lumimotion-ushas。')
    root = Path(__file__).resolve().parents[1]
    control = root/'output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控'
    receipt = control/'normal_origin_pair_launch.json'
    with (control/'normal_origin_pair_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            raise FileExistsError('正式启动已登记，禁止重复。')
        parity = read(root/'output/smoke_test/1019-51-ELDL-V3-原生零法线偏移浮点兼容复核/receipt.json')
        if parity['status'] != 'PASS_ZERO_OFFSET_EXACT_FLOAT_PARITY' or any(v for row in parity['frames'].values() for v in row.values()):
            raise ValueError('旧零偏移完整浮点兼容未通过。')
        parent = root/'output/1019-19-LH-static-jumpingjacks-ELDL-V3-S20固定方向共享法线SH/model_mlp'
        parent_geometry = read(parent/'three_stage/iteration_8100/receipt.json')['geometry_sha256']
        gates = dict(parity=parity)
        for name in ['1019-48-ELDL-V3-PBR法线起点基础门禁', '1019-49-ELDL-V3-PBR法线起点残差门禁']:
            folder = root/'output/smoke_test'/name
            if read(folder/'job_state.json').get('exit_code') != 0:
                raise ValueError('冒烟没有正常结束。')
            engineering = read(folder/'engineering_receipt.json')
            audit = read(folder/'independent_audit/receipt.json')
            marker = read(folder/'model_mlp/three_stage_training.json')
            origin = audit['actual_origin_check']
            if (engineering['actual_updates'] != 20 or not engineering['geometry_frozen'] or
                    audit['target_rgb_and_alpha_replacement_max_error'] != 0 or
                    audit['native_half_depth_max_abs'] >= 2e-5 or origin['points'] <= 0 or
                    origin['maximum_error'] >= 2e-5 or origin['offset'] != marker['config']['shadow_bias'] or
                    audit['pbr_origin_mode'] != 'normal_existing_tmin'):
                raise ValueError('实际射线起点/无目标/混合深度证明未通过。')
            if marker['parent'] != str(parent) or marker['parent_iteration'] != 8100:
                raise ValueError('不是同一正式Stage2父模型。')
            for step in [0, 4, 8, 20]:
                if read(folder/f'model_mlp/three_stage/iteration_{step}/receipt.json')['geometry_sha256'] != parent_geometry:
                    raise ValueError('冒烟几何没有冻结。')
            gates[name] = dict(engineering=engineering, independent=audit)
        snapshot = control/'snapshots/three_stage_v5_normal_origin'
        for relative, digest in read(snapshot/'snapshot_manifest.json')['files'].items():
            if hashlib.sha256((snapshot/relative).read_bytes()).hexdigest() != digest:
                raise ValueError('冻结源码变化：'+relative)
        cases = [(2, '1019-28-LH-static-jumpingjacks-ELDL-V3-A34法线起点PBR基础', '0'),
                 (3, '1019-29-LH-static-jumpingjacks-ELDL-V3-A35法线起点PBR小残差', '0.1')]
        if any((root/'output'/name).exists() for _, name, _ in cases):
            raise FileExistsError('正式输出已存在，禁止覆盖。')
        launches = []
        for gpu, name, weight in cases:
            command = [sys.executable, str(root/'scripts/run_three_stage_job.py'), '--output', str(root/'output'/name),
                '--gpu', str(gpu), '--snapshot', str(snapshot), '--entry', 'scripts.train_three_stage_normal_origin', '--',
                '--binding', str(control/'asset_binding.json'), '--stage', '3', '--parent', str(parent),
                '--parent_iteration', '8100', '--pbr_samples', '256', '--lambda_residual', weight]
            p = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            launches.append(dict(name=name, supervisor_pid=p.pid, gpu=gpu, command=command))
            receipt.write_text(json.dumps(dict(status='NORMAL_ORIGIN_PAIR_LAUNCHED', started=time.time(),
                parent=str(parent), parent_iteration=8100, budget=4050, launches=launches, gates=gates,
                train_only_mechanism_basis=True, gt_used_for_parameters_or_stopping=False,
                scientific_accuracy_pass=False), ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(launches, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
