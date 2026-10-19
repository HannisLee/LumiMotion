"""TRAIN终点核验后，从同一S20@8100启动原生PBR256残差消融。"""
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
    return json.loads(Path(path).read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    if socket.gethostname() != 'ushas':
        raise ValueError('本轮只登记ushas。')
    control = root/'output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控'
    receipt = control/'pbr_native_pair_launch.json'
    with (control/'pbr_native_pair_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            raise FileExistsError('正式PBR已登记启动；必须核验原PID，禁止重复启动。')
        parent = root/'output/1019-19-LH-static-jumpingjacks-ELDL-V3-S20固定方向共享法线SH/model_mlp'
        gates = {}
        for case, name in [('S20', '1019-19-LH-static-jumpingjacks-ELDL-V3-S20固定方向共享法线SH'),
                           ('S21', '1019-20-LH-static-jumpingjacks-ELDL-V3-S21微调方向共享法线SH')]:
            folder = root/'output'/name
            if read(folder/'job_state.json').get('exit_code') != 0:
                raise ValueError('Stage2未真实结束：'+case)
            train = read(folder/'engineering_receipt.json')
            diagnostic = read(control/f'train_endpoint_v2/{case}/results/receipt.json')
            if train['actual_updates'] != 8100 or diagnostic['iterations'] != [0, 4050, 8100]:
                raise ValueError('Stage2预算或全TRAIN节点不完整。')
            if diagnostic['status'] != 'PASS_TRAIN_DIAGNOSTIC_EXECUTION' or diagnostic['gt_read']:
                raise ValueError('完整TRAIN门禁不通过。')
            gates[case] = dict(engineering=train, diagnostic=diagnostic)
        for name in ['1019-42-ELDL-V3-原生混合深度PBR256基础门禁', '1019-43-ELDL-V3-原生混合深度PBR256残差门禁']:
            folder = root/'output/smoke_test'/name
            eng = read(folder/'engineering_receipt.json')
            meta = read(folder/'model_mlp/three_stage_training.json')
            independent = read(folder/'independent_audit/receipt.json')
            if read(folder/'job_state.json').get('exit_code') != 0 or eng['status'] != 'PASS_RELOAD_AND_LINEAR_COMPONENTS' or meta['config']['pbr_samples'] != 256 or meta['config'].get('pbr_surface_depth_mode') != 'native_half':
                raise ValueError('PBR256工程门禁不完整。')
            if independent['status'] != 'PASS_SELF_CONTAINED_RELOAD_TARGET_FREE_INFERENCE' or independent.get('pbr_surface_depth_mode') != 'native_half' or independent.get('native_half_depth_max_abs', 1.) >= 2e-5 or independent['target_rgb_and_alpha_replacement_max_error'] != 0:
                raise ValueError('原生混合深度或独立推理门禁未通过。')
            if meta['parent_iteration'] != 8100 or Path(meta['parent']) != parent:
                raise ValueError('工程门禁不是同一正式S20@8100父资产。')
            gates[name] = dict(engineering=eng, independent=independent)
        snapshot = control/'snapshots/three_stage_v3'
        for relative, expected in read(snapshot/'snapshot_manifest.json')['files'].items():
            if hashlib.sha256((snapshot/relative).read_bytes()).hexdigest() != expected:
                raise ValueError('冻结v2源码变化：'+relative)
        cases = [(2, '1019-23-LH-static-jumpingjacks-ELDL-V3-A32原生混合深度PBR256基础', '0'),
                 (3, '1019-24-LH-static-jumpingjacks-ELDL-V3-A33原生混合深度PBR256小残差', '0.1')]
        if any((root/'output'/name).exists() for _, name, _ in cases):
            raise FileExistsError('正式输出已存在，禁止覆盖。')
        launches = []
        for gpu, name, weight in cases:
            command = [sys.executable, str(root/'scripts/run_three_stage_job.py'), '--output', str(root/'output'/name),
                '--gpu', str(gpu), '--snapshot', str(snapshot), '--entry', 'scripts.train_three_stage_native_pbr', '--', '--binding', str(control/'asset_binding.json'),
                '--stage', '3', '--parent', str(parent), '--parent_iteration', '8100', '--pbr_samples', '256',
                '--lambda_residual', weight]
            child = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            launches.append(dict(name=name, gpu=gpu, supervisor_pid=child.pid, command=command))
            receipt.write_text(json.dumps(dict(status='PBR_PAIR_LAUNCHED', started=time.time(), parent=str(parent),
                parent_iteration=8100, launches=launches, gates=gates, gt_metrics_used_for_choice=False,
                budget=4050, pbr_samples=256, pbr_surface_depth_mode='native_half', scientific_accuracy_pass=False), ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(launches, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
