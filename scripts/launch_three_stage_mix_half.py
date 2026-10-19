"""单点lambda=.5的Stage2门禁通过后，从共同B00启动S22。"""
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
    receipt = control/'mix_half_formal_launch.json'
    with (control/'mix_half_formal_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            raise FileExistsError('S22已登记，禁止重复训练。')
        smoke = root/'output/smoke_test/1019-53-ELDL-V3-共享SH半混合权重门禁'
        if read(smoke/'job_state.json').get('exit_code') != 0:
            raise ValueError('20步smoke未正常结束。')
        marker = read(smoke/'model_mlp/three_stage_training.json')
        engineering = read(smoke/'engineering_receipt.json')
        independent = read(smoke/'independent_audit/receipt.json')
        initial = read(smoke/'initial_forward_audit.json')
        if (marker['stage'] != 2 or marker['config']['lambda_mix'] != .5 or marker['config']['direction_lr'] != 0 or
                engineering['actual_updates'] != 20 or independent['target_rgb_and_alpha_replacement_max_error'] != 0 or
                independent['normal_opacity']['protected_normal_opacity_max'] != 0 or
                independent['normal_opacity']['rgb_alpha_opacity_gradient_norm'] <= 0 or
                not initial['original_geometry_bitexact'] or initial['parent_optimizer_loaded']):
            raise ValueError('混合权重/独立梯度/B00来源门禁未通过。')
        snapshot = control/'snapshots/three_stage_v6_mix_control'
        for relative, digest in read(snapshot/'snapshot_manifest.json')['files'].items():
            if hashlib.sha256((snapshot/relative).read_bytes()).hexdigest() != digest:
                raise ValueError('冻结v6变化：'+relative)
        output = root/'output/1019-31-LH-static-jumpingjacks-ELDL-V3-S22半混合共享法线SH'
        if output.exists():
            raise FileExistsError('S22输出已存在，禁止覆盖。')
        plan = root/'LIHAN/DOC/训练/1019-27-TRAIN幅度诊断后的单点混合权重受控训练计划.md'
        command = [sys.executable, str(root/'scripts/run_three_stage_job.py'), '--output', str(output), '--gpu', '0',
            '--snapshot', str(snapshot), '--', '--binding', str(control/'asset_binding.json'),
            '--stage', '2', '--direction_lr', '0', '--lambda_mix', '.5']
        p = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        receipt.write_text(json.dumps(dict(status='S22_LAUNCHED', started=time.time(), supervisor_pid=p.pid,
            command=command, gpu=0, budget=8100, lambda_mix=.5, direction_lr=0.,
            plan=str(plan), plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
            gates=dict(engineering=engineering, independent=independent, initial=initial),
            gt_used_for_hyperparameters_or_stopping=False), ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(dict(pid=p.pid, command=command), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
