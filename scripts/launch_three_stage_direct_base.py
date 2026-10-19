"""三臂20步、独立审计、实际目检通过后启动同父4050步正式对照。"""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if socket.gethostname() != 'ushas':
        raise ValueError('本轮仅登记ushas/lumimotion-ushas。')
    root = Path(__file__).resolve().parents[1]
    control = root/'output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控'
    receipt = control/'direct_base_formal_launch.json'
    with (control/'direct_base_formal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            raise FileExistsError('正式三臂已登记，禁止重复。')
        state = read(control/'direct_base_smoke_state.json')
        if state['status'] != 'PASS_ENGINEERING_PENDING_ACTUAL_MANUAL_REVIEW':
            raise ValueError('三臂20步或独立审计未通过。')
        review = read(control/'direct_base_smoke_manual_review.json')
        if review['status'] != 'REVIEWED_ENGINEERING_ACCEPTED':
            raise ValueError('尚未实际查看20步四类与物理分量。')
        for row in review['files']:
            if sha(Path(row['path'])) != row['sha256']:
                raise ValueError('实际目检文件已变化。')
        parent = root/'output/1019-31-LH-static-jumpingjacks-ELDL-V3-S22半混合共享法线SH/model_mlp'
        selection = read(control/'appearance_parent_train_only_v1.json')
        if (selection['iteration'] != 8100 or Path(selection['model']) != parent or
                selection['gt_metrics_or_images_used_for_choice'] or
                sha(parent/'three_stage/iteration_8100/state.pt') != selection['checkpoint_sha256']):
            raise ValueError('固定父或TRAIN选择依据不匹配。')
        geometry = read(parent/'three_stage/iteration_8100/receipt.json')['geometry_sha256']
        snapshot = control/'snapshots/three_stage_v8_direct_base_runtime'
        manifest = read(snapshot/'snapshot_manifest.json')
        for path, digest in {**manifest['files'], **manifest['runtime_asset_sha256']}.items():
            if sha(snapshot/path) != digest:
                raise ValueError('冻结源码或BRDF资产已变化：'+path)
        specs = [
            ('A36', '33', '半混合父原生PBR控制', 'scripts.train_three_stage_normal_origin', 0., 1),
            ('A37', '34', '主要DL进入PBR基础', 'scripts.train_three_stage_direct_base', 0., 2),
            ('A38', '35', '主要DL基础小残差', 'scripts.train_three_stage_direct_base', .1, 3),
        ]
        outputs = []
        for name, number, label, entry, residual, gpu in specs:
            smoke = Path(state['folders'][name])
            marker = read(smoke/'model_mlp/three_stage_training.json')
            audit = read(smoke/'independent_audit/receipt.json')
            if (read(smoke/'job_state.json').get('exit_code') != 0 or
                    marker['config']['lambda_mix'] != .5 or marker['config']['lambda_residual'] != residual or
                    Path(marker['parent']) != parent or marker['parent_iteration'] != 8100 or
                    audit['target_rgb_and_alpha_replacement_max_error'] != 0 or
                    audit['native_half_depth_max_abs'] != 0 or audit['actual_origin_check']['maximum_error'] >= 2e-5):
                raise ValueError('三臂独立门禁未通过：'+name)
            for step in [0, 4, 8, 20]:
                if read(smoke/f'model_mlp/three_stage/iteration_{step}/receipt.json')['geometry_sha256'] != geometry:
                    raise ValueError('冒烟几何不等于共同父：'+name)
            if name != 'A36':
                if audit['pbr_base_mode'] != 'environment_plus_inherited_direct' or audit['direct_base_check'] is None:
                    raise ValueError('新基础独立分量证明缺失。')
            output = root/f'output/1019-{number}-LH-static-jumpingjacks-ELDL-V3-{name}{label}'
            if output.exists():
                raise FileExistsError('禁止覆盖：'+str(output))
            outputs.append((name, output, entry, residual, gpu))
        # 先写启动预留，再逐个记录实际PID；异常时不得自动重启已启动部分。
        plan = root/'LIHAN/DOC/训练/1019-28-TRAIN机制驱动的主要DL基础转换与同父残差计划.md'
        record = dict(status='LAUNCH_RESERVED', started=time.time(), parent=str(parent), parent_iteration=8100,
            budget=4050, plan=str(plan), plan_sha256=sha(plan), launches=[],
            supervisor_source_sha256=sha(root/'scripts/run_three_stage_job.py'),
            launcher_source_sha256=sha(Path(__file__)), gt_used_for_choice_or_stopping=False)
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        for name, output, entry, residual, gpu in outputs:
            command = [sys.executable, str(root/'scripts/run_three_stage_job.py'), '--output', str(output),
                '--gpu', str(gpu), '--snapshot', str(snapshot), '--entry', entry, '--',
                '--binding', str(control/'asset_binding.json'), '--stage', '3', '--parent', str(parent),
                '--parent_iteration', '8100', '--pbr_samples', '256', '--lambda_residual', str(residual)]
            child = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record['launches'].append(dict(name=name, output=str(output), gpu=gpu,
                supervisor_pid=child.pid, command=command, lambda_residual=residual, entry=entry))
            receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        record['status'] = 'THREE_ARMS_LAUNCHED'
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
