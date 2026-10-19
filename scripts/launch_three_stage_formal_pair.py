"""四组最终冒烟门禁通过后，按已登记预算启动Stage2唯一方向学习率对照。"""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    control = root/'output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控'
    receipt = control/'formal_pair_launch.json'
    if receipt.exists():
        raise FileExistsError('已登记正式启动，禁止重复启动。')
    smokes = ['1019-30-ELDL-V3-冻结源码固定SH最终门禁', '1019-31-ELDL-V3-冻结源码微调SH最终门禁',
              '1019-32-ELDL-V3-冻结源码PBR基础最终门禁', '1019-33-ELDL-V3-冻结源码PBR残差最终门禁']
    gates = {}
    for name in smokes:
        folder = root/'output/smoke_test'/name
        job = json.loads((folder/'job_state.json').read_text())
        engineering = json.loads((folder/'engineering_receipt.json').read_text())
        if job.get('exit_code') != 0 or engineering['status'] != 'PASS_RELOAD_AND_LINEAR_COMPONENTS':
            raise ValueError('正式前工程门禁未完成：'+name)
        gates[name] = dict(exit_code=0, engineering=engineering)
    for name in smokes[:2]:
        independent = json.loads((root/'output/smoke_test'/name/'independent_audit/receipt.json').read_text())
        if independent['status'] != 'PASS_SELF_CONTAINED_RELOAD_TARGET_FREE_INFERENCE':
            raise ValueError('尚未通过独立无目标推理／梯度检查。')
        gates[name]['independent'] = independent
    outputs = ['1019-19-LH-static-jumpingjacks-ELDL-V3-S20固定方向共享法线SH',
               '1019-20-LH-static-jumpingjacks-ELDL-V3-S21微调方向共享法线SH']
    if any((root/'output'/name).exists() for name in outputs):
        raise FileExistsError('正式输出已存在，不能覆盖或重复训练。')
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    snapshot = control/'snapshots/three_stage_v1'
    files = json.loads((snapshot/'snapshot_manifest.json').read_text())['files']
    for name, digest in files.items():
        if sha(snapshot/name) != digest:
            raise ValueError('冻结快照发生变化：'+name)
    launches = []
    for gpu, (name, lr) in enumerate(zip(outputs, ['0', '0.000001'])):
        command = [sys.executable, str(root/'scripts/run_three_stage_job.py'), '--output', str(root/'output'/name),
            '--gpu', str(gpu), '--snapshot', str(snapshot), '--', '--binding', str(control/'asset_binding.json'),
            '--stage', '2', '--direction_lr', lr]
        child = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        launches.append(dict(name=name, supervisor_pid=child.pid, gpu=gpu, command=command))
    receipt.write_text(json.dumps(dict(status='FORMAL_PAIR_LAUNCHED', started=time.time(), gates=gates,
        launches=launches, scientific_accuracy_pass=False,
        next_step='结束后先核验TRAIN全图机制与末段趋势，再推进同一S20父模型的PBR消融；GT不可反馈选参'),
        ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(launches, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
