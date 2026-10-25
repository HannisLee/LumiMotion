"""等待六个固定预算终点，执行只读审计、固定相机渲染与Exporter；人工验收单列。"""
import json
import os
import shlex
import subprocess
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT/'output/1025-02-Stage3-PPT配对总控'
NAMES = ['1025-03-staticDL-Stage3原生基线', '1025-04-staticDL-Stage3改进几何', '1025-05-dynamicDL-Stage3原生基线',
         '1025-06-dynamicDL-Stage3改进几何', '1025-07-dynamicDL-Stage3估计方向光', '1025-08-staticDL-Stage3已知方向光']
WRITE_LOCK = threading.Lock()


def write(path, value):
    with WRITE_LOCK:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(path)


def main():
    state = dict(status='WAITING_SIX_FIXED_ENDPOINTS', pid=os.getpid(), started=time.time(), commands=[])
    path = CONTROL/'finish_queue.json'
    environment = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       TORCH_EXTENSIONS_DIR=str(CONTROL/'native_cache'), PYTHONUNBUFFERED='1',
                       PYTHONPATH=str(ROOT/'output/archive/1011-07-六组训练总控/tools/nvdiffrast-sm86/extension'))
    write(path, state)
    while True:
        current = [json.loads((ROOT/'output'/name/'state.json').read_text()) for name in NAMES]
        if any(item['status'] == 'FAILED' for item in current):
            state.update(status='FAILED_UPSTREAM', states=current)
            write(path, state)
            return 1
        if all(item['status'] == 'READY_FOR_REVIEW' for item in current):
            break
        state['updated'] = time.time()
        state['upstream_status'] = {name: item['status'] for name, item in zip(NAMES, current)}
        write(path, state)
        time.sleep(30)

    def execute(experiment, label, command, env=None):
        record = dict(experiment=str(experiment), label=label, command=command, log=str(experiment/(label+'.log')))
        state['commands'].append(record)
        write(path, state)
        with (experiment/'README.md').open('a') as f:
            f.write('\n完整后处理命令（'+label+'）：\n```bash\n'+shlex.join(command)+'\n```\n日志：'+label+'.log\n')
        with (experiment/(label+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env or environment, stdout=log, stderr=subprocess.STDOUT).returncode
        record['exit_code'] = result
        write(path, state)
        if result:
            raise RuntimeError(label+' 失败：'+str(experiment))

    try:
        state['status'] = 'INDEPENDENT_AUDITS'
        write(path, state)
        for name in NAMES:
            experiment = ROOT/'output'/name
            for module, label in [('stage3_ppt_audit.py', 'independent_audit'), ('stage3_ppt_validate.py', 'independent_metric_validation')]:
                execute(experiment, label, [sys.executable, str(ROOT/'LIHAN/scripts'/module), '--experiment', str(experiment)])
        state['status'] = 'RENDER_FIXED_CAMERA_150'
        write(path, state)
        snapshot = CONTROL/'snapshot_v2'

        def render_fixed(spec):
            name, gpu = spec
            experiment = ROOT/'output'/name
            item = json.loads((experiment/'state.json').read_text())
            command = [sys.executable, str(snapshot/'LIHAN/scripts/bootstrap_1011.py'), str(snapshot),
                       'LIHAN.scripts.stage3_ppt_fixed_view', '--model', str(experiment/'model_mlp'),
                       '--iteration', str(item['iteration']), '--output', str(experiment/'fixed_view_0149'), '--frames', '150']
            execute(experiment, 'fixed_view_0149', command, dict(environment, CUDA_VISIBLE_DEVICES=str(gpu)))

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(render_fixed, [('1025-05-dynamicDL-Stage3原生基线', 1), ('1025-07-dynamicDL-Stage3估计方向光', 3)]))
        state['status'] = 'EXPORTING'
        write(path, state)
        command = [sys.executable, str(ROOT/'LIHAN/scripts/stage3_ppt_export.py')]
        state['commands'].append(dict(label='export', command=command, log=str(CONTROL/'export.log')))
        write(path, state)
        with (CONTROL/'export.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
        if result:
            raise RuntimeError('Exporter导出失败')
        state.update(status='READY_FOR_MANUAL_REVIEW', output=str(ROOT/'Exporter/1025-01-Stage3静动态实验展示'), finished=time.time())
    except Exception as error:
        state.update(status='FAILED_POSTPROCESS', error=str(error), finished=time.time())
    write(path, state)
    return 0 if state['status'] == 'READY_FOR_MANUAL_REVIEW' else 1


if __name__ == '__main__':
    sys.exit(main())
