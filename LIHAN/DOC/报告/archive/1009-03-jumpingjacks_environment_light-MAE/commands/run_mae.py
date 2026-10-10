"""等待最终55k权重落盘后，顺序执行用户要求的完整前景MAE。"""
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent.parent
TRAIN = REPO / 'output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练'


def state(status, step, **extra):
    data = dict(status=status, step=step, pid=os.getpid(),
                updated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(), **extra)
    temporary = ROOT / 'state.json.tmp'
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(ROOT / 'state.json')
    print(json.dumps(data, ensure_ascii=False), flush=True)


def main():
    (ROOT / '.launch_lock').mkdir()
    step = 'waiting_for_final_checkpoint_and_gt'
    try:
        state('WAITING', step, iteration=55000)
        while True:
            train_state = json.loads((TRAIN / 'state.json').read_text())
            if train_state['status'] == 'FAILED':
                raise RuntimeError('来源训练失败，禁止替换为其它权重。')
            checkpoints = [TRAIN / 'model_mlp/point_cloud/iteration_55000/point_cloud.ply',
                           TRAIN / 'model_mlp/deform/iteration_55000/deform.pth']
            ready = (train_state['step'] not in ('stage1', 'stage1_insights', 'stage2') and
                     all(p.is_file() and p.stat().st_size > 0 for p in checkpoints) and
                     (ROOT / 'gt_depth/manifest.json').is_file() and
                     (ROOT / 'gt_roughness/manifest.json').is_file())
            if ready:
                break
            for log in ('prepare_depth_gt', 'export_roughness_gt'):
                path = ROOT / 'logs' / (log + '.log')
                if path.is_file() and 'Traceback (most recent call last)' in path.read_text(errors='replace'):
                    raise RuntimeError('GT准备失败：' + log)
            time.sleep(5)
        for command in json.loads((ROOT / 'planned_commands.json').read_text()):
            step = command['name']
            with Path(command['log']).open('x') as log:
                child = subprocess.Popen(['bash', str(ROOT / 'commands' / (step + '.sh'))],
                                         cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
                state('RUNNING', step, child_pid=child.pid, iteration=55000)
                if child.wait() != 0:
                    raise RuntimeError(step + '退出码=' + str(child.returncode))
        (ROOT / 'EVALUATION_COMPLETED').write_text('全部评测计算完成；独立复算与真实目检待完成。\n')
        state('EVALUATION_COMPLETED_REVIEW_PENDING', 'finished', iteration=55000)
    except BaseException as error:
        (ROOT / 'FAILED').write_text(repr(error) + '\n')
        state('FAILED', step, error=repr(error))
        raise


if __name__ == '__main__':
    main()
