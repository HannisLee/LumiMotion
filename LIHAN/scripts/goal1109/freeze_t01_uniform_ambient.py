"""1109-01 T01 冻结脚本：登记规划/实现记录/训练计划/代码/输入/两臂拟合产物的 SHA256、大小与 UTC 时间，状态 UA_FROZEN_READY；不读取 GT。
模板：LIHAN/scripts/freeze_r6_env.py。冻结后不得重新拟合或改动任何已登记文件。

  python -m LIHAN.scripts.goal1109.freeze_t01_uniform_ambient --run output/1109-NN-... --selftest-json <numeric_tests.json> \
      --modify-doc LIHAN/DOC/修改/1109-01-....md --train-plan LIHAN/DOC/训练/1109-01-....md [--smoke-json <smoke_result.json>]

要求 <run> 下已有：fitting/{A0,A1}、regress/{E4,C01}、visual_inspection.json、exit_code_A0.txt/exit_code_A1.txt、run_A0.log/run_A1.log、launch_*.sh。
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from LIHAN.scripts.goal1109 import ua_core as C

SCRIPTS = [C.SCRIPT_DIR / f for f in ('fit_t01_uniform_ambient.py', 'ua_core.py', 'ua_diag.py', 'ua_figs.py', 'ua_selftest.py', 'freeze_t01_uniform_ambient.py')] + [
    Path(f'LIHAN/scripts/{n}.py') for n in ('static_rgb_light_fit_r3', 'static_rgb_light_fit_r5b', 'static_rgb_light_fit_r6_env', 'static_rgb_light_fit_goal1108_02_t01')]
EVAL = [C.SCRIPT_DIR / 'evaluate_t01_uniform_ambient.py', Path('LIHAN/scripts/static_rgb_light_evaluate.py')]
C01_PRED = C.C01_RUN_DIR / 'fitting/predictions.npz'
R6_PRED = C.R6_RUN_DIR / 'fitting/predictions.npz'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for v in iter(lambda: f.read(1 << 20), b''):
            h.update(v)
    return h.hexdigest()


def mean_ang(a, b):
    return float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())


def compute_tau(pred):
    """规划 8.4：tau = 1/2 * (有效 bootstrap 流程相对 A0 full 的平均夹角的均值)；两者都无效取 A/B 平均夹角的一半。"""
    full = pred['A0_full']
    vals = {k: mean_ang(pred[f'A0_{k}'], full) for k in ('boot103108', 'boot103109') if f'A0_{k}' in pred.files}
    if vals:
        return 0.5 * float(np.mean(list(vals.values()))), dict(source='bootstrap', per_flow=vals)
    if 'A0_A' in pred.files and 'A0_B' in pred.files:
        v = mean_ang(pred['A0_A'], pred['A0_B'])
        return 0.5 * v, dict(source='A_vs_B', value=v)
    raise RuntimeError('无法确定 tau：bootstrap 与 A/B 均无效')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selftest-json', type=Path, required=True)
    ap.add_argument('--modify-doc', type=Path, required=True)
    ap.add_argument('--train-plan', type=Path, required=True)
    ap.add_argument('--smoke-json', type=Path)
    a = ap.parse_args()
    C.check_root()
    run = a.run
    assert not (run / 'freeze_manifest.json').exists(), '已冻结，禁止覆盖'
    assert C.PLAN_SHA == sha(C.PLAN), '规划文件改变'
    for arm in ('A0', 'A1'):
        assert (run / f'exit_code_{arm}.txt').read_text().strip() == '0', f'{arm} 退出码非 0'
        fc = json.loads((run / f'fitting/{arm}/fit_complete.json').read_text())
        assert fc['gt_read'] is False and fc['status'] in ('AWAITING_REVIEW', 'INCOMPLETE_DIAGNOSTICS', 'NO_VALID_PREDICTION'), fc['status']
        pred = np.load(run / f'fitting/{arm}/predictions.npz', allow_pickle=False)
        assert np.array_equal(pred['frames'], np.arange(1, 151))
        for k in pred.files:
            if k.startswith(arm + '_') or k == 'reference_E4':
                assert pred[k].shape == (150, 3) and np.isfinite(pred[k]).all() and np.allclose(np.linalg.norm(pred[k], axis=1), 1, atol=1e-10), k
    for arm in ('E4', 'C01'):
        assert json.loads((run / f'regress/{arm}/regress.json').read_text())['ok'] is True, arm
    st = json.loads(a.selftest_json.read_text())
    assert st['status'] == 'PASS' and st['gt_read'] is False
    vi = json.loads((run / 'visual_inspection.json').read_text())
    assert vi.get('gt_read') is False and len(vi.get('items', [])) >= 4
    p0 = np.load(run / 'fitting/A0/predictions.npz', allow_pickle=False)
    tau, tau_detail = compute_tau(p0) if 'A0_full' in p0.files else (None, dict(source='A0 无有效 full'))
    frozen = run / 'frozen_code'
    frozen.mkdir(exist_ok=False)
    for p in [*SCRIPTS, *EVAL, C.PLAN]:
        shutil.copy2(p, frozen / p.name)
    files = {}
    for sub in ('fitting', 'regress'):
        for p in sorted((run / sub).rglob('*')):
            if p.is_file():
                files[str(p)] = sha(p)
    for p in frozen.iterdir():
        files[str(p)] = sha(p)
    extra = [*SCRIPTS, *EVAL, C.PLAN, C.GOAL_DOC, a.modify_doc, a.train_plan, run / 'visual_inspection.json', a.selftest_json, C.BASE / 'predictions.npz', C01_PRED, R6_PRED,
             C.MODEL / 'point_cloud/iteration_35000/point_cloud.ply', C.MODEL / 'deform/iteration_35000/deform.pth']
    from LIHAN.scripts import static_rgb_light_fit_r3 as r3
    extra.append(r3.OBS)
    for arm in ('A0', 'A1'):
        extra += [run / f'launch_{arm}.sh', run / f'run_{arm}.log', run / f'exit_code_{arm}.txt', run / f'process_{arm}.json']
    if a.smoke_json:
        extra.append(a.smoke_json)
    for p in extra:
        p = Path(p)
        assert p.is_file(), p
        files[str(p)] = sha(p)
    assert not any(Path(k).name == 'lights.json' for k in files)
    info = {k: dict(size=Path(k).stat().st_size) for k in files}
    man = dict(status='UA_FROZEN_READY', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gt_read=False, plan_sha256=C.PLAN_SHA,
               selftest_json=str(a.selftest_json), tau_deg=tau, tau_detail=tau_detail,
               arms={arm: json.loads((run / f'fitting/{arm}/fit_complete.json').read_text())['status'] for arm in ('A0', 'A1')},
               files=files, file_info=info,
               reporting_policy='README/DOC 可追加独立评测发现；本清单所列固定产物不得因 GT 结果而改动；冻结后增补须单独登记')
    (run / 'freeze_manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=2) + '\n')
    fm = sha(run / 'freeze_manifest.json')
    (run / 'eval_code_manifest.json').write_text(json.dumps(dict(status='EVALUATION_CODE_FROZEN', gt_read=False, frozen_utc=man['frozen_utc'], freeze_manifest_sha256=fm,
                                                                   files={str(p): sha(p) for p in EVAL}), ensure_ascii=False, indent=2) + '\n')
    print('UA_FROZEN_READY', len(files), fm, 'tau_deg=', tau)


if __name__ == '__main__':
    main()
