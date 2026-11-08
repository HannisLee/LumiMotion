"""T01(1108-01) R6-ENV 冻结脚本：登记规划/修改记录/训练计划/代码/输入/拟合产物的SHA256，状态 R6_FROZEN_READY；不读取GT。
模板：output/1031-12-…/freeze_r5b.py。冻结后不得重新拟合或改动任何已登记文件。
用法：python -m LIHAN.scripts.freeze_r6_env --run output/1108-NN-… --selftest-json … --modify-doc … --train-plan … [--smoke-json …]
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'LIHAN/DOC/规划/1108-01-T01环境光建模与RGB损失新Stage2规划.md'
E4 = ROOT / 'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角'
OBS = ROOT / 'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz'
MODEL = ROOT / 'output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp'
SCRIPTS = [ROOT / f'LIHAN/scripts/{n}.py' for n in ('static_rgb_light_fit_r6_env', 'static_rgb_light_fit_r3', 'static_rgb_light_fit_r5b', 'freeze_r6_env')]
EVAL = [ROOT / f'LIHAN/scripts/{n}.py' for n in ('static_rgb_light_evaluate_r6_env', 'static_rgb_light_evaluate')]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for v in iter(lambda: f.read(1 << 20), b''): h.update(v)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--selftest-json', type=Path, required=True)
    ap.add_argument('--modify-doc', type=Path, required=True); ap.add_argument('--train-plan', type=Path, required=True); ap.add_argument('--smoke-json', type=Path)
    a = ap.parse_args(); OUT = (ROOT / a.run).resolve() if not a.run.is_absolute() else a.run; rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
    ex = json.loads((OUT / 'exitcode.json').read_text()); assert ex['exitcode'] == 0, ex
    fit = OUT / 'fitting'; cfg = json.loads((fit / 'method_config.json').read_text()); done = json.loads((fit / 'fit_complete.json').read_text()); assert done['gt_read'] is False and cfg['gt_read'] is False
    pred = np.load(fit / 'predictions.npz'); assert np.array_equal(pred['frames'], np.arange(1, 151))
    for k in ['E0', 'E1', 'E2', 'E3', 'R6']: assert pred[k].shape == (150, 3) and np.isfinite(pred[k]).all() and np.allclose(np.linalg.norm(pred[k], axis=1), 1, atol=1e-10), k
    launch = json.loads((fit / 'launch.json').read_text()); code = SCRIPTS[0]; assert launch['code_sha256'] == sha(code) and launch['plan_sha256'] == sha(PLAN)
    assert json.loads((fit / 'e4_regression.json').read_text())['ok'] is True and json.loads((fit / 'r3_reproduction.json').read_text())['max_abs_diff'] and json.loads((fit / 'support_manifest.json').read_text())['equals_frozen'] is True
    st = json.loads(a.selftest_json.read_text()); assert st['status'] == 'PASS' and st['gt_read'] is False
    vi = json.loads((OUT / 'visual_inspection.json').read_text()); assert vi.get('gt_read') is False and len(vi.get('items', [])) == 5
    frozen = OUT / 'frozen_code'; frozen.mkdir(exist_ok=False)
    for p in [*SCRIPTS, PLAN]: shutil.copy2(p, frozen / p.name)
    files = {rel(p): sha(p) for p in sorted(fit.rglob('*')) if p.is_file()}
    for p in frozen.iterdir(): files[rel(p)] = sha(p)
    extra = [*SCRIPTS, *EVAL, PLAN, a.modify_doc, a.train_plan, OBS, OUT / 'visual_inspection.json', OUT / 'run.log', OUT / 'command.txt', OUT / 'exitcode.json', OUT / 'start.txt',
             MODEL / 'point_cloud/iteration_35000/point_cloud.ply', MODEL / 'deform/iteration_35000/deform.pth', a.selftest_json, E4 / 'fitting/predictions.npz', E4 / 'freeze_manifest.json']
    if a.smoke_json: extra.append(a.smoke_json)
    for p in extra:
        p = Path(p).resolve(); assert p.is_file(), p; files[rel(p)] = sha(p)
    assert not any(Path(k).name == 'lights.json' for k in files)
    man = dict(status='R6_FROZEN_READY', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gt_read=False, final_method='R6', coordinate_system='world', direction_convention='surface_to_light',
               plan_sha256=sha(PLAN), code_sha256=sha(code), observations_sha256=sha(OBS), prediction_sha256=sha(fit / 'predictions.npz'), env_sha256=sha(fit / 'env.npz'),
               full_converged=cfg['full_converged'], flags=cfg['flags'], files=files,
               reporting_policy='README/DOC may append independent evaluation findings; fixed artifacts listed here may not be modified in response to GT')
    (OUT / 'freeze_manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=2) + '\n'); fm = sha(OUT / 'freeze_manifest.json')
    (OUT / 'eval_code_manifest.json').write_text(json.dumps(dict(status='EVALUATION_CODE_FROZEN', gt_read=False, frozen_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), freeze_manifest_sha256=fm,
                                                                    files={rel(p): sha(p) for p in EVAL}), ensure_ascii=False, indent=2) + '\n')
    print('R6_FROZEN_READY', len(files), fm)


if __name__ == '__main__':
    main()
