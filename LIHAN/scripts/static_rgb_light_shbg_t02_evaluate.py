"""T02 冻结后审计与一次 GT 评测。仅读冻结结果，不拟合、不重投影。"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/1103-27-LH-static-ELDL-V3-T02-低频轨迹SH'
BASE = ROOT / 'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角'
EVAL = RUN / 'evaluation'
KEYS = ('full', 'A', 'B', 'boot103108', 'boot103109', 'hi', 'lo')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def audit():
    manifest = read(RUN / 'freeze_manifest.json')
    assert manifest['status'] == 'FROZEN_AWAITING_INDEPENDENT_EVALUATION'
    assert manifest['gt_read'] is False
    for category in ('files', 'external_inputs'):
        for name, entry in manifest[category].items():
            p = RUN / name if category == 'files' else Path(name)
            assert p.name != 'lights.json', '冻结清单包含禁止的 GT 输入'
            assert p.is_file() and p.stat().st_size == entry['size'], str(p)
            assert sha(p) == entry['sha256'], str(p)
    actual = {str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()
              and 'evaluation' not in p.relative_to(RUN).parts and p.name != 'freeze_manifest.json'}
    assert actual == set(manifest['files']), '存在未冻结文件或冻结缺件'
    choice = read(RUN / 'selection/candidate_table.json')
    assert choice['status'] == 'FALLBACK_E4' and choice['selected'] is None
    fallback = RUN / 'selection/direction.npy'
    baseline = BASE / 'fitting/runs/a1e+00_e1e+01_full/direction.npy'
    assert sha(fallback) == sha(baseline), '回退字节不一致'
    summaries = []
    for candidate in ('C00L', 'C01', 'C02', 'C03', 'C04'):
        for key in KEYS:
            folder = RUN / 'fit/runs' / candidate / key
            cert = read(folder / 'certificate.json')
            assert cert['converged'] and cert['reason'] is None
            assert max(abs(x) for x in cert['window']) < 1, '异常证书'
            with np.load(folder / 'parameters.npz', allow_pickle=False) as p:
                assert np.max(abs(np.tanh(p['U']))) <= .98
                assert np.max(abs(np.tanh(p['V']))) <= .98
                gamma = float(p['gamma'])
                assert gamma == 0 if candidate == 'C00L' else .01 <= gamma <= .99
            direction = np.load(folder / 'direction.npy', allow_pickle=False)
            assert direction.shape == (150, 3) and np.isfinite(direction).all()
            assert np.max(abs(np.linalg.norm(direction, axis=1)-1)) < 1e-10
            for file in ('checkpoint.pt', 'history.json'):
                assert (folder / file).is_file()
            summaries.append(dict(candidate=candidate, run=key, gamma=gamma, certificate=cert))
    templates = []
    for path in sorted((RUN / 'fit/templates').glob('*.npz')) + sorted((RUN / 'selection').glob('*holdout.npz')):
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['frames'], np.arange(1, 151))
            assert z['q'].dtype == np.float64 and np.isfinite(z['q']).all()
            assert np.max(abs(z['q'])) <= .1 + 1e-14
            assert np.max(z['error']) <= 1e-8 and np.max(z['relative']) <= 1e-6
            assert np.allclose(z['H'], z['q']-z['q'].min(0), atol=1e-14, rtol=0)
            if 'holdout' in path.name:
                assert not z['cal'][1::2].any()
            templates.append(dict(path=str(path.relative_to(RUN)), points=len(z['indices']),
                                  projection_error_max=float(z['error'].max()),
                                  projection_relative_max=float(z['relative'].max())))
    assert len(templates) == 16
    scores = choice['baseline']
    coverage = scores['E4_KKT_count'] / scores['OLS_count']
    for row in choice['candidates']:
        assert not row['all_gates']
        assert {g for g, ok in row['gates'].items() if not ok} == {'G4a', 'G4b'}
        assert row['G4a_mae'] > choice['C00L_OLS']
        assert row['G4b_mae'] > .995 * min(scores['E4_KKT'], choice['C00L_KKT'])
        assert max(v['max'] for v in row['eta'].values()) <= .2
    inspection = read(RUN / 'inspection.json')
    assert inspection['gt_read'] is False and len(inspection['figures']) == 4
    assert (RUN / 'freeze.exit_code').read_text().strip() == '0'
    return dict(status='PASS', checked_at_utc=now(), gt_read=False,
                freeze_manifest_sha256=sha(RUN / 'freeze_manifest.json'), frozen_utc=manifest['frozen_utc'],
                checked_frozen_files=len(manifest['files']), checked_external_inputs=len(manifest['external_inputs']),
                fallback_sha256=sha(fallback), baseline_sha256=sha(baseline), byte_identical=True,
                runs=summaries, templates=templates, coverage=coverage,
                coverage_count=scores['E4_KKT_count'], coverage_denominator=scores['OLS_count'],
                selection=choice, evaluation_code_sha256=sha(__file__))


def summarize(values):
    return dict(mean_deg=float(np.mean(values)), p50_deg=float(np.percentile(values, 50)),
                p90_deg=float(np.percentile(values, 90)))


def evaluate():
    audit_record = audit()
    saved = read(EVAL / 'integrity_audit.json')
    assert saved['freeze_manifest_sha256'] == audit_record['freeze_manifest_sha256']
    assert saved['evaluation_code_sha256'] == sha(__file__)
    visual = read(EVAL / 'independent_inspection.json')
    assert visual['status'] == 'PASS' and visual['gt_read'] is False and len(visual['figures']) == 4
    # 人工已知向量自检，不读取任何数据集 GT。
    vectors = np.array([[1.,0,0],[0,1.,0],[-1.,0,0]])
    np.testing.assert_allclose(np.degrees(np.arccos(vectors[:,0])), [0,90,180], atol=1e-12)
    np.testing.assert_allclose((np.array([350.,-350.])+180)%360-180, [-10,10], atol=1e-12)
    predictions = {'T02_FALLBACK_E4': np.load(RUN / 'selection/direction.npy', allow_pickle=False),
                   'E4': np.load(BASE / 'fitting/runs/a1e+00_e1e+01_full/direction.npy', allow_pickle=False)}
    for k, value in predictions.items():
        assert value.shape == (150,3) and np.isfinite(value).all()
        predictions[k] = value / np.linalg.norm(value, axis=1)[:,None]
    receipt = EVAL / 'gt_read_receipt.json'
    dump(receipt, dict(status='GT_READ_STARTED', started_at_utc=now(),
                       freeze_manifest_sha256=saved['freeze_manifest_sha256'], evaluation_code_sha256=sha(__file__),
                       policy='仅一次读取。异常保留收据与产物，禁止重读；仅最终回退和 E4 同帧评测。'))
    source = ROOT / 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3/lights.json'
    payload = source.read_bytes()  # 本脚本唯一的 GT 文件打开；哈希使用同一内存字节。
    records = json.loads(payload)
    assert set(records) == {f'{i:04d}' for i in range(1,151)}
    gt=[]
    for i in range(1,151):
        item=records[f'{i:04d}']
        assert item['source_type'] == 'SUN'
        if 'direction_convention' in item:
            assert item['direction_convention'] == 'Unit vector from surface toward sun; max(dot(normal, light_dir_world), 0)'
        value=np.asarray(item['light_dir_world'],dtype=np.float64)
        assert value.shape == (3,) and np.isfinite(value).all() and np.linalg.norm(value)>1e-12
        gt.append(value/np.linalg.norm(value))
    gt=np.stack(gt); metrics={}; rows=[]
    for method,pred in predictions.items():
        angle=np.degrees(np.arccos(np.clip(np.sum(pred*gt,axis=1),-1,1)))
        az=(np.degrees(np.arctan2(pred[:,1],pred[:,0])-np.arctan2(gt[:,1],gt[:,0]))+180)%360-180
        el=np.degrees(np.arcsin(np.clip(pred[:,2],-1,1))-np.arcsin(np.clip(gt[:,2],-1,1)))
        metrics[method]=dict(**summarize(angle), within_5deg_frames=int((angle<=5).sum()),
                             azimuth_abs_error=summarize(abs(az)), elevation_abs_error=summarize(abs(el)))
        for j in range(150):
            rows.append([j+1,method,*pred[j],*gt[j],angle[j],az[j],el[j],bool(angle[j]<=5)])
    assert metrics['T02_FALLBACK_E4'] == metrics['E4']
    with (EVAL/'per_frame.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['frame','method','pred_x','pred_y','pred_z','gt_x','gt_y','gt_z',
                                             'angular_error_deg','azimuth_error_deg_pred_minus_gt','elevation_error_deg_pred_minus_gt','within_5deg'])
        writer.writerows(rows)
    result=dict(status='FAILED', reason='无候选通过无 GT 门槛，E4 回退；同次 Mean 等于 E4，未严格改善。',
                evaluated_at_utc=now(),gt_read_count=1,gt_sha256=hashlib.sha256(payload).hexdigest(),
                metrics=metrics,frames=list(range(1,151)),coordinate_system='world',direction_convention='surface_to_light',
                axis_sign_time_search=False,gt_used_for_estimation=False,gt_feedback_parameter_tuning=False,
                freeze_manifest_sha256=saved['freeze_manifest_sha256'],evaluation_code_sha256=sha(__file__))
    dump(EVAL/'metrics.json',result)
    dump(EVAL/'completion.json',dict(status='COMPLETED',conclusion='FAILED',completed_at_utc=now(),gt_read_count=1,
                                    gt_sha256=result['gt_sha256'],metrics_sha256=sha(EVAL/'metrics.json')))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-gt-evaluation',action='store_true')
    args=parser.parse_args()
    if args.allow_gt_evaluation:
        evaluate()
    else:
        record=audit()
        EVAL.mkdir(exist_ok=False)
        dump(EVAL/'integrity_audit.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('runs','templates','selection')},ensure_ascii=False,indent=2))
