"""第二轮冻结方向的单次独立描述性评测；不执行估计或候选比较。"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import static_rgb_light_evaluate as base

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/1031-04-LH-static-ELDL-V3-第二轮RGB盲估计'
EXPECTED = 'cf1368b4fd322ba1a2bd28d2a3a3523f27d2739b55c33c26ffdcf11ba3f84c73'
PRED = '869755ee3e938183ae9d4fcff061331a70597e9601a039e1b1513a01d15d647c'
STATES = ['STOP_UNSTABLE', 'PROTOCOL_DEVIATION_BUDGET_OVERFLOW']

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def validate():
    base.require(base.sha(RUN/'delivery_manifest.json') == EXPECTED, '交付清单哈希错误')
    manifest = json.loads((RUN/'delivery_manifest.json').read_text())
    base.require(manifest['states'] == STATES and manifest['final_result']=='FAILED' and manifest['gt_read'] is False, '交付状态错误')
    checked = {}
    for group, root in [('files', RUN), ('source_documents', ROOT)]:
        for name, expected in manifest[group].items():
            path = root/name
            base.require(base.sha(path)==expected, f'交付哈希不符：{name}')
            checked[str(path.relative_to(ROOT))] = expected
    frozen = json.loads((RUN/'frozen_manifest.json').read_text())
    for name, expected in frozen['files'].items():
        base.require(base.sha(RUN/name)==expected, f'冻结哈希不符：{name}')
    inputs = json.loads((RUN/'input_manifest.json').read_text())
    for name, expected in inputs['inputs'].items():
        path=Path(name); path=path if path.is_absolute() else ROOT/path
        base.require(path.name!='lights.json', '输入中不应存在GT')
        base.require(base.sha(path)==expected, f'输入哈希不符：{name}')
    base.require(base.sha(RUN/'final_predictions.npz') == PRED, '预测哈希错误')
    s=json.loads((RUN/'selection.json').read_text())
    base.require(s['chosen_K']==8 and s['chosen_initialization']=='relative' and s['state']=='STOP_UNSTABLE', '固定选择变化')
    base.require(s['coordinate_system']=='world' and s['direction_convention']=='surface_to_light' and not s['final_refit'], '方向约定错误')
    best=min(s['scores'],key=lambda k:s['scores'][k]['clipped_mae'])
    upper=s['scores'][best]['clipped_mae']+s['scores'][best]['se']
    chosen=min(int(k) for k,v in s['scores'].items() if v['clipped_mae']<=upper)
    base.require(chosen==8 and abs(upper-s['one_se_upper'])<1e-12,'一标准误差选择不一致')
    deviation=json.loads((RUN/'execution_deviation.json').read_text())
    base.require(deviation['protocol_state']==STATES[1] and deviation['completed_optimizations']==4 and s['optimizations']==10,'预算偏差记录错误')
    with np.load(RUN/'final_predictions.npz',allow_pickle=False) as z:
        base.require(np.array_equal(z['frames'],base.FRAMES),'帧必须恰为1..150')
        direction=np.asarray(z['direction'],dtype=np.float64)
    base.require(direction.shape==(150,3) and np.isfinite(direction).all(),'方向形状或数值错误')
    norms=np.linalg.norm(direction,axis=1)
    base.require(np.max(abs(norms-1))<1e-4,'方向不是单位向量')
    with np.load(RUN/'K8_relative.npz',allow_pickle=False) as z:
        base.require(np.array_equal(direction,z['direction']),'最终方向不是固定K8 relative端点')
    return direction/norms[:,None], dict(validated_at=now(), gt_read=False, manifest_sha256=EXPECTED,
        predictions_sha256=PRED, checked_delivery_file_count=len(checked), checked_files=checked,
        input_file_count=len(inputs['inputs']), max_norm_deviation=float(max(abs(norms-1))),
        chosen_K=chosen, states=STATES, budget='dense10完整+原稀疏4完整及第5未完')

def self_test():
    p=np.array([[1.,0,0]]*3); g=np.array([[1.,0,0],[0,1.,0],[-1.,0,0]])
    actual=base.evaluate({'final':p},g)['final']
    np.testing.assert_allclose(actual,[0,90,180],atol=1e-12)
    return {'人工已知角度输入':[0,90,180], '实际角度':actual.tolist(), 'passed':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-gt-evaluation',action='store_true')
    args=parser.parse_args()
    test=self_test()
    pred, evidence=validate()
    if not args.allow_gt_evaluation:
        print(json.dumps(dict(integrity=evidence,self_test=test),ensure_ascii=False,indent=2)); return
    out=RUN/'evaluation'
    base.require(not out.exists(),'禁止覆盖既有评测')
    receipt=RUN/'gt_evaluation_receipt.json'
    dump(receipt,dict(started_at=now(),status='GT_EVALUATION_STARTED',output=str(out),integrity=evidence,self_test=test,
        code_sha256=base.sha(__file__),base_code_sha256=base.sha(base.__file__),failure_policy='失败保留收据，禁止二次运行'))
    out.mkdir()
    dump(out/'integrity_validation.json',evidence)
    dump(out/'self_test.json',test)
    # 此前未打开GT。该函数仅访问SUN类型、方向定义和light_dir_world；不访问其余GT属性。
    before=now()
    gt,gt_hash=base.read_gt(ROOT/'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3')
    after=now()
    errors=base.evaluate({'final':pred},gt)
    # 仅复用绘图/CSV函数，运行时限制为唯一冻结端点，原模块文件不修改。
    base.METHODS=('final',)
    metrics=base.write_outputs(out,{'final':pred},gt,errors,'final')['final']
    result=dict(status='FAILED',descriptive_gt_status='PASS' if metrics['mean_deg']<=5 else 'FAILED',
        scientific_protocol_status='FAILED',states=STATES,metrics=metrics,gt_read_started_at=before,
        gt_read_finished_at=after,gt_sha256=gt_hash,evaluated_at=now(),coordinate_system='world',
        direction_convention='surface_to_light',sign_or_coordinate_search=False,
        prediction_sha256=PRED,delivery_manifest_sha256=EXPECTED,gt_feedback_optimization=False,
        isolation='新规划与执行者未见真实GT/旧成绩；协调者见过旧轮GT但未参与本轮方法选择；本次仅描述性评测')
    dump(out/'results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
