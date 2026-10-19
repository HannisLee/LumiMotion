"""仅离线按用户指定ELV2精确三法线门槛核验V3固定终点，不修改计量或模型。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys


METHODS = ('gs_render', 'depth_normal', 'mesh')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(evaluation, reference):
    metrics_path = evaluation/'metrics/metrics.json'
    if not (evaluation/'PIPELINE_COMPLETED.json').is_file() or not metrics_path.is_file():
        return dict(evaluation=str(evaluation), numeric_status='INCOMPLETE', error='完整固定终点评测尚未完成。')
    data = read(metrics_path)
    expected = reference['required_test_frames']
    frames = sorted(row['source_frame'] for row in data['rows'])
    if data['split'] != 'test' or frames != expected or not data['all_predicted_meshes_available']:
        raise ValueError('不是完整原15TEST和同一模型的完整Mesh：'+str(evaluation))
    plan = read(evaluation/'plan.json')
    if Path(plan['model']) != Path(data['model']) or plan['iteration'] != data['iteration']:
        raise ValueError('统计与固定评测计划的模型/iteration不同。')
    comparisons = {}
    for method in METHODS:
        values = data['summary']['foreground']['normal'][method]
        value = values['mae_pixel_mean_deg']
        threshold = reference['methods'][method]['threshold']
        if (not math.isfinite(value) or not 0 <= value <= 180 or
                values['eligible_frames'] != len(expected) or values['valid_frames'] != len(expected) or
                values['missing_prediction_frames'] or not 0 < values['coverage'] <= 1 or
                not 0 < values['valid_pixels'] <= values['eligible_pixels']):
            raise ValueError('三法线主指标存在无效/缺失帧，不判完整数值PASS：'+method)
        comparisons[method] = dict(value=value, strict_upper_bound=threshold,
            difference_from_elv2=value-threshold, passed=value < threshold,
            coverage=values['coverage'], valid_pixels=values['valid_pixels'],
            eligible_pixels=values['eligible_pixels'])
    return dict(evaluation=str(evaluation), model=data['model'], iteration=data['iteration'],
        numeric_status='PASS' if all(v['passed'] for v in comparisons.values()) else 'FAILED',
        comparisons=comparisons, metrics_sha256=sha(metrics_path), plan_sha256=sha(evaluation/'plan.json'),
        pipeline_receipt_sha256=sha(evaluation/'PIPELINE_COMPLETED.json'),
        region='foreground', aggregation='mae_pixel_mean_deg', frames=frames,
        manual_review='本脚本只判定用户指定三项数值，实际目检由对应实验单独归档。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = read(args.reference)
    source = Path(reference['reference_source'])
    if (sha(source) != reference['reference_sha256'] or reference['region'] != 'foreground' or
            reference['aggregation'] != 'mae_pixel_mean_deg' or
            reference['comparison'] != 'strict_less_than_each_same_model_same_checkpoint'):
        raise ValueError('ELV2参考来源或计量约定变化。')
    original = read(source)['summary']['foreground']['normal_metrics']
    if any(reference['methods'][k]['threshold'] != original[k]['mae_pixel_mean_deg'] for k in METHODS):
        raise ValueError('阈值不是ELV2原始精确主值。')
    results = [inspect(path.resolve(), reference) for path in args.evaluation]
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='OFFLINE_USER_THRESHOLD_COMPARISON_COMPLETE',
        reference=str(args.reference.resolve()), reference_sha256=sha(args.reference),
        script_sha256=sha(Path(__file__)), results=results,
        no_metric_or_model_modified=True, no_training_or_hyperparameter_selection=True,
        note='每行三项均来自同一个模型及固定checkpoint；不拼接，不用舍入上限，不更改GT/掩膜/聚合。')
    (args.output/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    lines = ['# V3对用户指定ELV2三法线精确门槛的离线验收', '',
        '完整实际只读命令：', '', '```bash', shlex.join([sys.executable, *sys.argv]), '```', '',
        '参考来源：'+str(source)+'；SHA：'+reference['reference_sha256']+'。', '',
        '全15TEST、foreground、有向法线、有效像素加权；以下严格用原精度判定，显示小数不替代原值。', '',
        '|模型/iteration|GS°|Expected°|Mesh°|三项同时严格更低|', '|---|---:|---:|---:|---|']
    for result in results:
        if result['numeric_status'] == 'INCOMPLETE':
            lines.append('|'+result['evaluation']+'|N/A|N/A|N/A|INCOMPLETE|')
            continue
        values = result['comparisons']
        lines.append('|'+result['model']+' @'+str(result['iteration'])+'|'+
                     '|'.join(f"{values[k]['value']:.12f}" for k in METHODS)+'|'+result['numeric_status']+'|')
    lines += ['', '精确门槛：'+json.dumps({k:reference['methods'][k]['threshold'] for k in METHODS})+'。', '',
        'coverage、有效像素数、与ELV2差值以及来源SHA均在receipt.json；覆盖率/RGB未新增数值阈值。', '',
        '复用已完成的模型渲染与统计，本命令不新渲染、不读GT进入训练、不改模型或计量。原完整渲染命令：']
    for evaluation in args.evaluation:
        if (evaluation/'plan.json').exists():
            plan = read(evaluation/'plan.json')
            for step in plan['steps']:
                if step['action'] == 'render':
                    lines += ['', '```bash', shlex.join(step['command']), '```']
            lines += ['', '输出与日志：'+str(evaluation)+'、logs。代表图four_classes/four_classes_10_80_150.png；',
                '视频four_classes/{rgb,alpha,normal,separation}.mp4；各物理分量components。',
                '本只读脚本未进行新的RGB/Alpha/Normal/Separation或分量/视频人工目检，实际范围以原实验收据为准。']
    lines += ['', '最终数值结论逐行PASS／FAILED／INCOMPLETE；未通过行保留全部checkpoint、渲染、统计和日志。',
        '数值验收不冒充未执行的完整三阶段交付或实际目检，旧历史FAILED不追溯删除。', '']
    (args.output/'README.md').write_text('\n'.join(lines))
    print(json.dumps([dict(model=x.get('model'), iteration=x.get('iteration'),
                           status=x['numeric_status'], comparisons=x.get('comparisons')) for x in results],
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
