"""七模型冻结后统一指标的CPU只读汇总；不选模，不读训练输入，不替代人工验收。"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shlex
import socket
import sys


ORDER = ('ELV2', 'N00', 'A0', 'A1', 'B0', 'B1', 'DIR')
METHODS = ('gs_render', 'depth_normal', 'mesh')
DEPTHS = ('expected', 'median', 'mesh_depth')
REGIONS = ('foreground', 'human', 'leg', 'foot', 'shoe', 'platform')
TEST = list(range(10, 151, 10))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, evidence):
    path = Path(path).resolve()
    evidence[str(path)] = sha(path)
    def reject(value):
        raise ValueError('JSON含非有限值：' + value)
    return json.loads(path.read_text(), parse_constant=reject)


def average(values):
    present = [value for value in values if value is not None]
    return sum(present)/len(present) if present else None


def close(first, second):
    if first is None or second is None:
        assert first is second, (first, second)
        return
    assert first is not None and second is not None
    assert math.isfinite(first) and math.isfinite(second) and abs(first-second) <= 1e-8, (first, second)


def denominator(metrics):
    return [{name: (region['gt_pixels'], region['gt_depth_pixels'])
             for name, region in row['regions'].items()} for row in metrics['rows']]


def verify_summary_arithmetic(metrics):
    """核对原始逐帧统计的分母及汇总；不重新读取GT或推断预测是否正确。"""
    for region, summary in metrics['summary'].items():
        rows = [row['regions'][region] for row in metrics['rows']]
        for method in METHODS:
            items = [row['normal'][method] for row in rows]
            for row, item in zip(rows, items):
                n, count = item['eligible_pixels'], item['valid_pixels']
                assert n == row['gt_pixels'] and 0 <= count <= n and item['angular']['count'] == count
                expected = (item['angular']['sum']+180*(n-count))/n if n else None
                close(item['missing180_mae'], expected)
                close(item['coverage'], count/n if n else None)
            aggregate = summary['normal'][method]
            close(average([item['missing180_mae'] for item in items]), aggregate['missing180_frame_mean_deg'])
            close(average([item['coverage'] for item in items]), aggregate['coverage_frame_mean'])
        for field in DEPTHS:
            items = [row['depth'][field] for row in rows]
            for row, item in zip(rows, items):
                n, count = item['eligible_pixels'], item['valid_pixels']
                assert n == row['gt_depth_pixels'] and 0 <= count <= n and item['absolute']['count'] == count
                close(item['coverage'], count/n if n else None)
                close(item['clipped_relative_missing_mean'], item['clipped_relative_missing_sum']/n if n else None)
            aggregate = summary['depth'][field]
            close(average([item['absolute']['mean'] for item in items]), aggregate['absolute']['frame_mean'])
            close(average([item['clipped_relative_missing_mean'] for item in items]), aggregate['clipped_relative_missing_frame_mean'])
            close(average([item['coverage'] for item in items]), aggregate['coverage_frame_mean'])
    for region, aggregate in metrics['rgb'].items():
        rows = [row['rgb'][region] for row in metrics['rows']]
        assert all(row['valid_pixels'] == row['eligible_pixels'] for row in rows)
        close(average([row['mae'] for row in rows]), aggregate['mae_frame_mean'])
        close(average([row['mse'] for row in rows]), aggregate['mse_frame_mean'])


def collect(control):
    evidence = {}
    jobs = control/'eval_only/final_evaluation_jobs_v1'
    state = read(jobs/'state.json', evidence)
    assert state['status'] == 'READY_FOR_MANUAL_REVIEW_AND_REPORT', '七模型隔离执行尚未全部完成'
    lock = read(jobs/'selection_lock.json', evidence)
    assert lock['all_seven_frozen_before_test'] and not lock['candidate_gt_feedback_allowed']
    four = read(lock['four_selection'], evidence)
    final = read(lock['final_selection'], evidence)
    for path in (lock['four_selection'], lock['final_selection']):
        key = str(Path(path).resolve())
        assert evidence[key] == lock['locked_files'][key], 'DEV选择不再匹配冻结SHA：' + key
    assert four['mode'] == 'four' and final['mode'] == 'final'
    assert lock['selection_status'] == final['status']
    assert all(not row['gt_read'] and not row['test_read'] for row in (four, final))
    assert final['candidates'][final['selected']]['model'] == lock['selected_model_from_dev']
    assert final['candidates']['freeze']['variant'] == final['candidates']['directionlr']['variant'] == four['selected']
    assert final['candidates']['freeze']['model'] == four['candidates'][four['selected']]['model']
    pair_path = control/'four_pair_audit_v1/audit.json'
    pair = read(pair_path, evidence)
    assert evidence[str(pair_path)] == lock['locked_files'][str(pair_path)]
    assert pair['status'] == lock['pair_audit_status'] and pair['all_four_same_parent_inputs_config']
    entries = lock['entries']
    assert [entry['name'] for entry in entries] == list(ORDER)
    assert final['candidates']['directionlr']['model'] == entries[-1]['model']
    records = {}; v3_denominator = None; v3_source = None
    for entry in entries:
        name = entry['name']; folder = Path(entry['output']).resolve()
        assert folder.is_relative_to(control/'eval_only'), '指标仅允许来自隔离评测区'
        assert sha(entry['freeze']) == entry['freeze_sha256']
        frozen = read(entry['freeze'], evidence)
        assert frozen['evaluation_allowed'] and not frozen['selection_uses_test_or_gt']
        assert frozen['model'] == entry['model'] and frozen['iteration'] == entry['step']
        done = read(folder/'COMPLETED.json', evidence)
        assert done['status'] == 'PASS_EXECUTION_ONLY' and done['frames'] == 15 and done['video_frames'] == 150
        classes = ('rgb', 'normal', 'depth_mesh', 'separation') + (('components',) if entry['stage'] == 'stage2' else ())
        assert all(done['videos'][category] == 150 for category in classes)
        metrics = read(folder/'metrics.json', evidence)
        assert metrics['status'] == 'PASS_EXECUTION_ONLY' and metrics['frames'] == TEST
        assert metrics['model'] == entry['model'] and metrics['iteration'] == entry['step'] and metrics['stage'] == entry['stage']
        assert metrics['dataset'] == entry['dataset'] and metrics['source_bundle_sha256'] == frozen['source_sha256']
        assert not metrics['candidate_gt_feedback_allowed']
        assert [row['source_frame'] for row in metrics['rows']] == TEST
        assert metrics['protocol']['missing_normal_degrees'] == 180 and metrics['protocol']['alpha_threshold'] == .5
        assert metrics['protocol']['resolution'] == [400, 400]
        assert metrics['protocol']['signed_world_normal'] is True
        verify_summary_arithmetic(metrics)
        normal = metrics['summary']['foreground']['normal']
        for method in METHODS:
            items = [row['regions']['foreground']['normal'][method] for row in metrics['rows']]
            assert all(item['eligible_pixels'] > 0 for item in items)
            assert all(0 <= item['valid_pixels'] <= item['eligible_pixels'] for item in items)
            for item in items:
                close(item['missing180_mae'], (item['angular']['sum']+180*(item['eligible_pixels']-item['valid_pixels']))/item['eligible_pixels'])
            close(average([item['missing180_mae'] for item in items]), normal[method]['missing180_frame_mean_deg'])
        primary = average([normal[method]['missing180_frame_mean_deg'] for method in METHODS])
        close(primary, metrics['normal_composite_fixed_denominator_mae'])
        if name != 'ELV2':
            current = denominator(metrics)
            if v3_denominator is None:
                v3_denominator, v3_source = current, metrics['source_bundle_sha256']
            assert current == v3_denominator and metrics['source_bundle_sha256'] == v3_source, '同数据模型GT分母或source不一致'
        surface = read(folder/'surface_completeness.json', evidence)
        temporal = read(folder/'temporal_residuals.json', evidence)
        assert [row['frame'] for row in surface['rows']] == TEST
        assert temporal['active_track_protocol'] == 'persistent_vertices_v1'
        if temporal['status'].startswith('EVALUATED_'):
            assert temporal['units']['position'] == 'world'
            assert temporal['units']['velocity'] == 'world/' + temporal['time_unit']
            assert temporal['units']['acceleration'] == 'world/' + temporal['time_unit'] + '^2'
        precision = [average([row['precision_prediction_to_gt']['close_fractions'][i] for row in surface['rows']])
                     for i in range(len(surface['protocol']['distances_world_units']))]
        saturation = {region: average([row['rgb'][region]['saturated_channel_fraction'] for row in metrics['rows']])
                      for region in ('foreground', 'dark', 'middle', 'bright')}
        records[name] = dict(output=str(folder), model=entry['model'], iteration=entry['step'], stage=entry['stage'],
            composite_deg=primary, summary=metrics['summary'], rgb=metrics['rgb'], saturation_frame_mean=saturation,
            rgb_thresholds=metrics['protocol']['thresholds'],
            surface={key: value for key, value in surface.items() if key != 'rows'},
            precision_close_frame_mean=precision, temporal=temporal,
            unit_system=metrics['unit_system'], meters_per_world_unit=metrics['meters_per_world_unit'],
            denominator=denominator(metrics), completed_manual_review=done['manual_review'])
    selected = [name for name in ORDER if records[name]['model'] == lock['selected_model_from_dev']]
    assert len(selected) == 1
    selected = selected[0]; target = records[selected]['composite_deg']
    comparisons = dict(selected_from_dev=selected, final_selection_status=final['status'],
        four_selection_status=four['status'], mainline=four['selected'], pair_audit_status=lock['pair_audit_status'],
        selected_below_elv2=target < records['ELV2']['composite_deg'], selected_below_n00=target < records['N00']['composite_deg'],
        selected_minus_elv2_deg=target-records['ELV2']['composite_deg'], selected_minus_n00_deg=target-records['N00']['composite_deg'],
        ps_a1_minus_a0_deg=records['A1']['composite_deg']-records['A0']['composite_deg'],
        ps_b1_minus_b0_deg=records['B1']['composite_deg']-records['B0']['composite_deg'],
        dir_minus_frozen_mainline_deg=records['DIR']['composite_deg']-records[four['selected']]['composite_deg'],
        old_new_gt_denominator_equal=records['ELV2']['denominator'] == records['N00']['denominator'],
        inferential_significance='单次固定预算实验；没有重复种子或统计显著性结论',
        model_reselection_permitted=False, scientific_goal='NOT_DECIDED_PENDING_MANUAL_REVIEW')
    for path, digest in evidence.items():
        assert sha(path) == digest, '汇总期间输入改变：' + path
    return dict(status='PASS_COMPARISON_ARITHMETIC_ONLY', comparisons=comparisons, records=records,
                input_sha256=evidence, training_or_selection_modified=False,
                scope='核对各分区法线/深度分母、逐帧统计汇总及RGB等帧MAE/MSE；不重算渲染、GT误差、表面距离或时间轨迹')


def value(number):
    return '不可用' if number is None else f'{number:.8g}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def markdown(result, command):
    records = result['records']; comparison = result['comparisons']; parts = [
        '# 七模型统一指标只读汇总\n', '完整汇总命令：\n\n```bash\n' + command + '\n```\n',
        '本入口只读取七份完整隔离评测产物，不渲染、不重训、不重选模型。科学结论与四类人工验收仍待主Agent及独立Evaluation完成；算术检查PASS不能替代科学PASS。\n',
        '## 固定选择与主体法线\n',
        'DEV锁定模型：' + comparison['selected_from_dev'] + '；四组主线：' + comparison['mainline'] +
        '；最终DEV状态：' + comparison['final_selection_status'] + '；配对前缀：' + comparison['pair_audit_status'] + '。\n',
        '三法线按固定GT前景分母、缺失180度、15帧等权及三方法等权；覆盖列同为等帧均值，像素汇总值另见JSON。未按TEST数值排名选模。\n']
    rows = []
    for name in ORDER:
        record = records[name]; normal = record['summary']['foreground']['normal']
        rows.append([name, value(record['composite_deg'])] + [value(normal[m]['missing180_frame_mean_deg']) for m in METHODS] +
                    [value(normal[m]['coverage_frame_mean']) for m in METHODS])
    parts.append(table(['模型', '综合MAE度', 'GS度', 'Expected度', 'Mesh度', 'GS覆盖', 'Expected覆盖', 'Mesh覆盖'], rows))
    parts += ['锁定模型相对ELV2差值：' + value(comparison['selected_minus_elv2_deg']) + '度；相对N00：' + value(comparison['selected_minus_n00_deg']) +
              '度；A1−A0：' + value(comparison['ps_a1_minus_a0_deg']) + '度；B1−B0：' + value(comparison['ps_b1_minus_b0_deg']) +
              '度；DIR−冻结主线：' + value(comparison['dir_minus_frozen_mainline_deg']) + '度。负值仅表示该指标降低，不单独构成完整验收或显著性结论。\n',
              '## 人体与腿足鞋局部法线\n']
    rows = []
    for region in REGIONS[1:]:
        for name in ORDER:
            normal = records[name]['summary'][region]['normal']
            rows.append([region, name] + [value(normal[m]['missing180_frame_mean_deg']) for m in METHODS] +
                        [value(normal[m]['coverage_frame_mean']) for m in METHODS] + [str(normal['gs_render']['eligible_frames'])])
    parts.append(table(['分区', '模型', 'GS度', 'Expected度', 'Mesh度', 'GS覆盖', 'Expected覆盖', 'Mesh覆盖', 'GT有效帧'], rows))
    parts.append('## 三类正camera Z深度\n\n绝对MAE只在有效预测上计算；另列固定分母截断相对缺失罚分和覆盖，不能只比较条件误差。完整阈值命中率与其余分区见comparison.json。\n')
    rows = []
    for region in REGIONS:
        for name in ORDER:
            for field in DEPTHS:
                item = records[name]['summary'][region]['depth'][field]
                rows.append([region, name, field, value(item['absolute']['frame_mean']),
                    value(item['clipped_relative_missing_frame_mean']), value(item['coverage_frame_mean'])])
    parts.append(table(['分区', '模型', '深度', '条件绝对MAE world', '固定分母相对罚分', '覆盖'], rows))
    parts.append('## 表面完整性\n\n完整阈值与局部分区召回保存在comparison.json；距离为world，无GT尺度对齐。空mesh按既定缺失规则保留。\n')
    rows = []
    for name in ORDER:
        r = records[name]; surface = r['surface']
        for i, threshold in enumerate(surface['protocol']['distances_world_units']):
            rows.append([name, value(threshold), value(surface['gt_to_prediction_regions']['foreground']['close_fraction_frame_mean'][i]),
                value(r['precision_close_frame_mean'][i]), value(surface['fscore_frame_mean'][i]),
                value(surface['symmetric_capped_missing_chamfer_frame_mean_world'])])
    parts.append(table(['模型', '阈值world', 'GT召回', '预测精度', 'F-score', '双向截断缺失距离world'], rows))
    parts.append('## 动态残差\n\n15TEST、固定持久GT vertex ID、扣除真实GT运动的可见射线残差；不是150原始帧的高频材质点抖动。以下为等区间、固定分母截断缺失均值；时间单位逐模型读取协议，非秒协议不冒称物理速度。原始条件值、覆盖及腿足鞋分区见comparison.json。\n')
    rows = []
    for name in ORDER:
        temporal = records[name]['temporal']
        if not temporal['status'].startswith('EVALUATED_'):
            rows.append([name, '不可用', temporal['status'], '不可用', '不可用', '不可用', '不可用']); continue
        for region in ('foreground', 'human'):
            for field in DEPTHS:
                item = temporal['metrics'][region][field]
                rows.append([name, region, field] + [value(item[key]['equal_interval_capped_missing_mean']) for key in
                    ('position', 'velocity_residual', 'acceleration_residual')] + [temporal['time_unit']])
    parts.append(table(['模型', '分区', '深度', '位置world', '速度world/时间单位', '二阶world/时间单位²', '时间单位'], rows))
    parts.append('## RGB全前景与暗中亮\n\n固定TRAIN阈值、线性RGB，无逐帧曝光或对齐；原图饱和/截断限制不能解释成真实HDR恢复精度。这里为严格TEST MAE，与DEV删失MAE口径不同。原生按其实际码值渲染整体逆sRGB，Stage2直接线性，不对原生做Alpha反预乘或额外校色；原生训练预处理的边缘差异不能省略。\n\nELV2阈值：'+json.dumps(records['ELV2']['rgb_thresholds'], ensure_ascii=False)+'；V3阈值：'+json.dumps(records['N00']['rgb_thresholds'], ensure_ascii=False)+'。\n')
    rows = []
    for name in ORDER:
        for region in ('foreground', 'dark', 'middle', 'bright'):
            item = records[name]['rgb'][region]
            psnr = '+∞（含零误差帧）' if item['psnr_status'] == 'INFINITE_FRAME_PRESENT' else value(item['psnr_frame_mean'])
            rows.append([name, region, value(item['mae_frame_mean']), psnr,
                item['eligible_frames'], value(records[name]['saturation_frame_mean'][region])])
    parts.append(table(['模型', '亮度分区', 'MAE', '帧均PSNR dB', '有效帧', '帧均饱和通道比例'], rows))
    parts.append('## 资产、渲染及人工验收边界\n')
    for name in ORDER:
        r = records[name]
        parts.append('- ' + name + '：model=' + r['model'] + '；iteration=' + str(r['iteration']) + '；渲染输出=' + r['output'] +
            '；完整实际渲染命令及日志见该目录README/evaluation.log；定量metrics.json；代表图four_classes/*_contact_0010_0080_0150.png，四类视频four_classes/*.mp4，局部review_crops。\n')
    parts.append('\nMesh只在15TEST真实重建，其余135视频帧明确未重建；数值和视频解码通过不代表人工完整目检。新旧训练样本、时间约定和照明不同，跨数据变化不能全部归因于方法。单位证据与完整指标见comparison.json及原metrics.json。\n\n最终结论：PENDING（仅汇总算术PASS；四类/局部/连续视频实际目检及完整科学结论必须另行完成）。\n')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[2]
    control, out = args.control.resolve(), args.output.resolve()
    assert socket.gethostname() == 'ushas' and control.is_relative_to(root/'output')
    assert out.is_relative_to(control/'eval_only')
    out.mkdir(parents=True, exist_ok=True)
    assert all(path.name in ('launch.json', 'run.log') for path in out.iterdir()), '禁止覆盖既有汇总'
    command = shlex.join(list(getattr(sys, 'orig_argv', [sys.executable] + sys.argv)))
    try:
        code_digest = sha(__file__)
        result = collect(control)
        assert sha(__file__) == code_digest, '汇总运行期间代码改变'
        result.update(command=command, code_sha256=code_digest)
        (out/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        report = datetime.datetime.now(datetime.timezone.utc).strftime('%m%d') + '-01-七模型统一指标只读汇总.md'
        (out/report).write_text(markdown(result, command))
        (out/'README.md').write_text('# 七模型只读汇总\n\n完整命令：\n\n```bash\n'+command+'\n```\n\nsource/model/iteration、原渲染目录及日志见comparison.json和'+report+'；本入口不生成新渲染或执行优化。输出comparison.json及中文汇总。四类人工验收和科学结论PENDING。\n\n结论：PASS（仅读取完整七模型冻结产物、固定分母算术及身份一致性；非科学验收）。\n')
        print(json.dumps(dict(status=result['status'], output=str(out), scientific_goal='NOT_DECIDED'), ensure_ascii=False))
    except Exception as error:
        (out/'FAILED.json').write_text(json.dumps(dict(status='FAILED_COMPARISON', error=repr(error), command=command,
            code_sha256=sha(__file__), assets_retained=True), ensure_ascii=False, indent=2)+'\n')
        (out/'README.md').write_text('# 七模型只读汇总失败\n\n完整命令：\n\n```bash\n'+command+
            '\n```\n\n输入：'+str(control/'eval_only/final_evaluation_jobs_v1/selection_lock.json')+
            '所列冻结模型与统一指标；日志run.log，失败详情FAILED.json。本入口不产生新渲染或执行训练，原checkpoint/渲染/统计未修改。'+
            '\n\n四类人工目检与科学结论未完成。最终结论：FAILED（汇总输入/算术契约，不能替代科学结论）。\n')
        raise


if __name__ == '__main__':
    main()
