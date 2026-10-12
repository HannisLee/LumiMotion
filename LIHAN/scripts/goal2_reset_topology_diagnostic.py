#!/usr/bin/env python3
"""只读CPU：原拓扑事件与reset前预测高度带UID后代，不读取GT或改测量。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def analyze(model, output):
    inputs = {}

    def read_json(path):
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())

    marker = read_json(model / 'reset_audit_training.json')
    finished = read_json(model / 'reset_audit_finished.json')
    if (marker['smoke'] or marker['scheduled_reset_iteration'] != 15000
        or finished['status'] != 'TRAINING_FINISHED' or finished['iteration'] != 16000
        or finished['first_reset_iteration'] != 15000 or finished['optimizer_updates'] != 2000):
        raise AssertionError('本诊断只接受正式14000→16000自然15000 reset。')
    directory = model / 'topology_audit'
    events_path = directory / 'events.jsonl'
    inputs[str(events_path)] = digest(events_path)
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    active, ever = set(), set()
    cohort, family = None, {}
    birth_totals, replacement_totals, prune_totals = Counter(), Counter(), Counter()
    cohort_births = cohort_replaced = cohort_filtered = 0
    filtered_original, replaced_original = set(), set()
    deleted_by_root = Counter()
    event_rows, snapshots, by_iteration = [], [], defaultdict(Counter)
    previous_iteration = 14000
    final_arrays = None
    pre_count = None
    reset_count = 0

    for serial, event in enumerate(events):
        iteration, kind = int(event['iteration']), event['kind']
        if iteration < previous_iteration:
            raise AssertionError('事件顺序回退。')
        previous_iteration = iteration
        path = directory / event['arrays']
        inputs[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.array(data[key], copy=True) for key in data.files}
        ids = [int(uid) for uid in arrays['uid']]
        if len(ids) != len(set(ids)) or arrays['xyz'].shape != (len(ids), 3) or not np.isfinite(arrays['xyz']).all():
            raise AssertionError('UID重复／XYZ形状或有限性不符。')
        row = dict(serial=serial, iteration=iteration, kind=kind, count=len(ids),
                   phase='post_reset_recovery' if cohort is not None else 'before_pre_reset_selection',
                   tracked_descendants=0, low_opacity=0, big_screen=0, big_world=0, union=0, overlap=0)
        if kind == 'initial':
            if serial != 0 or iteration != 14000:
                raise AssertionError('初始事件必须为14000。')
            active, ever = set(ids), set(ids)
        elif kind in ('clone_birth', 'split_birth'):
            parents = [int(uid) for uid in arrays['parent_uid']]
            if len(parents) != len(ids) or any(uid in ever for uid in ids) or any(uid not in active for uid in parents):
                raise AssertionError('新生UID／真实父点排列无效。')
            active.update(ids)
            ever.update(ids)
            for uid, parent in zip(ids, parents):
                if parent in family:
                    family[uid] = family[parent]
                    row['tracked_descendants'] += 1
            if cohort is not None:
                birth_totals[kind] += len(ids)
                cohort_births += row['tracked_descendants']
                by_iteration[iteration][kind] += len(ids)
        elif kind in ('split_parent_replacement', 'filter_prune'):
            if not set(ids).issubset(active) or event['before'] != len(active) or event['after'] != len(active) - len(ids):
                raise AssertionError('删除UID／前后点数与真实活动点集不符。')
            row['tracked_descendants'] = sum(uid in family for uid in ids)
            if kind == 'filter_prune':
                low, screen, world = [arrays[key].reshape(-1).astype(bool) for key in ('low_opacity', 'big_screen', 'big_world')]
                if any(len(mask) != len(ids) for mask in (low, screen, world)) or not (low | screen | world).all():
                    raise AssertionError('真实过滤删除原因并集未覆盖全部删除点。')
                row.update(low_opacity=int(low.sum()), big_screen=int(screen.sum()), big_world=int(world.sum()),
                           union=len(ids), overlap=int((low.astype(int) + screen.astype(int) + world.astype(int) > 1).sum()))
                if any(event['reasons'][key] != row[key] for key in ('low_opacity', 'big_screen', 'big_world', 'union', 'overlap')):
                    raise AssertionError('原事件计数与原数组原因不同。')
                if cohort is not None:
                    prune_totals.update({key: row[key] for key in ('low_opacity', 'big_screen', 'big_world', 'union', 'overlap')})
                    cohort_filtered += row['tracked_descendants']
                    for uid in ids:
                        if uid in family:
                            deleted_by_root[family[uid]] += 1
                        if uid in cohort:
                            filtered_original.add(uid)
                    by_iteration[iteration].update({key: row[key] for key in ('low_opacity', 'big_screen', 'big_world', 'union', 'overlap')})
            elif cohort is not None:
                replacement_totals[kind] += len(ids)
                cohort_replaced += row['tracked_descendants']
                replaced_original.update(set(ids) & cohort)
                by_iteration[iteration][kind] += len(ids)
            active.difference_update(ids)
        elif kind in ('pre_reset', 'post_reset', 'after_topology', 'final'):
            if set(ids) != active:
                raise AssertionError('完整状态UID与事件重放点集不符。')
            if kind == 'pre_reset':
                if cohort is not None or iteration != 15000:
                    raise AssertionError('必须唯一自然15000 pre_reset。')
                band = (arrays['xyz'][:, 2] > .05) & (arrays['xyz'][:, 2] < .35)
                cohort = {uid for uid, selected in zip(ids, band) if selected}
                family = {uid: uid for uid in cohort}
                pre_count = len(ids)
            if kind == 'post_reset':
                if iteration != 15000 or cohort is None:
                    raise AssertionError('post-reset阶段不符。')
                reset_count += 1
            if cohort is not None:
                tracked_indices = [index for index, uid in enumerate(ids) if uid in family]
                roots = {family[ids[index]] for index in tracked_indices}
                z = arrays['xyz'][tracked_indices, 2]
                in_band = int(((z > .05) & (z < .35)).sum())
                snapshots.append(dict(iteration=iteration, kind=kind, total_active=len(ids),
                    tracked_descendants_alive=len(tracked_indices), families_with_any_descendant_alive=len(roots),
                    tracked_descendants_in_height_band=in_band, tracked_descendants_outside_height_band=len(z)-in_band,
                    selected_original_uids_alive=len(cohort & active)))
            if kind == 'final':
                final_arrays = arrays
        else:
            raise AssertionError('未知拓扑事件：' + kind)
        if event.get('count', len(ids)) != len(ids):
            raise AssertionError('事件count与UID数组不同。')
        event_rows.append(row)

    if reset_count != 1 or final_arrays is None or events[-1]['kind'] != 'final' or len(active) != finished['point_count']:
        raise AssertionError('reset／终点完整性不符。')
    final_ids = [int(uid) for uid in final_arrays['uid']]
    final_descendants = [uid for uid in final_ids if uid in family]
    alive_roots = {family[uid] for uid in final_descendants}
    in_band_roots = {family[uid] for uid, xyz in zip(final_ids, final_arrays['xyz'])
                     if uid in family and .05 < xyz[2] < .35}
    lost_roots = cohort - alive_roots
    if any(deleted_by_root[root] == 0 for root in lost_roots):
        raise AssertionError('谱系灭失没有真实filter删除证据；不能算split替换。')
    if len(cohort) + cohort_births != len(final_descendants) + cohort_filtered + cohort_replaced:
        raise AssertionError('高度带后代出生／替换／真实删除／存留不守恒。')
    tracked = dict(pre_reset_selected_original_uids=len(cohort), original_uids_alive=len(cohort & active),
        original_uids_truly_filter_pruned=len(filtered_original), original_uids_split_replaced=len(replaced_original),
        post_reset_descendant_births=cohort_births, tracked_uids_truly_filter_pruned=cohort_filtered,
        tracked_uids_split_parent_replaced=cohort_replaced, final_descendant_uids_alive=len(final_descendants),
        families_with_any_descendant_alive=len(alive_roots), families_truly_extinct=len(lost_roots),
        families_with_descendants_in_height_band=len(in_band_roots),
        surviving_families_only_outside_height_band=len(alive_roots - in_band_roots),
        conservation_pass=True)
    iteration_rows = [dict(iteration=i, window='pre_reset_topology' if i == 15000 else 'post_reset_recovery',
                          **{key: int(by_iteration[i][key]) for key in ('clone_birth', 'split_birth', 'split_parent_replacement',
                             'low_opacity', 'big_screen', 'big_world', 'union', 'overlap')}) for i in range(15000,16001,100)]
    # 15000结构事件在pre_reset选点之前，单独展示，不纳入reset后删除总量。
    for event in event_rows:
        if event['iteration'] == 15000 and event['phase'] == 'before_pre_reset_selection':
            for key in ('clone_birth', 'split_birth', 'split_parent_replacement'):
                if event['kind'] == key:
                    iteration_rows[0][key] += event['count']
            if event['kind'] == 'filter_prune':
                for key in ('low_opacity','big_screen','big_world','union','overlap'):
                    iteration_rows[0][key] += event[key]
    for path, sha in inputs.items():
        if digest(path) != sha:
            raise AssertionError('训练审计输入在读取时变化：' + path)
    for name, rows in [('events.csv', event_rows), ('post_reset_per_100_steps.csv', iteration_rows), ('height_cohort_snapshots.csv', snapshots)]:
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(output / 'height_cohort_uids.npz', selected_pre_reset_uid=np.array(sorted(cohort),dtype=np.int64),
        final_descendant_uid=np.array(final_descendants,dtype=np.int64),
        final_descendant_selected_ancestor_uid=np.array([family[uid] for uid in final_descendants],dtype=np.int64),
        extinct_selected_ancestor_uid=np.array(sorted(lost_roots),dtype=np.int64))
    summary = dict(status='PASS', model=str(model), iteration=16000, first_reset_iteration=15000,
        post_reset_normal_route=marker['post_reset_normal_route'], original_event_count=len(events),
        pre_reset_total_points=pre_count, final_total_points=len(active),
        post_reset_global_filter_prune=dict(prune_totals), post_reset_global_births=dict(birth_totals),
        post_reset_split_parent_replacements=dict(replacement_totals),
        per_100_steps=iteration_rows, predicted_height_cohort=tracked, input_sha256=inputs, input_unchanged=True,
        diagnostic_source_sha256=digest(Path(__file__)), no_gt_read=True, no_gpu_used=True,
        scope='预测世界高度0.05<z<0.35在15000 pre_reset选UID，按parent_uid追后代；不是GT鞋语义面积，不是像素覆盖，不进入训练。split父替换绝不计入真正filter删除。各组独立统计，不假设跨模型UID对应或逐位相同；未改变任何主评测。')
    write(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model, output = args.model.resolve(), args.output.resolve()
    if not output.is_relative_to(root / 'output'):
        parser.error('诊断输出必须output内。')
    output.mkdir(parents=True, exist_ok=False)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), '--model', str(model), '--output', str(output)])
    try:
        report = analyze(model, output)
    except Exception as error:
        write(output / 'FAILED.json', dict(status='FAILED_DIAGNOSTIC', error=repr(error), retained=True, command=command))
        raise
    (output / 'README.md').write_text('# 自然reset后只读CPU拓扑诊断\n\n'
        + f'模型：`{model}`；终点16000，reset15000；输出`{output}`。\n\n```bash\n{command}\n```\n\n'
        + report['scope'] + '\n\nUID重放、原因并集、点集与后代守恒及原输入SHA不变核验**PASS**，仅诊断完整性；不判科学Goal成功。原数据、每100步事件、高度带谱系均保留。\n')
    print('只读CPU拓扑诊断PASS：' + str(output / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
