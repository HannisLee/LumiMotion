"""全部真实MAE/实际会话退出及人工图像审核后回填，不自动给画质PASS。"""
from pathlib import Path
import json, hashlib, shutil, datetime
ROOT = Path(__file__).resolve().parent.parent
CWD = ROOT.parent.parent
NAME = ROOT.name
OLD = ROOT.parent/'1010-12-jumpingjacks_dir-PBR仓库原生光栅修复'
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
result = json.loads((ROOT/'metrics_summary.json').read_text())
state = json.loads((ROOT/'postprocess_state.json').read_text())
runtime = json.loads((ROOT/'training_runtime.json').read_text())
review = json.loads((ROOT/'inspection/visual_review.json').read_text())
assert result['status'] == 'COMPLETE_MAE_MANUAL_REVIEW_PENDING'
assert len(result['metrics']) == 7 and result['frames'] == list(range(10, 151, 10))
assert result['independent_audit']['status'] == 'PASS'
assert state['status'] == 'PIPELINE_COMPLETED_REVIEW_PENDING'
assert not (Path('/proc')/str(state['pid'])).exists()
steps = [json.loads(line) for line in (ROOT/'runtime_history.jsonl').read_text().splitlines()]
assert len(steps) == 9 and all(row['exit_code'] == 0 for row in steps)
assert runtime['training_exit_code'] == 0 and runtime['verified_training_session_closed']
assert runtime['steps_completed'] == 20000 and runtime['iteration'] == 55000
assert state['pipeline_exit_code'] == 0 and state['verified_session_closed']
assert review['status'] in ['PASS', 'FAILED'] and review['iteration'] == 55000
assert review['actual_images_viewed'] is True
assert review['reviewed_test_frames'] == [10, 80, 150]
assert review['reviewed_insight_frames'] == [1, 75, 150]
assert review['reviewed_hdr_frames'] == [1, 76, 150]
assert review['reviewed_directional_frames'] == [0, 36, 71]
for relative in review['images']:
    assert (ROOT/relative).is_file()
manifest = json.loads((ROOT/'code_manifest.json').read_text())
assert len(manifest) == 471
for relative, sha in manifest.items():
    for p in [CWD/relative, ROOT/'commands/code_snapshot'/relative,
              ROOT/'commands/code_snapshot_train_launch'/relative]:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha
native = json.loads((ROOT/'model_mlp/pbr_rasterizer.json').read_text())
assert native['source_sha256'] == '82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0'
config = json.loads((ROOT/'model_mlp/stage2_config.json').read_text())
assert config['pbr_directional_actual_ray_budget'] and config['wo_indirect'] and config['wo_indirect_relight']
normal = 'ACHIEVED_METRIC' if result['metrics'][2]['mae'] < 7 else 'FAILED'
quality = review['status']
result.update(status='PASSED_QUALITY' if quality == 'PASS' else 'FAILED_QUALITY',
    manual_review=review, normal_goal=normal, commit_allowed=quality == 'PASS',
    pipeline_exit_code=0, verified_postprocess_session_closed=True, source_code_manifest_verified=True,
    kernel_runtime_provenance=str(ROOT/'model_mlp/pbr_rasterizer.json'))
(ROOT/'metrics_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
artifact = json.loads((ROOT/'inspection/artifact_checks.json').read_text())
artifact.update(manual_review=quality, visual_review=review)
artifact['scope'] = artifact['scope'].replace('人工审阅待完成', '人工实际审阅'+quality+'；自动数值PASS不等于质量PASS')
(ROOT/'inspection/artifact_checks.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n')
state.update(status='FINISHED_QUALITY_'+quality, reviewed_at=now,
    final_gs_normal_mae_deg=result['metrics'][2]['mae'], commit_allowed=quality == 'PASS')
(ROOT/'postprocess_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
old = json.loads((OLD/'metrics_summary.json').read_text())
table = '| 指标 | B06原生+间接光 | B07直接SUN实际预算 | B07−B06 |\n| --- | ---: | ---: | ---: |\n'
for a, b in zip(old['metrics'], result['metrics']):
    assert a['metric'] == b['metric']
    table += '| {} | {:.8f} | {:.8f} | {:+.8f} |\n'.format(b['metric'], a['mae'], b['mae'], b['mae']-a['mae'])
common = result['normal_common_pixels']
final = '\n## B07完整验收与B06同55k比较（'+now+'）\n\n'+table
final += '\nGS覆盖率{:.8f}%→{:.8f}%；RGB PSNR{:.8f}→{:.8f}dB、SSIM{:.8f}→{:.8f}、LPIPS{:.8f}→{:.8f}。\n'.format(
    old['metrics'][2]['coverage_pct'], result['metrics'][2]['coverage_pct'],
    old['rgb']['psnr'], result['rgb']['psnr'], old['rgb']['ssim'], result['rgb']['ssim'],
    old['rgb']['lpips'], result['rgb']['lpips'])
final += '\n共同{}/{}（{:.8f}%），GS/Depth/Mesh共同MAE{:.8f}°/{:.8f}°/{:.8f}°；missing180={:.8f}°。\n'.format(
    common['valid_pixels'], common['gt_pixels'], common['coverage']*100,
    *[common['angular_deg'][k]['mae_pixel_mean_deg'] for k in ['gs_render', 'depth_normal', 'mesh']],
    result['gs_normal_missing_180_penalty_deg'])
final += '\n完整20k训练与9步后处理实际exit0且会话关闭，194 CPU/20步Direct Preset、真实SUN光线数/材质/XYZ/q梯度、共享几何与材质逐buffer均已验证。独立15帧七MAE复算与来源SHA、15time×135TRAIN预测Mesh、roughness来源复算、6视频全解码通过。471核心和双source快照逐SHA保持；native kernel SHA绑定，无GT训练材质/法线/几何/原能量/颜色。全部99827点/15测试帧及原评分定义保持，失败权重/图/数组/日志保留。\n'
final += '\n人工质量'+quality+'，原GS Normal <7°目标'+normal+'；本脚本不执行commit/push/新分支。\n'
body = (ROOT/'README.md').read_text()
body = body.replace('状态：完整20000步训练、渲染、全部15测试帧MAE与独立复算完成；人工质量审閱待完成。不能据此标记图像质量或7°目标PASS。',
    '状态：完整20000步训练、全量渲染和15测试帧七MAE/独立复算完成；人工质量'+quality+'，原GS Normal <7°目标'+normal+'。')
start = body.index('## 四类目检（待人工回填）')
end = body.index('## 审计及结论', start)
section = '## 四类实际目检：'+quality+'\n\n'
for key in ['rgb_background', 'alpha', 'albedo_static_dynamic', 'normal_depth']:
    section += '- '+review[key]+'\n'
section += '\nHDR/SUN：'+review['relighting']+'\n\n6视频逐帧解码通过；人工实际查看所列代表帧，不宣称人工逐帧观看所有视频。无重光照GT精度声明。\n\n'
body = body[:start]+section+body[end:]
body = body.replace('人工图像质量：PENDING', '人工图像质量：'+quality)
body += final
(ROOT/'README.md').write_text(body)
paths = [CWD/'LIHAN/DOC/训练'/(NAME+'训练报告.md'), CWD/'LIHAN/DOC/报告'/(NAME+'各项MAE评测报告.md')]
for p in paths:
    p.write_text(body)
goal = CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:
    f.write(final)
index = CWD/'LIHAN/README.md'
with index.open('a') as f:
    f.write('\n- [B07直接SUN实际预算完整七项MAE与验收](DOC/报告/'+paths[1].name+')：完整20k/全15七MAE/独立复算exit0，GS{:.5f}°/覆盖{:.5f}%，实际质量{}、7°{}；未自动提交。\n'.format(
        result['metrics'][2]['mae'], result['metrics'][2]['coverage_pct'], quality, normal))
archive = CWD/'LIHAN/DOC/报告/archive'/NAME
assert archive.is_dir()
for p in [ROOT/'README.md', ROOT/'metrics_summary.json', ROOT/'postprocess_state.json',
          ROOT/'runtime_history.jsonl', ROOT/'training_runtime.json']:
    shutil.copy2(p, archive/p.name)
for folder in ['commands', 'logs', 'inspection']:
    shutil.copytree(ROOT/folder, archive/folder, dirs_exist_ok=True)
sha = {str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
       for p in archive.rglob('*') if p.is_file() and p.name != 'archive_manifest.json'}
(archive/'archive_manifest.json').write_text(json.dumps(sha, ensure_ascii=False, indent=2)+'\n')
for checkout in [CWD, Path('/home/lihan/reproduce/LumiMotion-perlight'), Path('/home/lihan/reproduce/LumiMotion')]:
    if checkout != CWD:
        for p in paths+[goal, index]:
            shutil.copy2(p, checkout/p.relative_to(CWD))
        shutil.copytree(archive, checkout/archive.relative_to(CWD), dirs_exist_ok=True)
    for relative, digest in sha.items():
        assert hashlib.sha256((checkout/archive.relative_to(CWD)/relative).read_bytes()).hexdigest() == digest
print(json.dumps(dict(quality=quality, normal_goal=normal, GS_normal_MAE_deg=result['metrics'][2]['mae'],
                     archive_files=len(sha), SHA_verified_checkouts=3), ensure_ascii=False), flush=True)
