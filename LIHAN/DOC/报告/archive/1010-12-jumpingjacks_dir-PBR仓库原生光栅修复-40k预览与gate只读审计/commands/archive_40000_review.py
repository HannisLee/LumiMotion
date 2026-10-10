from pathlib import Path
import json,hashlib,datetime,re,shutil
ROOT=Path(__file__).resolve().parent.parent; CWD=ROOT.parent.parent
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
handles=json.loads((ROOT/'process_handles.json').read_text())
for label in ['train','postprocess']:
 h=handles[label]; proc=Path('/proc')/str(h['pid'])
 assert proc.exists() and (proc/'stat').read_text().split()[21]==h['start_ticks']
 assert (proc/'cwd').resolve()==CWD
manifest=json.loads((ROOT/'code_manifest.json').read_text())
assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
text=(ROOT/'logs/train.log').read_text(errors='replace')[-10000:]
steps=int(re.findall(r'(\d+)/20000',text)[-1])
checkpoint=json.loads((ROOT/'inspection/checkpoint_40000_audit.json').read_text())
new=json.loads((ROOT/'preview_40000/eval/metrics.json').read_text())
old=json.loads((ROOT.parent/'1010-10-jumpingjacks_dir-PBR来源平面几何投影/preview_40000/eval/metrics.json').read_text())
gate=json.loads((ROOT/'inspection/feature_gate_gradients.json').read_text())
assert len(new['frames'])==len(old['frames'])==15
assert gate['optimizer_steps']==0 and not gate['feature_values_changed'] and not gate['mlp_weights_changed']
assert not gate['gt_labels_read'] and not gate['current_training_modified']
record=dict(status='FAILED_SO_FAR_PREVIEW_ONLY',updated_at=now,iteration=40000,
 checkpoint=checkpoint,RGB=new['metrics'],previous_RGB=old['metrics'],preview_exit_code=0,
 checkpoint_audit_exit_code=0,feature_diagnostic_exit_code=0,
 actual_reviewed_frames=['0010','0080','0150'],comparison='inspection/preview_40000_comparison.png',
 visual_review=dict(background='黑背景正常，无全屏异常',foreground='人物/平台存在，头手细节欠完整、衣服斑驳，脚周平台伪凸起/灰黑残影仍明显；B05与B06同40k差异有限。0080的GT本来背光暗，不用暗单独判失败。',
 final_four_categories='尚待55k完整四类与HDR/方向光，不用RGB预览代替完整审核'),
 final_mae=False,final_quality='PENDING',training_modified=False,
 current_training=dict(status='TRAINING_VERIFIED_LIVE',steps=steps,total_steps=20000,iteration=35000+steps,process_handles=handles),
 code_manifest_verified_files=len(manifest),commit_allowed=False)
p=ROOT/'inspection/preview_40000_review.json'; assert not p.exists();p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
gate_table='| TRAIN帧 | RGB+SH gate梯度L2 | 非零点 | 来源动态plane候选RGB+SH L2 | 候选非零点 |\n| --- | ---: | ---: | ---: | ---: |\n'
for row in gate['rows']:
 g=row['gradient_groups']['native_photo']; gate_table+=f"| {row['frame']} | {g['all']['L2']:.8g} | {g['all']['nonzero_points']} | {g['dynamic_plane_candidates']['L2']:.8g} | {g['dynamic_plane_candidates']['nonzero_points']} |\n"
body=f"""
## B06 40k真实检查点、全15RGB预览与只读gate诊断（{now}）

完整训练仍运行：{steps}/20000，实际iteration{35000+steps}；train258688/startticks9357639与post258864/startticks9362140/cwd重新核验，核心{len(manifest)}来源/GLM源码SHA匹配launch，未改变本轮训练参数或源码。preview exec25842、checkpoint审计exec1420、gate诊断exec90456均实际exit0并关闭。

检查点：全部99827 Gaussian保留，opacity/scale/feature与source32NNtensor冻结不变，XYZ最大位移{checkpoint['max_displacement']:.8f}m、canonical旋转最大{checkpoint['max_rotation_deg']:.8f}°，DC RGB范围[.031,.969]、硬clamp0，训练RGB拟合光强{checkpoint['actual_light_scale']:.8f}。native CUDA marker来源与anchor/plane绑定通过；这些只是保存完整性，不是法线7°或质量PASS。

15 TEST帧40k RGB：PSNR{new['metrics']['psnr']:.8f}、SSIM{new['metrics']['ssim']:.8f}、LPIPS{new['metrics']['lpips']:.8f}、前景PSNR{new['metrics']['psnr_foreground']:.8f}、前景bbox SSIM{new['metrics']['ssim_foreground_bbox']:.8f}/LPIPS{new['metrics']['lpips_foreground_bbox']:.8f}。B05同40k PSNR{old['metrics']['psnr']:.8f}/LPIPS{old['metrics']['lpips']:.8f}，略改善，不能由此声称GS法线改善。

已实际view inspection/preview_40000_comparison.png中GT/B05/B06的0010/0080/0150：黑背景正常、人物和平台存在，但头手细节欠完整、衣服斑驳、脚周平台伪凸起/灰黑残影仍明显，两个预测总体接近；0080 GT本身暗，不以暗单独判失败。中间质量FAILED_SO_FAR，55k完整七MAE/四类/HDR仍PENDING，未commit/push/新建分支。

只读gate诊断：source NN隐藏层与32参数固定，只开放当前feature logit的输出乘法gate自动求导；当前位置/旋转形变与原FrozenScene路径逐bit相等。0001/0075/0149三TRAIN帧PBR/GGX/BVH的RGB+SH、normal-depth和自身输入Alpha梯度均finite且存在非零梯度，source SHA/状态fingerprint不变；0optimizer、feature/NN值不变、未读GT标签/测试RGB、未生成新训练prior。本轮仍冻结gate。来源预测筛选的6836动态plane候选不是GT物体标签，梯度方向不证明它们全部属于平台。

{gate_table}

该机制可作为本轮最终失败后的新候选，仅证明可导性，未实施新训练/未声称能到7°。不能从0075的大梯度推定正确几何或把候选全部静态化；需要重新读取本GOAL、独立完整训练与全量MAE验证。

完整诊断命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u {ROOT}/commands/audit_feature_gate_gradients.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u {ROOT}/commands/audit_checkpoint.py 40000
python {ROOT}/commands/make_preview_comparison.py
```

证据：{ROOT}/inspection/feature_gate_gradients.json、checkpoint_40000_audit.json、preview_40000_review.json、preview_40000_comparison.png；logs/audit_feature_gate_gradients.log、checkpoint_40000_audit.log、preview_40000.log、preview_render_40000.log。预览完整命令另在commands/preview_40000_argv.json；所有失败轮保持原权重与统计，最终MAE尚无。
"""
for p in [ROOT/'README.md',ROOT/'README_training_progress.md',CWD/'LIHAN/DOC/训练/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复训练报告.md',CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md']:
 assert '## B06 40k真实检查点、全15RGB预览与只读gate诊断' not in p.read_text();p.write_text(p.read_text()+body)
relative=Path('LIHAN/DOC/报告/archive/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复-40k预览与gate只读审计')
archive=CWD/relative;assert not archive.exists();archive.mkdir(parents=True)
files=[ROOT/'README.md',ROOT/'README_training_progress.md',ROOT/'process_handles.json',ROOT/'code_manifest.json']
for pattern in ['inspection/checkpoint_40000_audit.json','inspection/feature_gate_gradients.json','inspection/preview_40000*','logs/*40000*','logs/audit_feature_gate_gradients.log','commands/audit_feature_gate_gradients.py','commands/audit_checkpoint.py','commands/make_preview_comparison.py','commands/archive_40000_review.py','commands/preview_40000*','preview_40000/**/*.json','preview_40000/eval/*.png']:
 files.extend(ROOT.glob(pattern))
for p in set(files):
 if not p.is_file():continue
 target=archive/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
sha={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(sha,ensure_ascii=False,indent=2)+'\n')
for other in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 for rel in [Path('LIHAN/DOC/训练/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复训练报告.md'),Path('LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md')]:
  shutil.copy2(CWD/rel,other/rel);assert hashlib.sha256((CWD/rel).read_bytes()).digest()==hashlib.sha256((other/rel).read_bytes()).digest()
 assert not (other/relative).exists();shutil.copytree(archive,other/relative)
 for key,value in sha.items():assert hashlib.sha256((other/relative/key).read_bytes()).hexdigest()==value
print(json.dumps(dict(status='PASS_ARCHIVE_SYNC_ONLY',updated_at=now,steps=steps,archive_files=len(sha),checkouts_verified=3,final_mae=False,commit_allowed=False),ensure_ascii=False))
