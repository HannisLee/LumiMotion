"""B08固定35k独立完整20k；B07失败结果保留，原评分不改。"""
from pathlib import Path
import json,shutil,hashlib,shlex,datetime
CWD=Path.cwd();SMOKE=Path(__file__).resolve().parent.parent
OLDNAME='1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算'
NAME='1010-14-jumpingjacks_dir-PBR来源输出gate微调'
old=CWD/'output'/OLDNAME;root=CWD/'output'/NAME
for n in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients']:
 assert json.loads((SMOKE/f'inspection/{n}.json').read_text())['status'].startswith('PASS')
assert json.loads((SMOKE/'inspection/visual_review.json').read_text())['functional_status']=='PASS'
assert 'Ran 202 tests' in (SMOKE/'logs/cpu_tests_retry.log').read_text() and '\nOK\n' in (SMOKE/'logs/cpu_tests_retry.log').read_text()
assert json.loads((old/'metrics_summary.json').read_text())['status']=='FAILED_QUALITY'
root.mkdir(exist_ok=False)
for d in ['commands','logs','inspection']:(root/d).mkdir()
(root/'data_view').symlink_to(CWD/'output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/data_view')
shutil.copy2(old/'gt_directions_only.json',root/'gt_directions_only.json')
train=[x.replace(OLDNAME,NAME).replace('pbr_gt_directional_plane_native_direct','pbr_gt_directional_plane_native_direct_gate') for x in json.loads((old/'train_argv.json').read_text())]
(root/'train_argv.json').write_text(json.dumps(train,ensure_ascii=False,indent=2)+'\n')
plans=json.loads((old/'planned_commands.json').read_text())
for p in plans:
 p['argv']=[x.replace(OLDNAME,NAME) for x in p['argv']];p['log']=p['log'].replace(OLDNAME,NAME)
(root/'planned_commands.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
names=['render.sh','albedo.sh','predicted_meshes.sh','normal_depth.sh','roughness.sh','verify_roughness.sh',
 'verify_all_mae.py','verify_all_mae.sh','audit_artifacts.py','audit_artifacts.sh','write_results.py','write_results.sh',
 'postprocess.py','preview_40000.py','audit_checkpoint.py','make_final_rgb_comparison.py',
 'make_final_four_categories.py','make_final_relight_comparison.py','finish_manual_review.py']
for name in names:
 s=(old/'commands'/name).read_text().replace(OLDNAME,NAME)
 if name in ['audit_checkpoint.py','audit_artifacts.py']:
  s=s.replace("['opacity','fea_0']","['opacity']")
  s=s.replace("config['loss_preset']=='pbr_gt_directional_plane_native_direct'","config['loss_preset']=='pbr_gt_directional_plane_native_direct_gate'")
  needle="assert all(np.array_equal(v[f],initial[f]) for f in frozen)"
  lines=s.splitlines();out=[]
  for line in lines:
   if needle in line:
    indent=line[:len(line)-len(line.lstrip())]
    line=line.replace("'opacity/scale/feature应冻结'","'opacity/scale应冻结'")
    out.append(line)
    out += [indent+x for x in ["gate_delta=np.abs(v['fea_0']-initial['fea_0'])",
     "assert config['pbr_optimize_feature_gate'] and gate_delta.max()<=config['pbr_feature_gate_max_logit_delta']+1e-5",
     "assert (gate_delta>0).any()",
     "gate_record=json.loads((model/'pbr_feature_gate.json').read_text())",
     "assert gate_record['source_MLP_frozen'] and gate_record['source_files']==source" if name=='audit_artifacts.py' else "assert gate_record['source_MLP_frozen']",
     "gate_marker=json.loads((model/f'pbr_directional/iteration_{step}/lighting.json').read_text())",
     "assert gate_marker['feature_gate_metadata_sha256']==hashlib.sha256((model/'pbr_feature_gate.json').read_bytes()).hexdigest()"]]
   else:out.append(line)
  s='\n'.join(out)+'\n'
  s=s.replace('opacity_scale_feature_frozen=True','opacity_scale_frozen=True,feature_optimized=True,max_feature_logit_delta=float(gate_delta.max()),feature_changed_points=int((gate_delta>0).sum())')
  s=s.replace("'opacity_scale_feature_unchanged':True","'opacity_scale_unchanged':True,'feature_optimized':True,'max_feature_logit_delta':float(gate_delta.max()),'feature_changed_points':int((gate_delta>0).sum())")
  s=s.replace('opacity/scale/feature冻结','opacity/scale冻结与feature gate信赖域')
 if name=='write_results.py':
  s=s.replace('1010-13 jumpingjacks dir 原生PBR直接SUN与真实光线预算','1010-14 jumpingjacks dir 原生PBR来源输出gate微调')
  s=s.replace('B07全量结果','B08全量结果').replace('报告1010-13文档','报告1010-14文档')
  s=s.replace('194','202')
  s=s.replace('opacity、尺度和二值静动态参数冻结。','opacity、尺度冻结；feature原值初始化并优化输出gate，来源MLP全部32tensor冻结。')
  s=s.replace('feature/time/camera不变，源32tensor保持。','time/camera保持，feature gate按训练RGB/Alpha/预测ND优化，源32 MLP tensor保持。')
  s=s.replace('本次冻结静动态参数。','本次仅微调输出gate，32来源MLP tensor冻结。')
  s=s.replace('opacity/scale/feature冻结','opacity/scale冻结、feature gate信赖域微调')
  s=s.replace('本次Plane Native Direct Preset继承B06全部材质/几何参数','本次Plane Native Direct Gate Preset继承B07全部材质/几何参数')
  s=s.replace('本轮继承B06原生CUDA及Plane模型，只改关闭SH间接近似和实际SUN像素预算；材质/光强/loss/lr/几何参数保持。','本轮继承B07 Direct Native/Plane，只新增feature输出gate优化；其它材质/光强/loss/lr/几何参数保持。')
  start=s.index('## 本次Direct SUN与真实光线预算');end=s.index('## 各项MAE：全部15测试帧',start)
  section='''## 本次来源输出gate与Preset

Preset pbr_gt_directional_plane_native_direct_gate继承B07，显式pbr_optimize_feature_gate=True；全99827点feature原值初始化，source MLP全部32 tensor关闭梯度和optimizer，sourceXYZ输入冻结。原native Deform乘法gate可导；geometry warmup2000/ramp1000/decay同步gate，lr.001、source raw-logit最大偏移4、anchor.001，每4步Adam后投影source±4。不强制source候选当静态，不筛/删Gaussian或难像素，旧Preset flagFalse保持。固定source plane15187与B07相同；额外world plane投影不等于canonical .08m。

gate metadata保存来源SHA/config、checkpoint marker绑定其SHA，FrozenScene/GeometryRun/材质helper统一核验source32tensor、gate finite/trust/source配置。全部渲染/离线评测用保存PLY feature，不退回source gate。MLP hidden冻结、只有原生输出乘法feature梯度；原GGX/BVH/native raster/Direct实际SUN预算及原SH辅助loss不改。Directions白色单位×训练RGB拟合scale，无GT光强/颜色。原生Adam状态未完整保存，不声明无损resume。

202 CPU回归实际exit0，20步真实Trainer及sourceNN/gate保存检查、五参数GPU梯度/实际SUN计数/共享geometry-material逐buffer与目标time一致PASS。GPU photo gate L2三帧.00085692690/.00269499072/.00030556315；保存68101 feature变化、maxlogit1.85966492e−5，source32tensor/opacity/scale保持。首次201项中2个mock缺新增字段错误日志保留，仅修fixture/增metadata测试后202通过，训练生产源码未改。全部四类150/4视频及HDR16/SUN仅功能smoke，实际质量FAILED_SMOKE_ONLY，无正式7°指标。

独立完整35k→55k验证gate是否改善，不使用20步smoke或B07训练后权重。全15七MAE/coverage/common/missing180和四类、Depth/Mesh/HDR/SUN按原口径验收。GT材质/法线/几何只能离线评测；模型、输入、失败产物保留。

'''
  s=s[:start]+section+s[end:]
  s=s.replace("ROOT/'model_mlp/pbr_albedo_initialization.json',","ROOT/'model_mlp/pbr_albedo_initialization.json',ROOT/'model_mlp/pbr_feature_gate.json',")
 if name=='make_final_rgb_comparison.py':
  s=s.replace('1010-12-jumpingjacks_dir-PBR仓库原生光栅修复',OLDNAME)
  s=s.replace('B07','TEMP_NEW').replace('B06','B07').replace('TEMP_NEW','B08')
 if name=='finish_manual_review.py':
  s=s.replace("OLD = ROOT.parent/'1010-12-jumpingjacks_dir-PBR仓库原生光栅修复'", "OLD = ROOT.parent/"+repr(OLDNAME))
  s=s.replace('== 471','== 473').replace('471核心','473核心')
  s=s.replace('B07','TEMP_NEW').replace('B06','B07').replace('TEMP_NEW','B08')
  s=s.replace('B07原生+间接光','B07直接SUN实际预算').replace('B08直接SUN实际预算','B08来源输出gate')
  s=s.replace('194 CPU/20步Direct Preset','202 CPU/20步Gate Preset').replace('材质/XYZ/q梯度','材质/XYZ/q/feature梯度')
 (root/'commands'/name).write_text(s)
(root/'commands/train.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(CWD))+'\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4\n'+shlex.join(train)+'\n')
manifest=json.loads((SMOKE/'code_manifest.json').read_text())
for rel,h in manifest.items():
 p=CWD/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 for folder in ['code_snapshot','code_snapshot_train_launch']:
  q=root/'commands'/folder/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for n in ['code_manifest.json','code_manifest_train_launch.json']:(root/n).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for n in ['geometry_buffers','pbr_gradients','material_buffers','gate_checkpoint','visual_review']:
 shutil.copy2(SMOKE/f'inspection/{n}.json',root/f'inspection/smoke_{n}.json')
intro='''# 1010-14 jumpingjacks dir 原生PBR来源输出gate完整Stage2训练计划

状态PREPARED，完整B08未开训，无B08正式MAE/质量结论。已读GOAL及B07最终GS16.81871120°/覆盖95.00328573%、质量FAILED，全部失败产物保存。只新增原生feature输出gate，source35k MLP32tensor/XYZ输入冻结；本轮source/model35k→55000完整20000步，仅Stage2，135TRAIN/15TEST、400×400、黑背景、相机time原值。

输入仅原RGB/自身RGBA Alpha、源35k预测和GT方向白色单位，不用GT材质/法线/粗糙度/深度/mesh/blend/强度/颜色初始化/loss/prior；GT仅离线评分。源35k来自旧转换，但150帧原/新/旧Alpha逐bit一致已披露，不重训Stage1。

Preset pbr_gt_directional_plane_native_direct_gate：继承B07 Direct/native kernel/sourceXYZ/15187 source plane/DC内部投影/原geometry/loss/lr与全局RGB拟合E。只新增feature optimizer组，source32 MLP tensor冻结、opacity/scale冻结，全部99827点保留。原feature初始化，geometry warmup2000/ramp1000/decay同步，gate_lr.001、max_source_logit_delta4、weakanchor.001，每4步Adam投影source±4。gate metadata/lighting marker绑定来源SHA、加载实际gate而非source，FrozenScene/GeometryRun/材料helper共享。canonicalXYZ .08m/q60°保持，额外plane world投影不受canonical边界。原生没有完整Adam状态，不声明无损resume。

202 CPU retry/真实20步/GPU五参数梯度/源32tensor与gate保存/实际SUN预算/共享buffer PASS；初次201项2个旧mock缺字段错误只修fixture后202通过，原生产训练core保持。全部150四类及4视频解码PASS，实际smoke质量FAILED，只验证功能，无正式MAE。完整run从35k独立训练，不沿用smoke或B07权重。

保存40k/50k/55k；全量四类150×4、HDR150×2048、方向72、6视频。全部15帧七MAE，原GS有向world normal GTalpha≥.5全前景pixel-weighted，coverage/common/missing180保持；Depth Z米不拟合，Mesh每个测试time所有135训练姿态模型预测深度独立融合，无GT/testpose重建。质量未通过前不commit/push/新目标分支。

## 完整命令与日志

'''
intro+='### train\n\n```bash\n'+shlex.join(train)+'\n```\n\n日志：'+str(root/'logs/train.log')+'\n'
for p in plans:intro+='\n### '+p['name']+'\n\n```bash\n'+shlex.join(p['argv'])+'\n```\n\n日志：'+p['log']+'\n'
intro+='\n## 四类目检与代表产物：PENDING\n\nRGB/背景、Alpha、Albedo/静动态、Normal/Depth/Mesh及HDR/SUN待55k实际审核；model_mlp/renders_stage2/ours_55000/insights/{rgb,alpha,normal,albedo}各150及4视频；hdr150/directional72及2视频。inspection/{rgb_final_55000_comparison,four_categories_final_55000,relight_final_55000}.png；normal_depth/comparison_0010/0080/0150.png。质量/7°未PASS。473核心/GLM双snapshot SHA保留，所有失败权重/图片/浮点/日志保留。\n'
(root/'README.md').write_text(intro);(root/'README_training_progress.md').write_text('')
doc=CWD/'LIHAN/DOC/训练'/(NAME+'训练报告.md');doc.write_text(intro)
goal=CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:f.write('\n## B08完整训练准备（'+datetime.datetime.now().astimezone().isoformat()+'）\n\n'+intro.split('## 完整命令与日志')[0]+'\n完整命令见训练报告及train_argv/planned_commands，尚未开训。\n')
index=CWD/'LIHAN/README.md'
with index.open('a') as f:f.write('\n- [B08来源gate完整Stage2](DOC/训练/'+doc.name+')：202 CPU/20步/五参数GPU/共享buffer PASS，独立20k PREPARED，最终全15七MAE/质量待完成。\n')
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 for p in [doc,goal,index]:shutil.copy2(p,checkout/p.relative_to(CWD))
print(json.dumps(dict(status='PREPARED',run=str(root),source_files=len(manifest)),ensure_ascii=False),flush=True)
