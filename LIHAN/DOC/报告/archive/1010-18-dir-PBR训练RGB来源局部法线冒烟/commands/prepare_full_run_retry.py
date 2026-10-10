"""功能冒烟实际落档后准备B10原35k独立20k，完整评测沿用原口径。"""
from pathlib import Path
import json,shutil,hashlib,shlex,datetime,ast
CWD=Path.cwd();SMOKE=Path(__file__).resolve().parent.parent
OLDNAME='1010-15-jumpingjacks_dir-PBR概率gate优化';NAME='1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束'
OLD=CWD/'output'/OLDNAME;ROOT=CWD/'output'/NAME
GOAL=CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
assert 'B10真实Preset功能冒烟' in GOAL.read_text()
assert json.loads((OLD/'metrics_summary.json').read_text())['status']=='FAILED_QUALITY'
assert json.loads((SMOKE/'inspection/visual_review.json').read_text())['functional_status']=='PASS'
assert 'Ran 226 tests' in (SMOKE/'logs/cpu_tests.log').read_text() and '\nOK\n' in (SMOKE/'logs/cpu_tests.log').read_text()
for name in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients']:
 assert json.loads((SMOKE/f'inspection/{name}.json').read_text())['status'].startswith('PASS')
ROOT.mkdir(exist_ok=False)
for folder in ['commands','logs','inspection']:(ROOT/folder).mkdir()
(ROOT/'data_view').symlink_to(CWD/'output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/data_view')
shutil.copy2(OLD/'gt_directions_only.json',ROOT/'gt_directions_only.json')
preset='pbr_gt_directional_source_rgb_normal'
train=[s.replace(OLDNAME,NAME).replace('pbr_gt_directional_plane_native_direct_probability_gate',preset) for s in json.loads((OLD/'train_argv.json').read_text())]
train+=['--pbr_rgb_normal_prior_path',str(CWD/'output/smoke_test/1010-17-dir-训练RGB法线约束原型/candidate_assets')]
assert '--pbr_geometry_warmup' not in train and train[train.index('--load_iter')+1]=='35000' and train[train.index('--iterations')+1]=='55000'
(ROOT/'train_argv.json').write_text(json.dumps(train,ensure_ascii=False,indent=2)+'\n')
plans=json.loads((OLD/'planned_commands.json').read_text());assert len(plans)==9
for item in plans:
 item['argv']=[s.replace(OLDNAME,NAME) for s in item['argv']];item['log']=item['log'].replace(OLDNAME,NAME)
(ROOT/'planned_commands.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
section="""## 本次训练RGB来源局部法线与Preset

继承B09概率gate、仓库原生Direct SUN/GGX/BVH与SH辅助、材质、真实XYZ/q、信赖域、source-only plane15187及原photo/Alpha/ND损失和全部lr。独立Preset pbr_gt_directional_source_rgb_normal，新增pbr_source_local_rotation=True与lambda_pbr_rgb_normal_prior=2；正式warmup2000/ramp1000。source q/raw norm从原35k保存，固定NN32 tensor按当前gate查询来源world旋转，再按R_source_world(p) R_source_canonical^T R_current_canonical运输，随后执行既有plane。初值原d_rotation逐bit保持。实际Gaussian covariance、normal、depth、PBR/BVH共享真实q，绝非评测法线图覆盖。周期预览/FrozenScene/GeometryRun/Mesh/material/HDR/SUN都使用相同入口和顺序。

附加canonical轴loss全99827点平均confidence*(1-dot)，29621非plane点有候选支持（来源static17389/dynamic12232）。来自原135TRAIN RGB线性luminance与自身Alpha、来源预测可见性、源35k qXYZ/固定NN/相机time/白色单位GT方向的独立LS候选；9个全点数组、source6 SHA/135逐图SHA、TRAINkeys/TEST0、finite与CPU全点Gram/解/残差/confidence复算绑定模型。该候选是训练RGB派生，不是GT法线，也不保证GGX/阴影准确；其confidence只约束新增loss，不改变原photo、完整点/帧/难像素或原评分。既有plane点没有重复该轴约束，也没有删除。

模型保存pbr_rgb_normal{,_input}.{npz,json}，source quaternion/target/confidence和原独立input本地复制、各SHA与checkpoint metadataSHA绑定，所有加载严格复算并检查源32NN冻结。新增0 optimizer参数，仍用原canonical rotation Adam raw q lr.001/norm和60度信赖域；probability gate/native sigmoid/有限逆sigmoid原实现不变。canonical .08m不保证gate/plane引起的实际world位移同界。未保存完整Adam状态，不声明无损resume。

226 CPU、独立20步35000→35020实际Trainer/保存加载/GPU五参数梯度/共享buffer/HDR16/SUN/四类150×4和4视频全解码功能PASS，实际画质FAILED_SMOKE_ONLY；RGB约束loss .102662175894、q梯度L2 .00525612570345非零。476核心与双snapshot保持。此完整20k从原35k初始化，不续20步或B09学得checkpoint。GT材质/normal/roughness/depth/mesh/blend/原灯强度颜色全部不参与训练/初始化/约束；仅独立离线评测读取GT标签，原有完整七MAE/覆盖/common/missing180定义保持。

"""
names=['render.sh','albedo.sh','predicted_meshes.sh','normal_depth.sh','roughness.sh','verify_roughness.sh','verify_all_mae.py','verify_all_mae.sh','audit_artifacts.py','audit_artifacts.sh','audit_checkpoint.py','write_results.py','write_results.sh','postprocess.py','preview_40000.py','preview_40000_argv.json','make_final_rgb_comparison.py','make_final_four_categories.py','make_final_relight_comparison.py','finish_manual_review.py','launch_train.py','audit_checkpoints_when_saved.py','audit_launch.py']
for name in names:
 s=(OLD/'commands'/name).read_text()
 if name in ['make_final_rgb_comparison.py','finish_manual_review.py','audit_launch.py']:
  s=s.replace('1010-14-jumpingjacks_dir-PBR来源输出gate微调','OLD_RUN_KEEP_NAME')
 s=s.replace(OLDNAME,NAME).replace('OLD_RUN_KEEP_NAME',OLDNAME).replace('pbr_gt_directional_plane_native_direct_probability_gate',preset)
 if name in ['write_results.py','finish_manual_review.py','make_final_rgb_comparison.py','audit_launch.py']:
  s=s.replace('B09','TEMP_B10').replace('B08','B09').replace('TEMP_B10','B10')
  s=s.replace('211 CPU','226 CPU').replace('211项CPU','226项CPU').replace('211 CPU','226 CPU').replace('474核心','476核心').replace('== 474','== 476').replace('==474','==476')
 if name=='write_results.py':
  s=s.replace('原生PBR概率gate优化','原生PBR训练RGB来源局部法线约束').replace('# 1010-15 jumpingjacks dir','# 1010-16 jumpingjacks dir').replace('训练/报告1010-15文档','训练/报告1010-16文档').replace('本次Plane Native Direct Probability Gate Preset','本次Source RGB Normal Preset')
  start=s.index('## 本次概率gate与Preset');end=s.index('## 各项MAE：全部15测试帧',start)
  s=s[:start]+section+s[end:]
  s=s.replace('改为feature概率坐标优化；其它材质/光强/loss/lr/几何参数保持。','新增source-local真实旋转与lambda2训练RGB派生轴约束；其它材质/光强/原loss/lr保持。')
  s=s.replace('只将输出gate','继续优化输出gate')
  s=s.replace('source_plane_diagnostic_only=False)',"source_plane_diagnostic_only=False,source_RGB_derived_normal_constraint=True,RGB_constraint_weight=2.0,source_local_world_rotation=True)")
  needle="for folder in ['commands','logs','inspection']:"
  s=s.replace(needle,"for name in ['pbr_rgb_normal.npz','pbr_rgb_normal.json','pbr_rgb_normal_input.npz','pbr_rgb_normal_input.json']:\n shutil.copy2(ROOT/'model_mlp'/name,archive/name)\n"+needle)
 if name in ['audit_checkpoint.py','audit_artifacts.py']:
  # 同一个真实共享加载helper核验全部资产/输入复算与marker。此检查只读、0 optimizer。
  indent=' ' if name=='audit_artifacts.py' else ''
  needle=indent+"validate_feature_gate_model(model,SimpleNamespace(feature=torch.tensor(v['fea_0'].copy())[:,None]),SimpleNamespace(deform=SimpleNamespace(state_dict=lambda:"+('trained_state' if name=='audit_artifacts.py' else 'b')+")),Namespace(**config),step)"
  assert needle in s,name
  s=s.replace(needle,needle+'\n'+indent+'from audit_rgb_normal import audit_rgb_normal\n'+indent+'rgb_normal=audit_rgb_normal(model,step)')
  if name=='audit_checkpoint.py':
   s=s.replace("record=dict(status=", "record=dict(rgb_normal=rgb_normal,status=")
  else:s=s.replace("checkpoints[str(step)]={'gaussians':", "checkpoints[str(step)]={'rgb_normal':rgb_normal,'gaussians':")
 if name=='finish_manual_review.py':
  s=s.replace('B09 raw gate','B09概率gate').replace('B10概率gate','B10训练RGB轴约束')
  s=s.replace('226 CPU/20步Gate Preset','226 CPU/20步RGB轴Preset')
  needle="assert config['pbr_directional_actual_ray_budget'] and config['wo_indirect'] and config['wo_indirect_relight']"
  s=s.replace(needle,needle+"\nassert config['pbr_source_local_rotation'] and config['lambda_pbr_rgb_normal_prior']==2.0\nassert json.loads((ROOT/'model_mlp/pbr_rgb_normal.json').read_text())['confidence_points']==29621")
 if name=='make_final_rgb_comparison.py':
  s=s.replace('B10 Direct SUN + probability gate','B10 Direct SUN + source RGB normal').replace('B09 Direct SUN + raw gate','B09 Direct SUN + probability gate')
 if name=='audit_launch.py':
  s=s.replace("expected={'source_path','model_path','loss_preset','pbr_gt_lights_path','pbr_feature_gate_parameterization','pbr_feature_gate_max_probability_delta'}", "expected={'source_path','model_path','loss_preset','pbr_gt_lights_path','pbr_source_local_rotation','pbr_rgb_normal_prior_path','lambda_pbr_rgb_normal_prior'}")
  # B09当时不含3个新增legacy字段，其未出现与False/空/0当作同义default；显式差异仍应是预期7个。
  s=s.replace("diff={k:","for key,value in {'pbr_source_local_rotation':False,'pbr_rgb_normal_prior_path':'','lambda_pbr_rgb_normal_prior':0.0}.items():before.setdefault(key,value)\ndiff={k:")
  s=s.replace("assert cfg['pbr_optimize_feature_gate']", "assert cfg['pbr_source_local_rotation'] and cfg['lambda_pbr_rgb_normal_prior']==2.0\nassert cfg['pbr_optimize_feature_gate']")
  s=s.replace("result=dict(status=", "rgb=json.loads((model/'pbr_rgb_normal.json').read_text())\nassert rgb['confidence_points']==29621 and rgb['constraint_weight']==2.0 and rgb['source_MLP_frozen']\nassert all(rgb[k] is False for k in ['GT_normal_inputs','GT_material_inputs','GT_geometry_inputs','GT_light_photometry_inputs'])\nfor key,filename in [('asset_sha256','pbr_rgb_normal.npz'),('input_asset_sha256','pbr_rgb_normal_input.npz'),('input_metadata_sha256','pbr_rgb_normal_input.json')]:assert hashlib.sha256((model/filename).read_bytes()).hexdigest()==rgb[key]\nresult=dict(rgb_normal=rgb,status=")
 (ROOT/'commands'/name).write_text(s)
helper='''"""实际FrozenScene source-local RGB模型只读加载；不优化、不读GT。"""
from pathlib import Path
import sys,json,numpy as np,torch
CWD=Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0,str(CWD))
def audit_rgb_normal(model,step):
 from scripts.render_stage2 import Stage2RenderRun
 run=Stage2RenderRun(model,step);ctx=run.context;cfg=vars(run.args)
 assert cfg['pbr_source_local_rotation'] and cfg['lambda_pbr_rgb_normal_prior']==2.0
 assert len(ctx.pc.get_xyz)==99827 and len(ctx.train_cameras)==135 and len(ctx.test_cameras)==15
 assert ctx.pbr_rgb_normal is not None and int(ctx.pbr_rgb_normal.confidence.sum())==29621
 assert not any(p.requires_grad for p in ctx.deform.deform.parameters())
 record=json.loads((Path(model)/'pbr_rgb_normal.json').read_text())
 assert record['points']==99827 and record['confidence_points']==29621 and record['constraint_weight']==2.0
 for name in ['pbr_rgb_normal.npz','pbr_rgb_normal_input.npz']:
  with np.load(Path(model)/name,allow_pickle=False) as values:
   assert all(np.isfinite(values[k]).all() for k in values.files)
 with torch.no_grad():value=float(ctx.pbr_rgb_normal.loss(ctx.pc))
 assert np.isfinite(value)
 ctx.verify_frozen()
 result=dict(status='PASS_SHARED_RGB_MODEL_LOAD',points=99827,confidence_points=29621,
  constraint_weight=2.0,canonical_RGB_axis_loss=value,source_MLP_frozen=True,
  source_RGB_derived=True,GT_labels_read=False,optimizer_steps=0,source_files=ctx.source_files)
 del ctx,run;torch.cuda.empty_cache()
 return result
'''
(ROOT/'commands/audit_rgb_normal.py').write_text(helper)
(ROOT/'commands/train.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(CWD))+'\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4\n'+shlex.join(train)+'\n')
for p in (ROOT/'commands').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
manifest=json.loads((SMOKE/'code_manifest.json').read_text());assert len(manifest)==476
for relative,sha in manifest.items():
 p=CWD/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 for folder in ['code_snapshot','code_snapshot_train_launch']:
  dst=ROOT/'commands'/folder/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for name in ['code_manifest.json','code_manifest_train_launch.json']:(ROOT/name).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for name in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients','visual_review']:
 shutil.copy2(SMOKE/f'inspection/{name}.json',ROOT/f'inspection/smoke_{name}.json')
intro='# '+NAME+' 完整Stage2训练计划\n\n状态PREPARED，完整训练未启动，正式七MAE及7度和质量结论待执行。B09完整20k原GS MAE15.94231363度/覆盖95.47845260%，实际质量FAILED，全部失败产物保留。已读GOAL；B10实际226 CPU/20步/GPU/保存加载/共享buffer/4视频功能PASS，画质FAILED_SMOKE_ONLY。\n\n'+section
intro+='只从原Stage1 source35000→55000独立20000步，仅Stage2，135TRAIN/15TEST、400²/黑背景/原相机time。原RGB/自身Alpha/预测35k/GT direction白色单位，新增轴约束由135TRAIN RGB估计，没有GT normal/albedo/roughness/depth/mesh/blend/原光强颜色。sourceplane15187、全部99827点/原photo/评分保持。正式warmup2000/ramp1000，保存40k/50k/55k。训练初始化没有使用候选替换source q或学得B09权重。\n\n'
intro+='## 完整命令与日志\n\n### train\n\n```bash\n'+shlex.join(train)+'\n```\n\n日志：'+str(ROOT/'logs/train.log')+'\n'
for item in plans:intro+='\n### '+item['name']+'\n\n```bash\n'+shlex.join(item['argv'])+'\n```\n\n日志：'+item['log']+'\n'
intro+='\n## 可视化和验收：PENDING\n\n渲染model_mlp/renders_stage2/ours_55000/insights/{rgb,alpha,normal,albedo}各150帧/4视频；HDR150帧×2048samples和SUN72帧/2视频。全15TEST七MAE与coverage/common/missing180原评分；GS有向world shading normal、GTalpha≥.5全前景pixel-weighted，不按物体筛选；Expected/Median正camera Z米不拟合；Mesh15targettime各135TRAIN预测深度重建，不用GT或TEST位姿重建。代表图inspection/{rgb_final_55000_comparison,four_categories_final_55000,relight_final_55000}.png和normal_depth/comparison_0010/0080/0150.png。四类/HDR/SUN实际目检与6视频全解码待执行。失败checkpoint/浮点/图/日志保留。未通过质量不commit/push/新目标分支，不提前声称7度。\n'
(ROOT/'README.md').write_text(intro);(ROOT/'README_training_progress.md').write_text('')
doc=CWD/'LIHAN/DOC/训练'/(NAME+'训练报告.md');doc.write_text(intro)
with GOAL.open('a') as f:f.write('\n## B10独立完整20k准备（'+datetime.datetime.now().astimezone().isoformat()+'）\n\n'+intro.split('## 完整命令与日志')[0]+'\n全部9步原评分命令见train_argv/planned_commands/训练文档，完整训练尚未启动。\n')
index=CWD/'LIHAN/README.md'
with index.open('a') as f:f.write('\n- [B10训练RGB来源局部法线完整Stage2](DOC/训练/'+doc.name+')：226 CPU/20步/共享功能PASS，独立原35k→55k PREPARED，全15七MAE和质量待执行。\n')
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 for p in [doc,GOAL,index]:shutil.copy2(p,checkout/p.relative_to(CWD))
# smoke终端exit0以后补齐归档中的结束log，逐SHA同步。
archive=CWD/'LIHAN/DOC/报告/archive/1010-18-dir-PBR训练RGB来源局部法线冒烟'
shutil.copy2(SMOKE/'logs/finish_smoke.log',archive/'logs/finish_smoke.log')
m={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
(archive/'archive_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
for checkout in [CWD,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 if checkout!=CWD:shutil.copytree(archive,checkout/archive.relative_to(CWD),dirs_exist_ok=True)
 for k,v in m.items():assert hashlib.sha256((checkout/archive.relative_to(CWD)/k).read_bytes()).hexdigest()==v
print(json.dumps(dict(status='PREPARED',run=str(ROOT),source_files=len(manifest),smoke_archive_files=len(m),smoke_SHA_verified_checkouts=3),ensure_ascii=False),flush=True)
