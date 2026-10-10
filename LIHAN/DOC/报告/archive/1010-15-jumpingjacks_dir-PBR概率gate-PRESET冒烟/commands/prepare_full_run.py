"""B09从原35k独立完整20k；概率gate功能证据通过后准备，不自动判画质。"""
from pathlib import Path
import json,shutil,hashlib,shlex,datetime
CWD=Path.cwd();SMOKE=Path(__file__).resolve().parent.parent
OLDNAME='1010-14-jumpingjacks_dir-PBR来源输出gate微调'
NAME='1010-15-jumpingjacks_dir-PBR概率gate优化'
OLD=CWD/'output'/OLDNAME;ROOT=CWD/'output'/NAME
assert json.loads((OLD/'metrics_summary.json').read_text())['status']=='FAILED_QUALITY'
assert json.loads((SMOKE/'inspection/visual_review.json').read_text())['functional_status']=='PASS'
assert 'Ran 211 tests' in (SMOKE/'logs/cpu_tests.log').read_text() and '\nOK\n' in (SMOKE/'logs/cpu_tests.log').read_text()
for name in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients']:
    assert json.loads((SMOKE/f'inspection/{name}.json').read_text())['status'].startswith('PASS')
ROOT.mkdir(exist_ok=False)
for folder in ['commands','logs','inspection']:(ROOT/folder).mkdir()
(ROOT/'data_view').symlink_to(CWD/'output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/data_view')
shutil.copy2(OLD/'gt_directions_only.json',ROOT/'gt_directions_only.json')
preset='pbr_gt_directional_plane_native_direct_probability_gate'
train=[s.replace(OLDNAME,NAME).replace('pbr_gt_directional_plane_native_direct_gate',preset)
       for s in json.loads((OLD/'train_argv.json').read_text())]
assert '--pbr_geometry_warmup' not in train and '--stage1_model_path' in train
(ROOT/'train_argv.json').write_text(json.dumps(train,ensure_ascii=False,indent=2)+'\n')
plans=json.loads((OLD/'planned_commands.json').read_text())
for item in plans:
    item['argv']=[s.replace(OLDNAME,NAME) for s in item['argv']];item['log']=item['log'].replace(OLDNAME,NAME)
(ROOT/'planned_commands.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
names=['render.sh','albedo.sh','predicted_meshes.sh','normal_depth.sh','roughness.sh','verify_roughness.sh',
       'verify_all_mae.py','verify_all_mae.sh','audit_artifacts.py','audit_artifacts.sh','audit_checkpoint.py',
       'write_results.py','write_results.sh','postprocess.py','preview_40000.py','preview_40000_argv.json',
       'make_final_rgb_comparison.py','make_final_four_categories.py','make_final_relight_comparison.py',
       'finish_manual_review.py','launch_train.py','audit_checkpoints_when_saved.py']
for name in names:
    s=(OLD/'commands'/name).read_text().replace(OLDNAME,NAME).replace('pbr_gt_directional_plane_native_direct_gate',preset)
    if name in ['audit_checkpoint.py','audit_artifacts.py']:
        s=s.replace('gate_delta.max()<=config[\'pbr_feature_gate_max_logit_delta\']+1e-5',
                    "config['pbr_feature_gate_parameterization']=='probability'")
        s=s.replace("assert (gate_delta>0).any()", "assert (gate_delta>0).any()\n"+
            (" " if name=='audit_artifacts.py' else "")+
            "from scene.pbr_feature_gate import native_probability,validate_feature_gate_model\n"+
            (" " if name=='audit_artifacts.py' else "")+
            "probability_delta=(native_probability(torch.tensor(v['fea_0'].copy()))-native_probability(torch.tensor(initial['fea_0'].copy()))).abs()\n"+
            (" " if name=='audit_artifacts.py' else "")+
            "assert float(probability_delta.max())<=config['pbr_feature_gate_max_probability_delta']+1e-6")
        line=("assert set(source_state)==set(trained_state) and all(torch.equal(source_state[k],trained_state[k]) for k in source_state)"
              if name=='audit_artifacts.py' else "assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)")
        indent=' ' if name=='audit_artifacts.py' else '';weights='trained_state' if name=='audit_artifacts.py' else 'b'
        s=s.replace(line,line+'\n'+indent+'from types import SimpleNamespace\n'+indent+'from argparse import Namespace\n'+indent+
            "validate_feature_gate_model(model,SimpleNamespace(feature=torch.tensor(v['fea_0'].copy())[:,None]),"+
            'SimpleNamespace(deform=SimpleNamespace(state_dict=lambda:'+weights+')),Namespace(**config),step)')
        s=s.replace('max_feature_logit_delta=float(gate_delta.max()),',
                    'max_feature_probability_delta=float(probability_delta.max()),max_feature_logit_delta=float(gate_delta.max()),')
        s=s.replace("'max_feature_logit_delta':float(gate_delta.max()),",
                    "'max_feature_probability_delta':float(probability_delta.max()),'max_feature_logit_delta':float(gate_delta.max()),")
    if name=='write_results.py':
        s=s.replace('B08','B09').replace('202','211').replace('来源输出gate微调','概率gate优化')
        start=s.index('## 本次来源输出gate与Preset');end=s.index('## 各项MAE：全部15测试帧',start)
        section='''## 本次概率gate与Preset

继承B08 Direct/native CUDA/sourceXYZ/原平面15187/材质/XYZ与q/loss/白色单位SUN和全局RGB拟合E。只将输出gate的Adam变量从raw logit改为概率坐标，Preset pbr_gt_directional_plane_native_direct_probability_gate；lr.001、概率来源最大偏移1（同时投影[0,1]）、弱anchor.001，geometry warmup2000/ramp1000/decay和每4步更新保持。概率逐bit初始化为原35k native gate，不使用B08/20步训练后权重。

feature optimizer组只绑定概率叶子，原32NN全部关闭梯度/optimizer。前向逐bit使用真实PLY raw feature的native sigmoid值，梯度以概率坐标传递；更新后有限逆sigmoid写回原PLY（端点±80，未改概率保留source raw原值），再同步概率叶子到实际native量化值。所有FrozenScene/GeometryRun/材质/HDR/SUN直接加载保存PLY feature，没有另存teacher/概率prior。v2来源SHA/config/概率信赖域与checkpoint marker绑定，拒绝混用v1；旧Preset默认raw方式保持。

opacity/scale/来源NN32tensor原值、全部99827点保留，无强制静态/删点/筛难像素。canonical .08m/q60°只限制canonical参数，概率改变来源NN world形变须用实际RGB/Alpha/覆盖及完整MAE验证。GT材质/法线/粗糙度/深度/mesh/blend/灯强度颜色不参与训练、初始化、loss、prior，GT只离线评分。原生完整Adam状态未保存，不声明无损resume。

211 CPU、20步真实Trainer、五参数GPU梯度/真实SUN光线预算/保存加载/共享geometry-material和150×4/4视频PASS功能。smoke实际质量FAILED_SMOKE_ONLY、未测全15七MAE；69971点native概率变化、max偏移1.85966492e−5（短步ramp），source32NN/opacity/scale保持。三TRAIN photo概率梯度L2 .015394565/.035415199/.013251405，前向与实际PLY精确一致。功能PASS不能作为完整质量或7°达标证据。

'''
        s=s[:start]+section+s[end:]
    if name in ['make_final_rgb_comparison.py','finish_manual_review.py']:
        s=s.replace('1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算',OLDNAME)
        s=s.replace('B08','TEMP_B09').replace('B07','B08').replace('TEMP_B09','B09')
        s=s.replace('B09来源输出gate','B09概率gate').replace('B08直接SUN实际预算','B08 raw gate')
        s=s.replace('B09 Direct SUN + gate','B09 Direct SUN + probability gate')
        s=s.replace('B08 Direct SUN 55000','B08 Direct SUN + raw gate 55000')
        s=s.replace('== 473','== 474').replace('473核心','474核心').replace('202 CPU','211 CPU')
    if name=='finish_manual_review.py':
        needle="quality = review['status']\n"
        s=s.replace(needle,needle+"runtime.update(quality=quality,normal_goal=normal,final_gs_normal_mae_deg=result['metrics'][2]['mae'])\n(ROOT/'training_runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\\n')\n")
    (ROOT/'commands'/name).write_text(s)
(ROOT/'commands/train.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(CWD))+
    '\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4\n'+shlex.join(train)+'\n')
manifest=json.loads((SMOKE/'code_manifest.json').read_text());assert len(manifest)==474
for relative,sha in manifest.items():
    p=CWD/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    for folder in ['code_snapshot','code_snapshot_train_launch']:
        dst=ROOT/'commands'/folder/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for name in ['code_manifest.json','code_manifest_train_launch.json']:(ROOT/name).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for name in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients','visual_review']:
    shutil.copy2(SMOKE/f'inspection/{name}.json',ROOT/f'inspection/smoke_{name}.json')
intro='''# 1010-15 jumpingjacks dir 原生PBR概率gate完整Stage2训练计划

状态PREPARED，未启动完整B09，无正式MAE/7°或质量结论。已先读GOAL，B08完整20k/七MAE/实际质量FAILED：GS16.84133987°/覆盖95.01267352%，失败1055文件归档保留。sigmoid饱和诊断及211 CPU/真实20步/GPU/共享buffer/4视频功能PASS，20步实际质量FAILED_SMOKE_ONLY。

只从原Stage1 source35000→55000独立20000步，仅Stage2，135TRAIN/15TEST、400²/黑背景、原相机time。输入仅原RGB/自身RGBA Alpha、source35k PLY/MLP、GT方向白色单位；不用GT材质/法线/粗糙度/深度/mesh/blend/灯强度颜色训练或prior。GT仅离线评测。旧source转换Alpha来源已披露、150帧原/新/旧Alpha bit一致，不重训Stage1。

Preset pbr_gt_directional_plane_native_direct_probability_gate：继承B08 Direct/native CUDA/sourceXYZ、固定plane15187、DC内部投影、XYZ/q/loss、全局RGB拟合E，只有gate优化坐标不同。211 CPU/GPU短训核验，无后训练源码改动；474核心双snapshot。概率叶子native源概率逐bit初始化，feature Adam组仅该叶子，源NN32tensor/opacity/scale冻结；前向精确使用PLY native sigmoid，更新后概率来源±1∩[0,1]投影，有限逆sigmoid存PLY并同步native量化。lr.001/anchor.001、正式warmup2000/ramp1000/decay/每4步保持，v2 metadata/marker SHA绑定真实source。旧Preset raw/v1兼容。没有概率teacher/GT prior；canonical .08m/q60°不保证gate引起的world形变有相同上限，须实际全域验收。未存完整Adam状态，不声明无损resume。

保存40k/50k/55k；150×4四类/HDR150×2048/SUN72/6视频。全部15 TEST七MAE+coverage/common/missing180原评分：GTalpha≥.5全前景pixel-weighted有向GS；Expected/Median camera Z米不拟合，Mesh每个目标time全部135 TRAIN预测深度融合，不用GT/测试位姿重建。全部点/帧/难像素保留，源opacity/scale不优化。未通过质量前不commit/push/新目标分支。

## 完整命令与日志

'''
intro+='### train\n\n```bash\n'+shlex.join(train)+'\n```\n\n日志：'+str(ROOT/'logs/train.log')+'\n'
for item in plans:intro+='\n### '+item['name']+'\n\n```bash\n'+shlex.join(item['argv'])+'\n```\n\n日志：'+item['log']+'\n'
intro+='\n## 目检与代表产物：PENDING\n\n四类/前后景及Normal/Depth/Mesh/HDR/SUN待55k实际查看；model_mlp/renders_stage2/ours_55000/insights/{rgb,alpha,normal,albedo}各150/4视频，hdr150/directional72/2视频；inspection/{rgb_final_55000_comparison,four_categories_final_55000,relight_final_55000}.png与normal_depth/comparison_0010/0080/0150.png。失败权重/浮点/日志/图片全部保留，无质量或7°PASS。\n'
(ROOT/'README.md').write_text(intro);(ROOT/'README_training_progress.md').write_text('')
doc=CWD/'LIHAN/DOC/训练'/(NAME+'训练报告.md');doc.write_text(intro)
goal=CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:f.write('\n## B09完整20k准备（'+datetime.datetime.now().astimezone().isoformat()+'）\n\n'+intro.split('## 完整命令与日志')[0]+'\n全部命令见训练文档和planned_commands/train_argv，尚未启动。\n')
index=CWD/'LIHAN/README.md'
with index.open('a') as f:f.write('\n- [B09概率gate完整Stage2](DOC/训练/'+doc.name+')：211 CPU/20步/五参数GPU/共享buffer/4视频功能PASS，独立20k PREPARED，全15七MAE/质量待完成。\n')
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
    for p in [doc,goal,index]:shutil.copy2(p,checkout/p.relative_to(CWD))
print(json.dumps(dict(status='PREPARED',run=str(ROOT),source_files=len(manifest)),ensure_ascii=False),flush=True)
