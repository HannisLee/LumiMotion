"""B07从固定35k独立完整20k，继承B06评测且只改变Direct预算。"""
from pathlib import Path
import json, shutil, hashlib, shlex
from datetime import datetime, timezone, timedelta

CWD = Path.cwd()
SMOKE = Path(__file__).resolve().parent.parent
OLDNAME = '1010-12-jumpingjacks_dir-PBR仓库原生光栅修复'
NAME = '1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算'
old = CWD/'output'/OLDNAME
root = CWD/'output'/NAME
for name in ['geometry_buffers', 'pbr_gradients', 'material_buffers']:
    assert json.loads((SMOKE/f'inspection/{name}.json').read_text())['status'].startswith('PASS')
assert json.loads((SMOKE/'inspection/visual_review.json').read_text())['functional_status'] == 'PASS'
assert 'Ran 194 tests' in (SMOKE/'logs/cpu_tests.log').read_text()
assert '\nOK\n' in (SMOKE/'logs/cpu_tests.log').read_text()
assert 'Training complete.' in (SMOKE/'logs/train.log').read_text()
root.mkdir(exist_ok=False)
for folder in ['commands', 'logs', 'inspection']:
    (root/folder).mkdir()
(root/'data_view').symlink_to(CWD/'output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/data_view')
shutil.copy2(old/'gt_directions_only.json', root/'gt_directions_only.json')
train = [x.replace(OLDNAME, NAME).replace('pbr_gt_directional_plane_native', 'pbr_gt_directional_plane_native_direct')
         for x in json.loads((old/'train_argv.json').read_text())]
(root/'train_argv.json').write_text(json.dumps(train, ensure_ascii=False, indent=2)+'\n')
plans = json.loads((old/'planned_commands.json').read_text())
for plan in plans:
    plan['argv'] = [x.replace(OLDNAME, NAME) for x in plan['argv']]
    plan['log'] = plan['log'].replace(OLDNAME, NAME)
(root/'planned_commands.json').write_text(json.dumps(plans, ensure_ascii=False, indent=2)+'\n')
files = ['render.sh', 'albedo.sh', 'predicted_meshes.sh', 'normal_depth.sh', 'roughness.sh',
         'verify_roughness.sh', 'verify_all_mae.py', 'verify_all_mae.sh', 'audit_artifacts.py',
         'audit_artifacts.sh', 'write_results.py', 'write_results.sh', 'postprocess.py',
         'preview_40000.py', 'audit_checkpoint.py', 'make_final_rgb_comparison.py',
         'make_final_four_categories.py', 'make_final_relight_comparison.py']
for filename in files:
    text = (old/'commands'/filename).read_text().replace(OLDNAME, NAME)
    if filename in ['audit_artifacts.py', 'audit_checkpoint.py']:
        text = text.replace("config['loss_preset']=='pbr_gt_directional_plane_native'",
                            "config['loss_preset']=='pbr_gt_directional_plane_native_direct'")
        needle = "assert config['pbr_rasterizer_backend']=='lumimotion'"
        text = text.replace(needle, needle+"\nassert config['wo_indirect'] and config['wo_indirect_relight']\nassert config['pbr_directional_actual_ray_budget']\nfrom scene.pbr_ray_budget import validate_actual_ray_budget\nvalidate_actual_ray_budget(Namespace(**config))" if 'from argparse import Namespace' in text[:text.index(needle)] else needle+"\nassert config['wo_indirect'] and config['wo_indirect_relight']\nassert config['pbr_directional_actual_ray_budget']")
    if filename == 'write_results.py':
        text = text.replace('1010-12 jumpingjacks dir 原生PBR仓库原生光栅修复',
                            '1010-13 jumpingjacks dir 原生PBR直接SUN与真实光线预算')
        text = text.replace('B06全量结果', 'B07全量结果').replace('报告1010-12文档', '报告1010-13文档')
        text = text.replace('187', '194')
        text = text.replace('保留SH间接光近似及每4步优化', '关闭训练与重光照的Stage1 SH间接光近似，原生SH辅助RGB损失保留，每4步优化')
        text = text.replace('本次Plane Native Preset继承B05全部模型参数，包括既有来源平面几何投影。',
                            '本次Plane Native Direct Preset继承B06全部材质/几何参数，包括既有来源平面几何投影。')
        text = text.replace('本轮继承B05的Plane模型，只改变仓库原生CUDA backend与Native Preset；材质/光强/loss/lr/间接光/几何参数不改。',
                            '本轮继承B06原生CUDA及Plane模型，只改关闭SH间接近似和实际SUN像素预算；材质/光强/loss/lr/几何参数保持。')
        start = text.index('## 本次实际原生CUDA修复')
        end = text.index('## 各项MAE：全部15测试帧', start)
        text = text[:start]+'''## 本次Direct SUN与真实光线预算

新增pbr_gt_directional_native_direct及pbr_gt_directional_plane_native_direct，本轮后者。wo_indirect/wo_indirect_relight=True，只关闭PBR来源SH间接近似，native SH辅助RGB损失不改。pbr_directional_actual_ray_budget=True，SUN每像素L条BVH光线，启用间接光的显式覆盖则额外diffuse_sample_num条。单灯直接预算262144允许全部alpha>.9可用photo像素；该规则、全99827点、所有训练/测试帧与原GT评分域不改。旧Preset默认仍历史diffuse+light分母，显式HDR保留原预算，拒绝非法模式/非正整数/不足一个完整像素预算。

194 CPU discovery实际exit0；20步真实Trainer35000→35020实际exit0，保存Direct flags和native marker。三TRAIN帧实际BVH分别18843/15904/19137条单SUN，等于全部可用photo像素；真实3SUN预算512选170像素/510条，with-indirect成本257选1020像素/262140条，旧direct默认仍1024像素。材质/roughness/XYZ/q梯度非零finite；独立诊断0额外optimizer且参数SHA保持。首次诊断复用projection图导致重复backward失败，源码/日志保存；修正仅诊断每variant重建图，实际retry exit0。

GeometryRun与PBR alpha/worldnormal/ExpectedZ全buffer maxerror0、Mesh目标time一致，albedo/roughness材质helper一致。150×四类与4视频、三帧HDR16/替换SUN仅smoke功能；实际目检质量仍FAILED，不作为正式MAE或7°。正式完整150 HDR仍2048样本。

继承B06显式仓库LumiMotion native kernel，逐Gaussian按-dot(n_view,p_view)定向，forward/backward一致。pbr_rasterizer_backend=lumimotion，来源资产保存CUDA/Python/GLM逐文件SHA、综合源SHA和实际.so SHA，checkpoint marker与加载核验源，不静默回退installed。训练、全部RGB/Alpha/Normal、HDR/SUN、GeometryRun与逐time Mesh、材质评测共享入口。没有GT训练材质/法线/几何/能量；方向表白色单位×训练RGB拟合正scale，重光照metadata明确light-table而非GT photometry。

'''+text[end:]
        text = text.replace('原生SH间接光转灯仍是来源照明近似。', '本轮关闭来源SH间接近似，HDR/SUN仍使用原生GGX/BVH可见性；无重光照GT准确率声明。')
    if filename == 'make_final_rgb_comparison.py':
        text = text.replace('1010-10-jumpingjacks_dir-PBR来源平面几何投影', OLDNAME)
        text = text.replace('B05', 'B06').replace('B06原生', 'B07 Direct')
    (root/'commands'/filename).write_text(text)
(root/'commands/train.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(CWD))+
    '\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4\n'+shlex.join(train)+'\n')
manifest = json.loads((SMOKE/'code_manifest.json').read_text())
for relative in manifest:
    source = CWD/relative
    manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    for folder in ['code_snapshot', 'code_snapshot_train_launch']:
        destination = root/'commands'/folder/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
for name in ['code_manifest.json', 'code_manifest_train_launch.json']:
    (root/name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
for name in ['geometry_buffers', 'pbr_gradients', 'material_buffers', 'visual_review']:
    shutil.copy2(SMOKE/f'inspection/{name}.json', root/f'inspection/smoke_{name}.json')
intro = '''# 1010-13 jumpingjacks dir 原生PBR直接SUN与真实光线预算完整Stage2计划

状态PREPARED，完整B07尚未启动。已重新读取GOAL及B06最终FAILED：GS17.29779705°、覆盖94.10133523%，RGB PSNR19.68220260dB，材质烘光和脚周伪几何仍明显。本轮只改变Direct光照与实际光线预算，不同时放开gate/源NN或调整几何trust/loss/lr。

固定Stage1 model_mlp35000→55000共20000步，135train/15test，resolution2（400×400）黑背景。原RGB/自身RGBA Alpha、相机/time、来源35k预测、GT方向；禁止GT albedo/normal/roughness/depth/mesh/blend/强度/颜色进入训练/初始化/loss/prior，GT只离线评测。来源35k继承旧albedo Alpha转换但全部150帧已证明与自身RGBA及新转换逐像素一致，披露且不重跑Stage1。

Preset pbr_gt_directional_plane_native_direct，wo_indirect与wo_indirect_relight=True，pbr_directional_actual_ray_budget=True。关闭PBR Stage1 SH间接近似，保留native SH辅助RGB；SUN每像素L条BVH射线、with-indirect则L+diffuse_samples。单SUN预算262144覆盖全部alpha>.9可用photo像素，原选择规则和GT域保持。旧Preset及显式HDR预算不变。沿用B06原生CUDA/GGX/BVH、15187/99827 source plane支持、DC内部投影、固定sourceXYZ锚定、32来源NN/opacity-scale-feature冻结、geometry warmup2000/ramp1000、canonical .08m/60°、原loss/lr及RGB拟合全局正E。plane额外world投影不等于canonical位移边界。全部点保留，不筛掉难像素/帧。

194 CPU PASS、真实20步Trainer/GPU trace与所有材质/几何梯度PASS；三TRAIN直接SUN18843/15904/19137实际光线/all-eligible像素，3SUN170像素/510条，with-indirect1020像素/262140条；旧direct仍1024。诊断0额外优化/SHA保持，失败诊断及重试保留。共享GeometryRun/材料helper逐buffer PASS。150×4与三HDR16/SUN smoke功能PASS、实际质量FAILED；不是完整MAE或7°。

本轮来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp35k独立开始，不续B06的55k。计划保存40k/50k/55k，正式全量all render：四类各150，HDR150×2048，转灯72，6视频；全部15帧七MAE/coverage/common/missing180；每测试time全部135训练位姿预测Depth Mesh融合，固定原评分定义，无对齐/缩放/筛物体。原生无完整Adam state，不声称无损resume。

训练/各评测/日志完整命令如下。代码471文件双快照及native源SHA记录；旧失败run与本run失败权重/图/数组/日志全部保留。

## 完整命令与日志

'''
intro += '### train\n\n```bash\n'+shlex.join(train)+'\n```\n\n日志：'+str(root/'logs/train.log')+'\n'
for plan in plans:
    intro += '\n### '+plan['name']+'\n\n```bash\n'+shlex.join(plan['argv'])+'\n```\n\n日志：'+plan['log']+'\n'
intro += '''
## 四类实际目检与代表产物：PENDING

RGB/前后景、Alpha、Albedo/静动态、Normal/Depth/Mesh均待真实55k结果审阅；最终质量/7°未PASS。代表model_mlp/renders_stage2/ours_55000/insights/{rgb,alpha,normal,albedo}/*.png和4视频，directional/72及hdr/150/2视频；inspection/rgb_final_55000_comparison.png、four_categories_final_55000.png、relight_final_55000.png；normal_depth/comparison_0010/0080/0150.png。训练和MAE报告将真实回填，质量未验收前不commit/push/新目标分支。
'''
(root/'README.md').write_text(intro)
(root/'README_training_progress.md').write_text('')
doc = CWD/'LIHAN/DOC/训练'/f'{NAME}训练报告.md'
doc.write_text(intro)
goal = CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
now = datetime.now(timezone(timedelta(hours=9))).isoformat()
with goal.open('a') as f:
    f.write('\n## B07完整训练准备（'+now+'）\n\n'+intro.split('## 完整命令与日志')[0]+'\n完整命令见对应训练报告与train_argv/planned_commands，仍未开训或产生B07七MAE。\n')
index = CWD/'LIHAN/README.md'
with index.open('a') as f:
    f.write('\n- [B07直接SUN与真实预算完整Stage2](DOC/训练/'+doc.name+')：194 CPU及20步/真实GPU预算/共享buffer PASS，smoke图像仍FAILED；独立20k PREPARED，完整MAE/质量待完成。\n')
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'), Path('/home/lihan/reproduce/LumiMotion')]:
    for p in [doc, goal, index]:
        destination = checkout/p.relative_to(CWD)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
        assert destination.read_bytes() == p.read_bytes()
print(json.dumps(dict(status='PREPARED_FULL_DIRECT_RUN', run=str(root), code_files=len(manifest)), ensure_ascii=False), flush=True)
