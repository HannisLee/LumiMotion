"""B06完整训练与9步评测计划；只改Native Preset/光栅来源，独立输出保留B05。"""
from pathlib import Path
import json,shutil,shlex,hashlib,sys
CWD=Path.cwd();sys.path.insert(0,str(CWD))
from scene.pbr_rasterizer import source_manifest
OLDNAME='1010-10-jumpingjacks_dir-PBR来源平面几何投影'
NAME='1010-12-jumpingjacks_dir-PBR仓库原生光栅修复'
old=CWD/'output'/OLDNAME;root=CWD/'output'/NAME
smoke=Path(__file__).resolve().parent.parent
assert json.loads((smoke/'inspection/geometry_buffers.json').read_text())['status']=='PASS'
assert json.loads((smoke/'inspection/pbr_gradients.json').read_text())['status'].startswith('PASS')
assert json.loads((smoke/'inspection/material_buffers.json').read_text())['status'].startswith('PASS')
assert 'Ran 187 tests' in (smoke/'logs/cpu_final.log').read_text()
assert '\nOK\n' in (smoke/'logs/cpu_final.log').read_text()
root.mkdir(exist_ok=False)
for folder in ['commands','logs','inspection']:(root/folder).mkdir()
(root/'data_view').symlink_to(CWD/'output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/data_view')
shutil.copy2(old/'gt_directions_only.json',root/'gt_directions_only.json')
train=[x.replace(OLDNAME,NAME).replace('pbr_gt_directional_plane','pbr_gt_directional_plane_native')
       for x in json.loads((old/'train_argv.json').read_text())]
(root/'train_argv.json').write_text(json.dumps(train,ensure_ascii=False,indent=2)+'\n')
plans=json.loads((old/'planned_commands.json').read_text())
for plan in plans:
    plan['argv']=[x.replace(OLDNAME,NAME) for x in plan['argv']]
    plan['log']=plan['log'].replace(OLDNAME,NAME)
(root/'planned_commands.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
files=['render.sh','albedo.sh','predicted_meshes.sh','normal_depth.sh','roughness.sh','verify_roughness.sh',
       'verify_all_mae.py','verify_all_mae.sh','audit_artifacts.py','audit_artifacts.sh','write_results.py',
       'write_results.sh','postprocess.py','preview_40000.py','audit_checkpoint.py']
for filename in files:
    text=(old/'commands'/filename).read_text().replace(OLDNAME,NAME)
    if filename in ['audit_artifacts.py','audit_checkpoint.py']:
        text=text.replace("config['loss_preset']=='pbr_gt_directional_plane'","config['loss_preset']=='pbr_gt_directional_plane_native'")
        where=text.index('# 平面支持/marker绑定')
        extra="""# 仓库原生CUDA来源与checkpoint绑定，拒绝pip同名包或其它源码。
assert config['pbr_rasterizer_backend']=='lumimotion'
from argparse import Namespace
from scene.pbr_rasterizer import validate_backend_model
native=validate_backend_model(model,Namespace(**config))
for iteration in ITERATIONS:
 assert json.loads((model/f'pbr_directional/iteration_{iteration}/lighting.json').read_text())['rasterizer_source_sha256']==native['source_sha256']
""".replace('ITERATIONS','[step]' if filename=='audit_checkpoint.py' else '[40000,50000,55000]')
        text=text[:where]+extra+text[where:]
    if filename=='write_results.py':
        text=text.replace('1010-10 jumpingjacks dir 原生PBR来源平面几何投影','1010-12 jumpingjacks dir 原生PBR仓库原生光栅修复')
        text=text.replace('B05全量结果','B06全量结果').replace('179项CPU','187项CPU')
        text=text.replace("ROOT/'model_mlp/pbr_plane_projection.npz',ROOT/'training_runtime.json'", "ROOT/'model_mlp/pbr_plane_projection.npz',ROOT/'model_mlp/pbr_rasterizer.json',ROOT/'training_runtime.json'")
        text=text.replace('本轮仅改变Plane Preset及对应几何投影，材质/光强/loss/lr/其余参数不改。',
            '本轮继承B05的Plane模型，只改变仓库原生CUDA backend与Native Preset；材质/光强/loss/lr/间接光/几何参数不改。')
        text=text.replace('本次Plane Preset继承B04全部参数，新增来源平面几何投影。',
            '本次Plane Native Preset继承B05全部模型参数，包括既有来源平面几何投影。')
        needle='## 各项MAE：全部15测试帧'
        text=text.replace(needle,"""## 本次实际原生CUDA修复

B05使用环境installed hbb1 f7e2b68a的diff_surfel_rasterization0.0.1，以camera-space normal.z符号定向。实测来源plane上方相机0038/0075/0076被错误反向。B06显式构建仓库LumiMotion原始CUDA，逐Gaussian按实际观察射线-dot(n_view,p_view)定向，forward/backward同规则；保留原始T/光栅/BRDF/GGX/BVH实现，不在最终normal图或评分中翻转法线。独立namespace，不pip install，不改变Stage1/env/旧Preset默认installed后端，不覆盖B05结果。

新增pbr_gt_directional_native和pbr_gt_directional_plane_native，本轮后者；pbr_rasterizer_backend=lumimotion。来源资产pbr_rasterizer.json记录CUDA/Python/GLM每文件SHA与综合SHA、真实构建.so SHA和torch版本；checkpoint marker绑定综合SHA，加载/评分核验源，禁止静默回退。原嵌套GLM为空，使用同仓库已追踪stable目录的GLM头文件。仅支持scale/rotation CUDA路径，拒绝compute_cov3D_python=True。训练、PBR RGB/alpha/normal、HDR/方向光、GeometryRun/逐time Mesh、albedo与roughness材质helper均共享显式后端；actual albedo/rough缓冲首中尾相等验证PASS。

187 CPU最终discover实际exit0；20步真实Trainer smoke保存35020并加载marker，三TRAIN帧GGX/BVH/material/XYZ/rotation梯度非零finite、0额外optimizer；GeometryRun alpha/worldnormal/ExpectedZ与实际PBR全buffer maxerror0、Mesh targettime override一致。独立原生0步已知plane+三坏视角GPU法线camera-facing正确，合成normal梯度有限差分maxerror.00035429。HDR16三帧只作smoke，正式仍150帧2048；smoke不作全量MAE/画质/7°通过。

"""+needle)
        text=text.replace('源模型Plane/B04实际55k首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致。修复只改尚未执行的离线入口，训练源码不改，旧launch代码/manifest与当前修订已保存。',
            '本轮35020 native smoke首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致，材质helper一致；完整训练与离线使用同一native来源。完整launch源码/manifest和CUDA/GLM快照保存。')
    (root/'commands'/filename).write_text(text)
(root/'commands/train.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(CWD))+
    '\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4\n'+shlex.join(train)+'\n')
manifest=json.loads((old/'code_manifest.json').read_text())
manifest.update({x:'' for x in ['scene/pbr_rasterizer.py','tests/test_pbr_rasterizer.py',
                'gaussian_renderer/render_albedo.py','gaussian_renderer/render_roughness.py']})
manifest.update(source_manifest()['source_files'])
for relative in manifest:
    p=CWD/relative;destination=root/'commands/code_snapshot'/relative
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    manifest[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    destination=root/'commands/code_snapshot_train_launch'/relative
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
for filename in ['code_manifest.json','code_manifest_train_launch.json']:
    (root/filename).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for f in ['geometry_buffers.json','pbr_gradients.json','material_buffers.json']:
    shutil.copy2(smoke/'inspection'/f,root/'inspection'/('smoke_'+f))
intro='''# 1010-12 jumpingjacks dir PBR仓库原生光栅修复完整Stage2计划

状态PREPARED，B06未启动，质量/七MAE/7°待真实55k验收。已读取GOAL及B05全量FAILED：GS17.73081984°、common16.77003773°、missing18027.65368647°。先修复实际CUDA法线射线定向；本轮只有Native backend/对应Preset变化，其余与B05一致，不同时移除SH间接光或改ray预算。

仅Stage2，固定35k→55k全20000步；135train/15test，resolution2、400×400黑背景。原RGB/自身RGBA、原相机/time、既有Stage1预测、GT方向；禁止GT albedo/normal/roughness/depth/mesh/blend/光强/光色进入训练/初始化/损失。GT各项仅离线评测。固定35k继承早期albedo Alpha转换但150帧与原RGBA Alpha及转换RGBA逐像素相同已披露，不重跑Stage1。

Preset pbr_gt_directional_plane_native：保留Plane 15187/99827支持、sourceXYZ anchor、预测DC内部投影、2000 warmup/1000 ramp、原几何loss/lr、canonical .08m/60°、RGB拟合全局正光强及SH间接光。Canonical限制不是plane后actual world总位移界。只构建仓库原生CUDA逐Gaussian按观察射线定向，训练/全部渲染与几何/材质评测同backend；禁止评分后normal翻转、过滤物体或删点。来源资产/marker绑定CUDA/GLM综合SHA与.so SHA记录，不改环境installed包或其它分支源码。

187 CPU最终PASS、20步真实smoke及PBR/GGX/BVH/梯度、GeometryRun与材质helper逐bufferPASS。来源：output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp；本run/model_mlp，checkpoint40000/50000/55000，周期测试40k/55k。旧B05权重/渲染/统计保留，不续训旧55k，不称lossless resume。

全all render150×四类/72转灯/150 HDR2048/6视频；完整15帧七MAE、coverage/common/missing180；15targettime每time全部135train位姿预测深度Mesh融合，所有指标沿用原定义不对齐/拟合。失败全部保留。原生不保存完整Adam状态，不宣称支持无损resume。

## 完整命令与日志

'''
intro+='### train\n\n```bash\n'+shlex.join(train)+'\n```\n\n日志：'+str(root/'logs/train.log')+'\n'
for plan in plans:
    intro+='\n### '+plan['name']+'\n\n```bash\n'+shlex.join(plan['argv'])+'\n```\n\n日志：'+plan['log']+'\n'
intro+='\n## 四类目检与代表产物（PENDING）\n\nRGB/背景、Alpha、Albedo/静动态、Normal/Depth均待真实55k审阅；代表model_mlp/renders_stage2/ours_55000/insights/{rgb,alpha,normal,albedo}/*.png与4mp4，directional/72与hdr/150及2mp4；inspection/{rgb_comparison,four_categories,relight}.png，normal_depth/comparison_0010/0080/0150.png。最终质量与7°未PASS，不commit/push或新建目标分支。\n'
(root/'README.md').write_text(intro);(root/'README_training_progress.md').write_text('')
doc=CWD/'LIHAN/DOC/训练'/f'{NAME}训练报告.md';doc.write_text(intro)
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
    shutil.copy2(doc,checkout/doc.relative_to(CWD))
print(json.dumps({'status':'PREPARED_FULL_NATIVE_RUN','run':str(root),'code_manifest_files':len(manifest)},ensure_ascii=False))
