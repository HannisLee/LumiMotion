# 1010-02 原生PBR方向光运行验证与透明Roughness评测导出

日期：2026-10-10（JST）；用户要求完整训练验证后commit/push，再进入仅Stage2、仅GT方向的法线优化目标。

## 输入限制

B00只将原lights.json的方向去重后写白色单位光，原RGB/强度丢弃；固定倍率π来自纯数值约定，不是GT强度。训练只用RGB/输入RGBA、相机/time、Stage1预测。Albedo/Normal/Roughness/Depth/Mesh/Blend GT仅离线评测可用。

## 测试与执行

新增tests/test_pbr_directional.py，测试多SUN及精确帧匹配、非法方向与点光拒绝、面正对光时delta diffuse=albedo（辐照度π）、GGX roughness梯度、背光零直射、BVH透射率、原envmap默认preset与lighting marker哈希。与已有photometric、camera和roughness测试共35项通过。完整训练output/1010-02-jumpingjacks_dir-PBR方向光基线，35000→55000、20000步进行中，不能标训练通过。实际训练源码已快照在该输出commands/code_snapshot，哈希code_manifest.json。

## Roughness GT透明支持

scripts/export_roughness_gt.py原来拒绝Principled的Alpha<1或Alpha节点输入，dir的头发因此无法生成完整PBR Roughness评测。新增显式--alpha_composite；默认仍拒绝透明。开启时保留Alpha原socket或输入节点，以MixShader混合Transparent与r Emission，常量r/alpha逐帧刷新，保留原相机和源文件SHA。

不透明像素r定义与原来一致；有多个透明层时定义是可见层的透射率×alpha加权r，须在报告注明，不能声称唯一第一命中粗糙度。评测仍按全前景alpha≥.99内缩2像素，不跳过头发或按材质过滤。原失败日志保留，15帧EXR和manifest位于独立GT评测目录，不输入训练。本次不更改训练shader。

## 后续gate

完整渲染、全部15帧各项MAE、独立复算、四类目检、背景检查与checkpoint审计后才提交代码。GOAL文件为LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md；每次新修改必须先读取。

## 导出严格数值复核

Cycles16样本在低Alpha头发边缘产生少量r/alpha>1，原导出不能直接评分，文件保留。复核源dir记录的render_engine=BLENDER_EEVEE；新增--engine cycles/eevee/source，默认保持原Cycles，当前dir采用source保留原渲染器Alpha语义。最终15帧EXR均通过load_target的严格[0,1]/有限性检查，未裁剪数据。完整137项CPU unit tests通过。
