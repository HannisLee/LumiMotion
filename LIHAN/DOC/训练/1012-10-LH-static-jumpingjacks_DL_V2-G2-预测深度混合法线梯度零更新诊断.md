# G2｜预测深度混合与法线梯度冲突零更新诊断

## 依据与假设

已读取两度Goal、旧阴影Goal、G2-T05/T06训练与完整原15帧结果。T06@16000为本Goal最低GS／Mesh候选，三项为2.6683778384°／5.4045777992°／4.2084872870°，鞋仍未完整，GS尚未达到2°。衣物占其GS角度误差和约43.82%；仅消除鞋误差仍不足2°。另一个足部Goal已在延长无reset训练，本次不重复该实验。

假设：前后表面混合时，expected-depth计算的局部法线可能不可靠，法线自一致性会与RGB恢复几何的梯度冲突。先用模型自身expected与median深度差作诊断，不预设该置信度有用。

## 固定输入与零更新边界

- 主机ushas，Python `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。
- 模型：`output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp`，iteration16000。
- 训练源码：该实验正式命令所绑定的 `stage1-reset-audit-native-assets-20261012-retry1`；沿用真实renderer、原censored photo、normal掩膜M与normal→opacity阻断。
- 只读取原135训练RGB／自身Alpha、原训练相机／GT方向及保存的RGB估计光强；不读取测试帧、GT法线／深度／材质／阴影／能量或语义分区。
- 仅前向与autograd诊断，不执行optimizer.step、增密、裁剪、reset或checkpoint写回；输入模型前后SHA保持不变。不产生新的科学MAE，不将诊断PASS称作修复成功。

## 预注册诊断

比较同单位预测camera-Z的expected与median深度，令delta为其绝对差在3×3邻域的最大值。tau=0.002×原训练相机extent；不由GT或测试误差调参。候选权重 `w=stopgrad(0.1+0.9/(1+(delta/tau)^2))`，候选normal为 `sum(M*w*normal_residual)/sum(M*w)`，保持原normal系数0.1，原M不变。

在全部135训练帧报告delta/tau分布、w与有效支持，以及低差距(<tau/2)、中差距(tau/2至tau)、高差距(>tau)三档的像素数和normal残差。分XYZ、rotation参数各自统计真实normal梯度与原photo梯度的内积、cos和冲突比例；不混合不同单位参数伪造解释。另报原权重和候选权重的梯度量级，检验变化是否主要来自减弱整个normal项。

若高差距档未呈现更大的normal残差及更强几何梯度冲突，停止该方案正式训练。若支持，再由Astra提出同起点、同预算的单因素实验；本计划本身不批准任何新2k训练。

## 实际执行与结果

输出：`output/1012-12-LH-static-jumpingjacks_DL_V2-G2-预测深度混合法线梯度零更新诊断`。主机`ushas`，Python为`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。固定模型`output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp`@16000，使用`stage1-reset-audit-native-assets-20261012-retry1`快照。helper SHA256：`965e54c6f5e79636c23f7613e1c398b10a056542cd8cd46f0e7f86205b0af02b`。

完整命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal2_normal_confidence_profile.py --model /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp --iteration 16000 --snapshot /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-reset-audit-native-assets-20261012-retry1 --output /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-12-LH-static-jumpingjacks_DL_V2-G2-预测深度混合法线梯度零更新诊断
```

进程PID `2569688`，exit code `0`；运行于GPU1（UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`），结束后释放。实际加载native rasterizer SHA256为`f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3`，surfel tracer SHA256为`5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e`。完整执行日志为输出目录内`logs/profile.log`，启动资源为`logs/prelaunch_resources.json`，命令亦记录于`logs/command.txt`与`plan.json`。

零更新核验`PASS`：135/135原TRAIN帧完成，4,494,614个有效像素；`optimizer_created=false`、参数更新数0、未加载测试图，模型参数、输入、源码及CPU/CUDA随机状态前后不变。未运行训练或科学MAE。

观测结果为混合证据。tau为0.00858005米；delta大于tau的像素占14.28%，其法线残差均值为0.15670，低于tau/2档为0.000503，中间档为0.005293，显示高分歧像素有更大残差。但梯度冲突没有随分歧档一致增强。这里分别报告summary中的三种口径：对全体点聚合的负内积点比例（`negative_point_fraction`），高档XYZ/quaternion为55.51%/52.80%，低档为53.36%/43.62%；存在负内积的视角占比（`negative_view_fraction`），高档XYZ/quaternion为28.15%/41.48%，低档为49.63%/32.59%；负内积点-视角对占比（`negative_point_view_fraction`），高档XYZ/quaternion为49.28%/49.91%，低档为49.44%/48.65%。因此不能据此声称高分歧区存在更强冲突。候选权重下汇总normal梯度范数降至uniform的XYZ 0.416、quaternion 0.277，逐视角范数降至0.246/0.193，表明候选显著缩小了梯度量级；这不足以证明几何修复或训练收益。

本次仅为诊断，不启动基于该权重的训练，也不把诊断`PASS`解释为科学假设获支持或模型修复成功。规划Agent复核135帧后判定：高深度差像素承担96.54%的normal残差，但与RGB的梯度冲突并未一致增强；候选主要显著减弱normal梯度。按预注册门槛，置信度调权方案本轮科学结论为**FAILED／停止正式训练**，诊断执行本身PASS。完整本轮归档见`LIHAN/DOC/报告/1012-05-静态DL两度目标T01至T06与脚部机制归档报告.md`。
