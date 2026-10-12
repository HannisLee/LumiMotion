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

## 输出与执行状态

输出：`output/1012-12-LH-static-jumpingjacks_DL_V2-G2-预测深度混合法线梯度零更新诊断`。独立入口由规划Agent实现，完整命令、源SHA、GPU／活跃模块、日志及诊断汇总运行后追加。每卡任务上限2，尊重并发足部Goal及其他已有任务。

当前状态：预注册完成，尚未执行。常规执行由Luna负责；现有Agent唤醒曾被工具以agent thread limit reached拒绝，待空闲后重试，不视作科学失败。
