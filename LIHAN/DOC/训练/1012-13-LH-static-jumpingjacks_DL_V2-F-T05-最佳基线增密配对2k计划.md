# F-T05｜当前最佳基线增密统计配对2k

**优化思路：** 从G2-T06@16000延续相同法线/物理损失，仅为增密累计设置非normal支路范数下限，避免梯度抵消压低细节增密信号，不新增监督。

## 输入与控制变量

源`output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp`@16000；原135TRAIN data_view。起点无optimizer资产，明确新建零动量Adam、清零增密统计，seed1012及独立不放回相机序列相同，30k XYZ LR日程从16000继续，不挪用其他轨迹状态。

固定route=`non_normal_floor`；normal_opacity_route=detach_opacity、ambient0、reset_policy=skip_all；原censored RGB/Alpha/expected normal/distortion损失与权重、阈值0.0002及原拓扑算法保持。两组唯一实验变量为增密统计路线，输出路径属于运行元数据。

## 步数、输出及执行门槛

正式新增2000次optimizer更新，global16001—18000；保存16500/17000/17500/18000的PLY/deform/lighting与optimizer、随机状态、相机余栈、增密统计。输出`output/1012-17-LH-static-jumpingjacks_DL_V2-F-T05-非normal增密下限2k`，日志同目录`logs/`。16000边界不补结构或reset，终点18000不执行无后续优化的结构/重置。

正式前需新独立snapshot来源冻结、GPU1真实110步smoke至16110，跨16100增密；首次真实梯度、原拓扑算子、屏幕统计/分母/parameter.grad核验及存活runtime、fresh重载均PASS。正式从原G2-T06@16000重新初始化，不接续smoke。候选T05还需零更新诊断证实固定阈值下存在新增选择；若没有则登记未启动/方案否决，不改变阈值。

当前未启动训练；完整argv/命令、GPU/PID/真实退出码、snapshot SHA、日志与checkpoint待实际准备后记录。每卡最多2任务，重新检查显存，不中断其他任务。失败目录和日志不覆盖。

## 联合验收

原15测试10—150、原掩膜和MAE数学不变；完整六评测、四类150视角、正面/侧面/关键鞋部人工检查。三类主MAE须来自同一checkpoint，比较当前最佳G2-T06及旧冻结G2-T04；新更严格上限GS≤2.7683778383745303°、Depth≤5.604577799172339°、Mesh≤4.40848728697561°，旧上限同时报告不改写。必须腿足完整且数值保持才整体PASS，不以coverage或总点数代替几何。

训练结果：GS／Depth／Mesh未评测；足部恢复未评测；可视化未生成，最终状态PENDING。本计划不假定候选有效，完整命令/输出/指标/四类目检最终写入实验README。


## 本次收尾归档状态（2026-10-12）

按用户“继续，收尾实验，归档文档”要求，本计划以 **NOT RUN（未执行）** 归档。本轮未启动GPU诊断、smoke、正式训练或新模型评测；无本轮PID、训练日志、checkpoint、MAE及四类可视化。以上均为预注册方案，不能写作已执行结果；足部是否恢复与法线是否退化均无本轮证据。现存起点模型脚部仍为部分恢复、完整性FAILED。

命令草案、未冻结snapshot状态和资源核查见`output/1012-15-LH-static-jumpingjacks_DL_V2-F-T04-T05增密配对总控/README.md`。未来若另行继续，须先完成输入SHA／native来源／snapshot门禁审查、真实诊断及smoke/fresh验证；不能直接将本草案视为已通过训练验收。
