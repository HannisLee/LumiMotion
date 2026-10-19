# TRAIN机制诊断与PBR积分预算显式化

2026-10-19，ushas／lumimotion-ushas。Stage2冻结v1保持，不改其前向、优化或进行中的训练。

`scripts/diagnose_three_stage_train.py`升级诊断v2，增加原Alpha固定内缩分母上的预测GS／Expected深度法线内部一致性、缺失180度计罚及覆盖、Alpha IoU、实际软权重范围与高分位权重质量、亮区残差损失占比、共享SH有效系数和固定参考幅度。内部法线一致性不是GT法线精度。0／4050成对的旧漂移诊断保留；单独4050诊断的“相对首节点漂移0”不能冒充B00至4050没有变化。

`scripts/audit_three_stage_pbr_integration.py`新增TRAIN固定状态积分核验：同材质／几何／环境，无更新，比较16／128／256条光线，保存三视角图片、数值与父checkpoint SHA。256只作为数值参考，不使用GT。

`scripts/train_three_stage.py`新增显式 `--pbr_samples {16,128,256}` 并写入保存配置；保持默认16以重放旧工程。Stage3工程冒烟允许正式Stage2父模型，但正式训练仍拒绝冒烟父模型。下一轮Stage3正式积分预算256已在训练计划1019-23预登记，其余超参保持1019-21。

`scripts/eval_three_stage.py`保留冻结训练源码原始SHA检查；对实际推理模块继续要求当前与冻结版本逐字相同。仅优化入口当前CLI不再参与推理一致性判断，避免增加CLI选项后迫使已完成Stage2重训。未将修改后的前向冒充旧版本，也未改变旧checkpoint。

`scripts/launch_three_stage_pbr_pair.py`核验S20／S21正式退出、0／4050／8100完整TRAIN节点、256两组工程门禁及冻结v2快照后，以文件锁和新输出目录启动同一S20@8100的A30／A31。保存实际监督PID和完整参数，拒绝重复启动；不得仅凭启动收据认定进程仍活着。

当前4050节点的两组135TRAIN机制诊断均exit0；完整TRAIN 0→4050诊断、收据与图像保留在总控train_midpoint_v1和train_mechanism_v2。Stage3新256光线工程结果与正式运行状态以对应收据和训练启动记录为准，不预填PASS。
