# 原生PBR八组Adam与RNG末端全状态继续

2026-10-19，ushas／lumimotion-ushas。实现预登记计划1019-37，原A45@4050已完整归档FAILED；该计划及共同父/8100预算在读取A45 TEST指标和图片前按完整TRAIN冻结，不因失败数值改参。

新增scripts/train_three_stage_resume.py显式绑定唯一A45父checkpoint，完整8组参数/Adam moments/steps/actual lr/initial_lr原样恢复，Python/NumPy/CPU/CUDA RNG于构造和相机加载后恢复。原先的A42继续入口只有4组Adam、追加几何空moments，本入口不能复用该重置逻辑。成熟四几何保持A45末端常量，完整8100更新保存0/2700/5400/8100、累计63349；20步冒烟保存0/4/8/20、累计55269。normal→opacity保护按4050+local_step判断，始终在原保护期外，photo/Alpha始终可更新opacity。其余物理、损失、约束、冻结32MLP/SH/估计灯向及源资产不变。

新增scripts/audit_three_stage_resume.py独立核对四节点SHA、原PLY/全部32MLP/照明、全部8组父Adam/RNG初值、每组实际末端lr/initial_lr及继承step、后续有限更新、完整累计。独立复演父Python RNG产生全部相机顺序，并核验末点Python RNG；保留原Expected先验独立光栅重算及opacity保护/未保护/photo分支梯度核验。run_three_stage_job.py只增加新显式入口白名单，旧模式不变。

只读CPU门禁已实际PASS：加载真实父8组有限Adam，外观4组step8100、几何4组step4050，精确actual lr正确；光度/先验到优化器更新主体与既有入口相同，独立Expected审计主体相同，原V14所有scene/gaussian_renderer及原继续/审计入口SHA不变。源码AST通过。收据floor_resume_cpu_audit_v1.json；该检查不替代GPU冒烟重载/八组Adam与RNG逐值/无目标推理/原生实际起点/独立Expected先验/实际查看。

所有旧失败checkpoint、渲染、统计与日志保留，不修改V14快照。将从V14构造独立V15，仅新增两入口和监督白名单差异。当前未启动新GPU训练；后续须完整20步门禁及实际查看通过才正式，输出/日志按计划固定目录。修改源文件SHA256：

- `scripts/train_three_stage_resume.py`：`2b00f0f9b2f4345e1547b18297abfa69d3457467a499399f48dd4a1fbad24614`
- `scripts/audit_three_stage_resume.py`：`86cce819e3fd658f2edd99fa9bfdfadcbed4502e0addbf35af72e932fc8cb8ea`
- `scripts/run_three_stage_job.py`：`0b844629bd5412585fdf0da1c9c85839c4743eaf185325bc0dedac95d94f3d97`
- `output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控/audit_floor_resume_cpu_v1.py`：`7529203b58b259bd0768fb9b0623c10edc577888ad473020180d7fed8a79e846`
