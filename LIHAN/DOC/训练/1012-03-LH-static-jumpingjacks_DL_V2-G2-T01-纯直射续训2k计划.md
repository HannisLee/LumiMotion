# 1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k计划

## 实验问题与配对变量

G2-T01从共同的T13 Stage1 iteration10000检查点继续训练2000步，验证纯直射续训2k是否影响法线和脚部局部几何。配对T01/T02只改变`--ambient_mode`：T01=`zero`，T02=`learned`；其他来源、起点、seed、训练预算、光照上限、ambient学习率和T13继承参数相同。

## 当前计划状态

历史预注册状态：当时核心仍在稳定阶段、smoke与正式训练未启动。实际smoke、正式训练及最终评测状态以本文后续记录为准。

## 研究问题与对照

从共同的T13 Stage1 iteration10000检查点继续进行2000步可调几何训练，对比仅直接光照与学习受限全局间接底项。两组只允许 `ambient_mode` 取值不同（zero / learned），其余初始化、训练帧、损失、学习率日程、随机种子和优化配置完全一致。实验用来验证受限全局间接底项是否帮助法线精度，同时观察脚部局部几何；两组结果都必须在同一个检查点上检查GS、Depth、Mesh目标。

## 基线、约束与恢复语义

- 共同起点：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp`，iteration 10000；基线GS/Depth/Mesh分别为3.2553674619°、5.8876060528°、4.7507435938°。
- 初始化checkpoint SHA：PLY `72c5fa38f8ba4f4cdae8b0b78537fb31d795e5891c6802390a0eedb98099a90e`；deform `343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0`；lighting `35b8b543c24af2d2cc00d193ac6184badd7462faa4a357a2326397bc55cb353f`。T13 marker及完整审计见 `evaluation_10000/checkpoint_after_audit.json`，验收PASS。
- T13未保存optimizer state；这是从模型checkpoint开始的新optimizer，不是optimizer resume。正式配置必须记录Adam moments从零开始。
- T13 position LR为CLI init/final `1.6e-4 / 1.6e-6`、`max_steps=30000`，Gaussian spatial scale=5，xyz组为`8e-4 → 8e-6`；旧schedule在global step10000的值为`1.7235477520255056e-4`。新入口从global step10000调用schedule更新，XYZ参数组首步LR=0.00017235477520255056，max_steps保持30000，不因2000步预算压缩。optimizer moments重置；不声称从T13连续恢复优化器轨迹。
- T13 lighting checkpoint强度为6.6317067，必须由起点checkpoint恢复，不能重新按默认方向光初始化。
- 训练仅使用原`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view`下135个原训练RGB/RGBA相机与原GT方向。GT法线、深度、材质、阴影、光强及测试图均不得作为训练监督；完整15帧原协议仅作独立离线评测。
- 保持T13其余marker/训练参数不变；具体哪些参数自动从marker继承、入口怎样读取，待planner稳定后由完整CLI和启动前审计确认，不能仅凭本草稿推断。
- 每卡最多两个任务，GPU1/2按当前分配正式训练，smoke也计划GPU1/2；启动前重新读取GPU显存、compute PID和使用者，不迁移/打断已有任务。正确环境为ushas的`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`，经仓库native bootstrap和标准`goal_train_job.py`。

## 预算与待记录

计划训练2000步，新增optimizer更新编号10001–12000，保存global iteration10500/11000/12000。独立smoke各110步，保存relative step20/110，即global iteration10020/10110；step100对应global10100，必须实际覆盖一次增密。只有新ambient renderer fresh parity通过后才正式启动。source/model/iteration、完整argv与env、日志、PID/GPU UUID、runtime native/tracer SHA、输入和代码SHA、checkpoint稳定SHA、ROI脚诊断及完整15帧/四类评测路径均在输出README补记。训练与验收失败的日志和checkpoint保留，不覆盖。


## 计划输出与训练前检查

- 正式输出：`output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k`；模型：`output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp`。目录为root分配，失败结果不覆盖。
- 计划smoke：`output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步`；smoke模型路径：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步/model_mlp`。只有smoke真实分支审计和新的ambient renderer fresh parity通过，才进入正式训练。
- 训练文档编号：`LIHAN/DOC/训练/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k计划.md`。本文件为训练前计划，训练实际命令、PID、日志、检查点和评测在运行后追加。
- 输出README须记录完整训练/评测命令、source/model/iteration、日志与渲染目录、定量指标、代表图片/视频、四类可视化目检以及最终PASS/FAILED。

## CLI草稿（不可直接执行）

此处保留核心稳定前编写的草稿作为修订历史；其占位snapshot及smoke20参数已废止。当前唯一有效smoke命令为对应输出目录的`commands/train.json`，正式训练命令待smoke与fresh通过后落正式输出。

烟测argv草稿：

```text
/home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012 scripts.train_static_dl_ambient --init_model /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp --init_iteration 10000 --model_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步/model_mlp --source_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view --ambient_mode zero --ambient_cap 0.1 --ambient_lr 0.001 --steps 110 --save_steps 20 110 --seed 1012 --smoke
```

正式训练argv草稿：

```text
/home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012 scripts.train_static_dl_ambient --init_model /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp --init_iteration 10000 --model_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp --source_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view --ambient_mode zero --ambient_cap 0.1 --ambient_lr 0.001 --steps 2000 --save_steps 500 1000 2000 --seed 1012
```

runner草稿（正式训练GPU分配）：

```text
/home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal_train_job.py --command output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/commands/train.json --output output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k
```

`commands/train.json`将在稳定后替换为标准launcher schema，包含`gpu_physical`、带确切snapshot的argv和线程环境；本草稿尚未创建该文件。

## 当前状态

本段是早期草稿状态，已由后文真实smoke、正式训练和最终评测记录取代。


## 核心稳定与smoke配置修订（2026-10-12）

- planner已将ambient入口标记STABLE；核心SHA：`scene/static_dl_ambient.py`=`1c899faba045954d73dff1965960bf12411d5dbfa279b529fe552948304e3a0a`，`scripts/train_static_dl_ambient.py`=`198e365e177a8326b7b8e670ef807409e8298fd2cdcdc1d3cfed6db9b070a65e`。root已提交核心并完成95文件native snapshot：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012`。
- 独立smoke修订为110步，relative saves20/110，global iterations10020/10110；global10100实际触发原增密。T01/T02输出各自smoke目录见上；launcher真实argv与GPU声明在其`commands/train.json`。
- 新入口读取T13 `static_dl_training.json`继承科学参数，更新model/source/总iteration/device/save/test和ambient元数据；以T13 PLY、deform、lighting初始化。Adam状态不恢复，所有moment从零；XYZ LR在global10000继续原30k schedule，首步已审计值`0.00017235477520255056`。
- zero组ambient固定零且没有可训练optimizer组；learned组ambient从零开始学习并限制在[0,0.1]，LR0.001。两组共同执行a=0起点前向和六项photo梯度审计，随后执行110次真实更新。更新顺序为backward→屏幕统计→optimizer→拓扑更新→save，终点跳过无后续优化的拓扑/opacity reset；与T13原训练保存/更新次序不同，不能称为不中断续跑。
- 本计划已由后文smoke真实运行记录更新：两组110步smoke均exit0且GPU核验通过。正式训练仍须两组新ambient renderer fresh parity通过后才可启动。


## Smoke实际启动记录

- 入口与snapshot：`scripts.train_static_dl_ambient`，snapshot `output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012`，snapshot manifest SHA256=`89bb81edf36e9113a1bec29932561f0632d18b6de58ec343f90ad5790a0ba675`。入口SHA：`scripts/train_static_dl_ambient.py`=`198e365e177a8326b7b8e670ef807409e8298fd2cdcdc1d3cfed6db9b070a65e`；`scene/static_dl_ambient.py`=`1c899faba045954d73dff1965960bf12411d5dbfa279b529fe552948304e3a0a`。
- Smoke输出：`output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步`；T13@10000初始化、110个optimizer更新，global iterations10001–10110；保存相对step20与110（global10020/10110）。相对step100/global10100触发真实增密，终点更新后保存、不做无后续优化的reset。
- 命令文件：`output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步/commands/train.json`；launcher日志、状态和child环境核验在同目录。launcher完整命令：`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/goal_train_job.py --command output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步/commands/train.json --output output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步`。
- 实际启动时间UTC `2026-10-11T15:21:41Z`。

## Smoke真实训练记录（fresh parity最终通过）

- 训练实际正常结束：child PID `2392188`，supervisor PID `2392178`；GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`；启动时间UTC `2026-10-11T15:21:41.711358+00:00`，结束时间UTC `2026-10-11T15:22:22.018085+00:00`；`gpu_verified=true`、`training_exit_code=0`。launcher argv、env、command SHA与状态见`output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步/state.json`、`commands/train.json`和`launch_provenance.json`；command SHA=`f4175513d08cbd4ffec6abf5f7692a8394fcdcd8a5bf6a3c2bce45609a7a1f73`；日志`output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步/logs/train.log`。
- 输入T13文件SHA训练前后未变化；首帧`initial_checkpoint_gradient_audit.json`状态PASS、parameter_updates=0，前向对照最大差0；首训练步各开放几何与材质梯度有限。Adam从零初始化，起始XYZ LR `0.00017235477520255056`；optimizer更新110次。
- relative saves20/110分别对应global10020/10110。global10100真实增密，点数57465→57430，事件见`topology_events.jsonl`。T13光强由6.6317067恢复；终点light在`static_dl_lighting/iteration_10110/lighting.json`。终点PLY SHA=`215223ccdecd63c7bd29ddcaab42f6947ceba2c5b9b5b5cc0337fd186bb7e5fd`，lighting SHA=`700c65c95c1477156e8a1b1aed9c1bbbffdcd0bc4c60598349932e1083225f7e`，deform SHA与T13完全相同。
- T01纯直射ambient固定0，没有独立底项参数。这仅是smoke事实，需纳入两组正式方案解释，不将零值称为实际间接能量。
- smoke训练本身不产生Goal MAE。新ambient renderer fresh retry1已真实PASS，8类对照最大差为0；首轮helper字段读取失败原日志保留。
- 模块映射证据由evaluator在PID活跃时采集路径，并在进程退出后核SHA，见`output/1012-01-静态DL-两度目标总控/G2_T01_smoke_training_runtime_module_audit.json`；证据状态按`PASS_PATH_CAPTURED_HASH_AFTER_EXIT`，不称为活跃期hash读取。


## 正式启动门槛（已满足）

root接受T01从T13独立续训的几何探索价值，T01 smoke在真实ambient renderer fresh retry1 PASS后按授权独立启动正式2000步，不等待T02。正式训练及最终目标评测结果见本文后续记录。T02的a梯度和采样a均为零不改变T01参数。


## T01正式训练实际记录及最终评测

- T01 smoke fresh：`output/smoke_test/1012-01-staticDL-G2-T01-纯直射续训110步/render_parity_retry1/parity.json`为PASS。检查项含真实ambient/direct/visibility输出均通过，最大绝对差0；T01 ambient=0属于zero control，非零覆盖不适用。首轮fresh helper因命令snapshot字段读取假设失败，日志保留；修正路由后的retry1 PASS，未修改训练命令或checkpoint。
- 完整实际训练命令（由launcher启动）：

```text
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012' scripts.train_static_dl_ambient --init_model '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp' --init_iteration 10000 --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view' --ambient_mode zero --ambient_cap 0.1 --ambient_lr 0.001 --steps 2000 --save_steps 500 1000 2000 --seed 1012
```

- 标准持久launcher完整命令：`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/goal_train_job.py --command output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/commands/train.json --output output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k`。命令spec `output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/commands/train.json`，SHA256=`13e9057be263804d565bf98075e1178b04c661d37bcc8f357eba37a4460349fd`；日志`output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/logs/train.log`，状态`output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/state.json`。
- child PID `2396178`，supervisor PID `2396173`；GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`；开始UTC `2026-10-11T15:34:39.882682+00:00`，结束UTC `2026-10-11T15:36:50.871343+00:00`；`gpu_verified=True`、exit code `0`。任务为T13@10000参数初始化后的新Adam，执行2000次更新至global12000；不恢复Adam moments。
- 完成marker `model_mlp/ambient_finished.json`：steps=2000、iteration=12000、updates=2000、points=60077、ambient=0.0、T13输入SHA未变。三个检查点global10500/11000/12000保留；各文件SHA：`{'output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp/point_cloud/iteration_10500/point_cloud.ply': '4d0347d59e1e7a59db420ff53a17f5628329fedab511f1db1b34e3fbc14483f0', 'output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp/point_cloud/iteration_11000/point_cloud.ply': 'be080eb0f68f363181aa106ed432e71b1f68c8e8f4ed81c45b4ed843aefa03bd', 'output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp/point_cloud/iteration_12000/point_cloud.ply': '866862df27e8f8847085803e6712480cee0e1fee2e2ac440badabd1ea900320f', 'output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp/deform/iteration_12000/deform.pth': '343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0', 'output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp/static_dl_lighting/iteration_12000/lighting.json': 'b7f0eaadeb5b71316e08cc63c7a80f42693e3f1f830646456540bcd3ab5cb835'}`。
- ambient固定0。T13初始值和global XYZ schedule均按首步审计使用；training marker记录Adam moments清零。新增几何结构/点数允许变化，终点点数60077。
- 训练过程native/tracer运行时映射审计：正式PID退出后未捕获进程内maps，最终记录为`UNAVAILABLE_PROCESS_EXITED`；不将smoke或fresh模块证据代替正式训练maps。
- 原15帧、150方向四类渲染/解码与目检均已完成。G2三项数字结论FAILED、几何结论FAILED；见本文最终评测结果。


## 最终评测与G2验收（2026-10-11）

- 固定原15帧评测完成：`evaluation_12000/FINAL_REVIEW_COMPLETED.json`记录pipeline PASS、完整性PASS、G2数值FAILED/几何FAILED。GS/Depth/Mesh MAE为2.9169369838053854°／5.419012636909616°／4.43693964760927°；coverage为0.997858715857132／0.9658299045098029／0.9937330520128994。相对T13基线，Depth与Mesh未退化，但GS未达到G2 `<2.00°`要求。不能把静态EL阶段阈值的历史PASS误写成当前G2达标。
- 四类150方向可视化及原几何输出已完成，完整实际命令、渲染输出和日志见`evaluation_12000/README.md`、`plan.json`、`executed.json`。代表图：`evaluation_12000/four_classes/four_classes_10_80_150.png`和`evaluation_12000/normal_depth/comparison_0080.png`；视频为`evaluation_12000/four_classes/{rgb,alpha,normal,separation}.mp4`。600张PNG及四段各150帧、400×400视频全部解码通过，人工目检四类10/80/150和原几何0080已记录。
- 人工结论：RGB仍有暗部黑影与台面高亮，脚尖仍短缩；Alpha轮廓连续且鞋区100%覆盖不能证明鞋部几何完整；Normal平台较平，但鞋尖/脚掌形状仍短；Separation是均匀静态feature门控，不表示运动分离。离线鞋区GS/Depth/Mesh分别42.992467°／44.912316°／49.658132°，Expected同射线2cm保留率29.9434%（T13为5.2061%），距离中位数0.139881m（T13为0.220849m）；鞋部较T13有改善但仍未修复。
- 最终fresh parity 11项最大差为0；原GeometryRun与真实XYZ/q、Alpha和深度对齐。稳定检查点及源码来源审计见`evaluation_12000/checkpoint_after_audit.json`。训练活跃期native/tracer映射明确为`UNAVAILABLE_PROCESS_EXITED`，不冒称已采集；fresh/eval运行时映射不替代训练映射。
- 原15帧MAE报告：`evaluation_12000/goal_mae_summary.json`；空间诊断及逐对象数据：`evaluation_12000/spatial_diagnostic/`；最终视觉标记：`evaluation_12000/visual_review_completed.json`。最终G2结论：**FAILED**。Stage1静态EL阈值不是本轮G2目标，本轮也没有Stage2训练。
