# 1012-04-LH-static-jumpingjacks_DL_V2-G2-T02-受限间接底项续训2k计划

## 实验问题与配对变量

G2-T02从共同的T13 Stage1 iteration10000检查点继续训练2000步，验证受限间接底项续训2k是否影响法线和脚部局部几何。配对T01/T02只改变`--ambient_mode`：T01=`zero`，T02=`learned`；其他来源、起点、seed、训练预算、光照上限、ambient学习率和T13继承参数相同。

## 当前计划状态

历史预注册状态：该段记录smoke执行前的入口和启动门槛；实际烟测及fresh结果见后文。

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

- 正式输出：`output/1012-04-LH-static-jumpingjacks_DL_V2-G2-T02-受限间接底项续训2k`；模型：`output/1012-04-LH-static-jumpingjacks_DL_V2-G2-T02-受限间接底项续训2k/model_mlp`。目录为root分配，失败结果不覆盖。
- 计划smoke：`output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步`；smoke模型路径：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步/model_mlp`。只有smoke真实分支审计和新的ambient renderer fresh parity通过，才进入正式训练。
- 训练文档编号：`LIHAN/DOC/训练/1012-04-LH-static-jumpingjacks_DL_V2-G2-T02-受限间接底项续训2k计划.md`。本文件为训练前计划，训练实际命令、PID、日志、检查点和评测在运行后追加。
- 输出README须记录完整训练/评测命令、source/model/iteration、日志与渲染目录、定量指标、代表图片/视频、四类可视化目检以及最终PASS/FAILED。

## CLI草稿（不可直接执行）

此处保留核心稳定前编写的草稿作为修订历史；其占位snapshot及smoke20参数已废止。当前唯一有效smoke命令为对应输出目录的`commands/train.json`，正式训练命令待smoke与fresh通过后落正式输出。

烟测argv草稿：

```text
/home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012 scripts.train_static_dl_ambient --init_model /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp --init_iteration 10000 --model_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步/model_mlp --source_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view --ambient_mode learned --ambient_cap 0.1 --ambient_lr 0.001 --steps 110 --save_steps 20 110 --seed 1012 --smoke
```

正式训练argv草稿：

```text
/home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012 scripts.train_static_dl_ambient --init_model /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp --init_iteration 10000 --model_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-04-LH-static-jumpingjacks_DL_V2-G2-T02-受限间接底项续训2k/model_mlp --source_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view --ambient_mode learned --ambient_cap 0.1 --ambient_lr 0.001 --steps 2000 --save_steps 500 1000 2000 --seed 1012
```

runner草稿（正式训练GPU分配）：

```text
/home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal_train_job.py --command output/1012-04-LH-static-jumpingjacks_DL_V2-G2-T02-受限间接底项续训2k/commands/train.json --output output/1012-04-LH-static-jumpingjacks_DL_V2-G2-T02-受限间接底项续训2k
```

`commands/train.json`将在稳定后替换为标准launcher schema，包含`gpu_physical`、带确切snapshot的argv和线程环境；本草稿尚未创建该文件。

## 当前状态

本段为旧的启动状态，已由后文smoke与fresh真实记录取代；T02正式训练与MAE仍未产生。


## 核心稳定与smoke配置修订（2026-10-12）

- planner已将ambient入口标记STABLE；核心SHA：`scene/static_dl_ambient.py`=`1c899faba045954d73dff1965960bf12411d5dbfa279b529fe552948304e3a0a`，`scripts/train_static_dl_ambient.py`=`198e365e177a8326b7b8e670ef807409e8298fd2cdcdc1d3cfed6db9b070a65e`。root已提交核心并完成95文件native snapshot：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012`。
- 独立smoke修订为110步，relative saves20/110，global iterations10020/10110；global10100实际触发原增密。T01/T02输出各自smoke目录见上；launcher真实argv与GPU声明在其`commands/train.json`。
- 新入口读取T13 `static_dl_training.json`继承科学参数，更新model/source/总iteration/device/save/test和ambient元数据；以T13 PLY、deform、lighting初始化。Adam状态不恢复，所有moment从零；XYZ LR在global10000继续原30k schedule，首步已审计值`0.00017235477520255056`。
- zero组ambient固定零且没有可训练optimizer组；learned组ambient从零开始学习并限制在[0,0.1]，LR0.001。两组共同执行a=0起点前向和六项photo梯度审计，随后执行110次真实更新。更新顺序为backward→屏幕统计→optimizer→拓扑更新→save，终点跳过无后续优化的拓扑/opacity reset；与T13原训练保存/更新次序不同，不能称为不中断续跑。
- 本计划已由后文smoke真实运行记录更新：两组110步smoke均exit0且GPU核验通过。正式训练仍须两组新ambient renderer fresh parity通过后才可启动。


## Smoke实际启动记录

- 入口与snapshot：`scripts.train_static_dl_ambient`，snapshot `output/1012-01-静态DL-两度目标总控/snapshots/stage1-ambient-continuation-native-assets-20261012`，snapshot manifest SHA256=`89bb81edf36e9113a1bec29932561f0632d18b6de58ec343f90ad5790a0ba675`。入口SHA：`scripts/train_static_dl_ambient.py`=`198e365e177a8326b7b8e670ef807409e8298fd2cdcdc1d3cfed6db9b070a65e`；`scene/static_dl_ambient.py`=`1c899faba045954d73dff1965960bf12411d5dbfa279b529fe552948304e3a0a`。
- Smoke输出：`output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步`；T13@10000初始化、110个optimizer更新，global iterations10001–10110；保存相对step20与110（global10020/10110）。相对step100/global10100触发真实增密，终点更新后保存、不做无后续优化的reset。
- 命令文件：`output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步/commands/train.json`；launcher日志、状态和child环境核验在同目录。launcher完整命令：`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/goal_train_job.py --command output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步/commands/train.json --output output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步`。
- 实际启动时间UTC `2026-10-11T15:21:41Z`。

## Smoke真实训练记录（fresh前向通过、非零覆盖未通过）

- 训练实际正常结束：child PID `2392189`，supervisor PID `2392179`；GPU2 UUID `GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`；启动时间UTC `2026-10-11T15:21:41.716630+00:00`，结束时间UTC `2026-10-11T15:22:32.129815+00:00`；`gpu_verified=true`、`training_exit_code=0`。launcher argv、env、command SHA与状态见`output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步/state.json`、`commands/train.json`和`launch_provenance.json`；command SHA=`5b80cadde27da4ec384647e38b4550e2f7f82002a1e4c3a3f08c20699b93bb08`；日志`output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步/logs/train.log`。
- 输入T13文件SHA训练前后未变化；首帧`initial_checkpoint_gradient_audit.json`状态PASS、parameter_updates=0，前向对照最大差0；首训练步各开放几何与材质梯度有限。Adam从零初始化，起始XYZ LR `0.00017235477520255056`；optimizer更新110次。
- relative saves20/110分别对应global10020/10110。global10100真实增密，点数57465→57426，事件见`topology_events.jsonl`。T13光强由6.6317067恢复；终点light在`static_dl_lighting/iteration_10110/lighting.json`。终点PLY SHA=`ac2b03e530832cf5bd9f28b4b773cadf3bb06b01b40dc6af783938f70e5bddfa`，lighting SHA=`1bb2073bd5f1c50ada16a11e9d8faa9e90c142f84a8d12961a1c7e09a6b9abb2`，deform SHA与T13完全相同。
- T02 learned ambient首步梯度0.0635352有限非零，但受非负投影约束，终点ambient仍为0；本smoke没有学出正值。这仅是smoke事实，需纳入两组正式方案解释，不将零值称为实际间接能量。
- 新ambient renderer的fresh前向及其他对照检查通过，但保存ambient为0，验收标记`FAILED_NONZERO_AMBIENT_UNCOVERED`。这是smoke分支覆盖不完整，不是训练崩溃或数学错误；随后先执行固定T13的零更新profile。
- 模块映射证据由evaluator在PID活跃时采集路径，并在进程退出后核SHA，见`output/1012-01-静态DL-两度目标总控/G2_T02_smoke_training_runtime_module_audit.json`；证据状态按`PASS_PATH_CAPTURED_HASH_AFTER_EXIT`，不称为活跃期hash读取。


## 间接底项零更新剖面门槛（预注册历史；结果见文末）

初始配对设计保留为计划历史，但T02不立即进入正式2000步。110步smoke首训练帧ambient梯度为正且有限；保存日志采样step1/20/100/110的ambient pre/post均为0。有限采样不能证明110步中间从未暂时大于0，也不能称为训练获得非零间接项。

已stable的只读profile对固定T13与原135训练帧评估候选`a={0,0.001,0.01,0.05,0.1}`，未更新参数。具体数据、执行命令及科学结论见文末。该profile是预定启动门槛；未强制初始化正值。

该预注册修订不改smoke或失败helper日志。profile已经完成并否决当前固定起点的正ambient方案，T02正式2000步不启动。


## T02烟测fresh重载结果（正式试验继续暂停）

新入口ambient forward及其余重载/来源检查通过，但fresh验收marker=`FAILED_NONZERO_AMBIENT_UNCOVERED`，原因是saved `ambient_fraction=0.0`，故`nonzero_branch_coverage=UNCOVERED`。这不是训练/数学/Goal MAE失败；只说明当前smoke没有可供重载验证的正ambient checkpoint。输出`output/smoke_test/1012-02-staticDL-G2-T02-受限间接底项续训110步/render_parity/parity.json`和NPZ/log完整保留。

root已决定先对原T13 checkpoint与固定135训练相机做零更新ambient a-loss profile（0、0.001、0.01、0.05、0.1），计划输出`output/1012-05-LH-static-jumpingjacks_DL_V2-G2-间接底项零更新剖面`。profile裁定前不启动T02正式2k；不得强制positive initialization。


## 固定T13底项剖面结果与正式试验结论

规划门槛已执行，旧段落中“planner正在实现／等待profile”的状态由本节取代。`output/1012-05-LH-static-jumpingjacks_DL_V2-G2-间接底项零更新剖面/summary.json`记录T13@10000固定模型、135个原训练相机、零参数更新的真实结果：a=0、0.001、0.01、0.05、0.1平均censored photo loss分别为0.004115224390、0.004295548279、0.005157577629、0.007512970900、0.009949348586；a=0的135帧平均真实反向导数为+0.056146399565，只有1帧导数为负，网格最优值为a=0。增加正底项在这一固定起点上单调提高平均目标值。

数据完整性审计PASS：135行齐全；原模型参数、CPU/CUDA RNG、输入图片/方向/marker/Ply/deform/lighting SHA均保持不变；optimizer未创建、更新次数为0。固定a=0.05仅作独立进程保存／重载接口探针，11类渲染输出最大差均0，`trained_parameter=false`，不得当作训练学得了正a。完整逐帧数据和完整启动命令见output的`per_frame.jsonl`、`plan.json`和`execution.json`；总控及实验日志均保留。

按既定门槛，T02不启动正式2k，归档为“当前T13起点的受限间接底项假设不受固定RGB目标支持”。该结果不是T01/T02训练效果对照，也不证明联合几何优化下一定无效；不应称为训练失败、接口失败或Goal MAE失败。T02保留已完成110步smoke及其`FAILED_NONZERO_AMBIENT_UNCOVERED` fresh结果、首次helper失败日志和所有checkpoint，不覆盖或删除。T02正式训练与Goal MAE均未产生；本输出无四类可视化、代表视频或目检，不能据此宣称两度目标达成。
