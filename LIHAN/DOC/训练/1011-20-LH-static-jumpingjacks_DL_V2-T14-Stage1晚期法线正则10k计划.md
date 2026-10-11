# 1011-20-LH-static-jumpingjacks_DL_V2-T14-Stage1晚期法线正则10k计划

## 研究问题与控制变量

T14是T13的配对控制：两组从相同静态DL随机初始化、使用相同135帧、相同10000步预算和相同所有物理/损失/优化器设置。唯一变量是法线一致性正则的开始步数：T14在iteration>8000、即step8001启用，T13在iteration>2000、即step2001启用。该设计只检验normal-reg启用日程，distortion日程两组完全相同。

## T14配置

- 输出：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k`。
- 入口：`scripts.train_static_dl_early_geometry`；稳定快照为`output/1011-09-静态DL-Goal训练总控/snapshots/stage1-early-geometry-native-assets-20261011`，manifest SHA256=`e377849e28d803f6ab70282d666ebe8eaa8641bfb575aa7b9856bdd5b2eecfd2`；完整正式CLI记录到prelaunch train.json。
- 初始化：从头训练，使用与T13相同的100000点确定性随机`points3d.ply`（seed=0）及完全相同SHA；不从Stage2 checkpoint加载。
- 数据及监督：沿用`output/1011-09-静态DL-Goal训练总控/data_view`原135训练相机、RGB、Alpha与训练光照方向；只采样这135个训练相机进行优化。source中的15个测试相机仅由原加载器管理/评测，不作为训练监督或进入loss/优化。禁止使用GT法线、深度、材质、阴影、光强作训练监督。expected统计量只进入独立几何副本，RGB前向depth_ratio=1。
- 训练预算与保存：10000步，保存5000、8000、9000、10000。T13/T14共享增密截止20000及原Stage1其余训练设置。
- 正则日程：`start_normal_reg=8000`；两组共用`dist_start=8000`、dist系数1000、normal权重0.1。
- 物理与RGB设置：physical start=1000，visibility start=5000，shadow bias=0.03，BVH interval=1；censored RGB损失，shadow weight=0.25、highlight weight=0.5、residual threshold=0.08、alpha weight=0.1。
- 其他参数严格沿用T03 Stage1：deform-type mlp、resolution2、is_blender/eval/load2gpu_on_the_fly/gt_alpha_mask_as_scene_mask、train/test light folder images；lambda separation=0.001，d_xyz loss weight=0.001，binarization warm-up=1000，d_color regularizer=0.01；XYZ LR日程沿用原30k长度不压缩，rotation=0.001、albedo=0.01、light=0.001；seed、方向JSON、初始化文件及来源SHA由稳定入口和prelaunch审核记录。

## 启动门槛与状态

入口稳定后，T13/T14分别用独立模型目录、同一快照和共享初始化SHA。smoke必须实际越过物理、T14在step8001的normal-reg及dist-reg启用边界，并由梯度/状态检查确认，而非仅执行SH预热。首帧和边界审计、checkpoint来源、native实现及fresh parity均通过后才可正式运行；停止在T14专属normal启用前的共同行为及启用后差异均需如实记录。所有日志、checkpoint、配置和SHA留存，失败不覆盖。

T14计划输出目录为`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal-reg-8000-Stage1-10k`。T14烟测`output/smoke_test/1011-26-staticDL-T14-normal8000-smoke-schedule-20步物理日程冒烟`已于2026-10-11在GPU2 exit0/gpu_verified=true，child PID2195278、supervisor2195267，command SHA256=`bb64044c3d13fd77c97e17275bbc2c3f700efea9f1e47c64d0fad85895017d46`，保存iteration20。smoke schedule在step1触发物理/V但normal/dist关闭，step11二者均启用；两次真实梯度分支审计记录100000条trace rays且允许梯度有限非零。完整命令、状态、日志、审计见smoke output。独立fresh parity已通过：`output/smoke_test/1011-26-staticDL-T14-normal8000-smoke-schedule-20步物理日程冒烟/render_parity/parity.json`，8项输出最大差为0。正式T14已启动，详见下方执行记录。


## 正式启动记录

- 实际输出目录：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k`。预注册方案中的目录名曾写作`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal-reg-8000-Stage1-10k`；该名称未用于训练，实际目录以launcher state、marker和下列路径为准，没有在旧目录训练或覆盖结果。
- 启动时间：2026-10-11T12:42:04Z；child PID `2197353`，supervisor PID `2197344`；物理GPU `GPU2`（UUID `GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`），Conda Python `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。启动后`gpu_verified=true`；正式训练已完成10000步，launcher state记录`training_exit_code=0`。
- 正式marker：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/model_mlp/early_geometry_training.json`；实际记录`normal_start=8000`、`dist_start=8000`、`smoke_schedule=false`、`iterations=10000`、保存步数5000/8000/9000/10000。T13/T14启动argv唯一实验变量为`--start_normal_reg`（T13=2000，T14=8000），其余科学配置一致；root已完成`output/1011-09-静态DL-Goal训练总控/T13_T14_runtime_fairness_audit.json` 13项公平性审计，全部PASS。
- 完整命令：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/commands/train.json`；命令SHA256=`022e85003a3a0d2059d128fd98a09b43879ee456b8a9051d42421ec212a99d81`；训练日志：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/logs/train.log`；实时状态：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/state.json`。原生训练源码/.so运行时采集：`output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/T14_training_runtime_module_audit.json`，evaluator记录审计状态PASS_CAPTURED。
- Smoke及fresh：T13/T14均20步、实际物理/V/normal/dist计划分支覆盖并exit0；T13 fresh首个helper读取错误留档，`render_parity_retry1`实际exit0/PASS；T14 `render_parity` exit0/PASS。fresh八项最大差均为0。Smoke训练进程退出后未采集其maps，不把fresh运行maps当作训练maps。
- 本次为Stage1从头训练，允许原Stage1增密与opacity/scale更新；没有冻结它们。只优化原135训练相机；source中的15个测试相机由原加载器管理/评测，不作为训练监督或进入loss/优化。
- 终点评测以实际目录为准：稳定的iteration10000后由evaluator执行原15帧协议与150方向四类可视化；中间checkpoint是否补评不在本计划承诺范围。定量指标、解码、目检和最终PASS/FAILED待实际评测后记录。


## 当前训练边界审计

T14在step5000的审计显示normal_active=false、dist_active=false、trace_rays=44380，XYZ/rotation/albedo/light梯度均有限非零。对应日志`logs/train.log`与`model_mlp/static_dl_gradient_audit/`保留；step8001实际启用边界已在下文记录。


step8001边界审计：`normal_active=true`、`dist_active=true`，trace_rays=68383，mean_visibility=0.3881362678；normal_loss=0.0891332328、dist_loss=7.90873997e-08。XYZ/rotation/albedo/log_intensity梯度均有限非零，详见`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/logs/train.log`。

## 正式训练终态与评测交接

- 训练于2026-10-11T12:48:38Z完成，标准launcher state为`TRAINING_COMPLETED_PENDING_EVAL`、`training_exit_code=0`、`gpu_verified=true`。10000步结束日志、checkpoint与运行时审计保留在实际输出目录；训练未重启。
- step8001日程审计已证实normal与dist均启用，trace_rays=68383，mean_visibility=0.3881362678，normal_loss=0.0891332328，dist_loss=7.90873997e-08；XYZ/rotation/albedo/log_intensity梯度有限非零。
- 最终评测状态：evaluation_10000原15+150已由evaluator在GPU2启动。按实际目录运行原15帧协议与150方向四类可视化，不执行中间检查点评测。


原15帧最终数字结果：iteration10000三项GS/Depth/Mesh=4.168761017/6.408321834/5.591491479°；coverage=0.996954208/0.964929088/0.993026059；expected深度绝对误差=0.025322810m；阈值依次为3.51/7.35/5.46°。数字结论：数字阈值FAILED（GS与Mesh超阈值）；最终人工验收已完成，见下方最终验收段。完整来源、复算摘要与指标SHA见`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/goal_mae_summary.json`；最终视觉验收已完成，详情见本目录`FINAL_REVIEW_COMPLETED.json`。

## 最终验收（FAILED）

- 阶段：Stage1，iteration 10000；输出模型：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/model_mlp`；source：`output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`。本轮仅训练并验收Stage1；没有从该checkpoint继续训练Stage2。
- 渲染入口完整命令：`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/goal_eval_run.py --model /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/model_mlp --iteration 10000 --output /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000 --gpu 2 --execute`。四个实际子命令、环境、逐步退出码与日志记录在`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/README.md`、`plan.json`和`executed.json`；渲染产物在`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000`，子任务日志在`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/logs`。
- 原15帧GS/Depth/Mesh MAE：4.168761017° / 6.408321834° / 5.591491479°；coverage：0.996954208 / 0.964929088 / 0.993026059。阈值分别为3.51° / 7.35° / 5.46°。Expected深度绝对误差`0.025322810 m`，Median为`0.023696115 m`。数字判定：FAILED；15帧完整性：PASS。
- 四类代表图：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/four_classes/four_classes_10_80_150.png`；原几何代表图：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/normal_depth/comparison_0080.png`。四类视频：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/four_classes/rgb.mp4`、`alpha.mp4`、`normal.mp4`、`separation.mp4`。600张PNG与四段各150帧、400×400视频均完整解码通过；人工目检范围为四类10/80/150与原几何0080，不声称逐视频帧人工检查。解码证据：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/artifact_decode_audit.json`；目检证据：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/visual_review_completed.json`。
- RGB目检：人物与圆台可辨，长黑影与台面高亮持续；代表帧10／150腿足局部较短／截断，80俯视下肢与GT几何对比仍有明显差异。
- Alpha目检：人物与圆台组合轮廓连续，80俯视近满盘白；组合Alpha可掩盖台面区域内下肢几何缺失，不能凭轮廓完整替代原三指标FAILED。
- Normal目检：GS平台较平但小腿／足部与GT有明显差异，原0080ExpectedDepth及Mesh平台细密纹理和腿足误差持续；GS和Mesh原阈值失败。
- Separation目检：主体与圆台均匀橄榄黄绿feature门控，不代表真实运动分离。
- fresh parity、源码与实际native/tracer、checkpoint稳定性均通过；最终验收marker：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/evaluation_10000/FINAL_REVIEW_COMPLETED.json`。最终整体结论：**FAILED**。launcher的`state.json`仍保留训练结束时的`TRAINING_COMPLETED_PENDING_EVAL`状态；评测终态以此处FINAL_REVIEW_COMPLETED marker为准。失败checkpoint与所有渲染、数组、统计、日志均保留，没有覆盖或删除。

最终阶段说明：Stage1最终FAILED（GS与Mesh超阈值）；Stage2尚未训练。
