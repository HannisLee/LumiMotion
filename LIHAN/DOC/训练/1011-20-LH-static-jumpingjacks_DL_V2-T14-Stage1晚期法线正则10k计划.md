# 1011-20-LH-static-jumpingjacks_DL_V2-T14-Stage1晚期法线正则10k计划

## 研究问题与控制变量

T14是T13的配对控制：两组从相同静态DL随机初始化、使用相同135帧、相同10000步预算和相同所有物理/损失/优化器设置。唯一变量是法线一致性正则的开始步数：T14在step8000启用，T13在step2000启用。该设计只检验normal-reg启用日程，distortion日程两组完全相同。

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

入口稳定后，T13/T14分别用独立模型目录、同一快照和共享初始化SHA。smoke必须实际越过物理、T14 normal-reg及dist-reg启用边界，并由梯度/状态检查确认，而非仅执行SH预热。首帧和边界审计、checkpoint来源、native实现及fresh parity均通过后才可正式运行；停止在T14专属normal启用前的共同行为及启用后差异均需如实记录。所有日志、checkpoint、配置和SHA留存，失败不覆盖。

T14计划输出目录为`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal-reg-8000-Stage1-10k`。T14烟测`output/smoke_test/1011-26-staticDL-T14-normal8000-smoke-schedule-20步物理日程冒烟`已于2026-10-11在GPU2 exit0/gpu_verified=true，child PID2195278、supervisor2195267，command SHA256=`bb64044c3d13fd77c97e17275bbc2c3f700efea9f1e47c64d0fad85895017d46`，保存iteration20。smoke schedule在step1触发物理/V但normal/dist关闭，step11二者均启用；两次真实梯度分支审计记录100000条trace rays且允许梯度有限非零。完整命令、状态、日志、审计见smoke output。独立fresh parity已通过：`output/smoke_test/1011-26-staticDL-T14-normal8000-smoke-schedule-20步物理日程冒烟/render_parity/parity.json`，8项输出最大差为0。正式T14已启动，详见下方执行记录。


## 正式启动记录

- 实际输出目录：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k`。预注册方案中的目录名曾写作`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal-reg-8000-Stage1-10k`；该目录名与Goal/watch使用的实际目录不一致，未用于训练，实际目录以launcher state、marker和下面路径为准，未在旧目录训练或覆盖结果。
- 启动时间：2026-10-11T12:42:04Z；child PID `2197353`，supervisor PID `2197344`；物理GPU `GPU2`（UUID `GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`），Conda Python `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。启动后`gpu_verified=true`，训练仍在运行，当前尚无退出码。
- 正式marker：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/model_mlp/early_geometry_training.json`；实际记录`normal_start=8000`、`dist_start=8000`、`smoke_schedule=false`、`iterations=10000`、保存步数5000/8000/9000/10000。T13/T14启动argv唯一实验变量为`--start_normal_reg 8000`，其余科学配置一致；root公平性复核待marker齐全后进行。
- 完整命令：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/commands/train.json`；命令SHA256=`022e85003a3a0d2059d128fd98a09b43879ee456b8a9051d42421ec212a99d81`；训练日志：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/logs/train.log`；实时状态：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/state.json`。原生训练源码/.so运行时采集：`output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/T14_training_runtime_module_audit.json`，审计状态由evaluator记录为PASS_CAPTURED。
- Smoke及fresh：T13/T14均20步、实际物理/V/计划的normal/dist分支覆盖并exit0；T13 fresh首个helper读取错误保留，`render_parity_retry1`实际exit0/PASS；T14 `render_parity` exit0/PASS。fresh八项最大差均为0。Smoke阶段训练进程退出后未采集其maps，文档不将fresh运行maps冒充训练maps。
- 本次为Stage1从头训练，允许原Stage1的增密与opacity/scale更新；没有冻结它们。实际训练仅优化135训练相机；15个测试相机由原加载器管理/评测，不作为训练监督或进入loss/优化。
- 终点评测计划以实际目录为准：稳定的iteration10000后由evaluator执行原15帧协议与150方向四类可视化；中间checkpoint是否补评不在本计划承诺范围。定量指标、decode、目检和最终PASS/FAILED待实际评测后记录。


## 正式启动记录

- 实际输出目录：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k`。预注册方案中的目录名曾写作`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal-reg-8000-Stage1-10k`；该名称未用于训练，实际目录以launcher state、marker和下列路径为准，没有在旧目录训练或覆盖结果。
- 启动时间：2026-10-11T12:42:04Z；child PID `2197353`，supervisor PID `2197344`；物理GPU `GPU2`（UUID `GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`），Conda Python `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。启动后`gpu_verified=true`，训练仍在运行，当前尚无退出码。
- 正式marker：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/model_mlp/early_geometry_training.json`；实际记录`normal_start=8000`、`dist_start=8000`、`smoke_schedule=false`、`iterations=10000`、保存步数5000/8000/9000/10000。T13/T14启动argv唯一实验变量为`--start_normal_reg`（T13=2000，T14=8000），其余科学配置一致；root公平性复核待marker齐全后进行。
- 完整命令：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/commands/train.json`；命令SHA256=`022e85003a3a0d2059d128fd98a09b43879ee456b8a9051d42421ec212a99d81`；训练日志：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/logs/train.log`；实时状态：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal8000-Stage1晚期法线正则10k/state.json`。原生训练源码/.so运行时采集：`output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/T14_training_runtime_module_audit.json`，evaluator记录审计状态PASS_CAPTURED。
- Smoke及fresh：T13/T14均20步、实际物理/V/normal/dist计划分支覆盖并exit0；T13 fresh首个helper读取错误留档，`render_parity_retry1`实际exit0/PASS；T14 `render_parity` exit0/PASS。fresh八项最大差均为0。Smoke训练进程退出后未采集其maps，不把fresh运行maps当作训练maps。
- 本次为Stage1从头训练，允许原Stage1增密与opacity/scale更新；没有冻结它们。只优化原135训练相机；source中的15个测试相机由原加载器管理/评测，不作为训练监督或进入loss/优化。
- 终点评测以实际目录为准：稳定的iteration10000后由evaluator执行原15帧协议与150方向四类可视化；中间checkpoint是否补评不在本计划承诺范围。定量指标、解码、目检和最终PASS/FAILED待实际评测后记录。


## 当前训练边界审计

T14在step5000的审计显示normal_active=false、dist_active=false、trace_rays=44380，XYZ/rotation/albedo/light梯度均有限非零。对应日志`logs/train.log`与`model_mlp/static_dl_gradient_audit/`保留；后续step8001启用边界待实际到达后核对。
