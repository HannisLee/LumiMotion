# 1011-19-LH-static-jumpingjacks_DL_V2-T13-Stage1早期法线正则10k计划

## 研究问题与对照关系

T13/T14从同一静态DL Stage1随机初始化和同一135帧源数据从头训练，比较法线一致性正则开始步数。T13的`start_normal_reg=2000`，T14为8000；其余入口、初始化点云、训练图像、优化器、损失权重与日程完全一致。该实验衡量法线正则的启用日程效应，不与T11/T12续训组作单因素比较。

## T13配置

- 输出：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k`。
- 入口：`scripts.train_static_dl_early_geometry`；稳定快照为`output/1011-09-静态DL-Goal训练总控/snapshots/stage1-early-geometry-native-assets-20261011`，manifest SHA256=`e377849e28d803f6ab70282d666ebe8eaa8641bfb575aa7b9856bdd5b2eecfd2`；完整正式CLI记录到prelaunch train.json。
- 数据：沿用Goal原`output/1011-09-静态DL-Goal训练总控/data_view`的原135训练相机、RGB、原图Alpha和方向；只采样这135个训练相机进行优化。source中的15个测试相机仅由原加载器管理/评测，不作为训练监督或进入loss/优化。禁止GT法线、深度、材质、阴影或光强监督；相机与方向资产SHA启动前记录。
- 初始化：原Stage1共享的100000个随机点`points3d.ply`，seed=0；T13/T14必须使用同一初始化文件SHA，不从已有Stage2模型初始化。原点云由确定性随机初始化产生，实际路径及SHA待入口读取并记入prelaunch审计。
- 预算：10000步；保存相对step 5000、8000、9000、10000。densify截止步仍为20000，故本轮10k内不触发截止；沿用原增密和opacity reset设定，不新增阶段参数。
- 正则日程：`start_normal_reg=2000`；`dist_start=8000`、dist系数1000；normal权重0.1。T14除start_normal_reg外一致，使用8000。
- 物理与损失：静态方向光物理分支，physical start=1000、visibility start=5000、BVH interval=1、shadow bias=0.03；censored RGB损失；shadow weight=0.25、highlight weight=0.5、residual threshold=0.08；alpha weight=0.1。深度一致性统计量expected仅用于独立几何副本，RGB前向depth_ratio固定1。
- 其余沿用原T03 Stage1配置：deform-type mlp，resolution2，is_blender/eval/load2gpu_on_the_fly/gt_alpha_mask_as_scene_mask，训练和测试light folder均为images；lambda separation=0.001、d_xyz loss weight=0.001、binarization warm-up=1000、d_color regularizer=0.01；XYZ LR沿用原30k日程不压缩，rotation=0.001、albedo=0.01、light=0.001，随机seed及其他参数须由入口读取与启动命令锁定。

## 前置门槛与检查点

入口稳定后建立独立快照，先用最小必要smoke覆盖真正物理分支、法线早启分支、dist晚启分支；不接受只跑SH预热而未触发所需分支的冒烟。真实首步/边界步审计、checkpoint输入SHA、GPU/native来源与fresh parity通过后，才启动正式T13/T14配对。保存5000、8000、9000、10000；重点比较共同检查点8000、9000、10000，评测范围由root/evaluator安排，不预填指标。输出保留完整命令、配置、日志、状态、模型与SHA；失败结果不得覆盖。

## 状态

入口已stable，独立94文件快照已创建；共享source点云与随机初始化SHA一致。T13烟测`output/smoke_test/1011-25-staticDL-T13-normal2000-smoke-schedule-20步物理日程冒烟`已于2026-10-11在GPU1 exit0/gpu_verified=true，child PID2195277、supervisor2195268，command SHA256=`0f81ec625fa6038f62b07c780b88ef5563a27b014bc0be436996b10bb7d4d12e`，保存iteration20。smoke schedule在step1触发物理/V/normal，dist从step11触发；两个真实梯度分支审计显示100000条trace rays且XYZ/rotation/albedo/light梯度有限非零。完整命令、状态、日志、审计见smoke output。Fresh parity已通过：`output/smoke_test/1011-25-staticDL-T13-normal2000-smoke-schedule-20步物理日程冒烟/render_parity_retry1/parity.json`，8项输出最大差为0；首轮helper字段读取失败留档，修正后重试通过。正式T13已启动，详见下方执行记录。


## 正式启动记录

- 实际输出目录：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k`。预注册方案中的目录名曾写作`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal-reg-2000-Stage1-10k`；该目录名与Goal/watch使用的实际目录不一致，未用于训练，实际目录以launcher state、marker和下面路径为准，未在旧目录训练或覆盖结果。
- 启动时间：2026-10-11T12:42:04Z；child PID `2197354`，supervisor PID `2197343`；物理GPU `GPU1`（UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`），Conda Python `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。启动后`gpu_verified=true`，训练仍在运行，当前尚无退出码。
- 正式marker：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp/early_geometry_training.json`；实际记录`normal_start=2000`、`dist_start=8000`、`smoke_schedule=false`、`iterations=10000`、保存步数5000/8000/9000/10000。T13/T14启动argv唯一实验变量为`--start_normal_reg 2000`，其余科学配置一致；root公平性复核待marker齐全后进行。
- 完整命令：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/commands/train.json`；命令SHA256=`add7c3c12c1c718e1928c3278ff95df8e6c7b37253ae1fe503cd8ffa2a17cf3b`；训练日志：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/logs/train.log`；实时状态：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/state.json`。原生训练源码/.so运行时采集：`output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/T13_training_runtime_module_audit.json`，审计状态由evaluator记录为PASS_CAPTURED。
- Smoke及fresh：T13/T14均20步、实际物理/V/计划的normal/dist分支覆盖并exit0；T13 fresh首个helper读取错误保留，`render_parity_retry1`实际exit0/PASS；T14 `render_parity` exit0/PASS。fresh八项最大差均为0。Smoke阶段训练进程退出后未采集其maps，文档不将fresh运行maps冒充训练maps。
- 本次为Stage1从头训练，允许原Stage1的增密与opacity/scale更新；没有冻结它们。实际训练仅优化135训练相机；15个测试相机由原加载器管理/评测，不作为训练监督或进入loss/优化。
- 终点评测计划以实际目录为准：稳定的iteration10000后由evaluator执行原15帧协议与150方向四类可视化；中间checkpoint是否补评不在本计划承诺范围。定量指标、decode、目检和最终PASS/FAILED待实际评测后记录。


## 正式启动记录

- 实际输出目录：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k`。预注册方案中的目录名曾写作`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal-reg-2000-Stage1-10k`；该名称未用于训练，实际目录以launcher state、marker和下列路径为准，没有在旧目录训练或覆盖结果。
- 启动时间：2026-10-11T12:42:04Z；child PID `2197354`，supervisor PID `2197343`；物理GPU `GPU1`（UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`），Conda Python `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。启动后`gpu_verified=true`，训练仍在运行，当前尚无退出码。
- 正式marker：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/model_mlp/early_geometry_training.json`；实际记录`normal_start=2000`、`dist_start=8000`、`smoke_schedule=false`、`iterations=10000`、保存步数5000/8000/9000/10000。T13/T14启动argv唯一实验变量为`--start_normal_reg`（T13=2000，T14=8000），其余科学配置一致；root公平性复核待marker齐全后进行。
- 完整命令：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/commands/train.json`；命令SHA256=`add7c3c12c1c718e1928c3278ff95df8e6c7b37253ae1fe503cd8ffa2a17cf3b`；训练日志：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/logs/train.log`；实时状态：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal2000-Stage1早期法线正则10k/state.json`。原生训练源码/.so运行时采集：`output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/T13_training_runtime_module_audit.json`，evaluator记录审计状态PASS_CAPTURED。
- Smoke及fresh：T13/T14均20步、实际物理/V/normal/dist计划分支覆盖并exit0；T13 fresh首个helper读取错误留档，`render_parity_retry1`实际exit0/PASS；T14 `render_parity` exit0/PASS。fresh八项最大差均为0。Smoke训练进程退出后未采集其maps，不把fresh运行maps当作训练maps。
- 本次为Stage1从头训练，允许原Stage1增密与opacity/scale更新；没有冻结它们。只优化原135训练相机；source中的15个测试相机由原加载器管理/评测，不作为训练监督或进入loss/优化。
- 终点评测以实际目录为准：稳定的iteration10000后由evaluator执行原15帧协议与150方向四类可视化；中间checkpoint是否补评不在本计划承诺范围。定量指标、解码、目检和最终PASS/FAILED待实际评测后记录。


## 当前训练边界审计

T13在step2001的审计显示normal_active=true、dist_active=false；step5000 normal_active=true、dist_active=false、trace_rays=49353，XYZ/rotation/albedo/light梯度均有限非零。对应日志`logs/train.log`与`model_mlp/static_dl_gradient_audit/`保留。
