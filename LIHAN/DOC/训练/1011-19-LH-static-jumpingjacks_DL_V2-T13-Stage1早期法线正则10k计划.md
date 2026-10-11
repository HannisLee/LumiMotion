# 1011-19-LH-static-jumpingjacks_DL_V2-T13-Stage1早期法线正则10k计划

## 研究问题与对照关系

T13/T14从同一静态DL Stage1随机初始化和同一135帧源数据从头训练，比较法线一致性正则开始步数。T13的`start_normal_reg=2000`，T14为8000；其余入口、初始化点云、训练图像、优化器、损失权重与日程完全一致。该实验衡量法线正则的启用日程效应，不与T11/T12续训组作单因素比较。

## T13配置

- 输出：`output/1011-27-LH-static-jumpingjacks_DL_V2-T13-normal-reg-2000-Stage1-10k`。
- 入口：`scripts.train_static_dl_early_geometry`；入口实现尚未稳定，快照、完整CLI和smoke命令待稳定后记录。
- 数据：沿用Goal原`output/1011-09-静态DL-Goal训练总控/data_view`的135帧原图RGB、RGBA原Alpha、训练方向及正确训练相机；训练不读测试图、GT法线、深度、材质、阴影或光强监督。相机与方向资产SHA启动前记录。
- 初始化：原Stage1共享的100000个随机点`points3d.ply`，seed=0；T13/T14必须使用同一初始化文件SHA，不从已有Stage2模型初始化。原点云由确定性随机初始化产生，实际路径及SHA待入口读取并记入prelaunch审计。
- 预算：10000步；保存相对step 5000、8000、9000、10000。densify截止步仍为20000，故本轮10k内不触发截止；沿用原增密和opacity reset设定，不新增阶段参数。
- 正则日程：`start_normal_reg=2000`；`dist_start=8000`、dist系数1000；normal权重0.1。T14除start_normal_reg外一致，使用8000。
- 物理与损失：静态方向光物理分支，physical start=1000、visibility start=5000、BVH interval=1、shadow bias=0.03；censored RGB损失；shadow weight=0.25、highlight weight=0.5、residual threshold=0.08；alpha weight=0.1。深度一致性统计量expected仅用于独立几何副本，RGB前向depth_ratio固定1。
- 其余沿用原T03 Stage1配置：deform-type mlp，resolution2，is_blender/eval/load2gpu_on_the_fly/gt_alpha_mask_as_scene_mask，训练和测试light folder均为images；lambda separation=0.001、d_xyz loss weight=0.001、binarization warm-up=1000、d_color regularizer=0.01；XYZ LR沿用原30k日程不压缩，rotation=0.001、albedo=0.01、light=0.001，随机seed及其他参数须由入口读取与启动命令锁定。

## 前置门槛与检查点

入口稳定后建立独立快照，先用最小必要smoke覆盖真正物理分支、法线早启分支、dist晚启分支；不接受只跑SH预热而未触发所需分支的冒烟。真实首步/边界步审计、checkpoint输入SHA、GPU/native来源与fresh parity通过后，才启动正式T13/T14配对。保存5000、8000、9000、10000；重点比较共同检查点8000、9000、10000，评测范围由root/evaluator安排，不预填指标。输出保留完整命令、配置、日志、状态、模型与SHA；失败结果不得覆盖。

## 状态

仅完成预注册，尚未启动。训练入口和核心模块仍待planner标记stable；不得据此文档提前创建snapshot或输出目录。
