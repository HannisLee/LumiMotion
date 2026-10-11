# 1011-09-LH-static-jumpingjacks_DL_V2-T03-BVH方向光可见度训练计划

服务器ushas，环境lumimotion-ushas。目标是使用静态DL原135帧RGB/Alpha训练，允许读取相同训练帧的GT方向向量；禁止任何GT几何、材质、阴影、光强等监督。测试15帧方向只留在eval_only，训练不读取。

模式：`lambertian_visibility`；physical_start=1000；visibility_start=5000；BVH interval=1。约束为静态几何不发生时变形变；训练光方向取自`output/1011-09-静态DL-Goal训练总控/data_view/gt_directions_train.json`。输入点云使用共享100000点随机方块，seed=0，SHA256 `f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`。数据源仅原image PNG RGBA、相机和训练方向；alpha为0的RGB置黑。

训练总步数35000；checkpoint为10000、20000、30000、35000；训练代码快照`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/snapshots/T01-sh-native-assets-20261011`，启动通过仓库`LIHAN/scripts/bootstrap_1011.py`绑定native CUDA rasterizer。训练命令和参数完整记录在`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-12-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH/commands/train.sh`。输出目录`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-12-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH`，模型为`model_mlp`，日志为`logs/train.log`。

20步物理冒烟已完成：T02路径的xyz/rotation/albedo/log_intensity梯度有限非零；T03首步执行trace_rays=100000，XYZ/q/albedo/light总loss梯度有限非零。visibility值作为硬可见度结果，不对其反向传播，不能称visibility自身有非零梯度。正式输出需记录进程PID/starttime/exit code、每100步物理日志、checkpoint与native光栅源/so SHA。正式Goal评测按协议记录定量指标、实际渲染输出和四类可视化目检；当前计划状态：RUNNING；fresh/fresh parity七项最大绝对差为0，训练/评测source、.so与四个python实现SHA一致。正式目录分别为output/1011-14…和output/1011-15…；监督器保存PID、child环境、物理GPU UUID、pmon核验和最终退出码。


## 调度异常与新编号重试（2026-10-11T07:44:02.517559+00:00）

旧正式目录`output/1011-12-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH`的启动器未将`gpu_physical=2`映射到子进程`CUDA_VISIBLE_DEVICES`；`nvidia-smi pmon`确认其进程实际落在GPU0，与T01共卡。经核验PID与命令后发送SIGTERM，退出码-15。保留该目录日志、cfg、输入点云和梯度/物理诊断；这些中断结果不纳入正式实验。

新重试使用独立目录`output/1011-15-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH-GPU2重试`，不覆盖旧失败现场。标准启动器必须显式传递`child_env.CUDA_VISIBLE_DEVICES`，并记录子进程environ和pmon物理卡核验结果。参数不变：35000步，10000/20000/30000/35000保存；物理在1000步切换，可见度在5000步切换；基础normal/distortion regularization起点维持仓库默认8000；T03 BVH interval=1。共享点云及训练输入不变。


## 正式重试进度与10k评测（2026-10-11）

有效训练在独立目录`output/1011-15-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH-GPU2重试`，由`LIHAN/scripts/goal_train_job.py`监督；PID 1985466，物理GPU2 UUID `GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`，`gpu_verified=true`。旧#12错误GPU结果单独保留，未计入本实验。当前训练继续至35000步，完整启动器命令、子进程环境、日志、checkpoint和最终退出码记录于该目录。

10k checkpoint完整15帧评测：`output/1011-15-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH-GPU2重试/evaluation_10000/goal_mae_summary.json`显示完整性PASS，数值阈值FAILED。GS-render MAE 6.604177153°（coverage 0.998091304）；depth-normal MAE 12.780289326°（coverage 0.966031111）；mesh MAE 8.617989945°（coverage 0.995224567）。Goal阈值分别3.51°、7.35°、5.46°。15帧网格/法线数组、逐帧CSV和日志完整保留；代表mesh帧10、80、150已目检，四类渲染可视化暂缓至后续评测。中期10k数值未达标，训练继续到35000步，最终结论待最终checkpoint。
