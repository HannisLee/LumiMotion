# 1011-08-LH-static-jumpingjacks_DL_V2-T02-无遮挡方向Lambertian训练计划

服务器ushas，环境lumimotion-ushas。目标是使用静态DL原135帧RGB/Alpha训练，允许读取相同训练帧的GT方向向量；禁止任何GT几何、材质、阴影、光强等监督。测试15帧方向只留在eval_only，训练不读取。

模式：`lambertian`；physical_start=1000；visibility_start=5000；BVH interval=1。约束为静态几何不发生时变形变；训练光方向取自`output/1011-09-静态DL-Goal训练总控/data_view/gt_directions_train.json`。输入点云使用共享100000点随机方块，seed=0，SHA256 `f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`。数据源仅原image PNG RGBA、相机和训练方向；alpha为0的RGB置黑。

训练总步数35000；checkpoint为10000、20000、30000、35000；训练代码快照`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/snapshots/T01-sh-native-assets-20261011`，启动通过仓库`LIHAN/scripts/bootstrap_1011.py`绑定native CUDA rasterizer。训练命令和参数完整记录在`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-11-LH-static-jumpingjacks_DL_V2-T02-方向Lambertian无遮蔽/commands/train.sh`。输出目录`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-11-LH-static-jumpingjacks_DL_V2-T02-方向Lambertian无遮蔽`，模型为`model_mlp`，日志为`logs/train.log`。

20步物理冒烟已完成：T02路径的xyz/rotation/albedo/log_intensity梯度有限非零，T03路径trace_rays=100000、首步visibility梯度非零；峰值显存余量足够。等待fresh/fresh parity检查通过后启动。正式输出需记录进程PID/starttime/exit code、每100步物理日志、checkpoint与native光栅源/so SHA。正式Goal评测按协议记录定量指标、实际渲染输出和四类可视化目检；当前计划状态：RUNNING；已通过fresh/fresh parity验证，正式训练PID与GPU见对应output state.json。
