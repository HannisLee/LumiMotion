# JumpingJacks env阶段结果检查与方向光Stage1启动

日期：2026-10-09，venus / lumimotion。

对env完成的Stage1检查35k PLY属性有限性、7个150帧视频实际全解码，提取首中尾及frame10同视角同时间GT对照，读取TensorBoard原始精度子集指标；真实目检显示RGB拟合较好，但人体法线斑驳、SH基色含明暗残留、固定相机中间时刻有局部拖影。结果与限制写回1009-01训练报告与实验README。Stage2仍在训练，未伪报最终结果。

用户授权direction_light仅Stage1。独立1009-02运行与data_view，原版训练配置与env Stage1相同，从头默认10万随机方块；后台run_stage1.py顺序执行35k和insights，自动核查点云/deformation/初始化和7个视频完整帧数，失败停止并保留所有产物。脚本位于实验commands，本次未改原版训练或渲染代码。

[方向光训练报告](../训练/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1训练报告.md)。
