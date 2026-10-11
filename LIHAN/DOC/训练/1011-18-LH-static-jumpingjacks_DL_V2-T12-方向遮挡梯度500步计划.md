# 1011-18-LH-static-jumpingjacks_DL_V2-T12-方向遮挡梯度500步计划

## 实验目的与唯一变量

T12与T11组成同起点、同数据、同预算的方向梯度权重对照。T11使用逐点均匀权重，T12使用基于T10@13000原135帧方向遮挡诊断构造的逐点方向权重。除`direction_weight_mode=direction`这一变量外，入口、固定资产manifest、损失、路由、起点和优化器配置均与T11相同。

## 配置

- 起点：T10 `output/1011-22-LH-static-jumpingjacks_DL_V2-T10-expected续训-1k/model_mlp`，全局iteration 13000；新Adam优化器，动量为零。
- 数据：原135帧训练RGB、原图Alpha和对应训练方向，source=`output/1011-09-静态DL-Goal训练总控/data_view`。不读取测试帧或任何GT几何、材质、阴影和光强监督。
- 入口：`scripts.train_static_dl_direction_weighted`；`direction_weight_mode=direction`，route=`split`，loss=`censored`，`normal_depth_statistic=expected`；训练500步，保存相对步数250和500，即全局iteration 13250和13500；step100只写训练诊断日志，不额外保存检查点。
- 优化器：xyz LR `1e-5`、rotation LR `1e-4`、albedo LR `1e-3`、light LR `1e-4`；normal weight `0.1`、dist weight `1000`、alpha weight `0.1`；shadow bias `0.03`、shadow weight `0.25`、highlight weight `0.5`、residual threshold `0.08`、seed `1011`。
- 固定项：81323个点；opacity、scale、deform及增密流程冻结，不更新时间形变或时间颜色。T11/T12须引用同一份固定权重manifest SHA及同一核心快照SHA；路径和SHA待实际准备后记录。

## 资产与启动门槛

权重资产为`output/1011-24-LH-static-jumpingjacks_DL_V2-T10-原135帧方向遮挡诊断/fixed_direction_weights/manifest.json`，来自T10@13000原135训练帧零更新诊断，不以T08诊断或测试方向替代。逐帧逐点权重有限、形状正确、范围[0.5,2]，且以前景合成质量加权均值归一化为1；manifest记录公式及源检查点/方向/相机SHA。训练前须完成20步真实checkpoint梯度审计和fresh parity，两者均PASS后才可运行正式500步。稳定快照和完整CLI待实际执行后补录。

T12计划输出`output/1011-26-LH-static-jumpingjacks_DL_V2-T12-direction遮挡梯度500步`。独立快照为`output/1011-09-静态DL-Goal训练总控/snapshots/stage2-direction-weighted-native-assets-20261011`。输出目录由不覆盖启动器新建；保留13250、13500检查点、日志、state、marker、完整命令、诊断与SHA。评测范围由root/evaluator安排，不预填结果。任何失败输出均保留且不得覆盖。

## 状态

20步冒烟已于2026-10-11在GPU2正常结束，训练PID=2187194、supervisor=2187184，实际GPU UUID=`GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`；state记录exit0/gpu_verified=true，iteration13020。命令配置SHA256=`760f86631862be439675d536adefd8baa0cf383d29c9e3cdc848d1efca647426`。首帧真实梯度审计PASS，八项物理前向差0，weighted XYZ显式VJP、来源和冻结检查通过。evaluator fresh parity实际exit0/PASS，八项最大差0；见smoke的`render_parity`、`checkpoint_after.json`和`frozen_parameter_audit.json`。共享manifest SHA256=`e76bc81f6328505237e0dc628bcc9c77ad92c39d7f6dff3bc6ba4e00e2f05703`。

正式T12于2026-10-11T12:23:16.115022Z在GPU2启动，训练PID=2188348、supervisor=2188339、UUID=`GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`；2026-10-11T12:24:26.632839Z launcher exit0且gpu_verified=true。命令JSON SHA256=`aba5c38306269d34c55dbccc4c69e0e51f3e952938155d879cb4e04922262e6e`。与T11除model_path和direction_weight_mode外相同；runtime公平审计10项PASS，证据`output/1011-09-静态DL-Goal训练总控/T11_T12_runtime_fairness_audit.json`。真实native raster `.so` SHA256=`f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3`，tracer `.so` SHA256=`5b666e2b3b9fcae5b3137f3689b1ea43e44f2`，映射见`output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/T12_training_runtime_module_audit.json`。iteration13250/13500六个模型文件经10秒稳定SHA审计，见正式输出`checkpoint_completion_audit.json`。原15评测结果GS=5.267648622°、Depth=10.288382631°、Mesh=8.216063598°，coverage=0.999344693/0.967314036/0.996110616，完整性PASS、阈值FAILED。evaluator正在收尾150四类视频解码和目检。
