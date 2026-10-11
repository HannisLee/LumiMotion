# 1011-17-LH-static-jumpingjacks_DL_V2-T11-均匀遮挡梯度500步计划

## 实验目的与唯一变量

T11与T12使用同一T10@13000静态checkpoint、同一组135帧训练数据、同一份逐点权重资产及相同训练配置。两组都使用`route=split`和方向权重入口；T11设`direction_weight_mode=uniform`，T12设`direction_weight_mode=direction`。唯一实验变量是可见度到遮挡XYZ梯度的逐点乘数，物理前向、RGB损失、深度统计量和所有优化器参数保持一致。该对照不改变训练图像、GT监督或光照。

## 配置

- 起点：T10 `output/1011-22-LH-static-jumpingjacks_DL_V2-T10-expected续训-1k/model_mlp`，全局iteration 13000。两组各自加载同一PLY、deform和lighting检查点，新建Adam优化器且动量为零。
- 数据：`output/1011-09-静态DL-Goal训练总控/data_view`内原135帧RGB、原图Alpha和对应训练方向；不读取测试图像，不使用GT法线、深度、材质、阴影或光强监督。
- 入口：`scripts.train_static_dl_direction_weighted`；模式`uniform`；route=`split`；loss=`censored`；`normal_depth_statistic=expected`；训练500步，保存相对步数250和500，即全局iteration 13250和13500；step100只写训练诊断日志，不额外保存检查点。
- 优化器：xyz LR `1e-5`、rotation LR `1e-4`、albedo LR `1e-3`、light LR `1e-4`；normal weight `0.1`、dist weight `1000`、alpha weight `0.1`；shadow bias `0.03`、shadow weight `0.25`、highlight weight `0.5`、residual threshold `0.08`、seed `1011`。
- 固定项：81323个点；opacity、scale、deform、点数及增密流程冻结；不更新时间形变或时间颜色。训练新快照、方向权重manifest及资产SHA须由稳定代码和实际数据生成后补录。

## 启动门槛与输出

训练前先对T10@13000的135训练帧运行已验证的零更新方向诊断并生成固定逐点权重资产。资产采用前景合成质量加权均值归一化，逐帧均绑定T10@13000检查点与方向/相机SHA；两组共享manifest `output/1011-24-LH-static-jumpingjacks_DL_V2-T10-原135帧方向遮挡诊断/fixed_direction_weights/manifest.json`。固定权重公式为：令`A=(m>0.01)`，`g=smoothstep((mu-0.05)/0.2)*4V(1-V)`，对A内按前景质量m加权求`gbar`，`w_A=(1+g)/(1+gbar)`，A外权重为1；范围[0.5,2]且前景质量加权均值为1。manifest记录逐帧权重文件和源诊断/checkpoint SHA。

T11计划输出`output/1011-25-LH-static-jumpingjacks_DL_V2-T11-uniform遮挡梯度500步`。训练前先完成独立20步真实checkpoint梯度审计与fresh parity；均PASS后才能启动正式500步。独立快照为`output/1011-09-静态DL-Goal训练总控/snapshots/stage2-direction-weighted-native-assets-20261011`。输出目录应由不覆盖启动器新建，不得预建/复用失败目录。正式结束后保留13250、13500检查点、日志、state、marker、完整命令、训练诊断和每个产物SHA。评测范围待root/evaluator安排，不预填数值指标或验收结论。

## 状态

20步冒烟已于2026-10-11在GPU1正常结束，训练PID=2187193、supervisor=2187183、实际GPU UUID=`GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`；state记录exit0/gpu_verified=true，iteration13020。命令配置SHA256=`ca998309c3876fc0791e0582201944f0489891e3a15c7b225176dcf9fc576f0c`。首帧真实梯度审计PASS，八项物理前向差0，weighted XYZ显式VJP、来源和冻结检查通过。evaluator fresh parity实际exit0/PASS，八项最大差0；见smoke的`render_parity`、`checkpoint_after.json`和`frozen_parameter_audit.json`。共享manifest SHA256=`e76bc81f6328505237e0dc628bcc9c77ad92c39d7f6dff3bc6ba4e00e2f05703`。

正式T11于2026-10-11T12:23:16.129357Z在GPU1启动，训练PID=2188349、supervisor=2188338、UUID=`GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`；2026-10-11T12:24:26.669568Z launcher exit0且gpu_verified=true。命令JSON SHA256=`505585b0363f59768a82a48401fc325a30bf31224358d1d8d843ead58aa95219`。启动时与T12除model_path和direction_weight_mode外相同；root的runtime公平审计10项PASS，证据`output/1011-09-静态DL-Goal训练总控/T11_T12_runtime_fairness_audit.json`。真实native raster `.so` SHA256=`f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3`，tracer `.so` SHA256=`5b666e2b3b9fcae5b3137f3689b1ea43e44f2`，实际映射见`output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/T11_training_runtime_module_audit.json`。iteration13250/13500六个模型文件经10秒稳定SHA审计，见正式输出`checkpoint_completion_audit.json`。原15评测结果GS=5.275655755°、Depth=10.289938458°、Mesh=8.222855916°，coverage=0.999335464/0.967306652/0.996060776，完整性PASS、阈值FAILED。evaluator正在收尾150四类视频解码和目检。
