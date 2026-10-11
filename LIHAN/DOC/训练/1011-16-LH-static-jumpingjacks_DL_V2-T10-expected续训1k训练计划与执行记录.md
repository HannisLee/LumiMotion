# 1011-16-LH-static-jumpingjacks_DL_V2-T10-expected续训1k训练计划与执行记录

## 目标与控制变量

T10在T08 expected设置上固定配置再延长1000步，观察已见连续改善趋势能否保持。它是从T08@12000加载权重后以新优化器、零Adam动量开始的一段独立续训，不接续上一段优化器状态；保留T08原12000模型和本输出全部检查点。实验只沿用T08已验证的深度统计量选择，其他训练参数不变。

## 训练计划

- 初始化模型：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-20-LH-static-jumpingjacks_DL_V2-T08-深度统计expected-1k/model_mlp`，iteration 12000；新的优化器，Adam动量清零。
- 数据：原静态DL 135帧RGB、原图自身Alpha和对应训练光照方向，source=`output/1011-09-静态DL-Goal训练总控/data_view`。不使用EL图像、GT法线、深度、材质、阴影、光强等训练监督。
- 入口：`scripts.train_static_dl_depth_statistic`；复用已核验91文件snapshot `/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/snapshots/stage2-depth-statistic-native-assets-20261011`，经`bootstrap_1011.py`绑定仓库native CUDA。
- 参数：`route=control`、`loss_model=censored`、`normal_depth_statistic=expected`；xyz LR=1e-5、rotation LR=1e-4、albedo LR=1e-3、light LR=1e-4；normal=.1、dist=1000、alpha=.1、shadow_bias=.03、shadow=.25、highlight=.5、residual_threshold=.08、seed=1011。
- 冻结约束：保持81323点；opacity、scale、deform不更新，无增密，XYZ/旋转/材质/光照依既定LR训练。
- 预算：1000步，保存全局iteration12250、12500、13000。最终13000运行完整15帧协议、四类150帧可视化、完整解码及代表帧目检；中间检查点按需要补几何诊断，不预承诺重复视频。

## 20步起点冒烟与启动门槛

冒烟从T08@12000初始化，20步保存iteration12020，配置与正式训练一致。GPU1计划先运行，真实首帧审计通过后交evaluator执行fresh parity；仅两者PASS后才启动正式1k。冒烟输出为`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1011-22-staticDL-T10-expected-20步续训冒烟`，正式输出为`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-22-LH-static-jumpingjacks_DL_V2-T10-expected续训-1k`。独立目录已预建，任何失败材料均保留。

完整命令和child环境分别记录于各目录`commands/train.json`、`commands/train.sh`。正式评测完整命令由evaluator写入`evaluation_13000/plan.json`及`executed.json`；训练日志在`logs/train.log`，检查点、marker与SHA在`model_mlp/`及completion audit。

## 实际执行记录

当前状态：20步smoke、入口审计及fresh parity已PASS；正式T10训练已在GPU1启动。

冒烟启动命令：`bash /home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1011-22-staticDL-T10-expected-20步续训冒烟/commands/train.sh`。启动时间2026-10-11约11:48 UTC，训练PID=2173649、supervisor PID=2173644，GPU1 UUID=`GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`；启动器已实际核验gpu_verified=true，launcher exit0，iteration12020。首帧统计审计与公共审计PASS；stable模型SHA见`checkpoint_12020_ready.json`。fresh parity待evaluator执行。日志与退出码见冒烟目录`logs/train.log`和`state.json`。

Fresh parity证据：`output/smoke_test/1011-22-staticDL-T10-expected-20步续训冒烟/render_parity/parity.json`八输出均PASS且最大差0。evaluator另做独立10秒SHA稳定性复核；首轮三文件hash记录位于`checkpoint_12020_ready.json`，只作为hash引用，evaluator复核marker单独保存。

正式训练已于2026-10-11T11:50:24Z启动，GPU1 UUID=`GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`，child PID=2174378、supervisor PID=2174373，命令SHA=`c054bcdcb80a99381631f87253baae9159aaf89e56e6fef00ebda6f46f2aa899`；实时state与日志见`output/1011-22-LH-static-jumpingjacks_DL_V2-T10-expected续训-1k/state.json`、`logs/train.log`。

独立稳定性复核已完成：`checkpoint_12020_eval_ready.json`记录10秒READY_STABLE；随后GPU2运行`render_parity_stable_recheck`并exit0/PASS，8map max_abs=0，完整性与来源/实际native模块匹配。冻结项81323点、opacity、scale、deform均精确PASS。evaluator保存的二次审计、NPZ、完整命令和日志均保留；首次READY schema不符及拒覆盖记录不覆盖。

正式训练已于2026-10-11T11:50:24Z启动并于2026-10-11T11:51:55.297962+00:00正常结束，launcher exit0/gpu_verified=true，GPU1 UUID=`GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`，训练PID=2174378。12250/12500/13000共9个PLY/deform/lighting经过10秒稳定SHA审计，见`output/1011-22-LH-static-jumpingjacks_DL_V2-T10-expected续训-1k/checkpoint_completion_audit.json`。真实step1/250/500/1000训练诊断、snapshot和输入来源SHA见`output/1011-09-静态DL-Goal训练总控/T10_training_diagnostics.json`。终点评测原15指标已完成，完整性PASS/数值FAILED；四类视频解码与人工目检仍由evaluator收尾。

原15帧终点评测（iteration13000）真实完成，三个子步骤exit0、完整性PASS/阈值FAILED：GS=5.236901942°（coverage=0.999326234），Depth=10.437567496°（coverage=0.967290039），Mesh=8.242430432°（coverage=0.996055238）；expected depth absolute error=0.038771129米。数值摘要`output/1011-22-LH-static-jumpingjacks_DL_V2-T10-expected续训-1k/evaluation_13000/goal_mae_summary.json`。相较T08@12000三项略降但改善趋缓，未达到目标阈值。T10最终fresh八项差0和真实train/eval native+tracer映射一致证据已落`evaluation_13000/render_parity`与`native_binding_audit.json`；完整150四类目检待最终marker。
