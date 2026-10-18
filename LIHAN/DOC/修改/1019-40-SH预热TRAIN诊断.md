# 独立SH预热TRAIN诊断

2026-10-19（Asia/Tokyo；主机UTC2026-10-18）。新增`scripts/diagnose_eldl_v3_sh_warmup.py`，只读P00完整135已见TRAIN及post-step预登记端点；公共训练/评测/验证器和资产均未修改。

对应报告45固定输入/端点/原SH目标与稳定性统计。主要边界：真实SH+d_color重建、1000边界/升阶、原Concrete随机gate固定诊断seed、RGBA Alpha黑背景合成、PLY与纯几何逐张量、几何头权重/输出严格零、同一次native allmap提取expected-Z、signed GS、各自支持/固定分母缺失180°和relative1惩罚。CPU计划允许未落盘PENDING；正式run拒绝未完成16000或不一致来源。no_gt_guard加精确文件路径登记，禁止eval_only/GT/TEST，不优化不挑终点。

CPU自检和准备计划结果将回填；当前GPU完整135/六端点与四类实际目检PENDING。根任务审核协议并明确GPU ready前不运行。训练计数与拓扑/reset语义不能等同逐参数更新数；本脚本原SH全图sRGB L1/DSSIM/Alpha目标不得与物理linear censored损失混称。


## CPU准备结果

AST、signed反向180°/固定缺失分母、深度/法线支持独立、零共同深度支持为null、真实no_gt_guard拒绝未登记tensor均PASS。未导入CUDA前向、未解码TRAIN图、未读取GT/TEST。收据：`output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热/train_diagnostic_cpu_checks_v1/receipt.json`，包含源码SHA和独立guard拒绝日志。

初次plan_v1绑定初版源码后保留；追加实际native扩展对象/二进制与训练SHA一致核验、/proc/maps独立记录、原c2w中心及每帧固定Alpha支持SHA后，重新冻结plan_v2。新计划SHA `eeedf6e53017e58a945ae1435caed26b53a2dcf6add69e5566d50576ff86447c`；源码SHA `664764e8b996b8bd0000e5a8ad5789eaac67fca00ee11db79d8038884f0f7237`。两份计划均PENDING_TRAINING_COMPLETION，缺22项，完整GPU/实际图像/精度均未通过也未运行。完成训练且根任务ready后须新计划，不复用PENDING。


## 有限性与CPU schema补核

按根审查建议，在nan_to_num前显式保留raw expected-Z finite mask，NaN/正负Inf统一0但不能转成有效支持，并记录原Alpha内非有限数；仅新独立脚本修订，无科学阈值变化。报告45同步明确该存储/有效性区别。旧v1/v2计划保留，最终准备plan_v3仍PENDING缺22项，SHA `90fc2c881afd9f541023850f36a74cb263bc20624d90a332c7d90966b77d9099`。

CPU直接构造与DeformModel相同DeformNetwork参数（is_blender/hyper_dim/pred_color），严格加载真实1000端点32张量，missing/unexpected均空，三个几何头全零，torch.cuda.is_initialized=False。与有限性修订后自检均PASS，更新既有CPU receipt；完整GPU仍未运行。当前源码SHA `e8169fd332c38fbcdac8b789879bd4175dfcd9daeb95166a0b992488fd9a6a8a`，报告45 SHA `7d615d9f49731554dfb6c9763470b6ac41a6ead69bf3e2978b6df1de089c540f`。后续启动后不再改绑定源码/协议。
