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


## 首次GPU失败保留与设备修复

v1首个端点首帧BCE发现预测CUDA而Alpha仍CPU，exit1，无完整端点统计。Camera.load2device不移动Alpha；原训练显式.cuda。v1输出已冻结原e8169fd脚本、报告45、plan/环境/完整日志及FAILED_ENGINEERING.json。仅在独立脚本objective和统计两处将Alpha .to(预测device)，target也显式同render设备；值/dtype/阈值/目标与科学协议不变，公共Camera与旧文件未改。

根审查允许后CPU自检PASS，新READY v2 plan SHA `0c90f89a46c75cf59ab72dfe5d136116b9b8bcdc6050370c9a50ee0573a16e2d`（零PENDING）；报告45原SHA不变，源码SHA `b2943cf92bd72d550e46c8f92b1a455893f5243647e659e085c14a74343475bd`。新train_diagnostic_v2由supervisor_v2运行GPU0，完整数值/72图及实际目检仍PENDING，结束后报告48回填。


## 正式v2运行与实际查看完成

根任务审核父几何工程PASS与全落盘READY后，GPU0完整六端点×135已见TRAIN运行exit0（UTC19:33:03—19:34:01，session4268回收）；GPU0释放。72张RGB/Alpha/世界Normal/binary gate代表PNG完整，通过6张原分辨率联系图查看全部72面板，并绑定manual_review.json。原Alpha固定分母4884194、187输入SHA/全部mask/PLY↔geometry/严格零几何输出/实际native二进制均通过。

summary SHA `ed2a0be1ddffdf622c33fd651dd8fcf2f00d7f580056b67ff70c8fc4257b7c00`、manual SHA `ba367342f86564ff6bd71e00fa5ad1373d4b5f9c534d58f16c85e0ed241fb775`。数值原版另存numeric_summary_before_manual_review.json。结果与限制见报告48，完整命令/日志/环境/定量/四类实际结论见train_diagnostic_v2/README.md。16k外观较1k清楚、轮廓连续，平台normal仍起伏；12k/15k真实reset低覆盖不删。最终PASS仅TRAIN-only诊断与实际查看，无GT/TEST/准确性结论、不选择初始化、不导出灯。报告45与已启动源码SHA均未再改。
