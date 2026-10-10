
## B08完整20k实际启动（2026-10-10T09:11:05.296325+09:00）

真实PID282299/startticks10140426，exec76498，source35k→55k20000步、Preset pbr_gt_directional_plane_native_direct_gate；当前生产core473双snapshot SHA已锁定，不续B07/smoke权重。后处理PID282335/exec89846及40k预览PID282338/exec88932绑定同一真实train PID/ticks/argv，不自动重试。9步全量render/15七MAE/复算/文档已准备；真实train_runtime/process_handles保存。当前仍运行，无B08最终MAE或质量/7°结论，未commit/push/新分支。

## B08真实启动只读审计（2026-10-10T09:12:42.793456+09:00）

launch_audit实际exit0。473核心及双snapshot逐SHA一致；cfg仅8差异（3路径/Preset与4 gate字段），geometry warmup2000正式保持，source35k PLY/MLP SHA/32 NN optimizer0与feature组启用、方向表byte同B07白色单位、source plane全部数组bit同B07/15187支持、150帧自身RGBA Alpha provenance PASS；真实PID282299/ticks10140426/argv/cwd仍一致。该审计0优化、无GT标签或B08正式MAE。40k/50k/55k只读检查排入audit_checkpoints_when_saved.py，实际PID282503/exec3502绑定同train，未保存前退出则失败不重启；9步后处理exec89846等待本轮完成。

## B08训练期间B07全15原域误差分解（2026-10-10T09:14:41.800008+09:00）

只读取已完成B07的15张离线评测npz，实际exit0，原GS16.81871120°/GT415433/有效394675精确重组。按GT世界法线距固定source正plane轴有向夹角分箱，非GT物体语义：0–5°占有效75.53075315%，该箱MAE10.04343114°、贡献原MAE7.58587918°；5–30°占1.07607525%，MAE39.33107411°、贡献.42323195°；>30°占23.39317160%，MAE37.65885282°、贡献8.80960006°。这说明按B07分布，单把近plane方向组改善到0而保持其它组误差也不足7°，但不是B08预测或新评分，必须同时处理其它朝向；B08 gate对全部99827点开放，不仅平台候选。

≥30°误差占20.24881231%有效像素、贡献10.50594465°；完整各箱组成原域，没有筛帧/点/难像素，不生成任何GT训练监督/mask/prior，不修改live训练。命令commands/diagnose_b07_offline_error.py、日志logs/b07_offline_error_decomposition.log、证据inspection/b07_offline_error_decomposition.json及15来源npz SHA。GT只有本离线分析读取，0 optimizer。当前B08最终MAE/质量待完成。

## B08早期训练曲线实际快照（2026-10-10T09:16:37.128262+09:00）

固定事件字节快照到3663个Stage2步，100步均值photo L1/total loss及训练RGB拟合正E与B07曲线早期几乎重合。两轮Direct SUN/GGX/BVH/alpha>.9全可用photo一致，B08仅开放输出gate且total含新弱anchor；前2k仍warmup，早期曲线不能替代全15法线MAE或质量结论。实际view inspection/early_training/training_curves.png，原CSV、B07/B08事件字节snapshot、来源SHA与时刻保留metadata.json。作图exec59372实际exit0并关闭，命令commands/plot_early_training.py，日志logs/plot_early_training.log；0额外优化，无GT标签读取，不更改live训练。完整B08仍运行，40k预览与最终七MAE待完成。

## B08同40k检查点、全15 RGB与实际审核（2026-10-10T09:22:17.610367+09:00）

预览exec88932实际exit0并关闭，checkpoint等待审计exec3502已执行40k检查实际exit0（后续50k/55k仍绑定原训练等待）。99827点finite/全部保留，opacity/scale及32来源NNtensor保持；71553点raw feature变化，最大source logit偏移.92840385437（边界4），canonicalXYZ .03547000885m、q60.00005341°容差内；sourceXYZ/plane/native/gate marker与source35k SHA绑定，DC硬clamp0、边界41.16488191%，RGB拟合E5.49645996，非GT光强。

全部15 TEST RGB PSNR18.05530376dB（B07同40k18.05598159）、SSIM.89959086（B07.89944883）、LPIPS.11140581（B07.11159734），前景PSNR10.83801726。量化变化很小；实际view GT/B07/B080010/0080/0150图：GT/B07/B08实际首中尾拼图：黑背景正常，人物平台存在；人物头手/衣服/腿依旧破碎斑驳，脚周大块灰黑阴影污染和平台伪凸起明显。B08同40k外观几乎与B07相同，未通过质量。GT0080本身背光暗，失败不单凭暗。 当前FAILED_SO_FAR_RGB_ONLY，未测全15 Normal，不推断最终7°；正式训练持续进行，无commit/push/新分支。

命令commands/preview_40000.py282299、make_rgb_40000_comparison.py；日志logs/preview_40000_wait.log、preview_render_40000.log、checkpoint_40000_audit.log；证据preview_40000/eval/metrics.json、inspection/{rgb_40000_comparison.png,preview_40000_review.json,checkpoint_40000_audit.json}。

## B08同40k gate真实世界响应只读审计

commands/audit_gate_world_effect_40000.py实际exec29785 exit0并关闭；0 optimizer，未读GT标签/材质/法线/几何。固定同40k canonical XYZ/q/material/原32 MLP/sourceXYZ，仅比较saved gate与source35k gate的Deform+既有plane输出；不是B07另训权重，也不是GT normal准确率。4个source组完整分区全部99827点，没删点或改任何评分/训练参数、没生成标签/prior；source/checkpoint和473核心SHA保持。

概率偏移全点中位4.33647074e−8、P99.02914502、最大.25986132；source raw静动态阈值跨越91点（投影0、dynamic near-plane34、other dynamic14、static未投影43）。source dynamic near-plane6836点概率均值.97179770→.96945214、中位偏移2.45571136e−5、P99.08227499；组标签不是GT物体。TRAIN0001/0075/0149全点axis无向响应P99.13932712/.16727745/.12982376°、最大5.07943976/3.94833914/4.91278177°；近plane动态组P99.39499720/.41688162/.36503260°、XYZ P99.00792922/.00451554/.00697693m。多数source gate原已饱和，40k真实世界响应大多小；不能用raw feature变化数量宣称整体法线已修复，仍须本轮55k完整MAE/目检。完整JSON inspection/gate_world_effect_40000.json，日志logs/gate_world_effect_40000.log。

## B08训练期间原RGB拟合方向的原生四元数运输诊断（2026-10-10T09:30:29.890633+09:00）

只用原35k PLY/32 MLP、全部135 TRAIN相机/time/原RGB自身Alpha、白色单位GT方向及既有source-only plane；40000 FrozenScene仅相机/NN容器，learned40k XYZ/q/material/feature不用。重组source RGB对应/可见性Gram线性拟合，与B07统计同40948可靠点；仅RAM候选法线/四元数不写训练资产。已有硬plane支持11327可靠点的法线轴被投影固定，所以新旋转响应候选为其余29621点；是模型约束/源可观测性诊断，不改任何GT评分或筛Gaussian。候选四元数保留source norm/yaw、源轴最近无向符号及60°信赖域，全部点保留。

将RGB候选从canonical以source R_world R_canonical^T近似运输，逐135 TRAIN姿态与真实q_candidate+固定source NN偏置（并应用原plane）比较。全部候选的平均运输偏差中位0.00114575°、P999.84908840°、最大44.12652537°；最坏训练时刻偏差中位0.00442551°、P9927.51723807°、最大87.03892812°。这只是approx与exact原生模型差，不是GT normal准确率；未来RGB几何约束须考虑真实raw quaternion加NN偏置，线性Lambertian拟合仍有GGX/阴影/裁剪/对应误差，不能据此宣称7°或新训练改善。

首次exec86238实际exit1并关闭：第二轮相机fid卸载至CPU后未重新加载，造成CUDA/CPU Linear设备不匹配。原脚本/日志保留，独立retry脚本补齐逐camera load/unload；exec97008实际exit0并关闭。诊断0 optimizer、源35k PLY SHA8ebd88b6f847a20178458777d51aa191d5f67ba4bbc490fd1e99ef62f66d5529/40k fingerprint/473核心SHA保持。未读GT标签/光强/颜色，未保存任何法线teacher/prior/训练或评分mask，没有修改live B08。命令commands/audit_rgb_normal_native_transport{,_retry}.py，日志logs/rgb_normal_native_transport{,_retry}.log，统计inspection/rgb_normal_native_transport_diagnostic.json。B08正式最终七MAE与质量仍PENDING。

### 上述运输诊断的静动态分组补充

| 来源组（非GT物体） | RGB可靠非plane候选 | 平均运输偏差中位 | 平均偏差P99 | 最坏时刻偏差中位 | 最坏时刻偏差P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-static | 17389 | 0.00000734° | 0.65557366° | 0.00002910° | 2.39407082° |
| source-dynamic | 12232 | 1.61462750° | 12.19259415° | 5.54153050° | 32.80356603° |

总体中位很小由17389个近静态候选占比造成，不能概括动态12232点：动态平均偏差中位1.61462750°/P99 12.19259415°，最坏时刻中位5.54153050°/P99 32.80356603°、最大87.03892812°。这是后续方法选择的新证据：若使用原RGB生成的几何约束，需要直接拟合或检验原生q_raw+source NN偏置的各时刻法线，不能把简单R_world R_canonical^T运输当精确监督。当前只存诊断统计，尚未实现该prior/新Preset或改变B08；GT Normal仅离线评测。

## B08 50k保存与真实检查点审计（2026-10-10T09:37:01.357665+09:00）

原train PID282299已保存50k；audit_checkpoints_when_saved.py绑定原PID/ticks/argv，40k与50k子检查均实际exit0，父exec3502仍live等待55k，不重启本轮。50k全部99827点finite/保留，opacity/scale与32来源NNtensor原值、source35k SHA、sourceXYZ锚定和plane/native/gate marker核验。71123点raw feature变化，最大source logit偏移2.91029191（边界4）；canonicalXYZ .07999999821m、q60.00006485°容差内，额外world plane投影仍不等于canonical位移上限。DC [.031,.969]内部、硬clamp0、边界比例41.92152424%，RGB拟合E7.12114573，非GT能量。

本检查无GT标签/优化或正式MAE；只有权重有限性/来源/信赖域检查PASS，不判画质或7°。证据inspection/checkpoint_50000_audit.json、checkpoint_wait_state.json、model_mlp/pbr_directional/iteration_50000/lighting.json；命令commands/audit_checkpoint.py50000、audit_checkpoints_when_saved.py，日志logs/checkpoint_50000_audit.log、checkpoint_audits_wait.log。完整训练继续，最终55k全量render/15七MAE与目检PENDING，无commit/push/新目标分支。

## B08完整20k真实训练完成与55k审计（2026-10-10T09:40:57.309255+09:00）

固定35000→55000完整20000步，train exec76498实际exit0并关闭、原PID282299已退出；Training complete log 10/10 09:39:50，loop 28:59。检查点父exec3502实际exit0并关闭，40k/50k/55k三次子审计全部exit0。55k全部99827点finite/保留，opacity/scale及32源NNtensor原值、source35k SHA、sourceXYZ锚定、plane/native/gate marker绑定通过；70238点raw feature变化，最大source logit偏移3.157794（边界4），canonicalXYZ .08000001311m/q60.00005341°容差内。DC内部[.031,.969]硬clamp0，边界比例42.27981074%；RGB拟合E7.92472982，非GT光强/颜色。473核心与双snapshot逐SHA保持。

后处理真实PID282335/exec89846已开始all render，child287136实际argv/cwd/ticks核验；四类150×4、HDR150×2048、方向72/6视频及全部15七MAE/复算待完成。当前仅TRAIN_COMPLETED和权重审计，不判质量或7°，未commit/push/新目标分支。证据training_runtime.json、process_handles.json、inspection/checkpoint_55000_audit.json、checkpoint_wait_state.json、model_mlp/pbr_directional/iteration_55000/lighting.json及logs/train.log/checkpoint_55000_audit.log/postprocess.log/render.log。

## B08 55k raw gate与概率梯度只读比较（2026-10-10T09:46:01.299518+09:00）

commands/audit_probability_gate_gradients_55000.py实际exec81335 exit0并关闭，0 optimizer、原训练已结束；同55k源NN/sourceXYZ/native/plane/材质固定，TRAIN0001/0075/0149只读比较∂L/∂raw和∂L/∂概率，严格验证链式因子2g(1−g)。全点导数中位4.26760162e−5、dynamic source-near-plane6836点中位.00051473238，显示大量sigmoid饱和。没有读取GT标签/生成prior或修改任何权重、点数、评分。

| TRAIN帧 | 全点 raw梯度L2 | 概率梯度L2 | >1e−12 raw/概率梯度点数 | 同lr.001假设首步概率平均偏移 raw/direct |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.000849484 | 0.010315651 | 34923/49210 | 0.000004960/0.000232780 |
| 0075 | 0.001164197 | 0.015449737 | 34522/49417 | 0.000007103/0.000243716 |
| 0149 | 0.000726180 | 0.012277407 | 35059/49329 | 0.000004998/0.000231180 |

首步比较仅RAM中以lr.001/无动量估计并投影[0,1]，没有执行optimizer，真实训练采用ramp/decay、多步动量，不能据此预测MAE或7°改善。备选方案是使用概率变量优化但前向/保存仍为原native sigmoid PLY feature，需新协议/Preset与真实完整训练；尚未实施。最终B08全量七MAE与人工审核仍待完成。证据inspection/probability_gate_gradients_55000.json与logs/probability_gate_gradients_55000.log，473核心SHA与FrozenScene fingerprint保持。

## B08最终RGB与四类早期实际目检（2026-10-10T09:47:18.265752+09:00，七MAE/HDR仍进行中）

全15 TEST RGB PSNR19.24938275dB、SSIM0.91077023、LPIPS0.11025733、前景PSNR11.88729878；B07同55k19.23561417dB/.91049162/.11079910，改变量很小。

实际view GT/B07/B08首中尾0010/0080/0150与四类0001/0075/0150拼图：黑背景正常，人物与平台存在；头手/衣服/小腿仍稀疏斑驳破碎，脚周大块灰黑污染和伪凸起明显，B08与B07近乎相同。GT0080本身背光暗，失败不是单凭暗。 人物与平台主体大部饱和，身体/手指/平台边缘及脚周有薄雾和细碎残影；高Alpha不保证几何/颜色正确。 正背面色差和阴影继续烘入材质，平台棋盘格及脚周颜色污染、身体斑驳仍明显；没有GT材质先验。 平台整体上向，0075无旧kernel整面反转，但脚周/人体混合轴和伪起伏明显；最终Depth/Mesh未实际查看，七MAE待完成。 当前FAILED_SO_FAR_RGB_AND_FOUR_ONLY，无最终七MAE/7°结论。HDR仍生成；早期review只记录已实际看图，不提前填最终完整审核。未commit/push/新分支。

代表图inspection/rgb_final_55000_comparison.png、four_categories_final_55000.png及early_visual_review_55000.json；两作图命令实际exit0。RGB拼图仅修正继承B07的错误indirect标签为Direct，不修改图片/指标。
