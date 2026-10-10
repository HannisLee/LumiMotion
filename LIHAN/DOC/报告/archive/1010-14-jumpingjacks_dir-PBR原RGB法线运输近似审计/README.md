
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
