# ELDL-V3 几何尺度bias零优化诊断

2026-10-19（Asia/Tokyo启动日）。新增`scripts/diagnose_eldl_v3_bias_geometry.py`，在已有实际surfel接收点与完整BVH上对预登记几何尺度bias候选做零优化诊断。原cache/fit/validator不修改，所有方向/rho/EL/valid固定；不导出灯光。

旧三档读取已有exact并校验SHA，新中心0.01extent及0.5/1/2三档只重算V。报告V敏感性、rho×cos加权ΔV RMS、射线法向抬升以及预测低高度caster邻域的遮挡变化代理，保存固定fit代表图。没有新增科学通过阈值，也不按held选择bias。计划见报告14，实际结果追加于下方。

实际执行exit0，输出`output/1019-04-LH-static-jumpingjacks-ELDL-V3-L00连续混合估光/bias_geometry_candidate_v1`。旧/候选fit半档平均|ΔV|为0.07291154/0.02724986，双档为0.04384152/0.01403825；对应rho×cos加权能量RMS半档0.07409127/0.07197395（仅降2.86%），双档0.08903257/0.01383941。候选半档P99仍为1，不能只凭平均值下降宣称稳健。

新增独立CPU脚本`scripts/render_eldl_v3_bias_diagnostic.py`，不修改已运行诊断代码及其SHA。仅读白名单已存缓存与BVH数组，用相同原valid补绘固定fit1/81/149，五行包括旧中心/新半/新中/新双V、相对旧中心变化、各自中心偏差、低高度caster邻域和旧暗转亮叠图、>0.5/>0.9尾部位置。输出子目录`fit_spatial_review`，完整命令/SHA/统计/日志和逐图目检收据均已保存。语法检查与CPU实际执行exit0。

三图已实际查看：中央连续低V阴影保留，未见中央阴影整体消失；但旧暗点转新中心近全透射为94/280、98/272、84/270（33.6%、36.0%、31.1%），93/96/83点在低高度caster邻域。新半到中心>0.9尾部33/30/27点，全部在该邻域，空间上主要外圈。故存在绕过局部近面遮挡的风险，不能声称鞋接触正确。本次仅诊断完成，无科学PASS、不重新拟合、不导出；`internal_validation_passed=false`。
