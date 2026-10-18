# ELDL-V3 独立raster颜色梯度审计

2026-10-19（Asia/Tokyo启动日；主机执行日志2026-10-18 UTC）。新增`scripts/audit_eldl_v3_raster_color.py`，不修改主任务正在实现的`calibrate_eldl_v3_raster.py`或核心光栅/着色代码。仅读白名单TRAIN、I00@16000、cache_v3_geometry_bias与fit_v4_geometry_bias；不读外部GT/TEST，不运行校准拟合，不导出灯光。

调用新校准器的load/render/objective及验证器来源检查/no_gt_guard。load装载135个TRAIN相机容器，实际渲染、损失与梯度只用预先固定fit1/81/149；不统计或选择held。固定几何、opacity、原点序及TRAIN估计方向，固定原中心bias，只检查颜色和共享EL参数。工程rho代表值不来自材质或图像，严格处于生产可表达线性区间[sRGB_to_linear(.03),sRGB_to_linear(.97)]内。

独立GPU1实际执行，输出`output/smoke_test/1019-11-ELDL-V3-raster颜色梯度`；先确认新目录不存在，未覆盖旧结果。完整命令、模型/来源路径、SHA、规则与步长保存plan.json，日志run.log，三个逐帧统计fit_*.json，完整summary.json。关键源码按本次SHA另存source_snapshot及其manifest。README记录范围、命令和最终结论。没有模型四类可视化，颜色/EL数值工程审计不替代模型目检。

真实pc._albedo_dc经RGB2SH(linear_to_sRGB(rho))写入，使用真实get_albedo执行生产EL+DL着色，与局部线性颜色合成前向与rho/EL梯度配对。检查native override RGB三通道、线性rho三通道、log EL三通道及rho空间符号方向，每帧10项，步长固定0.01/0.003/0.001/0.0003，提前规定相邻两档相对误差≤1%、有效导数≥1e-6。零学习率Adam后参数逐值恢复，损失恢复，几何逐张量冻结且grad=None。

实际exit0，PASS_COLOR_EL_ENGINEERING_ONLY。三帧120/120个步长检查均低于1%，最大相对误差0.00134748（0.134748%）；真实生产线性前向最大差4.17233e-7，生产rho/EL梯度相对差最大3.58348e-7，零更新损失差0，几何全部原样。完整可表达线性rho边界[0.002321981359,0.933106899261]。GPU1已释放。

此PASS仅支持冻结几何条件下的颜色/EL梯度实现，既有几何exact FD仍FAILED且本次不重验；internal_validation_passed=false。完整校准保存/重载、matched constant-color对照、三初值及原三bias25%验收尚待主任务完成，不从此审计推导物理分解或灯光资产PASS。工程核查见报告22。
