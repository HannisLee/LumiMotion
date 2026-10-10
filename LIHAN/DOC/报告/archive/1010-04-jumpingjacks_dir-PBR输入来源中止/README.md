# 1010-04 jumpingjacks dir PBR预测几何与RGB光强：中止记录

状态：FAILED_INPUT_PROVENANCE，2026-10-10 02:03:55 JST。训练实际PID206152；完整20k计划在4177步（累计39177）被主动TERM，exit143。没有新的40k/50k/55k checkpoint，只有复制的来源35k，不能把它当训练结果评测。

## 原因与纠正

审计旧dataset_manifest.json与转换代码发现mask_source=albedo-alpha。此前“原始RGBA自带alpha、无GT材质输入”的声明不准确。原RGB实际有自身alpha，全150帧16-bit alpha与albedo alpha逐像素完全相同（差异0）；旧输入文件数值无额外轮廮信息，但来源链不满足严格不读取GT文件的流程。为消除文件来源歧义，立即停止，不覆盖/删除失败产物，新建1010-05重新完整训练。已有35k来源保持不变；其旧alpha与原RGB相同的继承限制明确披露。

## MAE和质量

本轮新权重各项Albedo/Roughness/GS/Depth/Mesh Normal/Expected/Median Z MAE均N/A：中止前未保存任何新checkpoint，没有可评分的新训练产物。不能用35k来源或B00 MAE冒充本轮结果；B02会重新完整20k并评测全部15帧七项MAE。40k预览及最终后处理均真实退出FAILED，未自动重启。

四类目检/前后景：N/A（未保存候选新权重、无离线渲染）。法线7°、图像质量、完整Stage2：均未通过。未commit/push，未建目标分支。

## 保留证据

源model_mlp/point_cloud/iteration_35000、deform/iteration_35000、方向灯表、训练TensorBoard、完整train.log、postprocess.log、preview_wait.log、POSTPROCESS_FAILED、源码快照/哈希、所有计划命令均保留。实际单元/单帧梯度检查通过仅证明之前代码接入，不能弥补输入来源声明错误。输入审计inspection/input_provenance_failure.json。

完整原训练命令train_argv.json与commands/train.sh；原渲染/MAE命令planned_commands.json（中止后没有执行）。日志均位于本输出logs。
