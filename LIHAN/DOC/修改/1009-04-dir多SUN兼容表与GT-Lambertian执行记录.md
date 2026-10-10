# 1009-04 dir多SUN兼容表与GT Lambertian执行记录

日期：2026-10-09（Asia/Tokyo）。本次使用已有photometric_lambertian/lambertian_gt训练器，不改训练、损失或渲染实现。用户已授权dir Stage2、完整MAE评测与报告归档。

## 新版多灯数据适配

原lights.json使用frames→帧号→lights数组，现有read_gt_directions读取顶层帧号→方向记录。新增实验内commands/prepare_gt_directions.py先核验全部150帧各两盏白色SUN、角大小0、两灯世界方向相同、单位向量、表面指向光源约定，再写独立gt_directions.json。仅满足同方向灯可用此适配，不将不相同的多灯误当单方向。两灯energy3+3只保存为来源元数据；当前训练强度仍按135训练RGB与SH代理初值标定固定，不混同为物理GT辐照度。

## 复用评测和来源保护

使用Stage1来源1009-02/35k，所有几何及形变冻结，仅优化材质。实验内GT深度适配器复用env已审核的几何/着色法线差诊断逻辑；实际读取dir自己的blend、相机、Alpha及EXR，不复制env真值。动态Mesh脚本使用15个独立时刻、各135训练相机预测，固定TSDF参数，与env同协议。

CPU独立复算脚本去掉roughness要求；Lambertian没有粗糙度输出，该项明确NOT_APPLICABLE。新增checkpoint审核核验有限值、来源哈希、固定方向/光强、冻结指纹和仅albedo优化组；视频审核改为本数据400×400、150帧（转灯72帧）。输出数组和checkpoint不删除，失败现场保留。

## 实际入口

- [训练文档](../训练/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian训练报告.md)。
- 输出：`output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/`；适配器、运行器、原代码快照及复算/审核脚本在commands，日志在logs，证据在inspection。
- Stage2训练代码提交a9e4d1061fec9986d82362f78b42c0c269e6146e，实际工作区差异保存在实验code_diff.patch；模型来源为原版main的35k权重，不能将此次Stage2标为原版IR训练。

完整55k训练、全15帧MAE、独立复算、六视频解码及人工目检已完成，流程PASS。详情见[评测报告](../报告/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian-各项MAE与三类Normal评测报告.md)；Roughness明确不适用，固定几何仍有局部碎片与法线误差。
