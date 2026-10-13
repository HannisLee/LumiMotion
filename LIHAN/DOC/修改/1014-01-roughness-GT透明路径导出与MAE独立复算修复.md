# roughness GT 透明路径导出与 MAE 独立复算修复

日期：2026-10-14（Asia/Tokyo）；服务器 ushas，环境 lumimotion-ushas。

## 根因与最终修改

旧 GT 不仅有低 alpha 边缘越界，第 0030 帧还有评分区内 alpha=1、r=1.0014025 的像素。原导出将 roughness 接入 Emission，并保留透明材质。源 Cycles 的 min_transparent_bounces=0，透明路径的轮盘终止与存活重加权会使有限采样的 RGB/Alpha 偏离有界材质参数。仅处理评分域外异常不能解决该问题；这一尝试已撤回，失败源码和结果完整保留。

scripts/export_roughness_gt.py 保持源材质 roughness、Alpha、相机、最大反弹数和源 blend 文件，令 min_transparent_bounces=transparent_max_bounces、min_light_bounces=max_bounces，关闭自适应采样，添加 --samples（默认仍为16），导出时记录实际设置。没有裁剪导出像素。最终采用128采样新GT；相同16采样预算的对照也在全15帧通过原整图严格校验，最大约0.95000054。

utils/roughness_metrics.py 保持原样，原整图范围检查、容差与前景MAE定义均未放宽。scripts/eval_roughness_gt.py 只增加原GT路径与复算策略记录。scripts/verify_roughness_eval.py 从原 EXR/NPZ 独立恢复浮点GT及Alpha，核对保存数组并复算MAE；旧报告兼容。

## 验证结果

最终15项回归测试通过，覆盖原指标行为，以及原GT重建、重采样、篡改拒绝、前景/低alpha越界和NaN拒绝、输入来源检查。旧16项评分域尝试测试日志保留，但该方案未作为最终代码。

37100 模型全15帧严格 MAE=0.170825119557，覆盖率100%，有效像素495065；独立CPU复算从15个原始GT恢复真值并核验40个输入哈希，PASS。新旧GT不同，历史其他GT分数不能直接作为该指标的对比基线；本轮37100与后续55000均用同一新GT。

完整命令、证据及失败产物见 [1014-02-LH-static-jumpingjacks_EL_V2-roughness严格评测修复](/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1014-02-LH-static-jumpingjacks_EL_V2-roughness严格评测修复/README.md)。完整训练及最终55000验收在 [1013-06-LH-static-jumpingjacks_EL_V2-1011-02原版Stage2续训](/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-06-LH-static-jumpingjacks_EL_V2-1011-02原版Stage2续训/README.md)，已完成验收。


## 最终55000结果

1011-02的35000检查点完整续训20000步到55000，训练、四类渲染、材质视频、NVS、roughness与独立复算全部PASS。最终MAE=0.176275439363，覆盖率100%，495065个有效像素，15帧原EXR重建、40项输入哈希核验通过。相同GT下37100参考MAE=0.170825119557，最终高0.005450319806（约3.19%）。本次修复证明评测有效，未证明roughness精度改善。RGB PSNR=39.83271128，SSIM=0.99627268，LPIPS=0.00352455。

代码提交：2533e3f（当前开发分支）；共享文档按要求同步全分支，original分支训练实现保持原样。
