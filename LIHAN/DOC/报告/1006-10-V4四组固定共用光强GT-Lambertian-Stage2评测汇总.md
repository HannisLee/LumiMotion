# 四组固定共用光强GT Lambertian Stage2评测汇总

更新：2026-10-06T17:26:36.873535+09:00

**4/4 COMPLETED，流程/产物/代表图验收PASS。**

CV4：raw Albedo MAE由原版IR的0.228857变为0.138418，下降39.52%；材质均衡bMAE由0.131870变为0.088483。前景SSIM 0.907423→0.830631。

CV4L：raw Albedo MAE由原版IR的0.198856变为0.137338，下降30.94%；材质均衡bMAE由0.117014变为0.083037。前景SSIM 0.909815→0.850182。

CV4D：raw Albedo MAE由原版IR的0.161104变为0.136366，下降15.35%；材质均衡bMAE由0.121246变为0.101979。前景SSIM 0.877510→0.808305。

CV4DL：raw Albedo MAE由原版IR的0.122299变为0.120561，下降1.42%；材质均衡bMAE由0.091330变为0.085033。前景SSIM 0.879943→0.828440。

| 数据集 | 阶段 | Albedo MAE | bMAE | Normal MAE° | Depth MAE | SSIM↑ | 前景SSIM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV4 | SH Stage1 35k | 0.192875 | 0.129177 | 7.4027 | 0.026094 | 0.999192 | 0.995041 |
| CV4 | 原版IR Stage2 55k | 0.228857 | 0.131870 | 7.9667 | 0.030203 | 0.985574 | 0.907423 |
| CV4 | GT Lambertian 55k | 0.138418 | 0.088483 | 7.4027 | 0.026094 | 0.973556 | 0.830631 |
| CV4L | SH Stage1 35k | 0.177255 | 0.114688 | 7.5799 | 0.027284 | 0.999261 | 0.995261 |
| CV4L | 原版IR Stage2 55k | 0.198856 | 0.117014 | 8.0912 | 0.031638 | 0.985740 | 0.909815 |
| CV4L | GT Lambertian 55k | 0.137338 | 0.083037 | 7.5799 | 0.027284 | 0.976506 | 0.850182 |
| CV4D | SH Stage1 35k | 0.158172 | 0.118708 | 11.5613 | 0.102184 | 0.997447 | 0.989456 |
| CV4D | 原版IR Stage2 55k | 0.161104 | 0.121246 | 11.9397 | 0.108388 | 0.975469 | 0.877510 |
| CV4D | GT Lambertian 55k | 0.136366 | 0.101979 | 11.5613 | 0.102184 | 0.961797 | 0.808305 |
| CV4DL | SH Stage1 35k | 0.138899 | 0.099402 | 12.4948 | 0.087812 | 0.995352 | 0.982812 |
| CV4DL | 原版IR Stage2 55k | 0.122299 | 0.091330 | 12.8886 | 0.093827 | 0.975389 | 0.879943 |
| CV4DL | GT Lambertian 55k | 0.120561 | 0.085033 | 12.4948 | 0.087812 | 0.964946 | 0.828440 |

Albedo主值未缩放；Stage1列为SH DC外观代理。15相同测试帧，640×360。几何/法线/opacity及形变冻结，材质与RGB质量分别解读，raw改善不代表所有材质及尺度诊断都改善。

- [CV4完整报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-06-CV4-固定共用光强GT-Lambertian-Stage2评测报告.md)。
- [CV4L完整报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-07-CV4L-固定共用光强GT-Lambertian-Stage2评测报告.md)。
- [CV4D完整报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-08-CV4D-固定共用光强GT-Lambertian-Stage2评测报告.md)。
- [CV4DL完整报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-09-CV4DL-固定共用光强GT-Lambertian-Stage2评测报告.md)。

[全部MAE CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/all_mae.csv)；[阶段CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/metrics.csv)；[逐帧CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/frames.csv)；[数值核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/final_verification.json)。

报告页面浏览器目检未执行：当前会话无可调用的浏览器检查工具；渲染代表图的真实目检与视频解码记录是已执行的单独检查。

## 可分享的报告附件

![同来源三阶段误差对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/figures/stage_comparison.png)

[SVG矢量图](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/figures/stage_comparison.svg)；[离线交互报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/report-app/.data-app-offline/exports/gt-lambertian-report.html)；[交付核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/delivery_verification.json)。
