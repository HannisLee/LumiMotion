# V4四版本原版LumiMotion训练结果

更新：2026-10-03T22:39:01.922610+09:00；正式验收完成：4/4。用户授权最多两组并行；所有产物在本项目output/。

| 数据集 | 正式状态 | NVS PSNR | SSIM | LPIPS | 输出文档 |
| --- | --- | --- | --- | --- | --- |
| 1003-01-only_clothV4_lambertian-LumiMotion原版两阶段 | COMPLETED | 33.61345456 | 0.98228605 | 0.01960011 | [README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-01-only_clothV4_lambertian-LumiMotion原版两阶段/README.md) |
| 1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段 | COMPLETED | 30.76417160 | 0.97216093 | 0.03076217 | [README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/README.md) |
| 1003-03-only_clothV4_Dynamic_Lambertian-LumiMotion原版两阶段 | COMPLETED | 31.73186073 | 0.97361401 | 0.02669440 | [README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-03-only_clothV4_Dynamic_Lambertian-LumiMotion原版两阶段/README.md) |
| 1003-04-only_clothV4-LumiMotion原版两阶段 | COMPLETED | 33.59008624 | 0.98218409 | 0.02084923 | [README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-04-only_clothV4-LumiMotion原版两阶段/README.md) |

## 冒烟与验收

| 冒烟 | 状态 | 人工目检 |
| --- | --- | --- |
| 1003-01-only_clothV4_lambertian-LumiMotion原版冒烟 | COMPLETED | PASS |
| 1003-02-only_clothV4_Dynamic-LumiMotion原版冒烟 | COMPLETED | PASS |
| 1003-03-only_clothV4_Dynamic_Lambertian-LumiMotion原版冒烟 | COMPLETED | PASS |
| 1003-04-only_clothV4-LumiMotion原版冒烟 | COMPLETED | PASS |

## 证据与解释

- 四组从头训练，统一15万点baseline、105/15划分、Stage1 35k、Stage2累计55k。日志、代码版本、模型、指标、图片/视频、完整命令及四类目检见各组README。
- Stage1原版训练报告只取最多4个测试视角和3个训练视角；本轮test为帧8/16/24/32（time约0.059～0.261），集中在前段，不能代表中后段动态质量或15视角全量指标；Stage2 NVS取15视角/帧。
- 静态与动态NVS条件不同、整图指标包含黑背景；不将四组数值排序直接解释为材质恢复优劣。HDR和roughness没有对应真值；动态BVH沿用原实现。
- PASS表示流程与产物验收，是否存在阴影残留、分离退化或动态细节丢失以人工目检说明为准。失败结果保留，不覆盖旧实验。
- 调度与显存记录：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-05-V4四版本执行汇总/batch_status.json`、`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-05-V4四版本执行汇总/logs/gpu_memory.jsonl`；调度脚本在commands/。

## 最终执行与核验

- 四组正式流水线全部完成，开始2026-10-03T16:52:23+09:00，结束2026-10-03T22:27:14+09:00，共5:34:51；各组目检和回填时间另见manual_review.json。
- 实验起止记录最大重叠2组；原版代码提交和脚本记录范围内的本地补丁一致，当前代码也与启动快照一致。
- 正式诊断图3360张，视频40个，共完整解码4800帧，NVS共60个测试视角/帧；35k/55k模型及55k神经网络/envmap数值核验通过。
- GPU显存采样峰值6873MiB（整卡统计，非本轮独占显存）；CUDA总进程数不可直接等同训练组数。
- 两个动态集已恢复主要布料运动和静动态分离，中末段仍有碎片、Alpha软边及法线混色；四组albedo/NVS仍有平台斑驳和明暗残留。流程PASS不代表这些质量问题消失。
- 汇总核验：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-05-V4四版本执行汇总/final_verification.json`；代码一致性：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-05-V4四版本执行汇总/code_consistency.json`。
