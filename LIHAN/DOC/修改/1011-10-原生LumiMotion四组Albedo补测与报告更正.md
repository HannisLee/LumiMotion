# 原生LumiMotion四组Albedo补测与报告更正

## 修改原因

用户要求补齐原生LumiMotion的Albedo MAE。此前把四组35k的Albedo直接标为N/A不够准确：原生模型保存`_albedo_dc`，提供`get_albedo`，已有Stage1 Albedo展示视频。因此应实际评分，并说明当前参数的训练阶段。

原生定义为`clamp(SH2RGB(_albedo_dc), 0.03, 0.97)`。Stage1把DC和高阶SH作为外观参数学习；35k的get_albedo可作为与GT Albedo对比的参考，不代表Stage2已经完成材质与光照分解。四组55k checkpoint仍不存在，不用35k分数替代55k。Stage1 Roughness尚未完成材质优化，仍记N/A。

## 实际执行

- 服务器：ushas；环境：lumimotion-ushas。四组评测并行，沿用调度器，每GPU最多2个计算任务。
- 只读取已有35k checkpoint，未执行训练，没有加入GT光照或GT材质监督。
- 使用原始世界相机的eval_view，全15测试帧（10、20、…、150）；训练135帧与测试15帧严格分离。
- 与05／06使用相同评分口径：原生get_albedo转换为线性RGB，按对应相机与时刻的几何光栅化；预测去Alpha预乘，在GT Alpha≥0.99并收缩2像素的前景核心上计算像素汇总MAE。没有拟合缩放，没有追加预测值裁剪，漏检GT像素也参与评分。
- 直接材质光栅化与原IR渲染器的material_only入口一致性检查通过，四组首测试帧最大线性RGB差均为0。
- 保存全部15帧NPZ、逐帧CSV、首中尾GT对比图、来源／权重哈希；独立复算校验GT核心、帧号、权重不变、GT哈希与分数，四组全部PASS。此PASS仅指评测数值核验。

执行入口：

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/backfill_1011_albedo.py
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/backfill_1011_report.py
```

每组完整Albedo命令、source／model／iteration、GPU、退出码及日志记于实验README和`review_albedo_1011/executed.json`。原始失败结果完整保留；旧MAE汇总和README在各组`review_albedo_1011`备份，总控另保留`mae_summary_before_native_albedo.json`。

## 补齐后的Albedo MAE

| 编号 | 数据集／方法 | 迭代与参数语义 | Albedo MAE | 前景预测Alpha覆盖率 |
|---|---|---|---:|---:|
| 01 | 静态DL原生 | 35k，SH/DC参考 | 0.18357946 | 99.9992% |
| 02 | 静态EL原生 | 35k，SH/DC参考 | 0.19744575 | 100.0000% |
| 03 | 动态DL原生 | 35k，SH/DC参考 | 0.16789033 | 99.9869% |
| 04 | 动态EL原生 | 35k，SH/DC参考 | 0.19197014 | 100.0000% |
| 05 | 静态DL PBR＋GT方向光 | 55k，Stage2材质 | 0.12618528 | 见原始指标 |
| 06 | 静态DL Lambertian＋GT方向光 | 55k，Stage2材质 | 0.20741376 | 见原始指标 |

以上为线性RGB绝对误差，越低越好，不是百分比误差。四组原生的参数语义与最终材质不同，不能据其较低分数宣称材质分解优于其他方法。单独看03参考MAE较低，但其几何／分离缺陷及Albedo污染仍明显。

## 实际目检

已查看四组Albedo GT对比帧10、80、150，共12张对比图：

- 01：明显地面黑色投影残留，棋盘偏白、颜色变淡，衣服有明暗斑块。
- 02：棋盘和衣服纹理较稳定，但整体偏暗。
- 03：圆盘偏白、棋盘变淡，人物有强烈明暗和彩色斑块，地面有黑色残留。
- 04：纹理较稳定，但人物和圆盘整体偏暗。

整批质量结论仍为FAILED：01—04原生Stage2缺失；05／06已有材质问题没有因本次补测而消失。

## 代码与文档

- 新增`LIHAN/scripts/backfill_1011_albedo.py`，只补四组Albedo和独立复算；分进程申请GPU，保留旧汇总及日志。
- `audit_1011_mae.py`对35k也要求Albedo评分，校验15个准确测试帧、原渲染器一致性和权重／GT哈希，并记录参数语义。
- `batch_1011.py`、`backfill_1011_evaluations.py`取消35k Albedo跳过逻辑。
- `backfill_1011_report.py`将35k Albedo N/A更正为实际数值，保留35k参考与55k材质的区别，加入完整补评命令、日志和目检结果。
- 回填六组训练文档、各实验README、总报告、总控CSV／JSON，并按AGENTS.md同步LIHAN及LIHAN.md到全分支。

证据：各原生输出`evaluation_35000/albedo/iteration_35000/metrics.json`、`evaluation_35000/mae_summary.json`、`review_albedo_1011/completed.json`；总控`native_albedo_backfill.json`。
