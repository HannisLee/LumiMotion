# only_clothV4 归档与原始复现执行记录

日期：2026-10-02。

## 修改范围

新增归档来源说明、逐文件哈希清单、训练计划及两组实验执行脚本；更新文档入口与 venus→lumimotion 环境映射。没有修改训练、损失、相机或渲染算法。现场已有 `scripts/loss.py` Python 3.8 兼容改动及 0930 文档修改均保留。

- `output/1001-01-onlyclothV4-FOVxy修复重训/`：原 main 实验完整副本，脚本和模型保留原绝对路径，README 说明原项目来源。
- `output/1002-01-onlyclothV4-perlight原始管线复现/`：`run_pipeline.sh`、`launch.sh` 、`preflight.py` 与 `inspect_outputs.py`；执行完整两阶段训练和评估。
- `output/smoke_test/1002-01-onlyclothV4-perlight原始管线冒烟/`：独立短训练执行脚本。
- 本次启动器记录主日志与逐步日志、实际命令、HEAD、工作区差异和服务器；已有 STARTED/COMPLETED/FAILED 时拒绝重跑，避免覆盖。
- 修正复制启动器中的命令记录换行：新实验 `commands/*.sh` 使用真实换行；归档的原脚本保留原字面量 `\n` 供追溯。

## 验证

两组 Bash 启动脚本语法检查通过。原始 SH 损失与 Blender 相机已有单测共 25 项通过；全量数据与默认参数预检查 PASS，证据见正式实验 `logs/preflight_tests.log`、`logs/preflight_data.log` 与 `preflight.json`。后续训练和视觉验收在对应实验 README 更新。

新增 `inspect_outputs.py` 核验检查点、每类 120 帧图像、10 个视频的首/中/末帧解码与 NVS 指标，并生成目检接触表。产物核验 PASS 与最终人工目检 PASS 分开记录。既有 TensorBoard callback 把 train 的部分指标 tag 写成 test 值，因此本次 train 指标取日志，test 指标可取 TensorBoard 原始精度；该既有问题未在本次修改。

首轮核验脚本的 `gt_image*.png` 同时匹配 NVS 普通 GT 和环境 GT，误计为 30 帧；改为限定数字后缀，并分别核验环境 GT 与 mask 各 15 帧。该问题属于核验 glob，不是训练失败；首轮日志和统计独立保留于冒烟目录 `logs/inspection_first_glob_failed.log`、`inspection_1002/首轮检查统计.json`。

新增 `finish_record.py` 在产物核验通过并提供四类人工目检记录后写入指标、代表路径和最终验收。冒烟完整训练/渲染/NVS 执行及产物核验 PASS，实际检查证据保留在冒烟目录。
