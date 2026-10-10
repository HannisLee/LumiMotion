# only_clothV4 几何评测流程图与训练视角

日期：2026-10-06。用户要求参考圆角方框流程图，补充训练集相机视角。本次沿用已实现的法线与深度 GT 评测管线，生成描述图与真实示例，不修改训练或评测核心实现。

产物目录：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-13-onlyclothV4-几何评测流程与训练视角/`。

- [流程图与训练视角 PNG](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-13-onlyclothV4-几何评测流程与训练视角/geometry_evaluation_with_train_views.png)：预测和独立 GT 两条支路、世界法线角度 MAE、相机 Z 深度 MAE、区域汇总和覆盖率，下方标注真实 TRAIN 0001/0041/0081。
- [训练视角预测/GT 对照 PNG](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-13-onlyclothV4-几何评测流程与训练视角/training_geometry_comparison.png)：同相机 RGB、世界法线和相机 Z 深度；共用裁剪与深度色标。
- [完整记录与命令](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-13-onlyclothV4-几何评测流程与训练视角/README.md)：模型来源、完整命令、日志、三帧指标、GT 审计、目检和适用范围。各图另有 SVG/PDF。

模型为归档的 **1002-02 original / iteration 55000**；数据为 `transfer-static/only_clothV4` 的 **105 个训练视角**。选取 source_frame 1/41/81，核实均属于 train 且不属于 15 个 test 帧。重新从原 `.blend` 导出三个对应帧的三角网格；相机误差均为 0，原文件保持只读。真值法线来自原 Normal EXR，Z 深度来自独立三角网格求交。

新增辅助脚本保存在产物目录的 `commands/render_examples.sh` 与 `commands/make_figures.py`。训练 RGB 保持原始画幅，另附对照图采用共同 GT 前景包围框裁剪，指标仍从 640×360 原始浮点数组计算。相机分布图使用实际 C2W/K 绘制全部训练相机位置和三例视锥。

三例全前景 Normal MAE 为 **7.712915° / 11.092766° / 8.137855°**，期望 Z 深度 MAE 为 **0.028052 / 0.033994 / 0.034736 米**；覆盖率联报。这些分数只解释训练视角，不替换此前正式测试集指标。

验收 **PASS**：真实三帧渲染、GT 投影/EXR 审计、train/test 核对、来源哈希复查、流程图与对照图实际目检、PNG/SVG/PDF 导出完成；本次未训练模型。

2026-10-07 补查世界尺度：读取Stage1来源模型保存的120个相机，与原GT相机的坐标尺度一致，位置最大差3.5379151e-7世界单位，焦距差0；世界尺度诊断s约等于1，未应用拟合。Blender训练加载器未执行空间归一化，因此本图采用原单位相机Z深度直接比较。详见[尺度核查记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-13-onlyclothV4-几何评测流程与训练视角/scale_audit_1007.json)。原记录中“baseline参与1002-02初始化”的表述已更正，保存input.ply实际是100000个默认随机方块点，baseline只用于GT审计。
