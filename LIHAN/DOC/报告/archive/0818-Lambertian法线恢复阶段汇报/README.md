# Lambertian 法线恢复审计与实验汇报

日期：2026-08-18  
服务器/环境：`garuda / lumimotion-garuda`

## 产物

- `0818-Lambertian法线恢复审计与实验汇报.pptx`：19 页中文技术汇报。
- `build_ppt.py`：PPT 可重复生成脚本。
- `source_notes.md`：数值、图片与结论的来源和口径。
- `qa_report.txt`：PPT 包结构、页数、图片链接和文本边界的自动检查结果。

## 汇报主线

1. 从“GT light + Lambertian 理论上应恢复 normal”的问题出发，整理三类初始假设。
2. 复盘坐标系、normal/light transform、手性、符号翻转和 sRGB/linear 链路。
3. 展示光强 `1.0 → 5.5` 的实际曝光恢复，以及 CV3/CV3L 分别标定光强的必要性。
4. 复盘前 500 步梯度、无效的固定 GS 对照、独立 normal 实现与首轮调度 bug。
5. 展示修复调度后 CV3/CV3L 在 iteration 10001、10500、35000 的同轮实验指标与图片。
6. 给出当前仍存的可辨识性、近场光、alpha 混合和梯度方向问题，建议下一步先做 N0 GT-normal oracle。

## 重新生成

`python-pptx` 仅用于生成产物，不进入训练环境依赖。若本机已安装，可直接运行：

```bash
python DOC/报告/0818-Lambertian法线恢复阶段汇报/build_ppt.py
```

PPT 使用 `Microsoft YaHei` 作为中文字体名。在 Linux 上若未安装该字体，PowerPoint/WPS 打开时会按本机字体策略替换。
