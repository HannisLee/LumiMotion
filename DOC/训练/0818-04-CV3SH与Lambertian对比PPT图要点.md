# 0818-04 CV3 SH vs Lambertian 对比 PPT 图要点

- 日期：2026-08-18
- 原目录：`output/0818-04-CV3-PPT_SH_vs_Lambertian`
- 类型：PPT 素材（非训练）

## 一句话总结

两张可直接放入 PPT 的四宫格对比图（每张含两张 Albedo + 两张 RGB），直观展示 0811 旧评测（实际走默认 `original_sh`）与 0813 显式线性 Lambertian 渲染的差别。

## 构图

- 上排：0811 旧评测 —— `render_stage1_insights` 未传 `--render_mode photometric_lambertian`，RGB 实际为默认 `original_sh`
- 下排：0813 显式线性 Lambertian（`render_mode=photometric_lambertian`）
- 两组均使用 CV3、测试相机 `frame_0008`；分别抽取 `t=0` 与 `t=60`

## 图片与原始素材

- `PPT_compare_t000.png`（t=0）、`PPT_compare_t060.png`（t=60）
- 旧版素材：`output/0811-only_clothV3-lambertian-modify/lambertian-modify_mlp/renders_stage1_insights/ours_35000/`
- 新版素材：`output/0813-only_clothV3-lambertian-directional-linear/renders_stage1_insights/ours_30000_linear_directional/`
- 背景详见 `0811-01`、`0813-01` 两份要点文档
