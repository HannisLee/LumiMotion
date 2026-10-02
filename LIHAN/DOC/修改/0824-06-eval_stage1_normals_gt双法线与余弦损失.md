# 0824-06 `eval_stage1_normals_gt` 双法线选择与 GT 余弦损失

- 日期：2026-08-24
- 服务器 / 环境：`garuda` / `lumimotion-garuda`
- 目的：为 0823（含）之后的实验统一补评 independent photometric normal 与 2DGS / GS raster normal 相对 Blender 真实法线的误差，避免评估器按 checkpoint 自动选源后把两类法线混为同一指标。

## 修改内容

1. `scripts/eval_stage1_normals_gt.py` 新增 `--normal_source auto|independent|gs`：
   - `auto` 为默认值，保持历史行为：checkpoint 含 independent normal 时评估 independent，否则评估 GS raster normal；
   - `independent` 显式评估 independent photometric normal；checkpoint 不含该参数时明确报错；
   - `gs` 无论 checkpoint 是否含 independent normal，均显式评估 `rend_normal`。
2. 指标新增逐帧 `cosine_loss`，定义与训练 GT-normal oracle 一致：有效像素上 `mean(1-cos(pred, GT))`。
3. `summary_mean_over_frames` 同步新增 `cosine_loss`，仍保持既有统计口径：120 帧逐帧指标的等权平均；角度 `mean_deg`、`median_deg`、`p95_deg` 的定义与数值不变。
4. `utils/normal_eval_utils.py` 新增 `resolve_normal_source()`，集中处理 source 选择和缺失 independent normal 的报错。

## 兼容性与影响

- 默认 `auto` 保持旧命令的法线选择逻辑；只在 JSON 中追加余弦损失字段，不改变既有角度指标。
- 修改仅影响显式运行的离线评估，不进入训练损失、反向传播、优化器或 checkpoint。
- 本轮统一复评写入各实验新的 `normal_gt_eval_dual_audit/<source>/ours_35000/` 目录，不覆盖历史评估产物。
- 强制评估 GS normal 时，Lambertian 前向仍可按 checkpoint 的 independent normal 完成着色；评估读取的是与颜色无关的 `rend_normal`，alpha mask 保持同 checkpoint、同阈值口径。

## 验证

```bash
conda run --no-capture-output -n lumimotion-garuda \
python -m unittest tests.test_normal_eval_utils tests.test_gt_normal_oracle -v
```

结果：9/9 通过。新增测试覆盖：

- `auto` 对含 / 不含 independent normal checkpoint 的历史选择语义；
- 含 independent normal 时强制选择 GS normal；
- checkpoint 缺少 independent normal 时显式请求 `independent` 的报错；
- 既有角度误差、alpha-normalized normal 与 masked cosine loss 测试继续通过。

随后执行完整回归：

```bash
conda run --no-capture-output -n lumimotion-garuda \
python -m unittest discover -s tests -v
```

结果：**62/62 通过**。

完整双法线复评结果与实验结论汇总见本轮 0823 后实验审计报告。
