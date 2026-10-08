# 0824-01 GS 几何自一致法线开关 `lambda_gs_normal` 与 `lambertian_normal2` 预设

- 日期：2026-08-24
- 修改序号：02
- 服务器/环境：garuda / `lumimotion-garuda`
- 概要：为 Stage1 的 GS 几何自一致法线项 `1-cos(rend_normal, surf_normal)` 新增权重参数 `--lambda_gs_normal`（原为 `scripts/loss.py` 内硬编码 0.02），使其可在**不影响 distortion** 的前提下单独开关；新增损失预设 `lambertian_normal2`（0824-01 二法线一致性消融组合）。默认值保持历史行为，既有实验零漂移。

## 1. 背景

0823-01 的“三法线一致性”包含：① GS 几何自一致法线（`start_normal_reg` 门控）、② `photometric_normal_live`、③ `photometric_normal_mv`。0824-01 消融实验要求**只去除①**、保留②③。但①与 distortion 共享同一门控（`tr.iteration > opt.start_normal_reg and not tr.pbr_active`），且法线权重硬编码，无法单独关闭：若改大 `--start_normal_reg`，distortion（λ=1000）也会被一并关闭。因此为①单独加权重开关。

## 2. 代码改动

### 2.1 `arguments/__init__.py`

- `OptimizationParams` 新增 `self.lambda_gs_normal = 0.02`（紧邻 `start_normal_reg`/`lambda_dist`，含中文注释）。默认 0.02 与原硬编码值一致；随 `ParamGroup` 自动暴露为 `--lambda_gs_normal`。注：本仓库 `cfg_args` 只落盘 ModelParams（既有行为），优化参数以各实验目录 `run.sh` 与 `train_stage1.log` 追溯。

### 2.2 `scripts/loss.py`

- `compute_stage1_loss()` 中 `lambda_normal` 由硬编码 `0.02` 改为读 `opt.lambda_gs_normal`；门控条件（`start_normal_reg`、PBR 关闭）不变。
- `LOSS_PRESETS` 新增：

| 预设名 | 兼容 render_mode | 覆盖的参数（未显式传时才生效） |
| --- | --- | --- |
| `lambertian_normal2` | photometric_lambertian | `lambda_photometric_normal_live=0.01`、`lambda_photometric_normal_mv=0.02`、`lambda_gs_normal=0.0` |

即 0823-01 组合去掉①（`lambertian_normal3` − GS 几何自一致法线）；distortion、alpha、d_xyz、二值化分离等其余项与 `lambertian_default` 完全一致。

### 2.3 `tests/test_stage1_loss.py`

- `_make_opt` / `LossPresetTest._make_args` 默认值补 `lambda_gs_normal=0.02`，原数值等价测试不改动、继续通过。
- `test_registry_complete` 注册表断言补 `lambertian_normal2`。
- 新增 2 例：`test_lambertian_normal2_applies_weights`（三项覆盖值与顺序）、`test_lambertian_normal2_requires_lambertian_mode`（render_mode 校验）。

## 3. 兼容性

- `lambda_gs_normal` 默认 0.02 = 原硬编码值：所有既有实验（含 0823-01）逐位不变。
- `lambertian_normal3` 及其他预设未改动；`auto` 行为不变。

## 4. 验证

- `python -m unittest tests.test_stage1_loss -v`：12/12 通过（含新增 2 例）。
- `--lambda_gs_normal` 出现在 `OptimizationParams` 帮助中。
- 冒烟/正式训练验收见 `DOC/训练/0824-01-*` 与 `output/0824-01-CV3s-GTlight_lambert_iter1_2ncons/README.md`。

## 5. 关联文档

- `DOC/指导/blender数据训练指导.md`：`--loss_preset` 预设列表补 `lambertian_normal2`。
- `DOC/训练/0824-01-CV3s静态布料从iter1起GT光照Lambertian二法线一致性消融训练方案与执行记录.md`。
