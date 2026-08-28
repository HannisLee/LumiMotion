# 0824-03 新增 `gs_wrapping` 损失预设（Gaussian Wrapping 论文权重方案映射）

- 日期：2026-08-24
- 修改序号：03
- 服务器/环境：garuda / `lumimotion-garuda`
- 概要：新增损失预设 `gs_wrapping`，将 Gaussian Wrapping（*From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians*，arXiv:2604.07337，2026）论文 §4.1「Loss Details」的损失权重方案映射到本仓库 Stage1 损失项，用于 0824-02 对照实验。不改任何默认值与既有预设，`auto` 与所有既有实验零漂移。

## 1. 背景

`lambertian_normal2`/`lambertian_normal3` 的两个独立法线一致性项（`photometric_normal_live`：独立法线 ↔ 同帧深度导出法线的 1−cos；`photometric_normal_mv`：多视角重投影 1−cos）与 Gaussian Wrapping 论文的法线对齐损失 `L_N = Σ_p 1 − N(p)·∇D(p)` 及其多视角几何一致性项 `L_gc` 在形式上同类。为检验"按论文权重方案配置损失"是否优于现有消融组合，新增 `gs_wrapping` 预设。

## 2. 论文 → 本仓库损失映射

| 论文项（§4.1） | 论文权重 | 本仓库对应项 | `gs_wrapping` 覆盖值 |
| --- | --- | --- | --- |
| L_RGB（photometric） | 1.0 | `rgb_l1`+`rgb_dssim`（photometric_rgb_loss_weight 默认 1.0） | 不动 |
| L_DN（深度-法线一致性，GOF/RaDe-GS 系） | 0.05 | `lambda_gs_normal`：`1−cos(rend_normal, surf_normal)`，`start_normal_reg` 门控 | **0.05** |
| L_N（法线对齐：渲染 oriented normal ↔ 渲染深度图像梯度） | 0.05 | `lambda_photometric_normal_live`：独立法线 ↔ 深度导出法线 | **0.05** |
| L_gc（多视角几何一致性） | 0.02 | `lambda_photometric_normal_mv` | **0.02** |
| L_pc（多视角光度一致性） | 0.6 | **无对应项**，不实现 | — |

调度沿用本仓库既有约定（控制变量，与 0823-01/0824-01 可比）：live@500 起、mv@1000 起 + 2000 步线性 ramp、gs_normal 受 `start_normal_reg=500` 门控；论文为从头施加，此差异在实验文档中注明。

## 3. 代码改动

### 3.1 `scripts/loss.py`

`LOSS_PRESETS` 新增：

| 预设名 | 兼容 render_mode | 覆盖的参数（未显式传时才生效） |
| --- | --- | --- |
| `gs_wrapping` | photometric_lambertian | `lambda_gs_normal=0.05`、`lambda_photometric_normal_live=0.05`、`lambda_photometric_normal_mv=0.02` |

terms 登记：`rgb_l1`、`rgb_dssim`、`light_smooth`、`normal`、`photometric_normal_live`、`photometric_normal_mv`、`alpha`、`deformation_xyz`、`binary_separation`。distortion 保持默认（λ=1000，`start_normal_reg` 门控），不在 terms 中单列（与既有 lambertian 系预设一致）。

### 3.2 `tests/test_stage1_loss.py`

- `test_registry_complete` 注册表断言补 `gs_wrapping`。
- 新增 2 例：`test_gs_wrapping_applies_weights`（三项覆盖值与顺序）、`test_gs_wrapping_requires_lambertian_mode`（render_mode 校验）。

## 4. 兼容性

- 纯新增预设：默认值、既有预设、`auto` 行为全部不变，既有实验零漂移。
- `gs_wrapping` 与 0824-01（`lambertian_normal2`）的差异即 0824-02 实验变量：①恢复并加大 L_DN（gs_normal 0→0.05）；②live 0.01→0.05；③mv 不变（0.02）。

## 5. 机制层面不可复现的差异（结论限定用）

1. 论文深度为其 occupancy 场的精确 0.5-等值面（改 CUDA rasterizer）；本仓库用 `surf_depth`。
2. 论文 L_N 经法线渲染与等值面深度双向可微；本仓库 `photometric_normal_live` 的几何目标 `surf_normal` 是 `.detach()` 的（单向把独立法线拉向几何）。
3. 论文用 L_N 误差驱动"翻转法线克隆"稠密化补洞；本仓库无此耦合。

因此 `gs_wrapping` 仅对齐**损失形式与权重**，实验结论限定在损失配置层面。

## 6. 验证

- `python -m unittest tests.test_stage1_loss -v`：14/14 通过（含新增 2 例）。
- 冒烟/正式训练验收见 `DOC/训练/0824-02-*` 与 `output/0824-02-CV3s-GTlight_lambert_iter1_gswrapping/README.md`。

## 7. 关联文档

- `DOC/指导/blender数据训练指导.md`：`--loss_preset` 预设列表补 `gs_wrapping`。
- `DOC/训练/0824-02-CV3s静态布料从iter1起GT光照Lambertian_gs_wrapping论文损失对照训练方案与执行记录.md`。
- 论文：arXiv:2604.07337（Gomez, Guédon, Maruani, Gong, Ovsjanikov，2026-04-08）。
