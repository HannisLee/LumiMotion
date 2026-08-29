# 0826-01 Stage1 Loss 对象与 Preset 组合式重构

- 日期：2026-08-26
- 修改序号：01
- 服务器/环境：garuda / `lumimotion-garuda`
- 结论：**PASS**
- 概要：将 `scripts/loss.py` 从集中条件分支重构为“原子 loss 对象 + 独立 preset 大 loss”组合。每个 preset 在自身 `__init__` 中完整声明损失组成，原子 loss 在构造时固定权重；保留原训练入口与 CLI 覆盖规则。按项目决策停用 Stage1 PBR loss。

## 1. Baseline 备份

重构前原文件已逐字节备份到：

```text
scripts/loss_legacy_20260826.py
```

重构前 `scripts/loss.py` 与备份文件的 SHA-256 均为：

```text
3f6cfb6f33499536ef775f946979b5f6726a62ec43746692e0db2925ee70d0f9
```

`cmp` 校验结果为完全一致。备份只用于追踪历史 baseline，新实现不从该文件导入任何代码。

## 2. 实现结构

### 2.1 原子 loss 对象

新增独立类：

- RGB：`RGBL1Loss`、`RGBDSSIMLoss`；
- 几何：`GSNormalLoss`、`DistortionLoss`；
- 独立法线：`GTNormalOracleLoss`、`PhotometricNormalInitLoss`、`PhotometricNormalLiveLoss`、`PhotometricNormalMVLoss`；
- 光照与 mask：`LightSmoothnessLoss`、`AlphaLoss`；
- 形变与分离：`DeformationXYZLoss`、`DeformationColorLoss`、`BinarySeparationLoss`。

每个对象的 `ratio` 默认值为 1.0，并在 preset 构造时读取最终优化参数后固定。iteration 门控、alpha threshold、MV interval/depth tolerance/ramp 等调度由对应对象保存。返回值已乘权重，preset 只做顺序加法。

live 与 MV 通过 `Stage1LossContext` 共享本次当前视角 world-normal render 缓存，避免两项同时启用时重复渲染。

### 2.2 独立 preset 大 loss

以下类互不继承，每个类的 `__init__` 都显式列出自己的 `self.losses`：

- `SHBaselineLossPreset`
- `LambertianDefaultLossPreset`
- `LambertianNormal2LossPreset`
- `LambertianNormal3LossPreset`
- `GSWrappingLossPreset`
- `N0GTNormalOracleLossPreset`
- `PBRDefaultLossPreset`
- 内部 `AutoLossPreset`

`sum_stage1_losses()` 按声明顺序逐项执行和累加，并依据 loss 对象的 `audit_name` 组装原有梯度审计项。总损失顺序保持为：

```text
RGB L1 → RGB DSSIM → GS normal → distortion
→ GT normal → normal init → live → MV
→ light smooth → alpha → d_xyz → d_color → separation
```

`DeformationColorLoss` 与 `BinarySeparationLoss` 仍只进入 total，不新增历史上不存在的 audit 项。

### 2.3 调用兼容

保留：

- `compute_stage1_loss(...)` 的参数和 `Stage1LossResult` 返回结构；
- `apply_loss_preset(args, argv=None)` 的调用方式、返回值和启动日志；
- `LOSS_PRESETS` 的全部历史键名；
- `.description/.render_modes/.overrides/.terms` 访问；
- `auto` 不覆盖参数、显式 CLI 参数优先于 preset 默认值。

新增 `build_loss_preset(opt, render_mode)`。正式训练在 `Trainer.__init__` 中只构造一次 `self.stage1_loss`，避免每个 iteration 重建对象；`compute_stage1_loss()` 对测试和旧的最小 Trainer mock 保留临时构造兼容路径。

## 3. PBR 停用说明

根据当前项目范围，`photometric_perlight_pbr` Stage1 loss 已放弃：

- `apply_loss_preset()` 在参数应用阶段检测到 PBR render mode 即抛出 `AssertionError`；
- `build_loss_preset()` 与 `PBRDefaultLossPreset.compute()` 均有防御性报错；
- 使用显式 `raise AssertionError`，不受 Python `-O` 移除 assert 语句影响；
- 历史 PBR 公式在当前文件保留只读参考函数，完整旧实现保存在 baseline 备份中。

这是有意的兼容性变化。SH 与 Lambertian 路径仍保持现有入口、公式和参数覆盖行为。

## 4. 数值与兼容性影响

- SH 手写参考测试继续使用 `atol=0, rtol=0`，重构后逐值一致；
- Lambertian 新增手写参考测试，覆盖 RGB 总权重、DSSIM、光照平滑和 alpha；
- 原子 loss 按旧顺序显式相加，未改用 `stack/sum` 等可能改变归约顺序的实现；
- `audit_terms` 的键名和顺序保持；
- photometric 启动前，Lambertian preset 中 photometric-only loss 自行关闭，`d_color` 仍按原 SH warm-up 行为生效；
- 没有执行训练，不产生 checkpoint 或 `output`，因此本次没有 `DOC/训练` 记录。

## 5. 验证

### 5.1 语法检查

```bash
conda run -n lumimotion-garuda python -m py_compile \
  scripts/loss.py scripts/loss_legacy_20260826.py \
  scripts/train_stage1.py tests/test_stage1_loss.py
```

结果：PASS。

### 5.2 Loss 定向单测

```bash
conda run -n lumimotion-garuda \
  python -m unittest tests.test_stage1_loss -v
```

结果：23/23 PASS。覆盖：

- SH 与 Lambertian 手写总损失；
- PBR 参数阶段和直接计算报错；
- preset 注册、模式校验、CLI 优先级；
- preset 无继承及 loss 组成顺序；
- ratio 构造后固定、默认 ratio=1；
- world-normal 缓存只渲染一次；
- baseline 备份 SHA-256。

### 5.3 全量单测

```bash
conda run -n lumimotion-garuda \
  python -m unittest discover -s tests -v
```

结果：71/71 PASS。仅出现既有 Kornia/Torchvision deprecated API warning，无测试失败。

## 6. 修改文件

- `scripts/loss.py`：对象组合式重构；
- `scripts/loss_legacy_20260826.py`：重构前逐字节 baseline；
- `scripts/train_stage1.py`：Trainer 构造一次 preset；
- `tests/test_stage1_loss.py`：更新 PBR 预期并扩展组合测试；
- `DOC/指导/blender数据训练指导.md`：补充对象组合与 PBR 停用说明；
- 本文档：记录实现与验证。

工作区原有的 normal evaluator 等无关修改未触碰、未纳入本次变更。

