# LumiMotion 文档库与 Agent 工作索引

更新日期：2026-08-29
适用范围：本仓库中的 LumiMotion 复现实验、数据准备、Stage 1 训练、离线评估与后续代码维护。

本文件是进入项目后的**第一阅读入口**。它不重复各实验的原始命令或指标；需要执行、判断或修改时，请沿下方链接回到对应的一手文档、脚本和实验产物。当前文档库的唯一入口目录为 `LIHAN/DOC/`。

## 先读什么

| 任务 | 首先阅读 | 然后查看 |
| --- | --- | --- |
| 了解仓库规则、命名、环境与产物保留要求 | [仓库协作约定](../AGENTS.md) | 本文“Agent 工作流程” |
| 首次理解项目代码与标准训练阶段 | [项目原始说明](../readme.md) | [Blender 数据训练指导](DOC/指导/blender数据训练指导.md) |
| 新建/检查 Blender 数据集 | [数据集转换指导](DOC/指导/blender数据集转换指导.md) | [训练集加载要求](DOC/指导/训练集加载要求.md) |
| 延续当前真实方向光实验 | [0829 冒烟训练记录](DOC/训练/0829-01-CV3-真实方向光-LambertianStage1冒烟训练.md) | 该实验目录的 `README.md`、`run.sh`、日志和指标 |
| 改 Stage 1 损失、预设或法线评估 | [0828 损失/评估完善](DOC/修改/0828-01-Stage1损失预设与法线评估完善.md) | [0826 loss 对象化重构](DOC/修改/0826-01-loss对象与preset组合式重构.md)、相关单测 |
| 判断 SH、Lambertian 或逐 GS 光方向的取舍 | [SH 优于 GT 光的审计](DOC/报告/08.24-CV3s-SH优于GT光Lambertian原因审计报告.md) | [坐标终审](DOC/报告/08.24-CV3s逐GS_GT位置光方向渲染坐标终审报告.md)、对应实验记录 |

## Agent 工作流程

1. **先确认现场。** 在仓库根目录运行 `hostname` 和 `git status --short`；不得覆盖、删除或回滚自己未产生的改动。当前文档库位于 `LIHAN/DOC/`，不要假设历史路径仍然有效。
2. **选择正确环境。** 服务器名与 Conda 环境一一对应：`mahadevi` → `lumimotion-mahadevi`、`minakshi` → `lumimotion-minakshi`、`parvati` → `lumimotion-parvati`、`ushas` → `lumimotion-ushas`、`garuda` → `lumimotion-garuda`。本机当前为 `garuda`，命令应使用 `conda run --no-capture-output -n lumimotion-garuda ...` 或等价的已激活环境。
3. **按任务读一手资料。** 训练前至少阅读“指导”中的训练与加载要求、目标训练记录，以及关联的“修改”文档；代码改动前至少阅读最新修改记录、目标脚本和对应单测。
4. **让产物成为证据。** 实验结论以 `output/<实验名>/README.md`、完整命令、日志、checkpoint、渲染和 JSON/CSV 指标为准；文档中的摘要不能替代产物核验。冒烟结果只能写在 `output/smoke_test/`，不能作为 35k 正式训练的验收结论。
5. **完成后补齐记录。** 代码修改写入 `LIHAN/DOC/修改/`；新训练或重新训练写入 `LIHAN/DOC/训练/`。除本入口及实验目录的 `README.md` 外，新增文档使用 `日期-当日序号-修改内容.md` 命名，并以中文记录。

训练验收必须保留完整渲染命令、source/model/iteration、渲染目录、日志、定量指标、代表图像/视频、RGB/alpha/albedo-separation/normal 四类目检和最终 `PASS`/`FAILED`。失败实验的 checkpoint、渲染、统计和日志必须保留，不能覆盖或删除。

## 文档库导航

### 1. 指导：执行前的规范与数据前置条件

| 文档 | 用途 |
| --- | --- |
| [Blender 数据训练指导](DOC/指导/blender数据训练指导.md) | CV3/CV3L 数据别名、Stage 1 调度、训练和验收的主操作规范；新训练优先读。 |
| [训练集加载要求](DOC/指导/训练集加载要求.md) | 开训前检查 `transforms`、alpha、帧号、相机与目录结构，避免静默数据错误。 |
| [Blender 数据集转换指导](DOC/指导/blender数据集转换指导.md) | 原始 Blender 导出到可训练数据集的转换命令与相机/alpha/PLY 校验。 |
| [Blender 训练错误报告](DOC/指导/blender训练错误报告.md) | only_cloth 早期失败的根因库；遇到相机、尺度、alpha 或固定单视角问题时查阅。 |

### 2. 训练：实验事实、命令和验收结论

#### 当前优先级

- [0829-01 CV3 真实方向光 Lambertian Stage 1 冒烟](DOC/训练/0829-01-CV3-真实方向光-LambertianStage1冒烟训练.md)：数据与管线通过；正式 35k 尚未启动，且短训 independent normal 不通过。后续工作必须先确认真实 SUN 光的方向/强度标定与 staged-training 调度，不能直接将该冒烟视为正式通过。
- [0823-01 CV3s 三法线一致性](DOC/训练/0823-01-CV3s静态布料从iter1起GT光照Lambertian三法线一致性训练方案与执行记录.md)：迭代 1 起 GT 方向光 Lambertian 的已通过基线。
- [0824-01 CV3s 二法线消融](DOC/训练/0824-01-CV3s静态布料从iter1起GT光照Lambertian二法线一致性消融训练方案与执行记录.md)：已通过；是去掉 GS 几何自一致项、保留 live/MV 项的对照。
- [0824-04 CV3s 逐 GS GT 位置方向三法线](DOC/训练/0824-04-CV3s逐GS_GT位置光方向三法线消融实验.md)：已通过的方向 oracle 消融；法线改善但 RGB 不优于 0823 基线，不应直接替代默认 RGB 重建方案。

#### 已完成/历史对照

| 文档 | 结论或查阅目的 |
| --- | --- |
| [0824-02 gs_wrapping 论文损失对照](DOC/训练/0824-02-CV3s静态布料从iter1起GT光照Lambertian_gs_wrapping论文损失对照训练方案与执行记录.md) | 正式实验 `FAILED`；保留为论文权重映射与失败证据。 |
| [0824-03 SH 默认 2DGS normal 基线](DOC/训练/0824-03-CV3s_SH默认2DGSnormal基线训练与对照方案.md) | 实验 A `PASS`；Lambertian 对照 B 未启动，不能把它当成成对结论。 |
| [0824-05 逐 GS 方向 + 默认损失](DOC/训练/0824-05-CV3s逐GS_GT位置光方向Lambert默认损失训练方案与执行记录.md) | `FAILED`：RGB 可通过而 independent normal 明显退化，是单灯 RGB 不足以约束法线的直接证据。 |
| [0809–0818 训练与可视化记录](DOC/训练) | CV3/CV3L 的历史训练、标定、阶段冻结与 PPT 素材；需要追溯旧路径、旧指标或失败原因时按日期阅读。标有“归档”的实验不可作为当前默认方案。 |

### 3. 修改：代码现状与兼容性边界

修改文档按时间顺序阅读；最新状态优先于旧文档，但旧文档用于解释参数和设计来源。

| 顺序 | 文档 | 需要知道的内容 |
| --- | --- | --- |
| 0823-01 | [loss 集中化与预设传参](DOC/修改/0823-01-loss集中化重构与预设传参.md) | `scripts/loss.py`、`--loss_preset` 的引入和历史兼容目标。 |
| 0824-01 / 03 | [GS normal 开关](DOC/修改/0824-01-lambda_gs_normal开关与lambertian_normal2预设.md) / [gs_wrapping 预设](DOC/修改/0824-03-gs_wrapping预设.md) | `lambda_gs_normal`、`lambertian_normal2` 与 `gs_wrapping` 的实验含义。 |
| 0824-04 / 06 | [SH 评估兼容](DOC/修改/0824-04-eval_stage1_normals_gt支持SH基线.md) / [双法线与余弦损失](DOC/修改/0824-06-eval_stage1_normals_gt双法线与余弦损失.md) | normal evaluator 的模式支持、诊断兼容与双来源评估。 |
| 0824-05 | [逐 GS GT 位置光方向](DOC/修改/0824-05-逐GS_GT位置光方向消融模式.md) | `gt_point_direction_only` 的语义、坐标与无距离衰减约束。 |
| 0826-01 | [loss 对象与 preset 组合重构](DOC/修改/0826-01-loss对象与preset组合式重构.md) | 当前 `scripts/loss.py` 架构、PBR Stage 1 停用、全量单测基线。 |
| 0828-01 | [Stage1 损失预设与法线评估完善](DOC/修改/0828-01-Stage1损失预设与法线评估完善.md) | 最新 loss preset 构造方式，以及 `--normal_source {auto,independent,gs}` 和掩码余弦指标。 |

[0824-02 损失级光强归一化方案 D](DOC/修改/0824-02-损失级光强归一化方案D设计（待实施）.md) 是**未实施设计**，不得误认为现有训练已经具备该能力。

### 4. 报告：解释实验结果，不替代复现记录

| 文档 | 应在何时阅读 |
| --- | --- |
| [CV3s 坐标与逐 GS 光方向终审](DOC/报告/08.24-CV3s逐GS_GT位置光方向渲染坐标终审报告.md) | 怀疑方向符号、坐标系、时间索引或“GT 灯位是否真等于 GT 成像”时。结论是坐标链路正确，但当前只是局部中心方向 oracle。 |
| [SH 优于 GT 光 Lambertian 原因审计](DOC/报告/08.24-CV3s-SH优于GT光Lambertian原因审计报告.md) | 比较 SH 与 Lambertian 时。核心限制是 GT 灯位被简化、SH 外观自由度更强、相机/时间/光照未解耦。 |
| [SH 主干与 Per-light 方向融合方案](DOC/报告/08.24-SH主干与Per-light方向融合方案报告.md) | 评估下一阶段方案时。它是设计提案，尚未改代码或启动训练。 |
| [报告归档](DOC/报告/archive) | 查早期 Stage 1/2、PBR、法线、光强、坐标审计的原始证据；仅用于追溯，不覆盖上述现行结论。 |

## 代码、数据与产物的定位图

```text
LIHAN/DOC/指导 ── 数据规范、训练前检查
LIHAN/DOC/训练 ── 每次实验的计划、命令、结果与验收
LIHAN/DOC/修改 ── 代码变更缘由、影响与验证
LIHAN/DOC/报告 ── 跨实验审计与后续方案

scripts/train_stage1.py ── Stage 1 训练入口与调度
scripts/loss.py ────────── 损失对象、预设和组装
scene/photometric_lambertian.py ── Lambertian / GT 光模式
gaussian_renderer/__init__.py ─── 渲染模式分派
scripts/eval_stage1_normals_gt.py ─ GT normal 离线评估
scripts/render_stage1_insights.py ── RGB/alpha/albedo/normal 等可视化
LIHAN/scripts/ ─────────── 本地数据转换与 alpha 专项评估工具
tests/ ─────────────────── 各模块的可执行回归测试
output/ ────────────────── 训练、渲染、日志、指标和实验 README
```

数据路径以目标训练文档及其 `run.sh` 为准。常用简称：`CV3` 为 clothV3 动态数据，`CV3L` 为其 Lambertian 数据对，`CV3s` 为 clothV3 静态布料/相机运动版本。不要混用各自的 source、`lights.json`、GT normal 或光强标定值。

## 变更后的最低验证

- 改损失、参数解析或训练调度：先运行 `tests/test_stage1_loss.py`，再按改动范围运行 `python -m unittest discover -s tests -v`。
- 改相机、数据或灯光解释：运行相应的 `test_blender_camera_convention.py`、`test_directional_light_calibration.py`、`test_photometric_lambertian.py` 和数据转换/加载检查。
- 改 normal evaluator：运行 `test_normal_eval_utils.py`，并在同一 checkpoint 上明确记录采用 `independent`、`gs` 或 `auto`。
- 启动正式训练前：完成目标配置的冒烟，确认 checkpoint、日志、四类可视化和定量指标齐全；未满足验收条件时记录 `FAILED` 并保留产物。

当文档、代码和产物存在冲突时，优先级为：**当前代码与可复现实验产物 > 最新修改/训练记录 > 审计报告 > 历史归档**。发现不一致时，不要静默修正文档；先定位差异，再在相应目录新增中文记录。
