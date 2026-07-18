# AI Coding Agents 工作入口

本文件是 AI coding agents 在本仓库中的工作入口说明。

关于 LumiMotion 的训练流程、数据结构、代码地图、评估方式以及常见参数陷阱，请优先阅读：

- [`lumimotion`](./readme.md)

## 工作定位

本仓库首先作为个人 baseline 仓库使用。

除非用户明确要求，所有改动都应遵循以下原则：

1. 保持改动范围小、逻辑清晰、便于解释和回退。
2. 避免影响原论文复现路径。
3. 不进行与当前任务无关的重构。
4. 保留已有实验结果、数据、checkpoint、日志以及未提交文件。
5. 修改训练、渲染、评估或数据读取逻辑时，必须明确说明行为变化及潜在影响。
6. 对 CUDA / native 扩展、依赖版本和 checkpoint 兼容性保持谨慎。

## 工作优先级

按以下优先级处理仓库任务：

1. 保持 baseline 可追踪。
2. 保证已有实验和复现流程不被破坏。
3. 优先实现用户明确提出的功能或修复。
4. 避免扩大修改范围。
5. 对可能影响训练结果、数值稳定性或兼容性的改动进行说明。

## 输出规定

1. 输出请统一输出到 output 文件夹下。
2. 如果是冒烟测试请统一输出到 `output/smoke_test` 文件夹内。
3. 标准测试如有指定按照指定的来；如未指定请输出为“日期-数据集-特征”，如 `0712-cloth_dynamic-V5`。
4. 输出文件夹内还应当新建一个 `README.md`，包含训练命令，主要存储超参数。
5. 每次测试都需要有 eval 图片。
6. 如果有 log 文件也请放在同目录内。

## 文档入口

### `readme.md`

原项目文档，主要包含环境安装、数据准备、训练与评估及原论文复现说明。

### `lumimotion.md`

本仓库的中文项目工作指南，主要包含两阶段训练流程、评估流程、数据目录布局、代码地图、关键参数说明及常见配置与参数陷阱。

## LH Dynamic 数据转换

逐帧相机的 LH Dynamic 数据使用 [`scripts/prepare_lh_dynamic.py`](./scripts/prepare_lh_dynamic.py) 转换为 LumiMotion 的 Blender transforms 格式。示例：

```bash
python scripts/prepare_lh_dynamic.py \
  data/LH-data/danamic/only_clothV2 \
  data/LH-data/transfer-dynamic/only_clothV2 \
  --test-stride 8 \
  --camera-extent 1.0
```

该脚本会校验 image、albedo、normal EXR、camera、light 和 object pose 的帧号一致性，从 albedo PNG 完整保留软 alpha mask，并为每帧写入独立的相机内参、FOV 和 camera-to-world 矩阵。输出目录必须为空，避免覆盖已有数据。转换结果的 `dataset_manifest.json` 记录数据划分、mask 统计及 Stage1 实际使用的信息。

## Stage 1 Eval 规范

Stage 1 训练完成后，必须使用 [`scripts/render_stage1_insights.py`](./scripts/render_stage1_insights.py) 进行完整时序渲染。参考产物为 `output/Baseline/0714-only_cloth-original-stage1-alpha-bg/only_cloth_stage1_mlp/renders_stage1_insights/ours_35000`。

### 1. 渲染前检查

1. 先用 `hostname` 确认服务器，选择对应 Conda 环境和空闲 GPU。
2. 确认目标 iteration 同时存在 `point_cloud/iteration_<ITER>/point_cloud.ply` 和 `deform/iteration_<ITER>/deform.pth`。
3. `--source_path`、`--train_light_folder`、`--resolution`、`--is_blender` 和 `--eval` 必须与训练一致。
4. `--model_path` 传入实际模型目录；MLP 模型通常为带 `_mlp` 后缀的路径。不要在未核对时传入无后缀路径。

### 2. 标准渲染命令

```bash
ROOT=/home/han.li/reproduce/LumiMotion
OUT="$ROOT/output/Baseline/<experiment>"
SOURCE="$ROOT/data/<dataset>"
MODEL="$OUT/<model>_mlp"
ITER=35000

CUDA_VISIBLE_DEVICES=<gpu> conda run --no-capture-output -n <server-conda-env> \
  python -m scripts.render_stage1_insights \
  --source_path "$SOURCE" \
  --model_path "$MODEL" \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --load_iter "$ITER" --depth_ratio 0.0 --quiet \
  2>&1 | tee "$OUT/render_stage1_insights.log"
```

`render_stage1_insights.py` 默认固定第一个 test camera，并遍历数据的所有 timestep。因此这是“固定视角的完整时序检查”，不等价于使用每帧原始移动相机进行 GT 对齐，也不代替 novel-view 定量评估。

### 3. 必需产物

渲染结果必须位于 `<MODEL>/renders_stage1_insights/ours_<ITER>/`，并核对帧数与数据 timestep 数相同。以 120 帧参考数据为例，应包含：

- 120 张 `full_t*_cam*.png`：完整 RGB 渲染。
- 120 张 `normals_t*_cam*.png`：表面法线。
- 120 张 `separation_small_t*_cam*.png`：小 Gaussian 的动态/静态分离。
- 120 张 `separation_large_t*_cam*.png`：原始尺度 Gaussian 的动态/静态分离。
- `full_render_cam*.mp4`、`alpha_cam*.mp4`、`Albedo_cam*.mp4`、`Normals_cam*.mp4`、`small_gaussians_cam*.mp4`、`Separation_cam*.mp4` 和 `Separation_large_cam*.mp4`。
- 实验根目录下的 `render_stage1_insights.log`。

每次正式测试还必须在实验根目录保存可直接打开的 eval 图片。至少从开头、中间和结尾 timestep 制作 RGB 和 alpha 接触表；不能只保存 MP4。参考命名为：

- `alpha_input_contact_sheet.png` 和 `alpha_input_stats.json`：输入 mask 的覆盖率及时序变化。
- `alpha_render_contact_sheet.png` 和 `alpha_render_stats.json`：渲染 alpha 的开头/中间/结尾帧与统计。
- `eval_rgb_contact_sheet.png`：完整 RGB 渲染的代表帧。

### 4. 检查顺序与验收

1. 先检查 RGB：FOV、宽高比、主体尺寸和位置正确，时序运动连续，不能出现主体缺失、拉伸、跳变、闪烁或明显拖影。
2. 再检查 alpha：背景应稳定为黑，轮廓应跟随主体，不能出现漂浮点、大面积条带、背景覆盖或布料消失。
3. 检查 normals：表面方向随运动连续，不应有大面积随机噪声、突然翻转或与 RGB 轮廓脱离。
4. 检查 separation：动态部分应有稳定响应，不能几乎全静态，也不能整个场景无区分地全动态。
5. 最后核对训练日志/TensorBoard 的 test PSNR、SSIM、LPIPS 和点数。定量指标不能代替 RGB、alpha、normal 和 separation 的目检。

只有上述四类可视化均正常，才能将 Stage 1 标记为通过。如果任一关键动态物体消失，alpha 存在明显漂浮 Gaussian/条带，或时序中出现崩坏，即使 PSNR/SSIM 较高也必须标记为验收失败。

### 5. README 记录要求

实验 `README.md` 必须记录：完整渲染命令、source/model/iteration、渲染输出目录、日志、定量指标、代表图片/视频路径、四类可视化的目检结论以及最终 `PASS`/`FAILED` 结论。验收失败时必须保留 checkpoint、渲染、统计和日志，不得覆盖或删除失败结果。

## 运行环境约定

本项目在多个服务器上运行。执行命令前，应先确认当前服务器名称，并使用对应的 Conda 环境。

| 服务器名称 | Conda 环境 |
| ---------- | ---------- |
| `mahadevi` | `lumimotion-mahadevi` |
| `minakshi` | `lumimotion-minakshi` |
| `parvati` | `lumimotion-parvati` |
| `ushas` | `lumimotion-ushas` |
| `garuda` | `lumimotion-garuda` |

可通过以下命令确认服务器名称：

```bash
hostname
```
