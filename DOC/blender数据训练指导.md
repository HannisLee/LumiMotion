# LumiMotion Blender 数据训练指导

更新日期：2026-07-18  
仓库：`/home/han.li/reproduce/LumiMotion`

本文整理 `only_cloth` 错误实验和 `only_clothV2` 训练的实际经验，目标是在不改变
LumiMotion Stage 1 模型主体和损失定义的前提下，避免错误 FOV、alpha 背景监督、
Gaussian 暴增或误删，并建立可复现的训练与验收流程。

相关材料：

- `DOC/blender训练错误报告.md`
- `scripts/prepare_lh_dynamic.py`
- `scripts/train_stage1.py`
- `scripts/render_stage1_insights.py`
- `output/Baseline/0717-only_clothV2-ply62-stage1/README.md`

> 注意：仓库入口说明引用的 `lumimotion.md` 当前不存在。本文暂时以最新
> `AGENTS.md`、错误报告和已完成实验为依据；恢复该文件后应再次核对参数约定。

## 1. 这次训练最重要的结论

1. **先证明数据和 PLY 对齐，再调训练参数。** 错误投影无法通过更多 Gaussian
   修复，只会诱导点云复制、拉伸和漂移。
2. **非方形 Blender 图像必须分别读取 `FovX/FovY`。** 优先从每帧 `fl_x/fl_y`
   和实际宽高计算，不要交换水平与垂直 FOV，也不要假设二者相同。
3. **alpha 必须来自真实物体 mask。** `only_clothV2` 使用 albedo PNG 的 soft
   alpha，并与黑色 renderer 背景保持一致；不能把灰色背景和全 255 alpha 一起训练。
4. **中间帧 PLY 通常比首帧更适合作为 canonical 初始化。** 中间帧更接近整个
   时序的平均状态，最大形变距离通常更小。本次使用 frame 62 的 4096 点完整场景 PLY。
5. **停止 densification 不等于冻结 deformation。** 停止的是点的 clone/split
   和点数变化；deformation MLP、位置增量、颜色、opacity 等仍会一直反向传播学习。
6. **不要在增密结束后长期进行带尺度阈值的 `prune-only`。** 本次从 4009 点逐步
   误删到 iteration 20000 的 267 点，证明 world/screen-size 阈值不适合无条件重复使用。
7. **高 PSNR 不代表动态几何正确。** 固定相机完整时序中的 RGB、alpha、normal
   和 separation 任一出现明显崩坏，Stage 1 都必须判为 `FAILED`。

## 2. 数据转换

### 2.1 环境检查

先确认服务器，再选择对应 Conda 环境：

```bash
cd /home/han.li/reproduce/LumiMotion
hostname
```

| 服务器 | Conda 环境 |
| --- | --- |
| `mahadevi` | `lumimotion-mahadevi` |
| `minakshi` | `lumimotion-minakshi` |
| `parvati` | `lumimotion-parvati` |
| `ushas` | `lumimotion-ushas` |
| `garuda` | `lumimotion-garuda` |

训练前还要检查 GPU。服务器存在故障 GPU 时，数字 ordinal 可能发生错误映射；优先使用
经过验证的 GPU UUID：

```bash
nvidia-smi -L
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

### 2.2 转换命令

目标目录必须为空，避免覆盖已有 transforms、mask 或手工放入的 PLY：

```bash
conda run --no-capture-output -n <server-conda-env> \
python scripts/prepare_lh_dynamic.py \
  data/LH-data/danamic/only_clothV2 \
  data/LH-data/transfer-dynamic/only_clothV2 \
  --test-stride 8 \
  --camera-extent 1.0
```

转换脚本应完成以下工作：

- 校验 image、albedo、normal EXR、camera、light 和 object pose 的帧号一致。
- 从 albedo PNG 完整保留 soft alpha。
- 为每帧写入独立 `fl_x/fl_y`、`camera_angle_x/y` 和 camera-to-world 矩阵。
- 写出 `transforms_train.json`、`transforms_test.json` 和
  `dataset_manifest.json`。
- 120 帧、`--test-stride 8` 时应得到 105 个 train frame 和 15 个 test frame。

### 2.3 PLY 命名和 canonical 帧

Blender reader 查找的是小写 `points3d.ply`。若导出文件是 `points3D.ply`，可以使用
硬链接避免复制和双份维护：

```bash
cd data/LH-data/transfer-dynamic/only_clothV2
ln points3D.ply points3d.ply
ls -li points3D.ply points3d.ply
cd /home/han.li/reproduce/LumiMotion
```

创建前先确认小写文件不存在；不要覆盖已有初始化点云。

canonical PLY 选择原则：

- 动态物体从一种极端状态移动到另一种极端状态时，优先选择中间帧。
- 选择主体可见、遮挡较少、拓扑完整且没有运动模糊的帧。
- PLY 应包含完整静态场景和动态物体，而不是只包含布料。
- PLY 只初始化 canonical XYZ；输入颜色、法线、速度和动态标签不会自动成为
  Gaussian 的动态先验。
- 最终 checkpoint 中的 `point_cloud.ply` 仍是 canonical Gaussian；各时刻形状由
  `deform.pth` 产生。因此不能只打开最终 PLY 判断全部动态时刻是否正确。

本次 `only_clothV2` 使用 frame 62：

```text
点数       = 4096
bbox min   = [-1.4454, -1.4498, -0.1800]
bbox max   = [ 1.4422,  1.4475,  1.9242]
```

### 2.4 训练前必须通过的检查

1. transforms 的 frame 数与 images 一致，train/test 不重叠。
2. `fl_x/fl_y`、FOV 和图像宽高一致；1280×720 原图不应被当作方形图像。
3. 输入 alpha 背景为 0，前景不是整幅图全 255。
4. 相机中心跨度、scene extent 和 PLY bbox 都是合理有限值。
5. 把 PLY 投影到至少首帧、中间帧和末帧，保存 overlay。
6. 点云投影包围盒应覆盖主体，不能出现统一偏移、上下颠倒或宽高比拉伸。

本次 frame 60 投影有 99.10% 的点落在 GT alpha 内，投影包围盒与 alpha 基本一致，
因此可以排除 PLY 整体 scale 或坐标系错位。

## 3. 推荐 Stage 1 训练策略

### 3.1 原则

`only_clothV2` 的初始化已经是场景表面点，不需要像随机点初始化那样长时间激进增密。
当前统一采用“第二档”方案，把训练分成两个阶段：

1. iteration 1500–5000：有限增密和正常 prune，补充初始化漏掉的区域。
2. iteration 5000–35000：停止点结构变化，固定点数，但继续完整训练 Gaussian 参数
   和 deformation MLP。

该方案在本次实验的 iteration 5000 实际得到 15304 个 Gaussian，画质较好且显存
可控。15304 是本次观测结果而不是严格保证值；换数据、相机或损失后，梯度分布变化会
导致最终点数变化。

推荐值：

| 参数 | 推荐值 | 说明 |
| --- | ---: | --- |
| `warm_up` | 500 | 早期先稳定 canonical 表示 |
| `densify_from_iter` | 1500 | 不要一开始立即 clone/split |
| `densify_until_iter` | 5000 | 在已验证的约 15k 点位置停止增密 |
| `densification_interval` | 200 | 降低结构变化频率 |
| `densify_grad_threshold` | 0.0004 | 比 0.0002 更保守 |
| `opacity_reset_interval` | 3000 | 仅在增密窗口内生效 |
| `min_opacity` | 0.01 | 与 reset 边界保持安全间隔 |
| `prune_from_iter` | -1 | 禁止后段重复 prune-only |
| `max_gaussians` | 20000 | 软上限，只作为异常保护，不是严格点数上限 |
| `lambda_separation` | 0.005 | 本次实际值；若全静态退化再单独降至 0.001 |
| `d_xyz_loss_weight` | 0.001 | 抑制无约束空间漂移 |
| `d_color_reg_loss_weight` | 0.01 | 抑制用颜色变化代替几何 |

`max_gaussians` 是软熔断器：当前实现在一次 densification 开始前检查点数，单次
clone/split 仍可能越过 20000，下一轮才停止继续增密。因此真正的主控制参数是
`densify_until_iter=5000`；一旦明显接近或越过 20k，应暂停检查 FOV、alpha、extent、
尺度和点数曲线，而不是把上限继续调高。

### 3.2 推荐的完整新训练命令

下面命令是当前确定采用的“第二档”稳定策略。运行时替换 GPU、环境和输出日期；正式
实验输出必须放在 `output/Baseline/` 下：

```bash
cd /home/han.li/reproduce/LumiMotion

SOURCE=data/LH-data/transfer-dynamic/only_clothV2
OUT=output/Baseline/<日期>-only_clothV2-ply62-stage1
MODEL="$OUT/only_clothV2_stage1"
GPU=<经过验证的GPU序号或UUID>
ENV=<当前服务器对应的Conda环境>

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n "$ENV" \
python -u -m scripts.train_stage1 \
  --source_path "$SOURCE" \
  --model_path "$MODEL" \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 \
  --iterations 35000 \
  --warm_up 500 \
  --densify_from_iter 1500 \
  --densify_until_iter 5000 \
  --densification_interval 200 \
  --densify_grad_threshold 0.0004 \
  --opacity_reset_interval 3000 \
  --min_opacity 0.01 \
  --prune_from_iter -1 \
  --max_gaussians 20000 \
  --binarization_warm_up 1000 \
  --lambda_separation 0.005 \
  --d_xyz_loss_weight 0.001 \
  --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 \
  --test_iterations 500 1000 5000 10000 20000 30000 35000 \
  --save_iterations 499 500 1000 5000 10000 20000 30000 35000 \
  --quiet \
  2>&1 | tee "$OUT/train_stage1.log"
```

`scripts/train_stage1.py` 会根据 deformation 类型生成实际带 `_mlp` 后缀的模型目录。
渲染前必须核对真实目录，不要把无后缀的 `$MODEL` 直接当作评估路径。

实验根目录还必须创建 `README.md`，记录实际展开后的命令、环境、GPU、数据集、所有
非默认参数、checkpoint 和日志路径。

### 3.3 训练中监控点

至少在 500、1000、5000、10000、20000、30000、35000 检查：

- Gaussian 点数和相对初始化的增长倍率。
- canonical XYZ bbox 是否离开场景范围。
- Gaussian scale 的 50%、90%、99% 和最大值。
- opacity 分位数以及低 opacity 点比例。
- separation 中动态点比例是否接近 0 或接近 100%。
- test PSNR、SSIM、LPIPS，但不能只依据这些指标决定是否继续。

必须暂停并审查的信号：

- 单次 densification 后点数成倍增长。
- 点数快速接近 `max_gaussians`。
- bbox 从约 ±1.5 漂移到数十或数百。
- 最大 scale 达到数十以上。
- 一次 prune 后点数接近 0。
- separation 几乎全静态或整个场景全动态。

## 4. 安全恢复训练

恢复点必须同时存在：

```text
<MODEL>_mlp/point_cloud/iteration_<ITER>/point_cloud.ply
<MODEL>_mlp/deform/iteration_<ITER>/deform.pth
```

先检查：

```bash
MODEL=output/Baseline/<experiment>/only_clothV2_stage1_mlp
ITER=10000

ls -lh \
  "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" \
  "$MODEL/deform/iteration_$ITER/deform.pth"
```

恢复命令的 source、model、resolution、mask、deformation 和损失参数必须与原训练一致。
下面是已经完成的 4009 点历史分支复盘命令，其中 `6000/150000` 是当时实际参数，
**不是当前第二档的新训练推荐值**。当前新实验应使用第 3.2 节的 `5000/20000`。
本次从安全的 iteration 10000 恢复到 35000 的命令为：

```bash
CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n "$ENV" \
python -u -m scripts.train_stage1 \
  --source_path data/LH-data/transfer-dynamic/only_clothV2 \
  --model_path output/Baseline/0717-only_clothV2-ply62-stage1/only_clothV2_stage1 \
  --load_iteration 10000 \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 --iterations 35000 --warm_up 500 \
  --densify_from_iter 1500 --densify_until_iter 6000 \
  --densification_interval 200 --densify_grad_threshold 0.0004 \
  --opacity_reset_interval 3000 --min_opacity 0.01 \
  --prune_from_iter -1 --max_gaussians 150000 \
  --binarization_warm_up 1000 --lambda_separation 0.005 \
  --d_xyz_loss_weight 0.001 --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 \
  --test_iterations 20000 30000 35000 \
  --save_iterations 20000 30000 35000 \
  --quiet \
  2>&1 | tee output/Baseline/0717-only_clothV2-ply62-stage1/train_stage1.log
```

恢复后 iteration 20000、30000、35000 都保持 4009 点，说明固定点数并没有冻结
deformation；指标和动态渲染仍在继续变化。

## 5. 已确认不能采用的做法

### 5.1 长期 prune-only

本次第一次尝试在 iteration 6000–35000 每 1000 步执行 prune-only：

```text
iteration 10000: 4009 points
iteration 20000:  267 points
```

该分支在 iteration 20230 主动停止，失败 checkpoint、日志和统计均已保留。问题不是
“形变被冻结”，而是 canonical Gaussian 被重复的尺度阈值误删。后段应设置：

```bash
--prune_from_iter -1
```

### 5.2 延长增密来追求细节

旧 `only_cloth` 从 4096 增长到 112827 点，并伴随 bbox 和 scale 爆炸。动态细节差时，
不要首先延长 `densify_until_iter`；应先检查 FOV、alpha、相机/物体同时运动造成的歧义，
以及 deformation 是否真正承担动态部分。

### 5.3 用大点数掩盖错误相机

错误 FOV、错误 alpha 或 extent=0 时，提高 `max_gaussians` 只会让模型用更多错误点
拟合二维图像。点数上限只能阻止资源失控，不能修复数据。

### 5.4 只看最终 PLY

dynamic Stage 1 的形变存储在 `deform.pth`。最终 canonical PLY 有物体轮廓是必要条件，
但不是充分条件；必须使用 Stage 1 insights 渲染全部 timestep。

## 6. 标准 Stage 1 Eval

### 6.1 渲染前检查

```bash
ROOT=/home/han.li/reproduce/LumiMotion
OUT="$ROOT/output/Baseline/<experiment>"
SOURCE="$ROOT/data/LH-data/transfer-dynamic/only_clothV2"
MODEL="$OUT/only_clothV2_stage1_mlp"
ITER=35000

hostname
ls -lh \
  "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" \
  "$MODEL/deform/iteration_$ITER/deform.pth"
```

### 6.2 完整时序渲染命令

```bash
CUDA_VISIBLE_DEVICES="$GPU" \
conda run --no-capture-output -n "$ENV" \
python -m scripts.render_stage1_insights \
  --source_path "$SOURCE" \
  --model_path "$MODEL" \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --load_iter "$ITER" --depth_ratio 0.0 --quiet \
  2>&1 | tee "$OUT/render_stage1_insights.log"
```

该脚本固定第一个 test camera，遍历全部 timestep。它检查的是固定视角时序连续性，
不是逐帧移动相机下的 GT 对齐，也不等价于真正 novel-view 几何评估。

120 帧数据必须得到：

- 120 张 `full_t*_cam*.png`。
- 120 张 `normals_t*_cam*.png`。
- 120 张 `separation_small_t*_cam*.png`。
- 120 张 `separation_large_t*_cam*.png`。
- full、alpha、albedo、normal、small-gaussians 和两种 separation 共 7 个 MP4。
- 实验根目录中的 `render_stage1_insights.log`。
- 实验根目录中的 `eval_rgb_contact_sheet.png`、输入/渲染 alpha 接触表和 JSON 统计。

接触表至少包含开头、中间和结尾 timestep。不能只保存 MP4，因为 MP4 不便于代码审查、
文档引用和快速对比。

### 6.3 严格验收顺序

1. **RGB**：检查 FOV、比例、主体位置、运动连续性、拉伸、跳变和拖影。
2. **alpha**：检查黑色背景、轮廓跟随、漂浮点、条带、背景覆盖和物体消失。
3. **normals**：检查连续性、随机噪声、突然翻转以及是否脱离 RGB 轮廓。
4. **separation**：动态物体应稳定响应，不能几乎全静态或全场景动态。
5. **定量统计**：最后检查 PSNR、SSIM、LPIPS、点数、bbox、scale 和 opacity。

只要关键动态物体消失，alpha 有明显漂浮 Gaussian/条带，或时序中出现明显崩坏，最终
结论必须是 `FAILED`，即使 PSNR/SSIM 很高也不能标为 `PASS with warning`。

## 7. only_clothV2 本次结果复盘

稳定训练分支完成到 iteration 35000：

```text
初始点数                    4096
最终点数                    4009
最终 bbox min               [-1.4427, -1.4140, -0.3772]
最终 bbox max               [ 1.4486,  1.4511,  1.9813]
最大 Gaussian scale        0.6187
scale > 1                  0
test PSNR                  38.99609
test SSIM                  0.989032
test LPIPS                 0.033271
```

训练过程采样指标在 iteration 30000 最好：PSNR 39.51547、SSIM 0.99025、LPIPS
0.03179。后续 Stage 2 不应只凭 iteration 编号选择 35000，应同时比较 30000 的完整
Stage 1 insights。

本次证明：

- PLY size 和场景坐标正确。
- FOV、alpha 和点云数量失控已解决。
- 圆台、椅子和布料具备基础形状。
- dynamic/static separation 对布料有响应，没有退化为完全静态。

仍存在的问题：

- timestep 119 的布料边缘有明显拖影和少量飞散点。
- alpha 和 normals 在该区域也出现对应的不稳定轮廓。
- 相机虽然有约 4.03 世界单位的总路径和约 22.86° 视向覆盖，但每个时间点仍只有
  一个视角；相机与布料同时变化，不能形成同一形态的真正多视角约束。

因此应区分两个结论：

- **训练稳定性：成功。** 点数、bbox 和 scale 均未失控，35000 checkpoint 完整。
- **当前严格 Stage 1 可视化验收：FAILED。** 末段关键动态物体仍有明显拖影，不能用
  PSNR 38.996 和 SSIM 0.989 覆盖该问题。

结果目录：

```text
output/Baseline/0717-only_clothV2-ply62-stage1/
```

重点文件：

```text
README.md
train_stage1.log
render_stage1_insights.log
eval/eval_rgb_contact_sheet.png
eval/alpha_input_contact_sheet.png
eval/alpha_render_contact_sheet.png
eval/normals_contact_sheet.png
eval/separation_small_contact_sheet.png
eval/separation_large_contact_sheet.png
eval/point_cloud_stats.log
only_clothV2_stage1_mlp/renders_stage1_insights/ours_35000/
```

## 8. 下一轮实验建议

下一轮一次只改变一个变量，并保留本次 checkpoint 作为对照：

1. 先对 iteration 30000 运行同样的完整 insights，确认末段拖影是否比 35000 少。
2. 若 separation 动态比例偏低，仅把 `lambda_separation` 从 0.005 降到 0.001，
   其他参数保持不变。
3. 如果能够重新采集，优先在若干关键物体形态上暂停或减慢布料，并让相机围绕该近似
   静态形态采集多个视角；“相机移动”和“物体移动同时放慢”仍不能提供严格多视角对应。
4. 对布料运动幅度大的数据，额外保存 canonical frame 到首尾帧的 deformation 位移
   分位数，用于区分网络欠拟合和空间漂移。
5. 未通过 Stage 1 严格可视化验收前，不进入正式 Stage 2；否则 Stage 2 会继承拖影和
   错误动态几何。

## 9. 最终检查清单

训练前：

- [ ] 服务器、Conda 环境和 GPU 已核对。
- [ ] 训练目标目录新建且包含 README。
- [ ] transforms、images 和 alpha 均为 120 帧且划分正确。
- [ ] `fl_x/fl_y`、FOV、宽高比正确。
- [ ] `points3d.ply` 命名正确。
- [ ] PLY bbox、相机中心跨度和 scene extent 合理。
- [ ] 首/中/末帧投影 overlay 已保存并检查。

训练中：

- [ ] 点数没有单次暴增或突然归零。
- [ ] 第二档在 iteration 5000 的点数约为 15k；若明显越过 20k，暂停审查而不是继续训练。
- [ ] bbox 和最大 scale 没有失控。
- [ ] 5000、10000、20000、30000、35000 checkpoint 按计划保存。
- [ ] deformation 持续更新，不能把停止 densification 误判成冻结形变。

训练后：

- [ ] point cloud 和 deform checkpoint 同时存在。
- [ ] Stage 1 insights 的四类 PNG 均与 timestep 数一致。
- [ ] 7 个视频和渲染日志存在。
- [ ] RGB 与 alpha 的首/中/末接触表已保存。
- [ ] RGB、alpha、normals、separation 已逐项目检。
- [ ] README 记录定量指标、目检结论和最终 `PASS`/`FAILED`。
