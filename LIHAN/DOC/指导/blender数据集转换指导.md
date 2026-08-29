# LumiMotion Blender 数据转换指导

数据转换只需对每个原始数据集执行一次。目标是把 Blender 导出的数据转换为 LumiMotion 可直接读取的格式，并确认相机、alpha 和初始化 PLY 正确对齐。

## 1. 执行数据转换

目标目录应为空，避免覆盖已有 transforms、mask 或 PLY。

```bash
cd /home/han.li/reproduce/LumiMotion

conda run --no-capture-output -n <server-conda-env> \
python scripts/prepare_lh_dynamic.py \
  data/LH-data/danamic/only_clothV2 \
  data/LH-data/transfer-dynamic/only_clothV2 \
  --test-stride 8 \
  --camera-extent 1.0
```

转换脚本应完成：

- 校验 image、albedo、normal、camera、light、object pose 帧号一致。
- 从 albedo PNG 保留真实 **soft alpha**。
- 每帧写入独立的 `fl_x/fl_y`、`camera_angle_x/y` 和 camera-to-world。
- 生成：
  - `transforms_train.json`
  - `transforms_test.json`
  - `dataset_manifest.json`

120 帧、`--test-stride 8` 时应得到：

```text
train: 105
test:   15
```

## 2. 准备 canonical PLY

LumiMotion Blender reader 默认读取：

```text
points3d.ply
```

如果文件名是：

```text
points3D.ply
```

可创建硬链接：

```bash
cd data/LH-data/transfer-dynamic/only_clothV2
ln points3D.ply points3d.ply
```

canonical PLY 建议选择**中间帧**，要求：

- 主体完整、遮挡较少；
- 无明显运动模糊；
- 包含完整场景和动态物体；
- 不要只导出布料。

`only_clothV2` 当前使用 frame 62 的 4096 点 PLY。

## 3. 转换后检查

训练前至少确认：

1. train/test frame 数正确且无重复。
2. `fl_x/fl_y`、FOV、图像宽高一致。
3. 1280×720 等非方形图像不能按方形相机处理。
4. alpha 背景为 0，前景使用真实 soft alpha，不能整张全 255。
5. 相机范围、scene extent 和 PLY bbox 数值合理。
6. 将 PLY 投影到**首帧、中间帧、末帧**进行 overlay 检查。

投影结果应覆盖 GT 主体，不能出现：

- 整体位置偏移；
- 上下翻转；
- 横纵比例拉伸；
- 明显 scale 不一致。

只有这些检查通过后，才进入训练阶段。