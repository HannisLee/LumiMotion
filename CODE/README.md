# Per-Light 3D 网页查看器

用于在浏览器中查看三维场景、逐帧光线方向轨迹（perlight JSON / 训练 `.pth` 导出）以及 PLY 点云的轻量查看器。基于 three.js（已本地化到 `vendor/`，无需外网）。

## 快速开始

```bash
cd CODE
bash serve.sh          # 默认端口 8321，可传参：bash serve.sh 8400
```

浏览器打开 `http://localhost:8321/`（远程服务器请用 `http://<服务器IP>:8321/`）。
启动后自动加载：

| 默认数据 | 内容 | 来源 |
| ---- | ---- | ---- |
| 初始 GT 光线 | `lights.json`（120 帧 GT 点光源世界位置） | `data/LH-data/danamic/only_clothV3/lights.json` |
| 数据集点云 | 下拉框默认选中 `only_clothV3`，加载其 `points3d.ply` | `data/LH-data/transfer-dynamic/<数据集>/points3d.ply` |

**数据集下拉框**：侧边栏「数据集点云」下拉框列出 `data/LH-data/transfer-dynamic/` 下的全部数据集，切换即替换当前初始点云（手动导入的点云不受影响）。新增数据集后运行 `python tools/update_dataset_index.py` 刷新 `data/datasets.json` 索引。

`data/` 下另有两个可手动导入的样例：`perlight.json`（0811 实验 gt_point 逐帧光线）、`perlight_learned_pbr.json`（0727 PBR 实验学习光线）。

## 支持的数据格式

**光线 JSON（直接导入或拖拽）**

1. **初始 GT 光线 `lights.json`**：`{"0001": {"light_pos_world": [...], "light_rgb", "intensity", ...}}` 的 GT 点光源位置序列（如 `data/LH-data/danamic/only_clothV3/lights.json`），查看器以世界坐标显示光源，箭头指向参考中心（点云包围盒中心）。
2. **perlight / `light_dirs.json`**：训练时 `save_weights` 导出的格式，含 `frames[].direction`（`light_to_surface` 约定）、可选 `light_position_world`、`exposure_log_delta` 等。

**`.pth` 训练检查点**

浏览器无法解析 PyTorch pickle，请先用导出工具转成 JSON：

```bash
conda activate lumimotion-garuda

# 单个检查点（输出到同目录 photometric_perlight.json）
python tools/export_photometric_pth.py /path/to/photometric.pth

# 指定输出
python tools/export_photometric_pth.py /path/to/photometric.pth -o data/perlight.json

# 实验目录：默认只导出迭代号最大的检查点，--all 导出全部
python tools/export_photometric_pth.py /path/to/experiment_mlp --all
```

支持两类光线参数化：

- `light_model._raw_light_dir_table` / `raw_light_dir`（逐帧方向表）；
- `StructuredDirectionalLightModel` 傅里叶基 + 切向残差（PBR v1，按训练代码同款公式重建，与 `light_dirs.json` 误差约 1e-7）。

若检查点含 `gt_light_positions`，会一并导出 `light_position_world`，可在“世界坐标位置”模式下查看真实光源轨迹。

**PLY 点云**

- 支持 `ascii` / `binary_little_endian` / `binary_big_endian`，动态解析任意属性布局；
- 颜色来源优先级：`red/green/blue` → `albedo_dc_*`（SH DC 转 RGB）→ `f_dc_*` → 法线可视化 → 灰色；
- 兼容标准 3DGS 点云与本仓库自定义属性（`albedo_dc_*`、`fea_*` 等）。

## 功能说明

- **多数据集对比**：可加载任意多条光线轨迹（如学习 vs GT），各自配色，支持隐藏/删除，点击列表项查看元信息。
- **两种显示模式**：
  - *单位球方向*：方向归一化后画在参考球上（与 `scripts/visualize_stage1_light_trajectory.py` 的单位球视图一致），球半径可调；
  - *世界坐标位置*：有 `light_position_world` 时显示真实光源位置，箭头为光线传播方向（光源 → 参考中心）。
- **时间轴**：滑杆/播放/循环/倍速，逐帧显示方向、位置、曝光信息；亮色短线为最近 N 帧轨迹尾迹（长度可调）。
- **点云**：点大小可调，多个点云可同时加载。
- **坐标系**：数据为 Blender 世界坐标（Z-up），默认 Z 轴朝上，可切换 Y 轴朝上。
- **视角**：鼠标左键旋转、右键平移、滚轮缩放；“适配视角”按钮自动框住全部物体。

## 目录结构

```
CODE/
├── index.html                  # 页面入口
├── serve.sh                    # 启动本地 HTTP 服务
├── lhdata -> ../data/LH-data   # 符号链接：让 HTTP 服务能访问 LH 数据集
├── css/style.css
├── js/
│   ├── main.js                 # 场景、UI、时间轴、数据集下拉框
│   ├── lights.js               # 光线 JSON 解析与轨迹渲染
│   └── ply_loader.js           # PLY 解析器
├── vendor/three/               # 本地化的 three.js 0.160（three.module.js + OrbitControls）
├── tools/
│   ├── export_photometric_pth.py   # pth → JSON 导出工具
│   └── update_dataset_index.py     # 扫描 transfer-dynamic 生成下拉框索引
└── data/                       # 索引与示例数据
    ├── datasets.json           # 数据集下拉框索引（由工具生成）
    ├── perlight.json           # 样例：0811 gt_point 光线
    └── perlight_learned_pbr.json   # 样例：0727 学习光线
```

## 常见问题

- **打开页面空白/数据未加载**：必须通过 HTTP 访问（`file://` 下 `fetch` 会被浏览器拦截）。
- **下拉框为空或 GT 光线 404**：检查 `CODE/lhdata` 符号链接是否有效（应指向 `../data/LH-data`）；重建：`cd CODE && ln -sfn ../data/LH-data lhdata`。
- **拖入 `.pth` 提示无法解析**：属预期行为，请先运行导出工具。
- **方向箭头看起来“反向”**：注意数据约定为 `light_to_surface`（从光源指向表面）。球面模式箭头表示存储向量（中心 → 球面标记），世界模式箭头为实际光线传播方向。
