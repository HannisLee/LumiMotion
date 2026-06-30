# AGENTS.md

本文件是 AI coding agents 在本仓库工作的入口说明。LumiMotion 的训练流程、数据结构、代码地图和参数陷阱请见 `lumimotion.md`。
回答都请使用中文


## 工作定位

该仓库首先作为个人 baseline 仓库使用。除非用户明确要求，所有改动都应保持小范围、可解释、可回退，并避免影响论文复现路径。

优先级如下：

- 保持 baseline 可追踪，不做无关重构。
- 保留用户已有实验结果、数据、checkpoint、日志和未提交文件。
- 修改训练、渲染、评估或数据读取逻辑时，明确说明行为影响。
- 对 CUDA/native 扩展、依赖版本和 checkpoint 兼容性保持谨慎。

## 文档入口

- `readme.md`：原项目安装、数据和论文复现说明。
- `lumimotion.md`：本仓库的中文项目工作指南，包含两阶段训练、评估流程、数据布局和代码地图。
- `CLAUDE.md`：Claude Code 相关说明。如与本文件冲突，优先遵循用户当前指令和 `AGENTS.md`。

## Git 约定
工作前先查看：

```bash
git status --short --branch
git remote -v
```

注意：

- 不要擅自执行 `git reset --hard`、`git checkout --` 或删除未跟踪文件。
- 不要回滚与当前任务无关的本地改动。
- 如果工作树中存在混合改动，只提交与当前任务直接相关的文件。
- baseline 文档或代码改动应使用清晰、原子化的提交。

### 分支与代码归属

当前 Stage1 光照实验分支采用“模型实验分支 + 通用工具分支”的结构：

- `PS-stage1-V2`：Stage1 V2 模型实验分支，只放 V2 相关的模型、训练、渲染或参数逻辑差异。
- `PS-stage1-V3`：Stage1 V3 模型实验分支，只放 V3 相关的模型、训练、渲染或参数逻辑差异。
- `PS-tools-common`：通用工具分支，放所有可被 V2/V3 共同使用的工具、诊断、导出、画图和分析脚本。

路径归属原则：

- 模型实验代码优先归入模型分支，例如 `scene/`、`scripts/train_stage*.py`、`arguments/` 中会改变训练、渲染、优化或 checkpoint 行为的改动。
- 通用工具代码优先归入 `PS-tools-common`，例如 `LH_Utils/` 下的 light direction 导出、画图、loss 诊断、CSV/JSON 分析脚本等。
- 通用文档和工作流说明可归入 `PS-tools-common`，例如 `AGENTS.md`、`CLAUDE.md`、`lumimotion.md` 中不绑定某个模型版本的说明。
- 某个模型版本专用的说明、命令或实验记录，可以留在对应 `PS-stage1-*` 分支，但提交信息必须说明版本范围。

工作方式：

```bash
# 修改通用工具时，先切到通用工具分支
git switch PS-tools-common
# edit LH_Utils/... or shared docs
git add LH_Utils AGENTS.md CLAUDE.md lumimotion.md
git commit -m "tools: update shared light diagnostics"

# 同步通用工具到模型分支
git switch PS-stage1-V2
git merge PS-tools-common

git switch PS-stage1-V3
git merge PS-tools-common
```

如果已经在 `PS-stage1-*` 分支上误改了通用工具，不要直接把工具改动混进模型提交。优先单独提交或转移到 `PS-tools-common`，再 merge 回模型分支：

```bash
# 只暂存通用工具，避免混入模型改动
git add LH_Utils AGENTS.md CLAUDE.md lumimotion.md
git commit -m "tools: update shared utility"

# 之后把该工具提交同步到 PS-tools-common，再从 PS-tools-common 同步到其他模型分支
```

比较 V2/V3 模型差异时，应排除通用工具路径，优先只看模型相关目录：

```bash
git diff PS-stage1-V2..PS-stage1-V3 -- scene scripts arguments
```

提交拆分建议：

- `tools:` 前缀用于 `LH_Utils/`、导出/画图/诊断脚本和通用工作流文档。
- `model:` 前缀用于会改变模型结构、训练行为、渲染行为、loss 或 checkpoint 语义的改动。
- `docs:` 前缀用于只改说明文档且不改变代码行为的改动。
- 同一次工作中同时有工具和模型改动时，必须拆成两个提交，避免模型分支之间的对比被通用工具噪声污染。

## 运行约定

本地实际环境为 `lumimotion-cu129`。运行训练、评估、渲染或 CUDA 扩展相关命令前，先激活环境：

```bash
conda activate lumimotion-cu129
```

### 新服务器环境：minakshi

当前新服务器 `minakshi` 已在 commit `52d085c` 上跑通，使用独立环境 `lumimotion-cu126`，不要覆盖旧服务器的 `lumimotion-cu129`：

```bash
conda activate lumimotion-cu126
```

环境要点：

- GPU：5 张 NVIDIA RTX 6000 Ada Generation，driver `560.28.03`。
- PyTorch：`2.1.0`，`pytorch-cuda=12.1`，`torch.version.cuda=12.1`。
- 编译工具：conda `cuda-toolkit=12.1`、`cmake`、`ninja`，系统 `/usr/bin/gcc-11` / `/usr/bin/g++-11`。
- NumPy 固定为 `1.24.4`；`setuptools` 固定为 `<70`，否则 PyTorch 2.1 的 C++ extension 可能找不到 `pkg_resources`。
- native 扩展已在该环境中编译安装：`diff_surfel_rasterization`、`simple_knn`、`surfel_tracer`、`nvdiffrast`。

在 `minakshi` 上重编 CUDA/native 扩展时，conda 激活脚本可能默认设置 `NVCC_PREPEND_FLAGS` 到 base conda 的 GCC 14，CUDA 12.1 不兼容。编译前必须覆盖为系统 g++ 11：

```bash
export NVCC_PREPEND_FLAGS=" -ccbin=/usr/bin/g++-11"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.9"
```

从仓库根目录执行脚本，并优先使用模块方式：

```bash
python -m scripts.train_stage1 ...
python -m scripts.train_stage2 ...
```

除非已确认导入路径不会出问题，不要优先使用 `python scripts/foo.py`。

### 输出目录约定

- 所有训练、评估、渲染、日志、checkpoint 和 smoke test 输出必须放在仓库根目录的 `output/` 下。
- `--model_path` 应使用 `output/<experiment>/...`，脚本中的输出根目录也必须以 `output/` 开头。
- 不要在仓库根目录新建 `outputs_*`、`results_*` 或其他平行实验输出目录。
- 数据转换产物仍放在对应数据集目录；只有模型和实验运行产物遵循本约定。

### Stage1 光源方向导出

`photometric_lambertian` 的光源 checkpoint 在 `output/<exp>/photometric/iteration_<iter>/photometric.pth`。V1 使用 `raw_light_dir`，V2 可能使用 `light_model._raw_light_dir_table` 或 B-spline `light_model._light_ctrl`；渲染实际使用单位化方向。导出六列 CSV：

```bash
python -m LH_Utils.export_light_directions --model_path output/<exp> --iteration 35000
```

再用 CSV 解耦画图；如有原始 `lights.json`，传入后会把 `light_pos_world` 相对默认目标点 `[0,0,0]` 单位化并作为 GT 对比：

```bash
python -m LH_Utils.plot_light_polar --csv output/<exp>/light_directions.csv --lights_json data/LH-data/static/<scene>/lights.json
python -m LH_Utils.plot_light_timeseries --csv output/<exp>/light_directions.csv --lights_json data/LH-data/static/<scene>/lights.json
```

## 修改原则

- 先读现有调用链，再改代码。
- 优先沿用现有参数名、目录结构和输出路径约定。
- 训练命令和复现实验优先参考 `bash_scripts/`，不要凭空拼复杂参数。
- 改动数据读取逻辑时，同时考虑 Blender、ENeRF 和 DNA 三类数据。
- 改动 CUDA/native 相关文件时，说明是否需要重新编译或重新安装扩展。
- 新增文档应保持中文、简洁，并和 `lumimotion.md` 分工清楚。

## 验证建议

仓库没有统一测试套件。根据改动范围选择最小有效验证：

- 文档改动：检查链接、文件名和命令是否准确。
- CLI 参数改动：运行 `python -m scripts.<entry> --help`。
- Python 工具函数改动：运行最小导入或小样例。
- 数据读取改动：用小场景或单个相机样本检查加载。
- 训练/渲染改动：尽量跑短迭代 smoke test。
- CUDA 扩展改动：重新 build/install 后做 import 检查。

常见环境检查：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
```

## 文档维护

如果发现本文件与实际工作方式不一致，优先更新本文件；如果是 LumiMotion 项目细节变化，更新 `lumimotion.md`。会影响复现结果的变更，也应同步检查 `readme.md` 和相关 bash 脚本注释。
