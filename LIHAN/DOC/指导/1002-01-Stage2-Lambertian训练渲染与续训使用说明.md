# Stage2 Lambertian 训练、渲染与续训

更新：2026-10-02。实现分支：`1001-stage2-perlight`；从原始 `9cd834f` 开始，保留 Stage1 SH 算法。venus 实现工作树：`/home/lihan/reproduce/LumiMotion-stage2-perlight`。

共享文档包含其他分支的历史功能。本指南命令只适用于本分支；Stage1 历史 Lambertian 开关与本次 Stage2 参数不可混用。

## 最快复现正式对照

先检查 `hostname` 并选择对应 Conda 环境。venus 使用 `lumimotion`。默认数据为 onlyclothV4，Stage1 来源为只读 iter35000 的 SH 结果。启动器完整记录命令、版本、配置与日志，输出位置必须是 `output/` 下未使用的实验目录。

```bash
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
hostname
bash bash_scripts/stage2_onlyclothv4.sh original output/1002-02-onlyclothV4-stage2_original
bash bash_scripts/stage2_onlyclothv4.sh GT output/1002-03-onlyclothV4-stage2_GT
bash bash_scripts/stage2_onlyclothv4.sh learned output/1002-04-onlyclothV4-stage2_learned
```

以上目录已用于本次实验，重跑时必须使用新序号。三行串行运行；并行时每张卡最多三个训练且先确认显存。`LM_STAGE1_MODEL`、`LM_STAGE2_DATA`、`LM_STAGE2_LIGHTS` 可更换来源，变更数据后不能照搬本次固定光强，须重新标定并记录。

## 手工训练

示例为新的 200 步 GT 冒烟。实验目录和当日序号应按实际运行调整；实际模型路径自动附加 `_mlp`。

```bash
conda run --no-capture-output -n lumimotion python -m scripts.train_stage2 \
  --source_path /home/lihan/data/LH-data/transfer-static/only_clothV4 \
  --stage1_model_path /home/lihan/reproduce/LumiMotion-perlight/output/1001-01-onlyclothV4-FOVxy修复重训/model_mlp \
  --model_path output/smoke_test/1002-06-onlyclothV4-stage2_GT/model \
  --load_iter 35000 --iterations 35200 --save_iterations 35100 35200 \
  --test_iterations 35200 --is_blender --eval --resolution 2 --load2gpu_on_the_fly \
  --train_light_folder images --test_light_folder images --depth_ratio 0 \
  --render_mode photo_lambertian --loss_preset lambertian_gt \
  --photometric_light_mode gt_directional \
  --photometric_gt_lights_path /home/lihan/data/LH-data/static/only_clothV4/lights.json \
  --photometric_light_intensity 6.032453522706779
```

learned 模式同时更换 `--loss_preset lambertian_default` 与 `--photometric_light_mode learned_directional`。learned 只为训练时刻创建方向参数，测试时刻插值；GT 为外部已知灯位的中心方向近似，参数固定。两者只有 albedo 和 learned 方向可以学习，几何、法线、opacity、形变固定。

`--photometric_light_intensity -1` 用训练帧、初始 SH 材质和 GT 灯位自动标定一个统一标量，写出 `calibration.json`。此值不表示恢复了 Blender 物理强度。其他数据须分别标定。

默认 Stage2 模式仍为 `original_ir`。`auto` 不覆盖已有损失权重；命名 Lambertian preset 默认是 0.8 L1 + 0.2 DSSIM，初始材质先验 0.001、learned 方向平滑 0.001。显式 CLI 权重优先。新模式强制真实 RGBA mask；不训练 alpha，因而不使用 Stage1 的 alpha loss 开关。

## 完整渲染

```bash
conda run --no-capture-output -n lumimotion python -m scripts.render_stage2 \
  --model_path output/1002-03-onlyclothV4-stage2_GT/model_mlp \
  --load_iter 55000 --task all \
  --hdr_filepath example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

默认写到 `model_mlp/renders_stage2/ours_55000/`，包括15帧自身相机/时间评估，120帧 RGB/alpha/normal/albedo 和四段视频、72帧固定相机转灯，以及120帧 HDR。可单独选择 `eval/insights/materials/directional/hdr`；重渲染须指定新的 `--output_path`，避免覆盖已有结果。

HDR 默认2048确定性球面样本，世界 Z 向上、经度 `atan2(-y,x)`，与原始脚本一致；支持 `--hdr_yaw` / `--hdr_exposure`。先积分线性 irradiance，再按 `rho/pi` 漫反射和线性背景合成，最后转换 sRGB。镜面、阴影和间接光不在本管线内。

旧 `scripts.render_materials` 和 `scripts.render_relight_with_hdr` 自动识别新模型；HDR 保留 `--hdr` 别名，`--diffuse_sample_num` 可作为 `--hdr_samples` 别名。恢复到新目录且只有resume配置的模型也能识别。实际参数从 checkpoint 恢复。新增统一入口记录渲染代码版本，旧 IR HDR 入口仍保持原始渲染流程。

## 续训与迁移

`--resume_iteration` 与 `--stage1_model_path` 互斥。续训使用原模型 basename（训练入口仍自动附加 `_mlp`），目标 iteration 大于已完成 iteration。例如将 GT 的55000续到56000：

```bash
conda run --no-capture-output -n lumimotion python -m scripts.train_stage2 \
  --model_path output/1002-03-onlyclothV4-stage2_GT/model \
  --render_mode photometric_lambertian --resume_iteration 55000 \
  --iterations 56000 --save_iterations 56000 --test_iterations 56000
```

原来的55000文件保留，新增56000文件。状态保存材质、方向/时间表、Adam、固定来源/光强、全部随机状态和相机队列。来源 SH 文件哈希与冻结指纹必须匹配。迁移到其他服务器时必须保留完整数据及来源模型，并映射保存配置中的绝对路径；不要替换数据或仅复制材质 checkpoint 后宣称能够完整恢复。CUDA 原子累加导致跨进程极小浮点差异，续训以有记录的数值容差验收。

## 结果验收

训练结束只表示模型已保存。必须完成 RGB、alpha、normal、albedo 四类目检、评估和重光照，将完整命令、source/model/iteration、日志、指标、代表图/视频和最终 `PASS/FAILED` 写入实验 README。失败产物保留并用新目录重试。整图 PSNR 容易受大面积黑背景影响，应同时看前景 PSNR；前景 SSIM/LPIPS 为遮罩后包围框协议。本次验证结果见 [三管线训练记录](../训练/1002-02-onlyclothV4-Stage2三管线对照.md)。
