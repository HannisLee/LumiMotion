# only_clothV4 原始实验归档与 perlight 原始管线复现

日期：2026-10-02；服务器：venus；环境：lumimotion（Python 3.8.18、PyTorch 2.1.0+cu121）；GPU 0：RTX 3090 Ti 24 GB。

## 原始实验归档

来源：`/home/lihan/reproduce/LumiMotion/output/1001-01-onlyclothV4-FOVxy修复重训`，原始项目 `main` 分支。该实验是在原始 LumiMotion 目录训练和评估，不是在当前 perlight 项目中训练。

完整复制至本仓库 `output/1001-01-onlyclothV4-FOVxy修复重训/`，970 个普通文件、7 个数据软链接；普通文件总字节数 176,515,943。所有文件在复制后逐一校验 SHA256，软链接校验目标，结果全部一致。`archive_manifest.json` 保存源文件哈希；仅副本 README 随后追加中文来源说明。原运行脚本、配置和日志的绝对路径保留原值，不能直接重跑归档脚本。

同时归档原项目的 1001-01 训练记录、FOVxy 修复记录，以及其引用的 0930-01 原始管线训练记录；均补充来源说明。0930 输出不在本次复制范围内。原始文件未修改。

原始 1001-01 已完成 Stage 1 35,000 iter 与 Stage 2 累计 55,000 iter；NVS PSNR 33.3726、SSIM 0.982744、LPIPS 0.019807，来源为归档 `model_mlp/results_nvs_static.json`。四类目检与 PASS 范围沿用原记录。

## 本次正式训练计划

状态：冒烟验收 PASS，正式两阶段训练于 2026-10-02 15:49:25（Asia/Tokyo）启动；tmux `lm_perlight_clothv4_1002`。

- 执行项目：`/home/lihan/reproduce/LumiMotion-perlight`，`perlight` 分支，HEAD `e936e739134baf2a041079ca0b15d4bc4caa3da9`。
- source：`/home/lihan/data/LH-data/transfer-static/only_clothV4`；105 个训练视角、15 个测试视角，同一份 100,000 点初始点云。
- 正式输出：`output/1002-01-onlyclothV4-perlight原始管线复现/`。
- 冒烟输出：`output/smoke_test/1002-01-onlyclothV4-perlight原始管线冒烟/`。
- 运行 source：实验目录 `data_view/`；CLI model：`model`，实际 model：`model_mlp/`。
- Stage 1：35,000 iter、resolution=2、densify_until_iter=20,000、depth_ratio=1.0；分离 0.001、d_xyz 0.001、d_color 0.01、binarization_warm_up=1,000。
- 显式 `--render_mode original_sh --loss_preset sh_baseline`，复现原始 SH 路径；perlight Stage 1 parser 默认的 Lambertian 路径在本次不会启用。
- Stage 1 诊断：加载 35,000 iter，depth_ratio=0.0，120 个时间点。
- Stage 2：加载 35,000 iter，训练到累计 55,000 iter；diffuse_sample_num=512、depth_ratio=0.0。
- 材质、HDR 重光照、静态 NVS：加载 55,000 iter；HDR/NVS diffuse_sample_num=2,048，使用仓库 Golden Bay HDR。
- 与原实验相同，两个光照目录都链接同一份采集 RGB，静态 NVS 使用 timestep0001 固定形变，整图指标包含黑背景；缺少第二种 HDR 观测和 roughness 真值，材质与 HDR 只作定性检查。

完整训练、渲染与评估命令见实验 `run_pipeline.sh` 和 README；逐步展开命令写入 `commands/`，日志写入 `logs/`，通过 tmux 保持后台执行。

## 复现边界与现有分支差异

原项目与当前项目的共有 Model/Pipeline/Optimization 默认参数全部一致，已通过预检查。当前分支已有正确的 Blender C2W 平移转换和 FOV/focal 优先级读取；原项目只修复 FOV 轴向，平移仍使用 `T=-w2c[:3,3]`。本次保留 perlight 相机实现，实际 120 个相机中心及 FOV 已与 JSON 核对通过。进一步逐相机比较，当前数据 main 与 perlight 的相机中心最大差异为 6.74e-07（look-at 原点使 w2c 的 X 平移接近零）；这一现有代码差异在本数据上没有显著改变相机位置。

perlight 的既有 opacity reset 下限为 `max(0.01, 2*min_opacity)`，本配置为 0.02；原项目为 0.01。这是已有防止全点剪除的修复。本次为原始算法路径在 perlight 上的复现，不宣称与 main 原实验数值逐位相同。已有 Python 3.8 注解兼容改动保留。

## 预检查与冒烟

- 原始 SH 损失和 Blender 相机已有单测：25 项通过，日志 `logs/preflight_tests.log`。
- 全量 120 帧路径、显式 RGBA、alpha 前景和背景、1280×720 尺寸、105/15 不重叠划分、实际读取器 FOV/相机中心、PLY 有限值及共有默认参数通过；记录 `preflight.json`、`logs/preflight_data.log`。
- 首/中/末 RGB 和 alpha 接触表：`preflight_rgb_alpha.png`。
- 冒烟从头训练 Stage 1 1,100 iter（覆盖 deformation 启动），Stage 2 训练到累计 1,120 iter，并执行与正式实验相同的四类可视化、HDR 与 NVS 流程。短训练指标不作为正式验收。

正式与冒烟 README 均记录 source/model/iteration、完整渲染命令、输出目录、日志、量化指标、代表图像/视频、四类目检与最终 PASS/FAILED。失败结果独立保留，不覆盖或删除。

## 冒烟完成记录

2026-10-02 15:39:57 至 15:46:25 完成全部六步；Stage 1 测试 PSNR 36.872074、训练 PSNR 37.19（日志精度）；Stage 2 NVS PSNR 30.595617、SSIM 0.978376、LPIPS 0.020039。120 帧 RGB/alpha/normal/分离/albedo/roughness/HDR、15 组 NVS、10 个 120 帧视频和检查点核验通过，四类代表帧人工目检完成。结论 PASS 仅用于冒烟流程完整性，详见冒烟 README。
