# V4 四版本原版 LumiMotion 训练核对

日期：2026-10-03（Asia/Tokyo）。状态：**四组冒烟及正式35k/55k训练、渲染、NVS、产物核验和代表帧目检全部完成，流程验收4/4 PASS。正式流水线16:52:23开始、22:27:14结束，共5小时34分51秒；实际最多两组并行。动态细节和材质质量限制已回填到结果及各组README，流程PASS不代表物理恢复质量达标。**

用户要求输出和文档统一留在当前项目，并把既有 V4 也重新训练，凑齐四个版本。本文件为本轮最终执行计划，取代先前仅有三组、输出放在原版目录的安排。

## 存储位置与执行代码

- 本轮存储项目：`/home/lihan/reproduce/LumiMotion-stage2-perlight`。
- 训练计划和数据总览：当前项目 `LIHAN/DOC/`；实验 README、脚本、预检查证据、日志、模型、图像、视频和指标：当前项目 `output/`。
- 正式根目录：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/`；冒烟根目录：同一项目的 `output/smoke_test/`。
- **执行代码仍使用原版** `/home/lihan/reproduce/LumiMotion` 的 `main`，准备时 HEAD `65287461075a1a4c0f54845325cc22039bd14e7d`，保留已验证的 FOVxy 修复和其他既有本地状态。全部命令通过绝对 `source_path/model_path` 写入当前项目。选择这一代码来源是为遵循用户“按照原版 LumiMotion”的要求；四组均不使用当前分支的 GT/learned-directional Lambertian Stage2。
- 没有修改原版或当前项目的算法代码；准备时保存 `prepared_code_*`，启动时重新保存 commit/status/diff。此前在原版目录准备但未启动的六个实验已搬入当前项目，旧路径不再用于启动。

## 四组数据与输出文档

原始四组均位于 `/home/lihan/data/LH-data/static/`。两个 Dynamic 数据集按导出信息和图像归为动态，原始上传目录保留，转换后放在 `transfer-dynamic`。

| 序号 | 数据集 | 几何与材质 | 转换 source |
| --- | --- | --- | --- |
| 01 | `only_clothV4_lambertian`（CV4L） | 静态、Diffuse BSDF Lambertian | `/home/lihan/data/LH-data/transfer-static/only_clothV4_lambertian` |
| 02 | `only_clothV4_Dynamic`（CV4D） | 动态布料、原始场景材质 | `/home/lihan/data/LH-data/transfer-dynamic/only_clothV4_Dynamic` |
| 03 | `only_clothV4_Dynamic_Lambertian`（CV4DL） | 动态布料、Diffuse BSDF Lambertian | `/home/lihan/data/LH-data/transfer-dynamic/only_clothV4_Dynamic_Lambertian` |
| 04 | `only_clothV4`（CV4） | 静态、原始场景材质；本轮从头重训 | `/home/lihan/data/LH-data/transfer-static/only_clothV4` |

保持已准备的前三组编号，新增旧 V4 为 04。各实验 README 是本轮命令和产物的核对文档：

| 序号 | 当前项目下的正式目录与 README | 冒烟目录名（在 output/smoke_test 下） |
| --- | --- | --- |
| 01 | [1003-01-only_clothV4_lambertian-LumiMotion原版两阶段/README.md](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-01-only_clothV4_lambertian-LumiMotion原版两阶段/README.md) | `1003-01-only_clothV4_lambertian-LumiMotion原版冒烟` |
| 02 | [1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/README.md](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/README.md) | `1003-02-only_clothV4_Dynamic-LumiMotion原版冒烟` |
| 03 | [1003-03-only_clothV4_Dynamic_Lambertian-LumiMotion原版两阶段/README.md](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-03-only_clothV4_Dynamic_Lambertian-LumiMotion原版两阶段/README.md) | `1003-03-only_clothV4_Dynamic_Lambertian-LumiMotion原版冒烟` |
| 04 | [1003-04-only_clothV4-LumiMotion原版两阶段/README.md](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-04-only_clothV4-LumiMotion原版两阶段/README.md) | `1003-04-only_clothV4-LumiMotion原版冒烟` |

每组已有 `README.md`、`run_pipeline.sh`、`commands/*.sh`、`planned_commands.json`、`data_view/`、`inputs/`、`logs/`、`inspection/`、`STATUS.json`。实际运行状态与模型以各组 `STATUS.json`、`STARTED`、`CURRENT_STEP`、`model_mlp/` 及汇总 `output/1003-05-V4四版本执行汇总/batch_status.json` 为准。CLI model 为本实验 `model`，实际输出为本实验 `model_mlp`。所有四组都从头训练；不加载旧 V4 训练 checkpoint，正式模型也不从冒烟模型续训。

## 统一数据与初始化

- 四组均为120帧、1280×720，105 train / 15 test，test-stride=8，test帧号8/16/…/120；time归一化到0～1，黑背景，使用真实albedo soft alpha。
- 四组原始 `baseline.ply` 完全一致，均为150,000点。为消除新旧初始化差异，本轮每个正式/冒烟实验都把对应baseline复制到自己的 `inputs/points3d.ply`，`data_view/points3d.ply` 指向这一副本。
- baseline SHA256：`01b2973c88ef1370f558e060f20dda3a701a8ee37a5ca4cff484b5182955930c`。
- **既有 CV4 转换目录内的100,000点 `points3d.ply` 保留，不覆盖；本轮运行视图显式使用上述15万点副本。** 四组的转换图像/划分保持不变，模型和日志独立。
- canonical点云是完整静态参考场景，不等同动态每帧GT几何。动态首帧布料悬空、中帧落下、末帧覆盖椅背，Stage1需要学习布料形变。
- Lambertian是数据材质标签。四组都用原版SH+IR算法，原始GT albedo RGB、EXR normal、lights/pose不作为训练监督。
- `data_view/chapel_day_4k_32x16_rot0` 只是链接到本组images的原版加载名称，不能将采集光误写成Chapel HDR。元数据光源为AREA，能量1800，不等同已标定的方向光强度。

## 原版两阶段参数与产物

| 项目 | 四组统一设置 |
| --- | --- |
| Stage1 | SH几何/形变训练35,000步；保存10k/20k/30k/35k；测试35k |
| Stage2 | 原版IR：albedo/roughness/opacity、保留MLP颜色head优化组及envmap，加载本组35k，训练到累计55k（新增20k）；保存40k/50k/55k；测试55k |
| 分辨率与背景 | resolution=2，640×360；黑背景；is_blender、eval、gt_alpha_mask_as_scene_mask、load2gpu_on_the_fly |
| Stage1参数 | densify_until_iter=20000，lambda_separation=0.001，d_xyz_loss_weight=0.001，binarization_warm_up=1000，d_color_reg_loss_weight=0.01，depth_ratio=1.0 |
| Stage2/离线渲染 | depth_ratio=0.0；Stage2 diffuse_sample_num=512；HDR/NVS=2048 |
| 冒烟 | 相同分辨率、点云与采样；Stage1到1100，Stage2到累计1120，再执行同样离线流程 |

六步顺序：Stage1 → Stage1 insights → Stage2 → materials → Golden Bay HDR → NVS。完整六步命令在每组 README 和 `commands/`，日志在每组 `logs/`。

| 正式输出 | 相对于各实验目录的位置 |
| --- | --- |
| RGB/Alpha/normal/静动态分离，120时刻图片与视频 | `model_mlp/renders_stage1_insights/ours_35000/` |
| albedo/roughness，各120帧和视频 | `model_mlp/trained_materials/ours_55000/` |
| Golden Bay HDR，120帧与视频 | `model_mlp/render_relight_with_hdr_golden_bay_4k_32x16_rot330/ours_55000/` |
| CV4/CV4L静态NVS，15视角 | `model_mlp/eval_nvs_static/ours_55000/`及`results_nvs_static.json` |
| CV4D/CV4DL动态NVS，15测试帧 | `model_mlp/eval_nvs_dynamic/ours_55000/`及`results_nvs_dynamic.json` |
| 每步日志 | `logs/{stage1,stage1_insights,stage2,materials,relight_hdr,eval_nvs}.log`；后台总日志`logs/pipeline.log` |

两个静态集沿用原版 `eval_nvs_static`，固定timestep0001形变；两个动态集用原版 `eval_nvs_dynamic`，使用测试帧实际时间。原动态脚本首视角构建BVH后复用，跨时间遮挡评估存在原实现限制。本轮保持原算法，结果中保留该说明。

原版Stage1训练报告最多计算4个测试视角和3个训练视角；本轮test实际为帧8/16/24/32，time约0.059～0.261，覆盖前段。不能把其高PSNR当作15帧全量或中后段动态质量证据。最终NVS使用完整15个测试视角/帧；HDR脚本则逐时刻更新BVH。

HDR没有独立目标光照GT，仅做定性重光照。NVS按GT Alpha掩码计算整图PSNR/SSIM/LPIPS，包含黑背景；单相机随时间运动的数据不等同同一时刻多视角GT。不同静态/动态协议的指标不能直接视为材质恢复优劣。

## 预检查和启动条件

1. 四组RGB/albedo/normal EXR各120帧，camera/light/pose帧号齐全，划分不重叠。逐帧RGBA尺寸、Alpha与原始albedo一致、真实软边、无rgba覆盖目录、RGB符合现有转换器一次Alpha合成已通过。EXR本轮核对非空及SHA256，未全量像素解码，不影响原版训练输入。
2. 四组原版实际loader均已读取105/15帧，核对运行视图图像路径、RGBA、FOV与相机中心；最大中心绝对误差9.5367431640625e-07，FOV误差0。迁移到当前项目后重新核验加载路径。
3. 首/中/末RGB/Alpha/点云投影已目检：椅子和平台的位置/尺度正常、Alpha对齐；两个静态集投影点全部在前景，动态约88.164%/100%/98.396%，差异与canonical布料形状不同一致。
4. 四组初始化点云同一SHA256、15万点、坐标有限。source、运行视图及初始化证据保存在各组 `preflight.json`、`loader_checks.json`、`input_sha256.json`、`inspection/preflight_rgb_alpha_projection.png`。
5. 八个实验（4正式+4冒烟）的shell语法、原版实际入口的CLI参数、source/model路径在启动前核查通过，见 `command_checks.json`；后续完整训练、渲染和评估也已完成，实际结果见各组README与四版本结果文档。

服务器venus；Conda lumimotion；GPU0 RTX3090Ti 24GB。用户授权最多同时两组，调度器并发上限2；启动下一组前检查至少9,000MiB空闲显存，不足则等待当前任务释放。未改变训练采样/分辨率/算法超参数。显存与阶段记录在汇总logs/gpu_memory.jsonl。原版参数与旧V4两阶段实验一致，初始化统一为本轮15万点。

**四组冒烟与正式流程验收均已完成，正式4/4 PASS。** 首批CV4L/CV4D从16:52:23启动，CV4DL于19:40:17、CV4于19:41:25自动补入，始终最多两个执行槽。所有目标模型、日志、诊断图、40个正式视频（完整解码4800帧）、60个NVS测试视角/帧及代表帧目检已核对。没有覆盖旧训练模型，失败保留机制未取消。

`run_pipeline.sh`拒绝覆盖已有STARTED/model_mlp；流程正常返回仅生成PIPELINE_COMPLETED，验收后才补齐README的定量指标、代表图片/视频、四类目检和最终PASS/FAILED。失败保留模型、日志、渲染与统计。

## 文档归属

本轮所有活动文档仅保留在当前项目，按用户指示不向其他项目复制。先前本聊天添加到原版main/perlight的计划、总览副本和README入口已撤回，其他项目原有文档和历史实验未改动；这次存储约定按用户指示优先于历史的全分支共享惯例。不做远端提交或推送。

## 执行与回填入口

- 调度与状态：[执行汇总](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-05-V4四版本执行汇总/README.md)。
- 指标与验收：[四版本训练结果](1003-02-V4四版本原版LumiMotion训练结果.md)，四组最终指标、验收状态、执行用时和质量限制已回填。
- 调度脚本：output/1003-05-V4四版本执行汇总/commands/scheduler.py，smoke/formal两阶段均已结束；四组正式automatic PASS和manual PASS已齐全。
- 冒烟调度、正式调度lm_v4_1003_formal_queue及显存记录lm_v4_1003_memory均已结束。启动/结束时间和六步日志见各实验目录，最终核验见汇总final_verification.json。
- 后台脚本已核对checkpoint、120时刻图像、每组10个120帧视频、15组NVS与有限指标；主代理已实际检查首/中/末四类、材质/HDR和NVS代表帧，再写最终流程PASS。当前四组状态均为COMPLETED。
