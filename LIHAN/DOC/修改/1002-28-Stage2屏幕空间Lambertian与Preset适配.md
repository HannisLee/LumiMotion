# Stage2 屏幕空间 Lambertian 与 Preset 适配

日期：2026-10-02；分支：`1001-stage2-perlight`；代码基准：`9cd834f` 加必要相机/运行兼容修复。

## 行为与接口

Stage1 保留原始 SH。Stage2 默认 `original_ir`，新增 `photometric_lambertian`（别名 `photo_lambertian`）。新模式固定全部 Gaussian、法线、opacity 和形变，独立 sigmoid albedo 与训练时刻方向表是唯一可学习参数。每步 Adam 更新，albedo LR=0.001、light LR=0.0001。原始 IR 模式保留原来的材质、opacity、EnvLight 优化及每4步更新行为。

光照模式为 `learned_directional` 和 `gt_directional`。方向约定统一为世界空间表面指向光源；learned 从训练相机到场景中心的方向初始化，仅训练帧建参数，测试帧使用时间插值。GT 从原始帧号读取灯位或方向，全部冻结。onlyclothV4 的灯为 AREA，当前 GT 是中心方向近似，不使用 1/r²，不把 Blender 1800 直接当 irradiance。

屏幕空间先栅格化线性 albedo 与固定几何，零背景、alpha归一化，再计算 `rho/pi × E × max(0,n·l)`；完成线性背景合成后只做一次 sRGB 转换。不使用粗糙度、镜面、间接光或阴影追踪。HDR 以世界 Z 向上、经度 atan2(-y,x)，2048个确定性球面样本积分；曝光/yaw/源文件哈希全部记录。轴向与原始脚本的回归及早期输出保留见 [1002-29](1002-29-Stage2-HDR轴向回归与来源写入保护.md)。

`loss_stage2.py` 提供 `auto/irgs_baseline/lambertian_default/lambertian_gt`。原始 IR 损失和梯度与初始提交逐项回归；Lambertian 为0.8前景L1+0.2DSSIM，albedo初始先验0.001，learned方向平滑0.001，GT无光照梯度。显式CLI权重优先，模式组合不匹配时报错。

首次训练用 `--stage1_model_path`、正的 `--load_iter`，输出单独 `--model_path`。恢复用 `--resume_iteration`，与首次初始化互斥。完整 checkpoint 包含材质、方向/时间表、优化器、数据配置、固定光强、来源文件哈希、冻结指纹、Python/NumPy/Torch/CUDA随机状态和未采样相机队列；恢复依赖原始数据及帧号不变。已有结果拒绝覆盖。

## 渲染与验证

`python -m scripts.render_stage2` 统一评估、四类可视化、固定相机转灯和HDR。原有 `render_materials` 和 `render_relight_with_hdr` 自动识别新模型，使用保存配置；HDR的 `--hdr` 保留为别名。新评估使用各帧自己的相机/形变/光照，输出整图及前景指标；前景SSIM/LPIPS明确为遮罩后包围框指标，不冒充逐像素前景均值。不与旧固定形变NVS直接比较。

CPU单测17项通过，覆盖原始IR损失及梯度、方向插值、GT冻结、CLI优先级、Lambertian平面响应、色彩空间、HDR白炉与轴向。两种模式各200步冒烟及完整渲染已完成。GPU检查覆盖真实来源哈希、冻结参数、材质/光照梯度、线性背景合成和完整续训。

首次GPU检查直接脚本调用遇到导入路径问题，改用模块调用；第一次严格浮点逐位比较出现最大4.77e-7的albedo logit差异，原因是原始CUDA rasterizer原子累加。完整随机状态/相机队列仍要求一致，浮点参数与优化器改用rtol=1e-5、atol=1e-6做等价验收。失败日志独立保留，未覆盖。

具体命令、完整指标和目检结论见训练记录及实验README。此文件中的命令面向本分支；全分支同步资料不表示其他分支实现了本功能。
