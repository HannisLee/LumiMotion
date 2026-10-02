# only_clothV4 FOVxy 修复后原始管线重训

> 归档来源：本记录对应的训练或代码修改发生在原始项目 `/home/lihan/reproduce/LumiMotion`（`main` 分支），不是在 `/home/lihan/reproduce/LumiMotion-perlight` 中执行。2026-10-02 将记录复制到本仓库；1001-01 的完整产物已复制至本仓库 `output/1001-01-onlyclothV4-FOVxy修复重训/`。原始绝对路径和命令保留用于追溯。0930 输出未在本次复制范围内。

2026-10-01 在 venus 上以 lumimotion 环境从头重训，修复记录见 `../修改/1001-01-Blender视角FOVxy轴互换修复.md`。训练参数沿用 0930 实验，运行目录为 `/home/lihan/reproduce/LumiMotion/output/1001-01-onlyclothV4-FOVxy修复重训`。

Stage 1 训练至 35,000 iter，生成诊断可视化；Stage 2 从 35,000 训练至 55,000 iter，随后生成材质、Golden Bay HDR 重光照及 15 个视角的静态 NVS 指标。完整命令在运行目录的 `run_pipeline.sh`，逐步日志在 `logs/`；模型输出 `model_mlp/`。使用原有的 105/15 训练测试划分和初始点云。

后台 tmux 会话名 `lm_clothv4_fovfix`。任务成功时生成 `COMPLETED`，失败时生成 `FAILED` 并保留日志和结果。

## 完成结果（2026-10-02 核查）

状态：**已完成，PASS（既定管线执行与定性可视化验收）**。实际运行时间为 2026-10-01 13:35:20 至 14:47:15（Asia/Tokyo），共 1 小时 11 分 55 秒。Stage 1 在 14:12:30 完成，Stage 2 在 14:40:04 完成，材质、HDR 重光照及 NVS 评估随后全部结束。存在 `COMPLETED`，无 `FAILED`；检查时无该训练进程。

| 指标 | 修复后结果 | 原错误 FOV 实验（仅供追溯） |
| --- | --- | --- |
| Stage 1 测试 PSNR | 51.02 dB | 39.23 dB |
| Stage 1 训练 PSNR | 52.89 dB | 42.20 dB |
| Stage 2 NVS PSNR | 33.3726 dB | 20.9591 dB |
| Stage 2 NVS SSIM | 0.982744 | 0.932674 |
| Stage 2 NVS LPIPS | 0.019807 | 0.065943 |

NVS 的 15 视角平均 PSNR 提升 12.4135 dB，LPIPS 下降约 70.0%。35,000 与 55,000 iter 检查点均已保留，阶段一 7 个诊断视频、材质 2 个视频、HDR 1 个视频，以及 15 组 NVS 图像全部生成；10 个视频各 120 帧，首/中/末帧均能解码。

四类代表可视化目检：完整 RGB 几何轮廓清晰，Stage 2 布料和底座仍有斑纹；alpha 稳定，二元分离几乎全绿，未形成明显静动态分界；法线主体连续；albedo 和 roughness 有不均匀纹理，HDR 暖色响应可见，无黑屏。PASS 仅涵盖既定管线完整性和定性结果可用性，不能据此认定材质真实性或动态分离质量达标。

Stage 1 逐帧重建与 Stage 2 固定形变 NVS 的评估条件不同，指标不能直接跨阶段比较。NVS 使用 GT alpha mask 后计算整图指标，包含大量黑背景，且使用 timestep0001 固定形变；材质与重光照缺少相应真值，仅作定性评价。

完整渲染命令、source/model/iteration、输出目录、日志、原始指标、代表图片/视频和验收结论见 [实验 README](../../../output/1001-01-onlyclothV4-FOVxy修复重训/README.md)。本次仅检查既有结果并更新记录，未重新训练或修改代码。
