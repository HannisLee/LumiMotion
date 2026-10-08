# 0930-01 jumpingjacks 论文默认 Stage 1 训练与评测

日期：2026-09-30  
数据集：`jumpingjacks150_v5_spec32`  
当前状态：用户要求停止全部训练；Stage 1 及 Stage 1 评测均未完成，不执行 Stage 2。

## 数据与运行环境

- source 绝对路径：`/home/lihan/datasets/LumiMotion/d-nerf-relight-spec32/jumpingjacks150_v5_spec32`
- 训练光照：`chapel_day_4k_32x16_rot0`
- 测试光照：`golden_bay_4k_32x16_rot330`
- 输出：`output/0930-01-jumpingjacks150_v5_spec32-official_pipeline/`
- 服务器：`venus`（当前服务器名称不在仓库环境映射表内）；选用已安装且验证可运行的官方 `lumimotion` 环境（Python 3.8.18、PyTorch 2.1.0+cu121）。
- GPU：NVIDIA GeForce RTX 3090 Ti，24 GB。

## 论文默认参数

| 参数 | 设置 |
| --- | ---: |
| iterations | 35,000 |
| resolution | 2 |
| `render_mode` | `original_sh` |
| `densify_until_iter` | 20,000 |
| `lambda_separation` | 0.001 |
| `d_xyz_loss_weight` | 0.001 |
| `binarization_warm_up` | 1,000 |
| Stage 1 `depth_ratio` | 1.0 |
| `d_color_reg_loss_weight` | 0.01 |

原仓库的 Stage 1 parser 默认覆盖成了 `photometric_lambertian`。为匹配论文官方合成脚本，本次明确传入 `--render_mode original_sh`。训练 `--eval` 收集 train/test PSNR、SSIM、LPIPS 等指标，完成后执行 `scripts.render_stage1_insights` 全时序渲染评测。没有加入另外两组 train/test light 组合。

## 执行过程

第一次启动遇到 Python 3.8 类型注解兼容问题，修复记录见 [0930-01 Python 3.8 延迟类型注解兼容](../修改/0930-01-Python3.8延迟类型注解兼容.md)。第二次启动在 iter 8278 被中止，checkpoint（最高保存 iter 5000）与日志保留于实验目录 `attempts/attempt1_aborted_before_10000/`。第三次从 iter 1 显式使用 `original_sh`，约到 iter 1780 时按用户要求停止，最后保存 checkpoint 为 iter 1000。没有达到目标 iter 35000，训练后的 Stage 1 渲染评测没有执行。所有失败/中止产物均保留。

完整命令、模型路径、渲染评测命令、日志、指标和最终目检记录见实验 README：

`output/0930-01-jumpingjacks150_v5_spec32-official_pipeline/README.md`

最终状态：`FAILED（用户要求停止；Stage 1 与评测未完成）`。训练过程 iter 1000 的临时 train/test 指标和 RGB、alpha、albedo/separation、normal 的目检均不能替代目标迭代的正式评测。
