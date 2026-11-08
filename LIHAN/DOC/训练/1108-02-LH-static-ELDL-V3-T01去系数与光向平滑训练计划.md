# T01 去系数与光向平滑无正则 Stage 2 训练计划

日期：2026-11-08（UTC）。执行服务器：`ushas`；Conda 环境：`lumimotion-ushas`；解释器：`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。本计划按冻结规划执行，不改变参数、门槛或输入。

## 目的与固定流程

在第五轮 E4 最优管线的观测、支撑与方向初始化上，执行一个 C01 无平滑/系数正则 Stage 2。目标固定为光度项加 `0.1*R_rev`；`L_coef、a1、a2、e1、e2` 的运行权重为零；KKT 分母 `1e-6`、E3 环绕方向和初始总绕行检查、40°–50°硬仰角带宽照规划保留。固定复用历史 E3 初值，不重新训练初始化，并在结果中披露历史系数项及 E2 三帧平均来源。禁止读取 GT，禁止重试和根据结果调参。

按入口固定顺序运行八流程：`full、A、B、boot103108、boot103109、hi、lo、repeat_full`。仅一个正式训练进程，使用 GPU 0；拟合期间不启动额外训练。输入/初值哈希、support/bootstrap 索引和数据配置必须由入口断言通过。

## 运行前核验记录

- 冻结规划：`LIHAN/DOC/规划/1108-02-T01去除系数与光向平滑正则预注册规划.md`，SHA256 `ad122b5d98d442391d866d3fe0e8b94f8e591d43eba5f2182c9369760d799854`。
- 实现入口：`LIHAN/scripts/static_rgb_light_fit_goal1108_02_t01.py`，SHA256 `81e7b7d1cf505583c9eda1ba281aec3b02a4eddbdcef6d71a5b66c591b8e8cb1`；r3 SHA256 `0359461d5c4436698d4eee4d417a564ca273edf0091bfa87153f1ef0eb834517`；r5b SHA256 `ea6a90084d61d88ee87dbcd5230793de65f56e8b1f562b2125a68d90dd6d12e2`。
- 固定观测 `output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz` SHA256 `eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47`。
- 第五轮 baseline `predictions.npz` SHA256 `e888f8758301ded62d075d78de719a8ba1da61761a440d6275a1c2f3fec52757`；`support.npz` SHA256 `35afe6f886c5df44b02ddb9faf5088330c01d3c79e9ef482359b0e8abc9703a7`。
- Stage 1 来源 `output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp`，iteration 35000：`point_cloud/iteration_35000/point_cloud.ply` SHA256 `adf94a9370f2251532d8885ef3f3e5a188ad2cdb6daa41539cb8d227d39222ff`；`deform/iteration_35000/deform.pth` SHA256 `5a2eb0cafbd9db551fcf99d19585efa510e200c40e9285a9a6b7d0923690fd6a`。
- E3、A、B、两组 bootstrap 初值及其 config SHA256 由规划及实现入口 `EXPECTED` 白名单固定；入口运行时逐项校验。五组方向实测形状均为 `(150,3)`、有限、最大单位范数误差不超过 `2.23e-16`；配置符合 stage2=false、geo_smooth=0、steps=250、delta=0.03、float64 及各自支撑数。
- `ushas` 的 GPU 0 为 NVIDIA A40，核验时显存占用 0 MiB；可用文件系统空间约 41 TiB。未发现同目标活动进程。核验未读取 GT 或 `lights.json`。

## 输出与命令

唯一正式输出目录：`output/1108-04-LH-static-ELDL-V3-T01去系数与光向平滑/`。选择依据：执行前该精确实验前缀无既有结果，日期 1108 的输出最大序号为 03。日志与 checkpoint、统计、图统一保留在该目录及其 `fitting/` 内；不覆盖既有目录。

完整训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python -u -m LIHAN.scripts.static_rgb_light_fit_goal1108_02_t01 --output output/1108-04-LH-static-ELDL-V3-T01去系数与光向平滑/fitting
```

通过 `nohup` 后台启动，stdout/stderr 均写入输出根目录 `run.log`，禁止 `tee`。启动 UTC、PID 和实际退出码写入 `process.json`、`exit_code.txt`；完整 shell 启动器保存在输出根目录 `launch.sh`。每五分钟核对 PID、日志增量、当前流程及最新 loss/证书，并向主会话报告实测进度。异常时保留已有 checkpoint、日志与统计并停止，不自动重试。

## 结果与交接

入口应生成八流程状态、参数/方向、优化 history 与证书、正则审计、候选/选择表、可辨识性与互偿统计、四类无 GT 图及完整输入哈希。拟合完成后逐图实际打开并记录目检结论；补齐 README 所需命令、路径、指标和最终状态。评测和冻结由主会话审核；本执行角色不读取 GT，不决定验收 PASS/FAILED。

## 实际启动记录与断点（2026-11-08 UTC）

按上文完整命令，使用 `nohup /bin/bash output/1108-04-LH-static-ELDL-V3-T01去系数与光向平滑/launch.sh </dev/null > output/1108-04-LH-static-ELDL-V3-T01去系数与光向平滑/run.log 2>&1 &` 启动。PID `1532765`，启动 UTC `2026-11-08T10:55:46Z`；服务器 `ushas`，环境 `lumimotion-ushas`，GPU 0。启动后短时观察 PID 存在；后续检查时 PID 消失。`run.log` 为 0 bytes，`exit_code.txt` 及临时文件缺失，`fitting/` 未创建。实际退出码、终止信号与根因均未知；现有证据不足以判断外部终止或其他原因，不推断为 OOM 或程序拟合错误。

正式流程进度 0/8，当前阶段、loss、证书未知；拟合预测、指标、history、checkpoint 与四类可视化均未生成。启动脚本、空日志、PID/启动时间元数据与本目录 README 保留。此断点按主会话要求停止，不自动重启。

## 管理者授权重启记录（2026-11-08 UTC）

主会话转达管理者明确授权：重新启动 T01，使用 `setsid` 完全独立会话（可配合 `nohup`）；必须选择全新输出目录，不覆盖或删除 `1108-04`；冻结规划的其他配置不变；启动后一分钟内确认实际拟合进程仍存在且 `run.log` 已写入，并将具体验活事实写入 GOAL；若再次瞬间死亡则保留现场、记录并上报，不再重试；存活后每五分钟报告实测进度，完成后按规划生成诊断供主会话审核冻结并交 Academic-analysis。

本次运行前复核 hostname=`ushas`，Conda=`lumimotion-ushas`；规划 SHA256=`ad122b5d98d442391d866d3fe0e8b94f8e591d43eba5f2182c9369760d799854`、入口 SHA256=`81e7b7d1cf505583c9eda1ba281aec3b02a4eddbdcef6d71a5b66c591b8e8cb1`、r3 SHA256=`0359461d5c4436698d4eee4d417a564ca273edf0091bfa87153f1ef0eb834517`、r5b SHA256=`ea6a90084d61d88ee87dbcd5230793de65f56e8b1f562b2125a68d90dd6d12e2`，均与冻结规格相同。观测、baseline predictions 和 support 哈希与冻结值一致。Stage1 model iteration 35000 文件在前次计划核验中已记载其哈希；入口会在运行开始时再次逐项验证全部输入白名单。

资源选择：复核时四张 A40 均已有外部计算进程。GPU 2 总显存 49,140 MiB，实测占用 28,332 MiB，剩余 20,808 MiB；一项外部任务正在该卡运行。根据项目单卡最多并行三个训练的规则，本次新增一个进程后同卡共两个计算任务，显存留量满足该小规模方向拟合入口运行需要。GPU 2 仅是设备分配变化，拟合方法、数据、初始化、优化器、预算、门槛和八流程顺序均不变。

新输出根：`output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑/`，选择因为日期内现有该实验前缀最大号为 04；旧目录全部保留。完整拟合命令：`CUDA_VISIBLE_DEVICES=2 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python -u -m LIHAN.scripts.static_rgb_light_fit_goal1108_02_t01 --output output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑/fitting`。后台会话命令为 `nohup setsid --fork /bin/bash output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑/launch.sh </dev/null > output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑/run.log 2>&1 &`。启动器记录 Python 子进程 PID 与真实退出码；启动 UTC 和 setsid wrapper PID 记入 `process.json`。本次是否验活、流程进展和结果仅根据此新目录的实测日志与进程状态追加。

### 重启验活实测（2026-11-08T13:00:42Z）

使用上述 `nohup setsid --fork` 启动方式，启动 UTC `2026-11-08T13:00:16Z`；launcher PID `1633940`，Python 拟合 PID `1633945`，SID `1633944`。启动后 26 秒实测 PID 仍存在、状态 `Rl`，`run.log` 为 7,254 bytes，故本次独立会话验活通过。日志显示 full 的 TA 已跑完 1000 步，当前 TB outer=20，最新 loss `0.0009148961713237829`；八流程完成 0/8，full 流程仍运行。无 GT 读取。进程/命令记录见 `output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑/process.json` 和 `README.md`。

### T01 重启实验完成记录（2026-11-08 UTC）

正式输出 `output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑/`。日志 `run.log`，启动 UTC `13:00:16Z`，launcher PID `1633940`，Python PID `1633945`，SID `1633944`；在 `13:04:51Z` 核验时 Python 已退出，`exit_code.txt=0`。固定八流程均生成终点及证书：full、A、B、两组 bootstrap、hi、lo、repeat_full；全部有效并最终收敛，A 按预注册规则触发 T-C 6000 步后通过。full 最终证书 C1=`8.310858605e-10`、C2=`0.0003011849°`、C3 位移=`0.08113022°`、下降=`9.60668e-7`，均通过；repeat 最大三维角差=`1.20742e-6°`。选择器选 C01/full，`AWAITING_REVIEW`，不是 GT 评测。

拟合数据项 full weighted MAE=`0.02736715`、RMSE=`0.04276668`；奇偶留出截断 MAE=`0.03028665`，未截断 MAE=`0.03072591`、RMSE=`0.04850116`。多初值、光度、照度分离、系数统计等详见新输出 `fitting/diagnostics.json`、`selection.json` 和 README。四张图均已实际打开：方向轨迹图显示仰角无平滑跳动并触及40°/50°带边；系数图与数值统计一致；不确定性/收敛图中 hi/lo 近 full、bootstrap 有限分歧且曲线趋稳。`01_support_residual.png` 的支撑曲线可见，但残差面板显示为空白白面；底层 residual 数组有限且非空。已在 README 明确标为可视化呈现缺陷，需主会话审核；未修改预测或据图调参。未读取 GT，未生成 GT 角度指标，冻结和最终验收待主会话及 Academic-analysis 完成。

### 残差图补充诊断（主会话指示）

因原始 `01_support_residual.png` 的残差面板打开后呈空白，按主会话要求仅重绘补充可视化，不修改实现入口、预测或选择。生成命令：`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑/plot_supplemental_residual.py`。脚本只读取已保存的 full residual、固定 fit support 及原始观测 weights；验证对齐后输出 `05_supplemental_weighted_residual_by_frame.png`、逐帧 CSV 和 summary JSON。加权 MAE/RMSE 与原 full 诊断一致（误差小于 `1e-12`），共 123,394 个正权重像素。实际打开补图：逐帧加权均值在 `-0.0259633` 至 `0.0150055`，绝对值帧均值 `0.0052211`；加权 q10–q90 区间较宽且逐帧波动，有效像素数 572–1,139。没有明显持续同向的大残差偏移，但逐像素残差分布较宽。未读取 GT；此项为拟合光度诊断，不能解释为方向精度。补图应与原空白图一并保留，冻结审查纳入新图及其 CSV/JSON/script 哈希。

日志中另有入口第 61 行记录阶段状态时触发的一条 PyTorch `UserWarning`（`requires_grad=True` 张量转 Python 标量提示）。其后八流程正常完成、证书写出、`fit_complete.json` 完成且真实退出码为 0；日志无 traceback。警告保留原样并交主会话审阅，不据此修改实现。
