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
