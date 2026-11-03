# T01 SH 背景外观代理真实冒烟与正式执行计划

日期：2026-11-03。服务器：`ushas`；Conda 环境：`lumimotion-ushas`。本计划严格执行 [T01 规划](../规划/1103-01-T01利用Stage1低阶SH分离背景外观代理规划.md) 和 [实现记录](../修改/1103-01-T01-SH背景代理数值核心.md)，不改代码、候选、参数、门槛或规划。

## 输入与隔离

- source：`data/LH-data/static/LH-static-jumpingjacks-ELDL-V3`
- model：`output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp`，iteration 35000
- observations：`output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz`
- baseline：`output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角`
- 规划锁定：T01 规划文件。只允许读取其规定字段；实验拟合过程不得加载 GT、evaluation、评测收据或历史 GT 结论。旧 baseline 只读。
- 已有最终合成收据：`output/smoke_test/1103-08-LH-static-ELDL-V3-T01-SHbg自检/receipt.json`，合成自检 PASS；此收据不代替真实冒烟。

## 固定执行次序

1. 在独立真实冒烟目录执行 `lock`、`extract`，再执行唯一一次真实 full C01 冒烟 `fit --smoke`。C01 固定 L1、lambda_bg=1e-4，按规划完整收敛预算；不计算候选排名或 GT。任何异常或未收敛均保留全部输出并停报主会话，不重试。
2. 真实冒烟通过后，正式目录执行 `lock`、`extract`、`fit --smoke-dir`（以已通过的 C01 真实冒烟作准入证据），随后 `choose`、`figures`。正式候选严格依规划固定为 C01–C04，每候选七流程；不通过规划门槛时按规定回退 E4。
3. 主会话确认冒烟证据后，依序运行正式流程。所有候选结束后，打开四类可视化并填写实际目检结论、UTC 时间及 `inspection.json`，完成 README 后执行 `freeze`。冻结前不做独立 GT 评测。

## 目录、日志和启动约定

- 真实冒烟：`output/smoke_test/1103-09-LH-static-ELDL-V3-T01-SHbg真实C01冒烟/`。该计数在既有 1103-01 至 1103-08 后递增。
- 正式实验：`output/1103-01-LH-static-ELDL-V3-T01-SHbg/`。当前 `output/` 下没有 1103 日期目录，故计数为 01。
- 两目录创建前均须检查不存在；运行期间所有产物、日志和 checkpoint 保留在各自目录。每条命令的 stdout/stderr 写到该目录 `logs/` 下专属日志，禁止 tee。执行记录须保存完整展开命令、PID、进程开始时间、退出码、环境版本、GPU 状态与输入摘要。
- 长任务使用 nohup 后台运行。启动后每 5 分钟记录一次 PID/进程状态、GPU、日志末尾、流程数和可见最新关键数值。不得因轮询超时重启。
- 训练计划以外的完整渲染不适用：本轮复用冻结 observations，SH 为逐点模板，不产生新像素渲染。上游 observations 的渲染来源/命令须从其已存清单原样记录，不捏造渲染命令。

## 固定命令模板

工作目录：`/home/han.li/reproduce/LumiMotion-stage2-perlight`。执行前须把公共参数按下列字面路径完整展开，并保存到 allocation/command 记录。

```bash
conda run --no-capture-output -n lumimotion-ushas python -m LIHAN.scripts.static_rgb_light_shbg_t01 lock --output output/smoke_test/1103-09-LH-static-ELDL-V3-T01-SHbg真实C01冒烟 --baseline 'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角' --observations 'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz' --source 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3' --model 'output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp' --iteration 35000 --plan 'LIHAN/DOC/规划/1103-01-T01利用Stage1低阶SH分离背景外观代理规划.md' --self-test-dir 'output/smoke_test/1103-08-LH-static-ELDL-V3-T01-SHbg自检'
conda run --no-capture-output -n lumimotion-ushas python -m LIHAN.scripts.static_rgb_light_shbg_t01 extract --output output/smoke_test/1103-09-LH-static-ELDL-V3-T01-SHbg真实C01冒烟 [同上全部公共参数]
conda run --no-capture-output -n lumimotion-ushas python -m LIHAN.scripts.static_rgb_light_shbg_t01 fit --output output/smoke_test/1103-09-LH-static-ELDL-V3-T01-SHbg真实C01冒烟 [同上全部公共参数] --smoke
```

各子命令必须作为独立受记录的启动执行；`[同上全部公共参数]` 是本计划排版缩写，实际执行记录不得保留缩写。`fit --smoke` 为长任务，须 nohup 后台运行并每五分钟核查。

真实 C01 冒烟通过后，正式执行命令及实际日志、PID 均追加记录于本文件的“执行记录”节；不能在启动前虚构结果。正式工作流命令固定为实现记录中的 `lock`、`extract`、`fit --smoke-dir <真实冒烟目录>`、`choose --smoke-dir <真实冒烟目录>`、`figures --smoke-dir <真实冒烟目录>`、`freeze --smoke-dir <真实冒烟目录>`，每步均使用同一组规划输入并写入正式目录。

## 验收信息与结果字段

执行记录须列 source/model/iteration、完整实际命令、输出目录、每步日志、数值诊断、C01 证书、耗时/峰值内存（工具可测时）和退出码。正式 README 须记录所有门槛及逐候选实际值、代表帧 1/38/75/112/150 的图路径、四类图逐图实际目检结论、冻结清单和最终 `PASS`/`FAILED`。目检只报告图上观察，不更改候选选择。冻结后由 Academic-analysis 按规划独立评测；本执行记录不读取或计算 GT 指标。

## 执行记录

- 计划创建时状态：尚未启动真实提取、C01 冒烟或正式拟合。仅有合成自检 PASS 收据（未读真实数据及 GT）。

### 2026-11-03 执行记录（异常停步）

- 服务器/环境：`ushas` / `lumimotion-ushas`；入口的 hostname/conda 与输入完整性门禁通过。合成收据 `receipt.json` 与当前实现代码哈希匹配（由 `lock` 门禁验证）；该收据只覆盖合成自检。
- 冒烟目录：`output/smoke_test/1103-09-LH-static-ELDL-V3-T01-SHbg真实C01冒烟/`。输入锁定命令为本计划模板中的 `lock` 命令，前台执行退出码 0，日志 `run.log`（空）；生成 `input_lock.json`。
- 实际提取命令：`nohup conda run --no-capture-output -n lumimotion-ushas python -m LIHAN.scripts.static_rgb_light_shbg_t01 extract --output 'output/smoke_test/1103-09-LH-static-ELDL-V3-T01-SHbg真实C01冒烟' --baseline 'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角' --observations 'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz' --source 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3' --model 'output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp' --iteration 35000 --plan 'LIHAN/DOC/规划/1103-01-T01利用Stage1低阶SH分离背景外观代理规划.md' --self-test-dir 'output/smoke_test/1103-08-LH-static-ELDL-V3-T01-SHbg自检'`，stdout/stderr：`extract.log`。
- 启动记录 PID `813401`，UTC 启动时间 `2026-11-03T07:49:54Z`，见 `launcher.pid`、`start_utc.txt`。启动后核查时 PID 已不存在，`pgrep` 未发现对应提取进程；`extract.log` 为 0 bytes，预期 `extraction/` 及其结果文件未生成。退出码无法从现存状态确认，记为未知，不推断具体异常原因。
- 依执行规则停在提取步骤；未重启、未运行 C01 拟合、未运行正式候选、未读取 GT。保留冒烟目录现有文件，等待主会话裁定下一步。

### 后续恢复尝试与程序异常（2026-11-03）

以下编号用于保留每次启动证据；不覆盖此前目录。当前实际候选目录为 1103-12，原计划的 1103-09 是首轮启动记录，现不作为成功冒烟目录。

- 1103-10：首次 `lock` 输出误写入 `lock.log`，该名称不在入口允许的初始文件名集合中；入口拒绝非空目录并退出 1，完整错误保存在该目录 `lock.log` 和 `lock.exit_code`。未运行提取。保留原目录。
- 1103-11：改用允许的 `run.log` 后 `lock` 退出 0。提取启动 PID 832292，UTC `2026-11-03T07:51:28Z`。后续 `ps` 和 `pgrep` 均查不到启动器/提取进程，`exit_code.txt`、`finish_utc.txt` 均不存在，未生成 extraction 结果，`extract.log`、`launcher.log` 各 0 bytes。退出码未知；系统内核日志无法读取，不能确认该进程的具体终止原因。该目录完整保留。
- 启动方式探针：在 1103-11 的 `supervisor_probe/` 下，用 `nohup setsid bash` 执行 8 秒等待脚本；PID 832655 启动后实测存活，随后 `exit.txt` 写入 0，证实该脱离方式可完成状态落盘。此为进程管理探针，不是模型实验。
- 1103-12：`lock` 退出 0。用 `nohup setsid bash launch_extract.sh` 启动真实提取，启动器 PID 854169，conda 子进程 PID 854175，启动 UTC `2026-11-03T07:52:58Z`。启动器完整写出 `exit_code.txt=1` 和 `finish_utc.txt`；进程均终止。`extract.log` 保留 Traceback：`static_rgb_light_shbg_t01_extract.py:192` 在 `fractions=clipped[valid].mean(axis=0)` 抛出 `IndexError: boolean index did not match indexed array along dimension 0; dimension is 1 but corresponding boolean dimension is 150`。未产生提取产物，GPU 使用为 0 MiB。此为真实程序异常，按规划停在提取步骤。
- 截至本记录：没有运行真实 C01 冒烟拟合、正式候选、选择、绘图或冻结；没有读取 GT；1103-09、1103-10、1103-11、1103-12 的现有输出与日志全部保留，等待主会话裁定修复与后续流程。

### 修复后重试（2026-11-03）

主会话确认提取器 L1/L2 通道维度索引 bug 已修复，修复提交已进入主工作树；新合成收据 `output/smoke_test/1103-13-LH-static-ELDL-V3-T01-SHbg自检/receipt.json` 为 PASS、`gt_read=false`，覆盖 L0/L1/L2 帧点通道维度、稀疏有效掩码与截断率。当前提取器 SHA256：`90317f8ce41237ecab6ad6db24b2f1bfab2a11b0b7184008ea280cd1ce769f7f`。本次按规划的“明确实现 bug 修复后另建目录”条款执行；此前 1103-09 至 1103-12 全部保留。

修复后唯一真实 full C01 冒烟的预分配目录为 `output/smoke_test/1103-14-LH-static-ELDL-V3-T01-SHbg真实C01冒烟/`（计数在 1103-13 后递增），先重新 `lock`，再用 `nohup setsid bash launch_extract.sh` 可靠记录 PID、UTC 起止时间和退出码。SH 提取成功后只允许一次规划预算的真实 C01 冒烟拟合；运行时不修改本训练计划，以免破坏输入锁定哈希。C01 终止并取得退出码后再补写本计划的实际结果。若提取/拟合报错或证书不通过，保留产物并停止，不重跑、不启动正式流程，报告主会话。
