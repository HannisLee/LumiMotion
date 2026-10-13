# G2-T07｜原135 TRAIN光度支持与几何梯度竞争零更新诊断计划

## 当前状态

**三帧probe已启动但在首帧前来源门禁FAILED；完整135帧尚未启动。** 本文记录恢复两度目标后的新一轮诊断，不改写历史Goal3-T07未执行归档。真实失败目录为`output/smoke_test/1012-23-staticDL-G2-T07真实native零更新探针/`，不可覆盖或重启。

本轮只做固定模型上的原训练集机制诊断，不是新训练：不创建优化器、不更新参数或点拓扑、不产生新MAE。用户最新指令恢复`LIHAN/DOC/GOAL/1011-02-PBR法线两度突破迭代.md`的持续迭代目标；历史报告11、训练17及修改19中“停止后续工作”的状态仅保留为当时事实。

## 问题与固定对象

检验原135 TRAIN图像、Alpha、相机和允许的纯GT光照方向中，光度分支在各像素的实际导数支持，以及photo、alpha、distortion、normal四分支对XYZ、scale、rotation、opacity的逐点余切和竞争关系。该诊断只用于决定后续假设，不直接证明某处几何缺陷成因，也不授权改损失权重或训练路线。

- source：原静态DL训练source，路径及SHA由运行时`plan.json`与完整冻结清单记录。
- model：`output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp`。
- checkpoint：iteration **16500**，PLY、deform、lighting及optimizer来源沿用修改19中的固定SHA；optimizer仅核对来源，不加载为可更新状态。
- 科学实现：冻结snapshot `output/1012-20-LH-static-jumpingjacks_DL_V2-F-T04-T05执行总控/snapshots/foot-densify-stable`。
- 数据范围：严格使用原135 TRAIN RGB／Alpha／相机和允许的纯方向；禁止读取GT test、GT法线、深度、材质、阴影、语义或其他额外监督。
- 固定损失路线：photo=1、alpha=0.1、distortion=1000、normal=0.1；normal到opacity继续detach。诊断不改变任何训练权重。

## 执行安排与资源门禁

仅在root完成独立代码审查、规划侧完成新输出路径适配，并完成两项独立审查准入后，才允许启动一次GPU零更新诊断。首选物理GPU0单进程；启动前重新读取`nvidia-smi`设备、显存、利用率和compute PID，确认GPU0资源可容纳任务且本任务同卡进程数为0。其他用户任务保留，不发送信号、不干预。按项目约束，本任务每卡最多2个进程、全项目任务总数最多8；本轮计划只用GPU0上的1个进程。

输出与日志均放在新目录`output/1012-25-LH-static-jumpingjacks_DL_V2-G2-T07光度支持与几何梯度零更新诊断/`。历史`output/1012-23-...`和`output/1012-24-...`维持原样，不把Goal3-T07的旧未执行记录当成本实验结果。

执行器：`output/1012-26-静态DL-两度目标恢复总控/code/diagnostic_runner.py`。helper冻结SHA为`321d22cbbc62efecac26baeacf194344bb5bb0f6e5167f983c05257bc5d52a01`；runner冻结SHA为`805b780f547d27a9c2b29fb25f8810d562568f4ab87d322c7df4570aa9c0c0b1`。两份spec位于总控26的`commands/g2_t07_probe_spec.json`和`commands/g2_t07_full_spec.json`，分别绑定smoke_test探针与正式output25。runner用GPU0固定UUID，不依据设备序号重排；需在正式启动前对helper/runner/spec SHA再做一次匹配。

三帧探针完整命令如下；目前仅冻结，未启动：

```bash
source /home/han.li/miniconda3/etc/profile.d/conda.sh
conda activate lumimotion-ushas
cd /home/han.li/reproduce/LumiMotion-stage2-perlight
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-26-静态DL-两度目标恢复总控/code/diagnostic_runner.py --spec /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-26-静态DL-两度目标恢复总控/commands/g2_t07_probe_spec.json --helper /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal3_photometric_support_profile.py --output /home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1012-23-staticDL-G2-T07真实native零更新探针 --gpu-index 0 --gpu-uuid GPU-f2584a61-4563-1230-045d-cc9dc2336e08
```

僅在三帧真实native检查PASS且root准入完整任务后，再执行原135帧命令：

```bash
source /home/han.li/miniconda3/etc/profile.d/conda.sh
conda activate lumimotion-ushas
cd /home/han.li/reproduce/LumiMotion-stage2-perlight
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-26-静态DL-两度目标恢复总控/code/diagnostic_runner.py --spec /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-26-静态DL-两度目标恢复总控/commands/g2_t07_full_spec.json --helper /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal3_photometric_support_profile.py --output /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-25-LH-static-jumpingjacks_DL_V2-G2-T07光度支持与几何梯度零更新诊断 --gpu-index 0 --gpu-uuid GPU-f2584a61-4563-1230-045d-cc9dc2336e08
```

执行器记录实际命令、source冻结清单、完整环境、source/model/iteration、GPU UUID、PID、native/tracer运行映射、启动前后资产/RNG/参数摘要及真实退出码。诊断自身`runtime_active.json`必须包含活跃进程通过`/proc/maps`捕获的PBR cache `.so`和tracer `.so` SHA，并与checkpoint provenance吻合，否则判失败并保留现场。

## 预期记录与判定边界

逐帧保存原始像素级光度导数及四分支对四类几何参数的向量余切；normal分支另拆为`rend_normal`与`surf_normal`两条真实VJP，并核验分解和与原normal梯度一致。跨135帧统计逐点支持、范数、方向一致性和分支点积。范数与点积在CPU以float64计算，逐帧向量落CPU文件；不得跨帧把全部像素向量留在GPU。

计划检查输入/source/checkpoint哈希、真实加载的135 TRAIN相机和方向、真实native来源、诊断前后参数/拓扑/RNG及`.grad`状态、零优化器更新和测试输入未加载。真实native VJP守恒或零更新检查失败时，保留完整checkpoint来源、渲染/统计（若已生成）、日志和失败现场；README记`FAILED`，不得删除或覆盖。

本诊断不执行测试渲染或MAE评测。代表图片/视频、四类测试可视化目检和三项MAE均记为“不适用：本任务未渲染/未评测”，不得伪造结果。机制解释仅作为下一轮实验设计证据；不得据此直接宣告鞋部修复或两度Goal成功。

## 启动前状态

- 当前主机已确认：`ushas`。
- 指定解释器：`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`，环境名`lumimotion-ushas`。
- 本次资源盘点：GPU0有外部进程占用，GPU1也有外部高利用率任务，GPU2/3由外部重训练占用；必须在实际启动前重查，不能沿用旧资源快照。
- 项目自有G2-T07/T04进程：盘点时未发现运行进程；外部进程未干预。
- 三帧probe child PID 2959605，runner session 2576，实际UTC启动约2026-10-13T02:13:42Z、退出2026-10-13T02:19:19Z，真实return code=1；完整现场及准确时间见smoke目录的command JSON。
- 失败原因：GPU0 live map中的PBR cache SHA为`ee4a9e00b8aceef8fc9d888d88530eeaea61e24d48a3609357a84962602723d6`，而iteration16500 marker要求`f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3`。tracer SHA吻合。未加载首帧、没有逐帧NPZ、没有更新、没有MAE；不能当作训练科学失败。
- 隔离缓存调查计划：在总控26/native_cache用旧运行记录中的CXX=`/home/han.li/miniconda3/bin/x86_64-conda-linux-gnu-c++`、CC=`/home/han.li/miniconda3/bin/x86_64-conda-linux-gnu-cc`、CUDA_HOME=`/home/han.li/miniconda3/envs/lumimotion-ushas`、TORCH_CUDA_ARCH_LIST=8.6，独立构建同一冻结PBR source；不得覆盖或引用当前共享cache中的binary作为checkpoint provenance。结果须由evaluation复核后才能安排新日期独立probe。
- 当前状态：**PROBE FAILED / FULL135 NOT STARTED**；参数更新0；新MAE无。probe目录及cache调查产物保留，重试必须用新的1013 smoke计数与独立输出。


## 隔离native构建结果（2026-10-13）

首次probe的失败由PBR `.so` SHA与固定checkpoint marker不一致触发。为避免共享扩展缓存被其他任务改写，已在总控26/native_cache中用旧成功runtime登记的CXX/CC、ushas CUDA_HOME与`TORCH_CUDA_ARCH_LIST=8.6`单次隔离构建；设置`MAX_JOBS=4`仅是本次可复现控制参数，旧runtime未登记该值。构建结束时间为`2026-10-13T04:41:15.482720Z`，`cuda_initialized=false`。实际GCC为14.3.0、nvcc为12.9。

构建结果为`BINARY_MISMATCH_RETAINED`：隔离`.so` SHA256=`9e414f5c7a447d51d28d993903f305c2bb7ab7c45b653b5ea9e72956ee4ef7bd`，期望SHA=`f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3`；build.ninja SHA256=`1f6598e114e416813f388fa4977cc35a97c9e5f02ad66f90c10a6587cc2bc822`。源SHA与冻结manifest一致，但产物SHA不匹配，因此不继续重编、不取消来源门禁，也不启动probe或full135。日志、完整环境、provenance与二进制均保留在`output/1012-26-静态DL-两度目标恢复总控/native_cache/`。下一步由evaluation复核差异并由root决定是否建立兼容验证路径；任何重试使用新日期和新smoke目录。


### 2026-10-13最终审计收尾

用户要求停止新增实验并完成收尾。当前实际优化更新为0，无新checkpoint或新GPU评测；科学目标未达成。真实探针／smoke均在渲染前因native二进制身份变化退出1，私有重编译也未恢复历史SHA。所有失败与历史证据原位保留，不启动预留retry或正式训练。原会话现为idle，目标状态库已无本Goal行；本文旧ACTIVE／进行中描述只代表当时历史，不代表现在仍在运行。

运行审计和最终归档见`LIHAN/DOC/报告/1013-05-两个Goal长时低产出审计与最终收尾.md`。审计总控为`output/1013-05-两个Goal运行审计与最终收尾`；原位清单和复核凭据在两个原总控`archive_closeout/manifest.json`、`ARCHIVE_COMPLETED.json`，全分支同步凭据为`shared_sync_selected.json`。归档PASS不等于科学目标PASS。
