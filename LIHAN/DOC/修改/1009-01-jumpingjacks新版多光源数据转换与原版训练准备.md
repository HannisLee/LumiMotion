# JumpingJacks 新版多光源数据转换与原版训练准备

日期：2026-10-09，venus / lumimotion。

新增 `LIHAN/scripts/prepare_jump_lh_dataset.py`，以既有LumiMotion-perlight转换器为基础，兼容新版 `lights.frames`，核对bone_pose帧号、双光源、相机逆矩阵与FOV，保留 `frame/150` 时间和物理秒数。两套150帧数据转换至共享 `transfer-dynamic`，各135/15划分；保留原始多光源和骨骼元数据、albedo、normal EXR、blend快照。

输出8-bit straight RGBA，Alpha来自原始albedo，仅清零全透明RGB；原版reader在加载时乘一次Alpha，避免双重预乘。新增独立NVS合成输入view以适配原版评测脚本的原始PNG GT读取。原始PNG/EXR和正式训练、渲染代码未改动；原版main已有FOV修复保留并记录。

实际输入审计由实验 `commands/audit_conversion.py` 完成，检查全部300帧RGBA/相机/时间/划分，使用原版reader核对两套各15测试相机并生成首中尾目检图。记录见实验 `inspection/conversion_audit.json` 与 `logs/conversion_audit.log`。本次未运行短训练。

训练与报告：[1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练报告.md](../训练/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练报告.md)。environment_light正式训练入口、状态和完整命令已准备，报告名字已核对，用户已授权并于18:04启动正式后台训练，direction_light本次仅转换。
