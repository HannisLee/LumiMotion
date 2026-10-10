# 原版材质配置与逐帧 Blender PLY 评测支持

日期：2026-10-05；venus / lumimotion。

为评测1003-01/02/03/04四组最终55k原版模型，本次修复独立评测配置恢复并增加逐帧真值PLY导出与时间核验。四组完整评测和独立复算已通过，结果见[报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1005-01-V4四版本Albedo与动态法线深度真值评测报告.md)，执行产物见[README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-01-V4四版本材质与动态几何真值评测/README.md)。未修改训练算法或来源权重。

## 修改

- `scripts/eval_albedo_gt.py`：原版模型没有stage2_config.json，使用几何入口的安全AST解析读取cfg_args或续训配置；补齐PipelineParams默认项后执行原有albedo评分。保存实际配置文件哈希。已有Stage2配置/checkpoint路径仍可用。
- `scripts/export_blender_geometry.py`：显式 `--export_ply` 生成逐帧evaluated世界三角面，保留角点法线和物体ID；manifest记录PLY哈希、源Blender材质/光源/显示/shape-key审计、scene_frame。静态NPZ按几何hash去重，逐帧PLY仍独立留存。只读源blend。
- `scripts/prepare_geometry_gt.py`：存在逐帧PLY时逐项核对其三角坐标、法线和ID与NPZ完全一致，再以PLY实际求交生成GT；manifest记录source_frame/time/PLY审计。增加 `--skip_baseline_audit`，显式允许独立新网格作为真值，不依赖旧canonical baseline通过表面距离检查。未提供PLY时仍支持历史NPZ入口。
- `scripts/eval_geometry_gt.py`：读取GT条目中的source_frame/time并校验与当前预测帧一致，避免动态错时刻。兼容未带time的旧静态GT清单。
- 本轮审计、尺度统一、浮点独立复算、PLY法线对照、阈值敏感性、作图和报告脚本均位于新实验 `commands/`，只写新实验输出与本仓库文档。

## 口径与验证

四组均无已有光强/albedo标定；raw绝对线性指标为主，105训练帧拟合单一scalar后固定评15测试帧，仅作诊断。两个动态集共享经各自GT验证的15帧新PLY，不用旧baseline点云评分；模型使用真实time形变。法线主表沿用EXR，另报新PLY插值法线对照；深度固定expected Z，median为辅助。

34项现有单测通过；4组×15帧实际评测通过；首/中/末12组原渲染器对照最大几何差约1.28e-7；独立浮点复算通过，来源权重与GT哈希不变。动态允许单帧材质完全被遮挡，单帧null不填0，四材质宏平均使用各自测试可见像素汇总。
