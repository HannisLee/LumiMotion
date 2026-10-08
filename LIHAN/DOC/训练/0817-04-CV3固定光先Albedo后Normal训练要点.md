# 0817-04 CV3 固定 GT 光先 Albedo 后 Normal 训练要点

- 日期：2026-08-17 ~ 08-18
- 服务器/环境：garuda / `lumimotion-garuda`
- 原目录：`output/0817-04-CV3-GTlight_i5p5_A500_Nonly_lr1e4`
- 状态：训练已到 35000 并完成离线法线评估；README 验收占位未填写，**不能标 PASS**

## 一句话总结

“先冻结几何、再两阶段训材质”的覆盖性实验：1–10000 走原 SH Stage-1；10001–10500 冻结几何/形变/光/法线只训 albedo；10501–35000 冻结 albedo 只训独立法线。结果显示独立法线未向 GT 收敛。

## 实验定义

- 数据集：CV3，source `data/LH-data/transfer-dynamic/only_clothV3`
- GT directional 辐照度：标定值 `5.5043499`
- normal LR `1e-4`；初始化 normal trust-region 权重 `0.01`
- 不使用 GT-normal loss，GT EXR 仅离线验收
- 模型目录：`CV3_A500_Nonly_lr1e4_mlp`

## 执行记录

- 08-17 启动；外层会话在 iter ~9687 结束（无 Python/CUDA/OOM traceback），最新有效 checkpoint 5000
- 08-18 用 `resume_from_5000.sh` 从 global iter 5001 恢复，日志 `train_stage1_resume_from_5000.log`；进程监视 `process_monitor_0818.log`
- checkpoint 齐全：499/1000/5000/10000/10001/10500/10501…11000/20000/30000/35000
- 训练命令见同目录 `run.sh`；日志 `train_stage1.log`

## 离线法线评估（独立法线 vs world GT，`normal_gt_eval_independent/`）

| iter | mean | median | P95 |
| ---: | ---: | ---: | ---: |
| 10001 | 25.598° | 15.896° | 74.921° |
| 10500 | 25.598° | 15.896° | 74.921° |
| 35000 | 28.690° | 22.610° | 68.809° |

- 10001 = 10500：albedo 阶段法线被冻结，符合预期
- 法线阶段训到 35000 后 mean 反而恶化 `+3.09°`（仅 P95 改善）→ 独立法线未收敛到 GT

## 命令与产物

- 渲染命令（README）：`scripts.render_stage1_insights --render_mode photometric_lambertian --load_iter 35000 --depth_ratio 1.0`（计划输出 `train/ours_35000`、`test/ours_35000`）
- 法线评估日志：`eval_normals_{10001,10500,35000}.log`
- 验收占位（定量指标、代表图片/视频、四类目检、最终结论）均未填写

## 关联实验

- 同批姊妹实验：0817-05（CV3L 版，见 `0817-05-CV3L固定光先Albedo后Normal训练要点.md`）
- 前身（独立 normal、LR 1e-3）已归档：见 `0817-04-CV3独立Normal训练要点（归档）.md`
