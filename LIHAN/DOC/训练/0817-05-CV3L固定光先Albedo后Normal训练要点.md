# 0817-05 CV3L 固定 GT 光先 Albedo 后 Normal 训练要点

- 日期：2026-08-17 ~ 08-18
- 服务器/环境：garuda / `lumimotion-garuda`
- 原目录：`output/0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4`
- 状态：训练已到 35000 并完成离线法线评估；README 验收占位未填写，**不能标 PASS**

## 一句话总结

0817-04 的 CV3L 姊妹实验：同样 1–10000 SH、10001–10500 只训 albedo、10501–35000 只训独立法线；关键修正是改用 CV3L 独立标定辐照度 `7.8434867`，不再错误复用 CV3 的 5.5。

## 实验定义

- 数据集：CV3L，source `data/LH-data/transfer-dynamic/only_clothV3_lambertian`，训练图像已核实来自 `image_lambertian`
- GT directional 辐照度：`7.8434867`（CV3L 独立标定）
- normal LR `1e-4`；初始化 normal trust-region 权重 `0.01`；不使用 GT-normal loss
- 模型目录：`CV3L_A500_Nonly_lr1e4_mlp`

## 执行记录

- 08-17 启动；外层会话在 iter ~9760 结束（无 traceback），最新有效 checkpoint 5000
- 08-18 用 `resume_from_5000.sh` 从 5001 恢复，日志 `train_stage1_resume_from_5000.log`；进程监视 `process_monitor_0818.log`
- checkpoint 齐全：499/1000/5000/10000/10001/10500/10501…11000/20000/30000/35000
- 训练命令见同目录 `run.sh`；日志 `train_stage1.log`

## 离线法线评估（独立法线 vs world GT，`normal_gt_eval_independent/`）

| iter | mean | median | P95 |
| ---: | ---: | ---: | ---: |
| 10001 | 24.809° | 18.496° | 65.867° |
| 10500 | 24.809° | 18.496° | 65.867° |
| 35000 | 31.024° | 24.154° | 81.136° |

- 10001 = 10500：albedo 阶段法线冻结，符合预期
- 法线阶段训到 35000 后 mean 恶化 `+6.22°`、P95 恶化 `+15.27°` → 独立法线明显未收敛，情况比 CV3 版更差

## 命令与产物

- 渲染命令（README，GPU3）：`scripts.render_stage1_insights --render_mode photometric_lambertian --load_iter 35000 --depth_ratio 1.0`
- 法线评估日志：`eval_normals_{10001,10500,35000}.log`
- 验收占位均未填写

## 关联实验

- 前身（独立 normal、LR 1e-3、错误复用辐照度 5.5）已归档：见 `0817-05-CV3L独立Normal训练要点（归档）.md`
