# G2-T06｜opacity reset 后阻断 normal→opacity 梯度续训计划

## 目标与配对设计

与 G2-T05 构成同起点、同预算的单因素配对。两组在 global 14001–15000 均使用 normal→opacity 全梯度；在 global 15000 执行原生 opacity reset 后，本组从 global 15001–16000 仅切断 normal 一致性损失到 opacity 的梯度。其他损失仍可更新 opacity，避免将该处理误写成冻结 opacity。核心比较脚掌/鞋尖局部重建，并要求同一 checkpoint GS MAE <2°、Depth 与 Mesh 不劣于 T13 基线（5.88760605°、4.75074359°）。

## 起点、预算与固定项

- 起点：G2-T03 `full` checkpoint，`output/1012-06-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k/model_mlp`，iteration 14000。PLY SHA256=`9b43deb28405aa9e07c41026d102e550c1713b8a90b8fee48a3ead8b0cd250e2`；deform SHA256=`343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0`；lighting SHA256=`d190cc031d3ba94cfbeeae4e3c9f4c92e8c9e5dfd7922ddac8c3ef313b947e51`。起点点数 63230、ambient a=0。
- global 14001–16000 共 2000 次优化器更新，以全新 Adam 状态运行，并沿用 T03 的 30k 全局 LR 日程。T03 检查点没有可恢复 optimizer state；本轮不是无缝 Adam 续跑。
- global 14001–15000 路由为 `full`；global 15000 执行 opacity reset 后，global 15001 起路由切为 `detach_opacity`。审计必须分别验证 reset 前后的 normal→opacity 单项梯度路径，并验证其他应保留的 opacity 梯度可用。
- 继承 T03 的训练相机、损失、权重、学习率、物理设置、原图 RGB/Alpha、随机种子 1012 与 `ushas` 环境。source 为 `output/1011-09-静态DL-Goal训练总控/data_view`；训练使用 135 个训练相机。禁止任何 GT 法线、深度、材质、阴影或光强监督；评测固定原 eval_view 和 15 帧协议。
- 两组 ambient a 均为 0，不改 opacity 起始值，不加先验，不改变除 normal→opacity 路由外的科学因素。checkpoint 保存 global 14900、15000 reset 后、15100、15200、16000；另留 15000 reset 前独立副本，并写入 UID/点数与 topology 事件审计。

## 预注册输出与命令状态

- 正式输出：`output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k`。
- 训练文档：`LIHAN/DOC/训练/1012-08-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k计划.md`。
- 冒烟目录：`output/smoke_test/1012-07-staticDL-G2-T06-reset阻断透明度110步`。110 步 smoke 从 global 14000 开始，在 global 14010 人工触发 reset，global 14011 验证候选路由，global 14100 检查原生拓扑更新；保存 global 14020、14110。该人工边界用于执行路径审计，不代表正式训练在 global 15000 的自然 reset 效果。
- 稳定入口为 `scripts.train_static_dl_reset_audit`（scene SHA256=`8c2f3757f820f56730015ae066aa474cfa43b3c9740473b776273034f56789e0`，entry SHA256=`dc0da9bde13931b52ff14244e77f00a9fc5f5e075cbde21f6e31278d78a1e22a`）。正式 CLI 草案为 `--init_model <T03/model_mlp> --init_iteration 14000 --model_path <独立/model_mlp> --source_path <原data_view> --post_reset_normal_route detach_opacity --steps 2000 --save_steps 900 1000 1100 1200 2000 --seed 1012`；最终完整 argv 须绑定 root 核验过的 snapshot 和 launcher 配置。入口初始审计18项，global15001 reset 后审计19项；marker 为 `reset_audit_training.json`、`reset_audit_metrics.jsonl` 与 `reset_audit_finished.json`（实际路径以稳定实现为准）。
- smoke 使用 `--smoke --steps 110 --save_steps 10 11 100 110 --seed 1012`，从14000开始，14010人工reset、14011验证后续路由、14100检查原生拓扑。人工 reset 只是执行分支测试，不是正式15000自然 reset 的证据。
- 正式前需 root 完成 snapshot/manifest 核验、smoke 两阶段路由审计和 fresh reload 通过，并实时复核GPU资源及同卡总训练数。启动时记录完整命令、环境、GPU UUID、PID、实际 native/tracer SHA、日志与 checkpoint SHA；每卡最多两个本目标训练任务并尊重外部任务。

## 评测与状态

采用固定原 15 帧完整几何评测，检查同一个 checkpoint 的三项数值及脚掌/鞋尖局部几何。只有 GS <2°、Depth ≤5.88760605°、Mesh ≤4.75074359° 且局部几何符合目标，才判为通过。route 审计通过不等于模型达标；任何失败的 checkpoint、渲染、统计和日志均完整保留。

**当前状态：训练 exit=0，完整16000最终评测与终审已结束，科学目标FAILED。** retry1 smoke训练及14110 fresh parity PASS（11项输出 max_abs=0；17/18/19项路由审计逐项通过）。正式 PID2500425 在 GPU1 完成 2000 更新，state 的 `gpu_verified=true`。首轮 smoke 因 bootstrap snapshot cwd 的相对路径校验 exit=2，失败目录保留。正式 marker/19项自然reset后审计已由 root 核验；global15000前 route=full，global15001起 detach_opacity。14900/15000/15100/15200/16000 checkpoint、reset前后副本和UID/topology audit均已落盘。训练native/tracer见总控实际maps审计；最终16000稳定checkpoint评测/目检均完成，详见下方最终记录。smoke人工reset@14010不证明正式机制。


## 实际16000最终验收（evaluator收尾回填）

训练exit0/gpu_verified=true；最终完整15原measurement completeness PASS，六子步骤均exit0。GS/Depth/Mesh=2.6683778383745302/5.404577799172339/4.20848728697561°；coverage=0.9980285418408767/0.9660477247933015/0.9931515826120344。旧Goal数字PASS，新GS<2数字FAILED；鞋尖/鞋掌几何仍短缩或偏低，整体几何FAILED，最终**FAILED**。

唯一实际评测：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/evaluation_16000`；完整六子命令与source/model/iteration16000、日志、三指标、代表图片/四视频及目检在其README。真实fresh11输出max_abs=0、原GeometryRun/六零头、12源码/5起点资产、native/tracer与真实17/18/19审计PASS。每类150张PNG/四视频各150帧400×400全部实际解码PASS；人工四类及整个GT足部bbox目检10/80/150，不声称人工逐150帧。

GT对象4 Sneakers 6185原mask交集像素、Alpha和三normal coverage100%不能代替正确脚部高度。独立CPU拓扑UID重放/原因并集/后代守恒PASS；height cohort只是预测世界0.05<z<0.35范围的谱系，不能算语义鞋面积；split父替换非filter删除。

终态依据`FINAL_REVIEW_COMPLETED.json`、`goal2_final/goal2_acceptance.json`、`geometry_review.json`、`visual_review_completed.json`、`artifact_decode_audit.json`、`fresh_parity/parity.json`、`topology_diagnostic/summary.json`与`checkpoint_after_audit.json`。检查点前后SHA不变；原失败smoke和最终checkpoint/渲染/统计/日志均保留。不补评中间或重复GPU渲染。
