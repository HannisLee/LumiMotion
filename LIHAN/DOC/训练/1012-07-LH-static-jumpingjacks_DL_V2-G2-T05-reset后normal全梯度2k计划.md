# G2-T05｜opacity reset 后 normal→opacity 全梯度续训计划

## 目标与配对设计

继续改善静态 DL 角色几何，重点检查脚掌/鞋尖局部重建，同时要求同一 checkpoint 的 GS MAE 小于 2°，Depth 与 Mesh 不劣于既定 T13 基线（Depth 5.88760605°、Mesh 4.75074359°）。本组作为 reset 后全梯度对照：normal 一致性损失对 opacity 的梯度在整个训练区间保持开启。配对组 T06 在 global 15000 的原生 opacity reset 后，从 15001 起仅阻断 normal→opacity 梯度；其他 opacity 梯度保持可用。两组以同一起点、同一快照和同一预算运行，不能拼接不同模型各自的最佳指标。

## 起点、预算与固定项

- 起点：G2-T03 的 `full` checkpoint，`output/1012-06-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k/model_mlp`，iteration 14000。记录的 PLY SHA256 为 `9b43deb28405aa9e07c41026d102e550c1713b8a90b8fee48a3ead8b0cd250e2`，deform SHA256 为 `343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0`，lighting SHA256 为 `d190cc031d3ba94cfbeeae4e3c9f4c92e8c9e5dfd7922ddac8c3ef313b947e51`。T03 的最终点数为 63230，ambient a=0。
- 正式预算：global 14001–16000，共 2000 次优化器更新；采用 T03 原 30k 全局学习率日程并以全新 Adam 状态开始。T03 checkpoint 未保存可恢复的 optimizer state，因此本实验不是 Adam 连续恢复。
- T05 normal→opacity 路由始终为 `full`。T06 在 global 14001–15000 与 T05 相同；在 global 15000 执行既有 opacity reset 后，global 15001–16000 切为 `detach_opacity`。reset 前后分别保存审计数据，明确按 global iteration 定义边界。
- 固定训练输入和科学参数继承 T03 marker；随机种子 1012，环境为 `ushas`，Python 为 `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`，source 为原 `output/1011-09-静态DL-Goal训练总控/data_view`。loader 使用原图 RGB/Alpha 与 135 个训练相机，不使用 GT 法线、深度、材质、阴影或光强监督。评测继续使用既定 eval_view 与原 15 帧协议。
- 两组共同从 ambient a=0 起步；不得人为抬高 a、改变 opacity 初值、增加先验或改动 T03 的损失权重与 LR。除指定的 normal→opacity 梯度路由外，其余监督及参数更新保持一致。
- checkpoint 计划：保存 global 14900、15000（reset 后）、15100、15200、16000；另独立保存 global 15000 的 reset 前副本。记录 reset 前后点 UID/数量审计及 topology 事件，使 reset、裁剪和保存顺序可追溯。

## 预注册输出与命令状态

- 正式输出：`output/1012-08-LH-static-jumpingjacks_DL_V2-G2-T05-reset后normal全梯度2k`。
- 训练文档：`LIHAN/DOC/训练/1012-07-LH-static-jumpingjacks_DL_V2-G2-T05-reset后normal全梯度2k计划.md`。
- 冒烟目录：`output/smoke_test/1012-06-staticDL-G2-T05-reset全梯度110步`。计划在 T03@14000 起点运行 110 步，global 14010 人工触发 reset、global 14100 覆盖一次原生拓扑更新；保存 global 14020、14110。人工 reset 与短 smoke 仅核对代码路径，不代表正式 global 15000 机制的训练效果。
- 稳定入口为 `scripts.train_static_dl_reset_audit`（scene SHA256=`8c2f3757f820f56730015ae066aa474cfa43b3c9740473b776273034f56789e0`，entry SHA256=`dc0da9bde13931b52ff14244e77f00a9fc5f5e075cbde21f6e31278d78a1e22a`）。正式 CLI 草案为 `--init_model <T03/model_mlp> --init_iteration 14000 --model_path <独立/model_mlp> --source_path <原data_view> --post_reset_normal_route full --steps 2000 --save_steps 900 1000 1100 1200 2000 --seed 1012`；准确命令仍须以 root 快照和 launcher 配置核验后落档。入口执行初始18项路由审计；global15001执行19项 reset 后审计。marker 为 `reset_audit_training.json`、`reset_audit_metrics.jsonl` 与 `reset_audit_finished.json`（实际路径以稳定实现为准）。
- smoke 使用独立目录与 `--smoke --steps 110 --save_steps 10 11 100 110 --seed 1012`；从 global14000 起，在14010人工触发reset、14011审计后续路由、14100覆盖一次实际拓扑更新。它只验证分支和执行路径，不模拟或证明正式 global15000 opacity reset 的训练效果。
- 正式启动前置：root 完成快照/manifest核验，smoke 的初始与 reset 后路由审计、拓扑检查和 evaluator fresh reload 均通过，且启动前重查GPU/显存/同卡训练数。启动后记录完整命令、环境、GPU UUID、PID、实际 native/tracer 映射与 SHA、日志、source/代码/checkpoint SHA 和退出状态；同时尊重其他用户进程和其他实验任务。

## 评测与状态

最终采用固定原 15 帧完整几何评测，并按计划完成人工检查。只有同一 checkpoint 同时满足 GS <2°、Depth ≤5.88760605°、Mesh ≤4.75074359°，且鞋尖/脚掌局部几何改善，才可判为目标通过。训练、完整性、fresh parity 与科学目标分别记录；若目标未达到，保留所有模型、统计、图像、视频和日志，不覆盖失败结果。

**当前状态：训练 exit=0，完整16000最终评测与终审已结束，科学目标FAILED。** retry1 smoke训练及14110 fresh parity PASS（11项输出 max_abs=0；17/18/19项路由审计逐项通过）。正式 PID2500372 在 GPU1 完成 2000 更新，state 的 `gpu_verified=true`。首轮 smoke 因 bootstrap snapshot cwd 的相对路径校验 exit=2，失败目录保留。正式 marker/19项自然reset后审计已由 root 核验；global15000触发reset，route全程full。14900/15000/15100/15200/16000 checkpoint、reset前后副本和UID/topology audit均已落盘。训练native/tracer见总控实际maps审计；最终16000稳定checkpoint的原15/四类完整评测与目检均完成，详见下方最终记录。smoke人工reset@14010不是正式机制证据。


## 实际16000最终验收（evaluator收尾回填）

训练exit0/gpu_verified=true；最终完整15原measurement completeness PASS，六子步骤均exit0。GS/Depth/Mesh=2.9371346095106157/5.498365383313657/4.442241136602189°；coverage=0.9979731638026991/0.9659757333436706/0.9936573686940566。旧Goal数字PASS，新GS<2数字FAILED；鞋尖/鞋掌几何仍短缩或偏低，整体几何FAILED，最终**FAILED**。

唯一实际评测：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-08-LH-static-jumpingjacks_DL_V2-G2-T05-reset后normal全梯度2k/evaluation_16000`；完整六子命令与source/model/iteration16000、日志、三指标、代表图片/四视频及目检在其README。真实fresh11输出max_abs=0、原GeometryRun/六零头、12源码/5起点资产、native/tracer与真实17/18/19审计PASS。每类150张PNG/四视频各150帧400×400全部实际解码PASS；人工四类及整个GT足部bbox目检10/80/150，不声称人工逐150帧。

GT对象4 Sneakers 6185原mask交集像素、Alpha和三normal coverage100%不能代替正确脚部高度。独立CPU拓扑UID重放/原因并集/后代守恒PASS；height cohort只是预测世界0.05<z<0.35范围的谱系，不能算语义鞋面积；split父替换非filter删除。

终态依据`FINAL_REVIEW_COMPLETED.json`、`goal2_final/goal2_acceptance.json`、`geometry_review.json`、`visual_review_completed.json`、`artifact_decode_audit.json`、`fresh_parity/parity.json`、`topology_diagnostic/summary.json`与`checkpoint_after_audit.json`。检查点前后SHA不变；原失败smoke和最终checkpoint/渲染/统计/日志均保留。不补评中间或重复GPU渲染。
