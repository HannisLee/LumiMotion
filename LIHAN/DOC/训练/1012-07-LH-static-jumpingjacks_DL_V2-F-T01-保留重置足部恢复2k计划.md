# G2-F-T01｜保留15000步透明度重置的脚部恢复对照

## 研究问题与方案

检验G2-T04法线到opacity梯度阻断后，原训练日程中的15000步opacity reset是否仍与脚部局部退化有关。本组从G2-T04@14000独立续训2000步，保留15000步原有reset及其它拓扑更新；与F-T02比较时，唯一科学变量是是否执行这一次reset。

## 固定起点、训练范围与数据边界

- 起点：`output/1012-07-LH-static-jumpingjacks_DL_V2-G2-T04-normal阻断透明度梯度2k/model_mlp`，iteration 14000。冻结起点文件SHA：PLY `b88ca49b9ffc0c7c80209d5e052e169877aa23dc2f7703d15ccf8b1479e4dc9b`；deform `343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0`；lighting `2fa6506f0a96584388c4d3caf16707f07e4f8487bc69d9bf9b76648e7b85c0f5`。启动前再次核验文件存在及SHA。
- source：`output/1011-09-静态DL-Goal训练总控/data_view`；启动前核验135帧训练loader及source文件SHA，并确认未加载测试图像。
- global 14001–16000，共2000次optimizer更新；seed=1012；法线到opacity梯度路由固定为`detach_opacity`。从14000参数重新初始化本轮优化器，不能描述成完整Adam状态恢复。
- 与F-T02共同冻结训练配置、代码快照、source、相机、loader、随机数、优化器及调度器状态构造、loss、渲染和拓扑规则。F-T01唯一处理为`reset_policy=standard`。
- 只用原135张训练RGB、各自原始Alpha、原相机参数及允许的GT光照方向。不得加载测试图像参与训练，不得使用GT法线、深度、材质、阴影、鞋部语义或审计标签作为训练监督或在线策略。
- 计划保存global 14999、15000、15100、15200、15500、16000。须从实际日志和代码路径核实15000步的optimizer、densify/prune、reset、checkpoint保存次序及reset执行次数。

## 输出及启动状态

- 独立输出：`output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k`。
- 配对总控：`output/1012-08-LH-static-jumpingjacks_DL_V2-G2-F脚部reset单因素配对总控`。
- 主机应为`ushas`，解释器为`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。当前准备快照GPU1空闲；GPU0有两项外部服务，GPU2/3由一项外部双卡任务占用。启动前重新核对资源与全机任务数量；GPU1若被其它授权任务调度占用，再依root指示评估GPU2/3各增加一个任务的可能性，不干涉外部任务。
- planner正在实现`scene/static_dl_foot_recovery.py`与`scripts/train_static_dl_foot_recovery.py`；预告参数包括`--init_model`、`--init_iteration 14000`、`--source_path`、`--model_path`、`--reset_policy standard`、`--steps 2000`、`--save_steps 999 1000 1100 1200 1500 2000`、`--seed 1012`。本文件当前只记录接口草案；以提交的稳定实现和完整argv为准，不据此提前启动。
- smoke预告：`--smoke --smoke_clock_start 14990 --steps 120 --save_steps 9 10 110 120`，应越过reset事件到15110，仅作事件与路由诊断，不替代正式训练，不与正式训练检查点或产物混用。入口稳定后核实该clock参数是否确实绑定optimizer/global训练步。
- bootstrap `argv[2]`必须是加入新scene、entry及依赖的不可变代码快照路径，不能指向主repo。smoke、fresh和单项reset路由审计通过后才可正式启动。

## 调度、来源与保留要求

GPU上限为每卡最多2个任务、全机最多8个任务，按实时进程与实际显存调度。启动记录需包含完整shell命令与argv、`commands/train.json`、环境、快照清单及SHA256、source SHA、起点checkpoint SHA、GPU UUID、PID、真实退出码，以及训练PID存活期间采集的native/tracer来源和SHA。训练命令与日志必须放在本实验输出目录。失败时完整保留checkpoint、渲染、统计和日志，不覆盖或删除。

## 评测要求

沿用原固定15帧、原掩膜和像素加权MAE，评测Agent安排确定实际对比checkpoint，避免重复G2-T03/T04已有终点评测。和F-T02在相同global step比较GS／Depth／Mesh Normal MAE及脚部局部诊断；退化界限冻结为GS 2.8561103847°、Depth 5.6240755738°、Mesh 4.4873535823°，同时要求正面、侧面及鞋部关键视角下腿足视觉完整。数值和局部完整性分别报告。尚未完成评测时标记`PENDING`；训练exit 0不等同最终PASS。

实验README最终必须记载完整渲染命令、source/model/iteration、渲染目录、日志、定量指标、代表图／视频路径、原pipeline RGB／Alpha／Normal／Separation四类目检结论及最终`PASS`／`FAILED`；Depth／Mesh作为额外几何与脚部诊断记录。训练执行记录在接口与运行结果稳定后追加，不预写结果。

## 实际训练执行记录（2026-10-11）

本节以实际启动器和模型内记录为准，覆盖上文的“计划/启动前”状态。正式训练从G2-T04@14000独立初始化，global 14001–16000完成2000次optimizer更新；seed 1012、`detach_opacity`、clock_shift_diagnostic=false。仅使用原135张训练图像及其RGB/Alpha、相机和允许的GT光照方向；`test_images_loaded=false`、`forbidden_gt_supervision=false`。

- 完整命令与child argv：本输出目录`commands/train.sh`、`commands/train.json`；bootstrap `argv[2]`为不可变快照`output/1012-08-LH-static-jumpingjacks_DL_V2-G2-F脚部reset单因素配对总控/snapshots/foot-recovery-stable`。
- GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`，训练PID 2497856，supervisor PID 2497842，真实退出码0，结束于2026-10-11T21:39:21Z。训练日志为本输出目录`logs/train.log`；活跃PID期间runtime capture为`PASS_CAPTURED`，native raster/tracer来源SHA见`actual_runtime_capture.json`。
- 六个目标检查点14999、15000、15100、15200、15500、16000均已保存。15000步的optimizer与densify/prune后执行一次透明度reset（performed=1，skipped=0）；reset明细为`model_mlp/reset_events.jsonl`。16000点数56589，ambient_fraction=0。
- 训练监督和模型输入审计、初始化梯度审计、初始法线-opacity审计、拓扑算子对照均通过；配对来源审查与root smoke审查记录在总控目录。
- 两组16000 fresh评测均已通过：11项真实前向输出最大绝对差为0，科学源码SHA一致。完整原15及150步评测、600张PNG与四类视频均通过解码审计；本组整模型最终结论为FAILED（T01数值和足部完整性均失败；T02数值通过但鞋尖仍短、足部完整性失败）。详见下方正式终审记录与配对报告。Depth／Mesh保留为额外几何诊断。

## 正式16000独立评测终审



完整15原GS／Depth／Mesh MAE：**2.960431106973° / 5.779198766501° / 4.302605163168°**；相对冻结基线差值为 `{'gs_render': 0.20432072229341447, 'depth_normal': 0.3551231926519405, 'mesh': 0.015251580863300696}`。数值门槛：**FAILED**；足部完整性：**FAILED（部分恢复，鞋尖仍短）**；最终整模型：**FAILED**。
原15及150六步骤exit0，600PNG与4×150视频完整解码PASS，正式fresh11项差值0。source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`；model：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/model_mlp`；iteration16000；输出：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000`；评测日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/logs`；launcher日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/logs/evaluation_16000_launcher.log`；fresh日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/logs/fresh_parity_16000.log`。

定量：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/goal_mae_summary.json`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/spatial_diagnostic/summary.json`；终审：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/FINAL_REVIEW_COMPLETED.json`。

代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/four_classes/four_classes_10_80_150.png`；足部10／40／80／150：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/spatial_diagnostic/figures`。四类视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/four_classes/rgb.mp4`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/four_classes/alpha.mp4`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/four_classes/normal.mp4`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-F-T01-保留重置足部恢复2k/evaluation_16000/four_classes/separation.mp4`。

- RGB目检：两腿与鞋主体连续；暗面较黑且投影模糊。鞋尖较14000增加，但正面150与侧面10仍短缩。
- Alpha目检：人物与台面合并轮廓连续，鞋object4覆盖100%；不能区分鞋和后方台面。
- Normal目检：台面近均匀，人体仍有色带。Expected与Mesh鞋尖在10/80/150及40可见末端仍短且偏低；足部局部几何部分恢复但不完整。
- Separation目检：人物与平台近同橄榄色，静态门控不能当作运动分离或鞋部独立保护。

实际人工检查四类10／80／150与鞋部10／40／80／150；没有声称人工逐帧看完视频。失败checkpoint、渲染、统计与日志全部保留，未覆盖。

完整配对报告：`LIHAN/DOC/报告/1012-04-足部恢复重置配对完整评测报告.md`。
