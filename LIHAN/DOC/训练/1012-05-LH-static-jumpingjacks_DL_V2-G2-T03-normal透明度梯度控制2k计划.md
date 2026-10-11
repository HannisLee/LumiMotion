# G2-T03｜法线损失到透明度梯度全量控制2k计划

## 研究问题与配对设计

检验法线一致性损失流向Gaussian opacity的梯度是否会削弱脚部/鞋尖点。两组共同从G2-T01的iteration12000开始，以新Adam状态训练2000步。唯一科学变量是梯度路径：本组保留normal loss到opacity的正常梯度；另一组只切断该项梯度。其余渲染、RGB/几何损失、参数组、学习率、训练图像、随机种子、环境和入口保持一致。比较相同global checkpoints 12500/13000/14000，不拼接不同模型各自的最佳单项指标。

## 起点、固定项与预算

- 起点模型：`output/1012-03-LH-static-jumpingjacks_DL_V2-G2-T01-纯直射续训2k/model_mlp`，iteration12000；PLY SHA256=`866862df27e8f8847085803e6712480cee0e1fee2e2ac440badabd1ea900320f`，deform SHA256=`343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0`，lighting SHA256=`b7f0eaadeb5b71316e08cc63c7a80f42693e3f1f830646456540bcd3ab5cb835`。
- 预算：global12001–14000，实际2000次optimizer更新，保存相对step500/1000/2000，即global12500/13000/14000。T01没有保存可恢复Adam状态；两组都以零Adam moment开始，不称为无缝连续恢复。沿用原30k全局LR日程，global12000起步；不因本轮2k预算压缩schedule。
- 共同固定项：G2-T01几何、光强与材质检查点；seed=1012；环境`ushas`与Python `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`；source使用原`output/1011-09-静态DL-Goal训练总控/data_view`；新loader只构建恰好135个训练相机（`test_images_loaded=false`），使用其RGB/原图Alpha、相机和原GT方向。测试图片不加载、不进入训练loss，评测另行读取原固定eval_view。禁止GT法线、深度、材质、阴影或光强监督。
- 初始化ambient固定a=0（沿纯直射T01起点），除本实验指定的梯度通道外所有物理、权重、拓扑、normal/alpha设置从G2-T01训练marker继承。具体完整继承清单须写进prelaunch fairness audit；不得改变Opacity初值、添加先验、重置Adam以外的优化器状态或额外饱和处理。
- 正式输出：`output/1012-06-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k`；训练文档：`LIHAN/DOC/训练/1012-05-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k计划.md`。
- 冒烟：分别用独立目录`output/smoke_test/1012-04-staticDL-G2-T03-normal透明度梯度控制110步`（首次ROOT目标的无效记录）及独立重试`output/smoke_test/1012-04-staticDL-G2-T03-normal透明度梯度控制110步-retry1`，以global12000起点跑110步，global12100执行一次既有拓扑更新，保存global12020/12110；smoke与正式模式必须由入口明确区分。梯度路由审计、原生栅格来源、fresh reload及GPU核验通过后才可启动正式训练。

## 实验前置审计与运行记录

normal-opacity入口已由planner标记STABLE，root已审阅并完成97文件native snapshot。本计划已进入独立smoke验证阶段；正式训练仍须真实路由审计与fresh通过后才可启动。记录完整snapshot、CLI、launcher `commands/train.json`、环境变量、GPU UUID、训练PID、actual native/tracer.so SHA、日志、参数/代码/source SHA与稳定checkpoint SHA。每卡最多两个本目标任务；启动前重新检查GPU进程及显存，尊重外部任务。若 smoke、fresh或训练失败，另开编号保留，不覆盖。

训练输出必须解释梯度路径审计：full组核对normal损失对opacity的实际有限非零梯度；detach组核对该分量严格为零，同时其它相应训练损失梯度保持预期。此审计要区分单项损失与合并梯度，不能用总梯度掩盖分流失败。

## 评测及目标

终点采用固定原15帧完整协议并作四类150方向可视化；保持原掩膜与像素加权MAE。检查同一checkpoint上的GS<2.00°，Depth不得高于T13基线5.8876060528°、Mesh不得高于4.7507435938°；同步人工审查鞋尖/脚掌局部几何，不以完整Alpha或整体MAE代替局部重建。未到目标如实FAILED。中间checkpoint是否补评等待root/evaluator安排，不预承诺。

**最终状态：FAILED。** 正式训练exit=0、gpu_verified=True，global14000/2000次更新完成；点数=63230，a=0。终点评测marker=`output/1012-06-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k/evaluation_14000/FINAL_REVIEW_COMPLETED.json`：同一个checkpoint原15帧完整性PASS，但两度Goal FAILED（GS=2.7576517427°未达<2°；Depth=5.3275671046°、Mesh=4.2877150228°满足不退要求），局部鞋尖仍缩短。四类150方向/600PNG/4×150视频完整解码PASS，10/80/150帧及鞋部特写已人工目检；视觉记录见`output/1012-06-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k/evaluation_14000/visual_review_completed.json`，代表图与视频见该目录`four_classes/`。训练命令、启动状态、日志、完整checkpoint和训练native/tracer映射记录在`output/1012-06-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k`。正确snapshot smoke retry1 fresh PASS；早期argv[2]=ROOT的冒烟仅以`FAILED_BOOTSTRAP_TARGET_PROTOCOL`保留历史，未作本结论依据。目标失败产物不删除、不覆盖。
