# 1011-10-LH-static-jumpingjacks_DL_V2-T04-control-静态分流2k训练计划与执行记录

## 目的与配对设计

对比静态光照模型在控制路由（`control`）与阴影分流路由（`split`）下的短程微调。两组共同起点是T03的10k checkpoint，固定数据、seed、学习率、loss权重和训练步数；唯一实验变量为route。研究起点由规划审查后选为C10k，以保留已学方向Lambertian与可见度前向；单帧初始photo loss不作为完整质量排名。

## 真实checkpoint审计依据

B10k与C10k在 `output/smoke_test/1011-11-staticDL-B10k-真实checkpoint分流审计/model_mlp/initial_checkpoint_gradient_audit.json` 和 `output/smoke_test/1011-12-staticDL-C10k-真实checkpoint分流审计/model_mlp/initial_checkpoint_gradient_audit.json` 均审计PASS，11项检查为true且参数更新数为0。C10k首训练相机初始photo为0.0116060162，visibility均值0.5037023、零值占比0.4111383；控制路由XYZ梯度零，split/真实几何路由可得有限XYZ与q梯度，相机正常路径XYZ/q梯度为零，旁路负控非零。基于保留已学光照与真实遮挡，选择C10k为两组共同初始化。

## 训练配置

- checkpoint：`output/1011-15-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH-GPU2重试/model_mlp`，iteration 10000。
- 数据：`output/1011-09-静态DL-Goal训练总控/data_view`，仅原135帧训练RGB及原图自身Alpha；无GT几何和测试监督。
- 源码：`output/1011-09-静态DL-Goal训练总控/snapshots/stage2-shadow-split-native-assets-20261011`，经 `bootstrap_1011.py` 使用仓库native raster。
- route：`control`；steps=2000；相对保存步数500、1000、2000（全局checkpoint iteration 10500、11000、12000）。seed=1011。
- 学习率：xyz=1e-5、rotation=1e-4、albedo=1e-3、light=1e-4。权重：normal=.1、dist=1000、alpha=.1、shadow_bias=.03、shadow=.25、highlight=.5、residual_threshold=.08。
- 不增密、不更新scale/opacity/deform，不新增饱和处理因素。

## 启动前门槛

先运行并审计 `1011-13-staticDL-T04-control-20步冒烟` 的20步冒烟，保存全局iteration 10020；control与split两组均通过真实checkpoint梯度路由/有限非零检查后，由evaluator对新renderer和现有物理wrapper做fresh/fresh parity。两组parity通过后才启动本计划正式2k训练。正式输出目录：`output/1011-16-LH-static-jumpingjacks_DL_V2-T04-静态分流control-2k`。

## 完整启动命令

实际配置文件：`output/1011-16-LH-static-jumpingjacks_DL_V2-T04-静态分流control-2k/commands/train.json`；启动脚本：`output/1011-16-LH-static-jumpingjacks_DL_V2-T04-静态分流control-2k/commands/train.sh`。其完整argv以json内记录为准，训练日志为同目录 `logs/train.log`，状态/GPU/PID/退出码保存在 `state.json` 与 `launch_provenance.json`。

## 执行记录

计划创建时尚未启动正式训练。完成后记录三个checkpoint稳定性和SHA、训练状态、fresh parity、Goal评测输出目录、量化指标、代表图/视频路径、四类目检结论及PASS/FAILED。验收失败保留所有checkpoint/渲染/统计/日志，不覆盖。


## 20步冒烟执行

`T04` `control`冒烟已退出码0，标准launcher对GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`核验通过。输出 `output/smoke_test/1011-13-staticDL-T04-control-20步冒烟`，checkpoint iteration 10020（81,323点），真实checkpoint梯度审计 `PASS`，11项检查全true、参数更新数0；control XYZ梯度严格为零，split XYZ/q/材质/灯光梯度有限。日志、命令、checkpoint及launcher状态均保留在冒烟目录。Fresh/fresh parity尚待evaluator确认；正式2k未启动，必须等待两组parity全部PASS。


## 启动前复核

T04与T05两组iteration10020 smoke的fresh/fresh renderer parity均由evaluator复核PASS，8项对照（含逐surfel visibility）最大差均为0；详细文件位于各smoke输出的 `render_parity/parity.json`。共同checkpoint四文件SHA与源码快照匹配，现具备正式2k启动条件。


## 正式训练执行记录

T04 `control`于2026-10-11T09:10:30.346339+00:00至2026-10-11T09:13:11.623112+00:00完成，PID 2017851，GPU1，实际UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`；launcher确认gpu_verified=true、exit code 0。训练首相机真实梯度审计PASS（11项true，参数更新数0），点数保持81,323。iteration 10500、11000、12000 PLY/PTH/lighting完整保存，逐文件SHA见`output/1011-16-LH-static-jumpingjacks_DL_V2-T04-静态分流control-2k/training_completion_audit.json`。训练metrics最后一条step photo=0.01136760仅为训练值，不替代测试评测。当前训练结束、完整15帧Goal评测及四类目检待执行；尚无最终PASS/FAILED。


## iteration 12000最终评测与目检

完整15帧Goal评测四阶段全部exit 0，完整性PASS，600张PNG及4个各150帧MP4全部解码；fresh RGB parity PASS（八项含逐surfel visibility最大差0），训练/评测renderer来源、native .so SHA及稳定checkpoint核对通过。全参数、原始命令、渲染日志和逐帧数组见`output/1011-16-LH-static-jumpingjacks_DL_V2-T04-静态分流control-2k/evaluation_12000/README.md`。

- GS-render：6.91798432° / coverage 0.99938900；阈值<3.51°。
- Depth-normal：13.66069230° / coverage 0.96730850；阈值<7.35°。
- Mesh：8.87916647° / coverage 0.99651672；阈值<5.46°。
- completeness PASS，三项数值阈值均FAILED；visual integrity PASS，但存在明确几何缺陷。最终结论：**FAILED**。
- RGB：人物与圆台可辨，脚周出现灰白延伸／拖影，投射阴影边界有明显黑色伸展；不能把图像拟合视为准确几何。
- Alpha：人体和圆台总体连续，代表帧无大块缺失，脚与台面接触区域有薄外扩；80俯视圆台近满盘白。
- Normal：平台整体相对平坦但脚／小腿有异常抬高和延伸伪表面，帧80原Depth/Mesh误差在脚周突出；人体表面仍有法线偏差。
- 分离：人物与台面均为黄绿／橄榄色连续binary feature门控；保存零时变几何为显式约束，不代表学会运动分离。

代表图：`output/1011-16-LH-static-jumpingjacks_DL_V2-T04-静态分流control-2k/evaluation_12000/four_classes/four_classes_10_80_150.png`；四类视频分别在`evaluation_12000/four_classes/`。保留全部失败checkpoint、评测、统计及日志。


## 中间与最终checkpoint趋势

10500/11000仅geometry_only全15帧评测，四个中间pipeline完整性PASS，数值阈值FAILED；视觉目检按DEFERRED_INTERMEDIATE处理。三点指标（GS/Depth/Mesh MAE，单位度）为：

- iteration 10500: 6.60908960° / 12.77645969° / 8.57978193°；coverage 0.99927824 / 0.96719405 / 0.99575435。完整摘要和逐帧数据在`evaluation_10500/goal_mae_summary.json`及同目录。
- iteration 11000: 6.67739958° / 13.00208654° / 8.64794811°；coverage 0.99936131 / 0.96728266 / 0.99614938。完整摘要和逐帧数据在`evaluation_11000/goal_mae_summary.json`及同目录。
- iteration 12000: 6.91798432° / 13.66069230° / 8.87916647°；coverage 0.99938900 / 0.96730850 / 0.99651672。完整摘要和逐帧数据在`evaluation_12000/goal_mae_summary.json`及同目录。

10500→11000→12000三项误差均缓慢上升；control/split变化接近，最终12000的完整目检结论仍以`evaluation_12000/FINAL_REVIEW_COMPLETED.json`为准。
