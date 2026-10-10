# 1010-18 dir 原生PBR训练RGB来源局部法线 Preset冒烟

状态：226 CPU实际exit0；真实20步Trainer/GPU/保存加载/共享buffer与实际四类/HDR/SUN待完成。2026-10-10T10:58:17.196226+09:00

已先读GOAL，B09完整20k/15七MAE/实际质量FAILED，原GS Normal15.94231363度/覆盖95.47845260%，1085文件SHA三checkout保留，未commit/push/新分支。独立训练RGB原型135 TRAIN、全99827点，有29621非投影来源点约束（静态17389/动态12232），其source/原RGB SHA与独立CPU全点LS/confidence复算PASS，首路径失败保留。约束由原RGB/自身Alpha/源35k/相机time/单位灯方向派生，绝不来自GT normal/material/depth/roughness/mesh/blend/灯颜色能量；GT只最终离线评分。

本Preset pbr_gt_directional_source_rgb_normal继承B09概率gate/native Direct/原GGX/BVH/SH辅助/material/geometry/lr/warmup/ramp信赖域与plane15187，新增source-local quaternion运输和lambda2 RGB轴约束。source q35k独立常量、原NN32tensor固定；对当前p gate查询原NN的world qsource，按R_world_source(p) R_canonical_source^T R_canonical_current运输；source初值raw deformation逐bit原值。只修改真实Gaussian渲染world quaternion，XYZ/depth来自原位置与真实raster，plane继续在运输后应用。原RGB拟合canonical轴约束平均1-dot，原photo/SH/Alpha/预测几何损失及GT评分不变；可靠性仅附加项权重，不删点/帧/难评分像素，existing plane支持不重复施加。B09概率gate前向/保存仍原native PLY sigmoid，无强制静态。

create/load均核验local35k来源/NN无梯度、135 TRAIN无TEST、全部观测图SHA/camera/白色单位方向、input协议/SHA与完整LS数组复算；模型复制输入asset+metadata并保存source quaternion/轴/confidence本地npz，checkpoint绑定其metadataSHA。FrozenScene/周期预览/GeometryRun/Mesh/材质/HDR/SUN共享运输入口，模型原数值/资产任何变化报错。canonical .08m/q60度保持，不称保证world位置最大.08（可学gate改变原NN motion）。原生Adam仍无完整状态，不声明无损resume。

独立原35k→35020，20步，仅冒烟warmup0/ramp1000；正式完整必须从原35k另训20k、恢复warmup2000，不续本smoke或B09。当前476核心两snapshot锁定；新增15 CPU实际PASS（初次CPU矩阵引用repo CUDA工具造成device mismatch失败保留并修正测试独立CPU矩阵），全部226 PASS。不提前声称图像/7度通过。

## 完整训练命令

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-18-dir-PBR训练RGB来源局部法线/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-18-dir-PBR训练RGB来源局部法线/model' --load_iter 35000 --iterations 35020 --save_iterations 35020 --test_iterations 35020 --render_mode original_ir --loss_preset pbr_gt_directional_source_rgb_normal --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-18-dir-PBR训练RGB来源局部法线/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_geometry_warmup 0 --pbr_rgb_normal_prior_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-17-dir-训练RGB法线约束原型/candidate_assets'
```

日志logs/train.log；来源source35k/model_mlp/point_cloud与deform SHA保持。后续命令commands/check_gate_checkpoint.py、check_pbr_gradients.py、check_shared_buffers.py和insights渲染，输出inspection与insights；真实四类首中尾/HDR16/SUN目检和4视频全帧解码待完成，最终PASS/FAILED待填。正式15七MAE尚未测。

## B10实际冒烟完成（2026-10-10T11:08:01.965361+09:00）

226 CPU、真实20步Trainer及保存加载/GPU/共享buffer/insights均实际exit0且会话关闭；476核心和双snapshot SHA保持。35000→35020，全部99827点finite，opacity/scale及来源NN32tensor冻结，概率gate改变69981点、maxdelta1.85966491699e-5；raw变化70033/max74.534210205只是有限逆sigmoid编码。独立源RGB候选29621点，lambda2，canonical轴loss0.102662175894、其q梯度L2 0.00525612570345非零，三TRAIN photo q梯度L2 .022683673/.602086246/.024002507，概率 .015508413/.038564317/.013200786；albedo/roughness/XYZ/q/gate均finite且非零。不能从全局梯度L2推断完整训练质量。

真实DirectSUN/HDR16梯度预算、三SUN/间接/legacy只读0 optimizer检查PASS；共享Geometry alpha/GSnormal/ExpectedZ误差0，material albedo0/roughness≤5.96e−8，plane真实世界位置约2.7e−8m/轴cos≥.99999964。source-local运输应用在真实Gaussian q，所有入口同顺序先运输再plane。4类150×4及4视频各150帧全部解码；Depth/Mesh及全15七MAE尚未在smoke生成或评分。

- RGB黑色背景正常，首尾人物和平台保留；中帧人物接近黑色但Alpha主体完整，不能解释为缺失。首尾人物/裤腿有明显黑斑，平台足周黑灰杂点，20步画质不通过。
- 首中尾人物和完整椭圆平台均有Alpha；手臂、身体及脚周有少量模糊边缘和孤立点。没有删去难点或平台。
- 人物首尾和中帧均可见，但明暗及纹理斑块混入反照率，手臂/裤腿碎斑，平台颜色受阴影污染；不认定已经完成材质分解。
- 实际渲染GS法线可见人物和平台，首尾平台主要紫蓝，中帧人物黄绿且表面杂色，足周法线噪点明显。此拼图只检查GS法线，未额外生成Depth/Mesh，不称其已验收。
- 同首中尾的方向光与HDR16均通过共享入口有实际亮度/色彩响应；HDR有橙色轮廓与高光、平台蓝色响应。方向光中帧人物近黑，不能仅据此宣称代码异常，也不能据重光照有响应认定几何正确。

最终功能PASS，画质FAILED_SMOKE_ONLY，原GS Normal<7度NOT_EVALUATED。GT方向白色单位；新轴来自原135TRAIN RGB/自身Alpha和原35k预测，不是GT normal；GT材质/法线/粗糙度/深度/mesh/blend/灯能量颜色全部未参与训练/初始化/约束。全部点/帧/原photo/评分保留。正式B10需独立原35k→55k完整20k，恢复warmup2000/ramp1000；不用本smoke或B09学得参数继续训练。不commit/push/新目标分支。
