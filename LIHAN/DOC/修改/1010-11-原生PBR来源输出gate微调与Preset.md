# 1010-11 原生PBR来源输出gate微调与Preset

## 修改依据和范围

已先读GOAL，B07完整GS16.81871120°/覆盖95.00328573%，质量与7°失败，保留全部失败产物。新增pbr_gt_directional_plane_native_direct_gate，只开放PLY单维feature的原生输出乘法gate，源35k MLP全部冻结/sourceXYZ输入锚定不变，全部99827点保留。B06真实只读gate梯度存在、B07source joint贡献图有动态near-plane平台贡献；该source组不是GT语义，不做强制静态/删点/评分域筛选。

Stage2OptimizationParams默认flagFalse，新增gate_lr.001、max_source_logit_delta4、anchor.001。Trainer只对gate候选保留feature optimizer组并冻结MLP全部组，允许Deform输出乘法对feature可导；source输入仍detach，MLP hidden不可导。warmup2000/ramp1000/decay同geometry，只在几何启用后加source logit归一化平方anchor；每4步Adam后将raw feature投影到source±4。初始feature来自真实原35k且逐bit校验，不重置/二值硬化/筛点。原生sigmoid(2*feature)、NN结构、GGX/BVH/SH辅助loss、Direct预算/材质/原geometry/loss与评分均保持。固定source NN32tensor不得变化。

scene/pbr_feature_gate.py保存本地source PLY/MLP SHA、metadata配置与无GT协议；checkpoint lighting marker绑定metadata SHA，加载核验source权重所有tensor、gate shape/finite/trust及配置。FrozenScene统一校验，GeometryRun/材质helper记录同metadata SHA，渲染/评测直接用保存PLY feature，禁止退回source gate。其它Preset旧no_grad/optimizer范围保持，Stage1无改动。原生Adam未完整保存，不声明无损resume。

训练只用原RGB/自身RGBA Alpha、相机time、原35k预测和GT方向白色单位灯；不用GT albedo/normal/roughness/depth/mesh/blend/光强/颜色初始化/loss/prior。GT仍仅离线评分。

修改文件：scene/pbr_feature_gate.py、arguments/stage2.py、scripts/{train_stage2,loss_stage2,stage2_runtime,eval_geometry_gt,eval_albedo_gt,eval_roughness_gt}.py；tests/test_pbr_feature_gate.py和共享Runtime测试。Trainer保留CRLF，diff --check用cr-at-eol设置通过。


## 实际冒烟验收（2026-10-10T09:08:06.181380+09:00）

202 CPU discovery实际exit0；真实20步Trainer实际exit0并关闭，35k→35020保存全部99827点。新增gate参数组真实训练68101点raw feature发生变化，max delta1.85966492e−5，低步ramp仅验证更新；opacity/scale不变，source32 MLP tensor完全原值，初始feature原值，保存source metadata/lighting marker并共享加载核验。无额外诊断优化、无GT材质/法线/几何/能量/颜色训练输入。

真实native三TRAIN0001/75/149 SUN18843/15904/19137条，all-eligible alpha>.9；albedo/roughness/XYZ/q/feature梯度非零finite。gate photo L2 .00085692690/.00269499072/.00030556315。3SUN/with-indirect/历史预算variant同checkpoint零优化全部finite，参数SHA保持，不宣称不同variant已独立训练。GeometryRun alpha/worldnormal/Expected Z全buffer一致、目标timeoverride一致；albedo/roughness helper一致。

150×四类及4视频全解码PASS，实际查看0001/0075/0150 RGB/Alpha/Normal/Albedo及三HDR16/SUN。功能PASS、实际质量FAILED_SMOKE_ONLY：人物破碎/背面暗、材质烘光、平台脚周伪起伏仍明显；不作为正式全15七MAE/7°，未commit/push/新分支。正式候选须从35k独立完整20k。

首次CPU201项有2个旧GeometryRun mock缺少新增FrozenScene字段的错误，失败日志和原source snapshot保留；真实20步Trainer生产源码未改，之后仅修mock/增加metadata用例，202 retry全PASS。code_manifest_train_launch与现manifest唯一差异是tests/test_geometry_eval_source_runtime.py，生产核心完全相同。

完整训练命令：
```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-15-dir-PBR来源输出gate/model' --load_iter 35000 --iterations 35020 --save_iterations 35020 --test_iterations 35020 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct_gate --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_geometry_warmup 0
```

诊断命令commands/{check_gate_checkpoint,check_pbr_gradients,check_shared_buffers,check_geometry_buffers,check_material_buffers,make_comparison,finish_smoke}.py；对应日志logs/{gate_checkpoint,pbr_gradients,shared_buffers,geometry_buffers,material_buffers}.log；渲染命令python -m scripts.render_stage2 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-15-dir-PBR来源输出gate/model_mlp --load_iter35020 --task insights --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-15-dir-PBR来源输出gate/insights，日志logs/insights.log。图inspection/four_categories.png、relight.png；JSON和视频完整路径见inspection/visual_review.json。
