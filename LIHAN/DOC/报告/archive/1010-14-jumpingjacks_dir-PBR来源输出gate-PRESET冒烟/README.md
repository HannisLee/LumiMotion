# 1010-15 dir PBR来源输出gate功能冒烟

状态PREPARED。固定原35k→35020真实20步，warmup0仅冒烟，正式仍2000。原值feature初始化，source MLP所有32tensor冻结，只新加feature optimizer组/乘法梯度；raw-logit source±4投影及.001弱anchor，lr.001。其余B07 Direct Native/来源XYZ/plane/材质/原评分不改。原RGB/自身Alpha/相机time/source预测/GT单位方向only，无GT材质/法线/几何/能量/颜色训练输入。全部99827点保留。

完整命令见commands/train.sh和train_argv.json；CPU日志logs/cpu_tests.log，真实训练logs/train.log。source/core双快照及checkpoint/sourceNN/gate差值、共享几何/材质buffer检查待完成。四类/HDR/SUN图及视频目检待完成。此20步冒烟不产生或宣称完整七MAE/7°。失败权重/图/日志保存，未commit/push。

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
