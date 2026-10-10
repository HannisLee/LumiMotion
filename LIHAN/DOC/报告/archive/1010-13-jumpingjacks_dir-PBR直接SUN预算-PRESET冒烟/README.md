# 1010-14 原生SUN真实预算与Direct Preset冒烟

状态PREPARED。194 CPU discovery实际exit0/OK。仅功能验证，不是全量MAE或质量PASS。

新Preset pbr_gt_directional_native_direct / pbr_gt_directional_plane_native_direct，原生CUDA+GGX/BVH，训练/重光照关闭来源SH间接近似，native SH辅助RGB损失保留。实际预算开关True：每像素L条SUN，若间接光启用再加diffuse_sample_num。显式HDR仍采用原预算。旧Preset默认预算不变。新增非法模式/零预算/不足单像素预算拒绝。

固定Stage1 35000→35020，geometry warmup0仅smoke；正式仍2000/1000、独立35k→55k。仅原RGB/自身Alpha、相机/time、Stage1预测、GT方向白色单位表；不读GT材质/normal/geometry/color/intensity。

## 命令与路径


```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-14-dir-PBR原生SUN预算与DirectPreset/model' --load_iter 35000 --iterations 35020 --save_iterations 35020 --test_iterations 35020 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_geometry_warmup 0
```

日志logs/train.log；model_mlp/iteration35020；四类/HDR/转灯、共享buffer、真实trace形状与梯度待完成。

完整source快照471文件见code_manifest.json，CUDA源与GLM继承B06；结果/日志全部保留。最终功能PENDING，七MAE/质量/7°未评测。

## 实际冒烟结论（2026-10-10T08:13:31.388800+09:00）

194 CPU/20步真实Trainer/三帧all-eligible SUN及L3/with-indirect真实光线预算、材质与XYZ/q梯度、GeometryRun与材质helper逐buffer一致均PASS。参数SHA不变、0额外诊断优化，源码471 SHA核验。首次诊断重复backward图失败、保存后修正仅诊断图重建，retry实际exit0。35020 checkpoint flags/native来源绑定通过。

四类150×4和4视频逐帧解码PASS，已实际看首中尾RGB/Alpha/normal/albedo及SUN/HDR16。实际质量FAILED_SMOKE_ONLY：人物破碎/背面暗、albedo烘光、脚周伪几何仍有；不是完整MAE/7°，不commit/push。功能允许下一步独立完整20k训练，不沿用20步权重。

命令：check_pbr_gradients.py（失败和retry日志）、check_shared_buffers.py依次调用check_geometry_buffers.py/check_material_buffers.py；render_stage2 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-14-dir-PBR原生SUN预算与DirectPreset/model_mlp --load_iter35020 --task insights --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-14-dir-PBR原生SUN预算与DirectPreset/insights；日志logs/{pbr_gradients_retry,geometry_buffers,material_buffers,shared_buffers,insights}.log。代表图inspection/four_categories.png、relight.png；各检查JSON同inspection。
