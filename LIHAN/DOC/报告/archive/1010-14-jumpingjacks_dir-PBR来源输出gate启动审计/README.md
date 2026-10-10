# 1010-14 jumpingjacks dir 原生PBR来源输出gate完整Stage2训练计划

状态PREPARED，完整B08未开训，无B08正式MAE/质量结论。已读GOAL及B07最终GS16.81871120°/覆盖95.00328573%、质量FAILED，全部失败产物保存。只新增原生feature输出gate，source35k MLP32tensor/XYZ输入冻结；本轮source/model35k→55000完整20000步，仅Stage2，135TRAIN/15TEST、400×400、黑背景、相机time原值。

输入仅原RGB/自身RGBA Alpha、源35k预测和GT方向白色单位，不用GT材质/法线/粗糙度/深度/mesh/blend/强度/颜色初始化/loss/prior；GT仅离线评分。源35k来自旧转换，但150帧原/新/旧Alpha逐bit一致已披露，不重训Stage1。

Preset pbr_gt_directional_plane_native_direct_gate：继承B07 Direct/native kernel/sourceXYZ/15187 source plane/DC内部投影/原geometry/loss/lr与全局RGB拟合E。只新增feature optimizer组，source32 MLP tensor冻结、opacity/scale冻结，全部99827点保留。原feature初始化，geometry warmup2000/ramp1000/decay同步，gate_lr.001、max_source_logit_delta4、weakanchor.001，每4步Adam投影source±4。gate metadata/lighting marker绑定来源SHA、加载实际gate而非source，FrozenScene/GeometryRun/材料helper共享。canonicalXYZ .08m/q60°保持，额外plane world投影不受canonical边界。原生没有完整Adam状态，不声明无损resume。

202 CPU retry/真实20步/GPU五参数梯度/源32tensor与gate保存/实际SUN预算/共享buffer PASS；初次201项2个旧mock缺字段错误只修fixture后202通过，原生产训练core保持。全部150四类及4视频解码PASS，实际smoke质量FAILED，只验证功能，无正式MAE。完整run从35k独立训练，不沿用smoke或B07权重。

保存40k/50k/55k；全量四类150×4、HDR150×2048、方向72、6视频。全部15帧七MAE，原GS有向world normal GTalpha≥.5全前景pixel-weighted，coverage/common/missing180保持；Depth Z米不拟合，Mesh每个测试time所有135训练姿态模型预测深度独立融合，无GT/testpose重建。质量未通过前不commit/push/新目标分支。

## 完整命令与日志

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct_gate --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/write_results.log

## 四类目检与代表产物：PENDING

RGB/背景、Alpha、Albedo/静动态、Normal/Depth/Mesh及HDR/SUN待55k实际审核；model_mlp/renders_stage2/ours_55000/insights/{rgb,alpha,normal,albedo}各150及4视频；hdr150/directional72及2视频。inspection/{rgb_final_55000_comparison,four_categories_final_55000,relight_final_55000}.png；normal_depth/comparison_0010/0080/0150.png。质量/7°未PASS。473核心/GLM双snapshot SHA保留，所有失败权重/图片/浮点/日志保留。

## B08完整20k实际启动（2026-10-10T09:11:05.296325+09:00）

真实PID282299/startticks10140426，exec76498，source35k→55k20000步、Preset pbr_gt_directional_plane_native_direct_gate；当前生产core473双snapshot SHA已锁定，不续B07/smoke权重。后处理PID282335/exec89846及40k预览PID282338/exec88932绑定同一真实train PID/ticks/argv，不自动重试。9步全量render/15七MAE/复算/文档已准备；真实train_runtime/process_handles保存。当前仍运行，无B08最终MAE或质量/7°结论，未commit/push/新分支。

## B08真实启动只读审计（2026-10-10T09:12:42.793456+09:00）

launch_audit实际exit0。473核心及双snapshot逐SHA一致；cfg仅8差异（3路径/Preset与4 gate字段），geometry warmup2000正式保持，source35k PLY/MLP SHA/32 NN optimizer0与feature组启用、方向表byte同B07白色单位、source plane全部数组bit同B07/15187支持、150帧自身RGBA Alpha provenance PASS；真实PID282299/ticks10140426/argv/cwd仍一致。该审计0优化、无GT标签或B08正式MAE。40k/50k/55k只读检查排入audit_checkpoints_when_saved.py，实际PID282503/exec3502绑定同train，未保存前退出则失败不重启；9步后处理exec89846等待本轮完成。
