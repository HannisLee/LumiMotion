# 1010-13 dir PBR原生CUDA与Preset冒烟

最终功能PASS；画质FAILED_SO_FAR，仅20步功能检查，不是全量训练/7°结果。Preset pbr_gt_directional_plane_native，固定35k source→35020，warmup显式0仅smoke；完整B06恢复2000。原RGB/自身RGBA/相机/time/source预测/GT方向，无GT材质、Normal、几何或灯强度/颜色。训练exec59239实际exit0；20步保存/加载绑定marker。Source35k未修改，不重跑Stage1。

187 CPU discover实际exit0（exec96419）。真实三TRAIN帧0001/0075/0149 GGX/BVH photo对albedo/roughness/xyz/rotation梯度均非零finite，0额外optimizer，sourceSHA/fingerprint不变。GeometryRun三测试帧0010/0080/0150 alpha/worldnormal/ExpectedZ逐buffer maxerror0，targettime .375 override一致，453来源SHA核验。真实材质首中尾albedo maxerror0、roughness黑背景0/白背景5.96046e−8。四类150帧和4视频、HDR16/替换direction三帧生成且已实际看首中尾拼图；功能PASS不代表画质PASS。

## 四类实际目检

- 黑背景正常，人物平台存在；人物偏暗/斑驳，脚周伪结构仍有。20步源初始化不能作完整训练结果。
- 主体及平台基本覆盖，边缘仍少量残影。
- 光照仍烘入，脚周污染。
- 0075平台不再整体反向；人体与脚周局部轴仍斑驳。

HDR/方向光：两种灯替换均响应且背景正常；HDR16噪声和斑驳不作质量准确性。

## 完整命令与日志

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/model' --load_iter 35000 --iterations 35020 --save_iterations 35020 --test_iterations 35020 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_geometry_warmup 0
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/logs/train.log

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/commands/check_geometry_buffers.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/logs/check_geometry_buffers.log

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/commands/check_pbr_gradients.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/logs/check_pbr_gradients.log

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/commands/check_material_buffers.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/logs/check_material_buffers.log

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/model_mlp' --load_iter 35020 --task insights --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/insights'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-13-dir-PBR原生CUDA与Preset冒烟/logs/insights.log；四类目录insights/insights/，代表inspection/four_categories.png与relight.png。
