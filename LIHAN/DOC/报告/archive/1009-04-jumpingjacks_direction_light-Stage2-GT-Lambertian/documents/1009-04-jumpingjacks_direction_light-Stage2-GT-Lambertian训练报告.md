# 1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian训练报告

日期：2026-10-09（Asia/Tokyo），venus / lumimotion。状态：COMPLETED，训练、渲染、完整15帧MAE与最终审核完成，流程及产物PASS。

## 来源、输出与参数

- Stage1只读来源：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp`，iteration35000，99827个Gaussian。原7视频各150帧已解码，首中尾四类及静动态分离目检已回填1009-02。
- source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view`；原始GT：`/home/lihan/data/LH-data/danamic/jumpingjacks_direction_light`。135训练帧/15测试帧，测试10、20至150，time=frame/150，400×400，黑背景、真实Alpha。
- 输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian`；实际模型：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp`。Stage1原模型不覆盖。
- photometric_lambertian / lambertian_gt；35000→55000，新增20,000步。每Gaussian sigmoid albedo，LR0.001；几何、旋转、尺度、opacity、SH、法线和形变网络冻结，动态按真实time计算，无增点裁剪。
- 固定GT方向，0.8前景L1+0.2遮罩DSSIM+0.001初始材质先验；GT alpha>0.5监督RGB，不用GT albedo/normal/depth训练监督。保存35000/40000/50000/55000。

## GT方向与光强

原lights.json各帧两盏SUN：白色、角大小0、逐帧方向相同、表面指向光源，各energy3。兼容表gt_directions.json直接使用light_dir_world，不使用灯位或中心近似；保留两灯名字及energy总和6，源数据不修改。

沿用现有管线，仅135训练RGB、初始SH材质和冻结法线标定非负光强标量，随后固定；不使用测试RGB或GT材质拟合，不解释为Blender能量的物理恢复。标定写入model_mlp/calibration.json。

## 完整执行命令

执行目录：`/home/lihan/reproduce/LumiMotion-stage2-perlight`；CUDA_VISIBLE_DEVICES=0，OMP_NUM_THREADS=4，OPENBLAS_NUM_THREADS=4，PYTHONPATH为仓库根目录。

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --load_iter 35000 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --depth_ratio 0 --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode photometric_lambertian --loss_preset lambertian_gt --photometric_light_mode gt_directional --photometric_gt_lights_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_directions.json --photometric_light_intensity -1 --photometric_albedo_lr 0.001 --lambda_dssim 0.2 --lambda_photometric_albedo_prior 0.001 --lambda_photometric_light_smooth 0 --photometric_log_interval 100 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 55000
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/train.log`。

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/render.log`。

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/albedo --save_float
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/albedo.log`。

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/predicted_meshes
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/predicted_meshes.log`。

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/normal_depth --pred_mesh_manifest /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/predicted_meshes/manifest.json --save_arrays
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/normal_depth.log`。

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/commands/verify_all_mae.py
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/verify_all_mae.log`。

### export_gt_meshes

```bash
/home/lihan/reproduce/LumiMotion-stage2-perlight/output/Archive/1003-06-onlyclothV4-geometry_gt_eval/tools/blender-5.2.2-linux-x64/blender --factory-startup --disable-autoexec -b /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/scene_snapshot.blend --python /home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/export_blender_geometry.py -- --camera_json /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/camera.json --frames 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 --output_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_meshes --export_ply
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/export_gt_meshes.log`。

### prepare_depth_gt

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/commands/prepare_jump_depth_gt.py --mesh_manifest /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_meshes/manifest.json --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --transforms '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view/transforms_test.json' --output_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --resolution 2 --skip_baseline_audit
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/logs/prepare_depth_gt.log`。

## 评测和验收

完整15测试帧：Albedo线性RGB全前景MAE（GT alpha≥0.99内缩2像素，不做尺度拟合）；GS Render/Depth/Mesh Normal世界有向夹角MAE、覆盖率、共同有效像素；Expected/Median Z Depth MAE。GT Normal为Blender着色法线EXR，可能包含法线贴图。逐时刻TSDF各融合135训练相机、固定目标time，无GT或测试相机位姿参与；参数与env一致。GT深度从dir自己blend逐帧导出三角面生成，不复用env真值。

Roughness MAE不适用：Lambertian没有粗糙度或镜面输出，不能拿加载器默认参数评分。另存完整15帧RGB PSNR/SSIM/LPIPS、前景指标、150帧RGB/Alpha/Normal/Albedo、72帧转灯与GoldenBay150帧HDR示例（无对应GT）。

完成后检查checkpoint有限/冻结/来源哈希、完整视频解码、首中尾四类实际图、MAE独立CPU复算；回填指标、质量限制、PASS/FAILED，并归档证据和训练文档、同步本地分支。所有失败现场保留；自动流水线结束不等于最终目检验收。

## 正式启动核验

2026-10-09 19:56:27（Asia/Tokyo）启动正式流水线，主PID144878，训练PID144881；CPU真值生成PID144877。来源99,827 Gaussian各PLY属性有限，32形变Tensor有限；完整150帧GT方向表核验通过。135训练帧标定固定光强3.0964135864977482；初始化真实checkpoint只含albedo优化组，GT方向最大浮点差1.1921e-7，冻结指纹和来源哈希通过。

15份dir逐帧GT三角网格及400×400 Z深度已生成，真值流水线19:56:48结束。正式训练、最终渲染与MAE仍在进行；不将初始化检查写成最终验收PASS。

## 最终结果与归档

2026-10-09T20:12:46.051089+09:00完成最终审核。真实55000 checkpoint finite、仅albedo优化组、Adam20,000步、初末冻结指纹和光照表不变，来源35k PLY/形变哈希不变；固定光强3.096413586498。完整15测试帧MAE、独立CPU复算、GT审核与六视频完整解码通过，首中尾RGB/Alpha/Albedo-separation/Normal真实目检完成，流程与产物验收PASS，不代表达到模型质量阈值。

- Albedo MAE：0.23996675 linear RGB，覆盖率96.159989%。
- GS Render Normal MAE：21.52504346 degree，覆盖率92.937489%。
- Depth Normal MAE：27.60512758 degree，覆盖率88.323268%。
- Mesh Normal MAE：22.88577862 degree，覆盖率93.647351%。
- Expected Z Depth MAE：0.08970386 m，覆盖率92.937489%。
- Median Z Depth MAE：0.09224996 m，覆盖率92.937489%。
- GS Normal XYZ component MAE (supplementary)：0.17537827 unit normal XYZ，覆盖率92.937489%。
- Roughness MAE：不适用，Lambertian没有粗糙度输出。

RGB PSNR 17.01860567 dB，SSIM 0.88039081，LPIPS 0.12505545。

质量结论：本次训练与完整评测执行成功，但RGB重建和材质恢复质量较差：中间帧明显偏暗，albedo总体偏亮、保留照明结构；冻结几何仍有平台突起、肢体碎片和漏覆盖。此结果不能说明GT Lambertian只训练材质能够修复Stage1几何。

完整[评测报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian-各项MAE与三类Normal评测报告.md)，命令/日志/代表图/视频、SHA256清单及训练文档副本随1009-04归档附件保存；原checkpoint、浮点数组和GT/预测Mesh保留。
