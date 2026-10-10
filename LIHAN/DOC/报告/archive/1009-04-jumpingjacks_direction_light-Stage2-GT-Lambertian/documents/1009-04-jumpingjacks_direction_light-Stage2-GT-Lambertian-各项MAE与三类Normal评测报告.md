# 1009-04 jumpingjacks_direction_light GT Lambertian Stage2 各项MAE与三类Normal评测报告

日期：2026-10-09（Asia/Tokyo），venus / lumimotion。归档时间：2026-10-09T20:12:46.051089+09:00。

**状态：COMPLETED；35k→55k训练、完整15测试帧评测、独立CPU复算、视频解码和首中尾人工检查完成，流程与产物验收PASS。** PASS不表示模型达到质量阈值。

## 来源与设置

Stage1来源1009-02原版SH/35k，99,827 Gaussian；本次Stage2使用当前仓库photometric_lambertian / lambertian_gt，累计55k、新增20,000步。source为1009-02/data_view，135训练帧/15测试帧，测试0010、0020至0150，time=frame/150，resolution2、400×400、黑背景。来源模型、相机世界坐标保持不变，不做尺度或姿态拟合。

- 模型：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp`；checkpoint：`stage2_photometric/iteration_55000/checkpoint.pth`。
- 完整输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian`。
- 原始GT：`/home/lihan/data/LH-data/danamic/jumpingjacks_direction_light`。
- [训练文档](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian训练报告.md)；Stage2代码提交a9e4d1061fec9986d82362f78b42c0c269e6146e，源码快照与工作区差异随证据保存。

正式流水线19:56:27启动，最终55000权重20:07:32保存；训练循环实际耗时646.552秒，新增20,000步。

几何、尺度、旋转、opacity、SH、法线及完整形变网络冻结，仍按真实time求动态形变，仅训练跨时间共享sigmoid albedo，LR0.001。0.8前景L1+0.2遮罩DSSIM+0.001初始材质先验，无GT材质/法线/深度训练监督。Adam实际新增20,000步，冻结指纹、方向表、来源哈希检查通过。

两盏逐帧重叠白色SUN各energy3，角大小0，GT方向直接使用原light_dir_world（表面指向光源），不用灯位中心近似；兼容表保留两灯及energy总和6。固定着色光强3.096413586498来自135训练RGB、SH初值及冻结法线标定，不用测试帧或GT材质拟合，不当作Blender能量的物理恢复。简化Lambertian没有投射阴影、镜面或间接光，albedo可能吸收这些误差。原版Stage1颜色形变不参与此材质着色。

## 各项MAE：完整15测试帧

| 指标 | MAE（有效像素加权） | 覆盖率 |
| --- | ---: | ---: |
| Albedo MAE | 0.23996675 | 96.159989% |
| GS Render Normal MAE | 21.52504346° | 92.937489% |
| Depth Normal MAE | 27.60512758° | 88.323268% |
| Mesh Normal MAE | 22.88577862° | 93.647351% |
| Expected Z Depth MAE | 0.08970386 m（8.9704 cm） | 92.937489% |
| Median Z Depth MAE | 0.09224996 m（9.2250 cm） | 92.937489% |
| GS Normal XYZ component MAE (supplementary) | 0.17537827 | 92.937489% |
| Roughness MAE | 不适用：Lambertian无粗糙度输出 | — |

Albedo在线性RGB评分，GT alpha≥0.99、内缩2像素，全部内部前景，三个通道绝对误差均值；不做材质分区或尺度拟合，漏检预测记0，覆盖率按预测alpha≥0.9诊断。原16-bit sRGB PNG经既有PIL入口转8-bit，原GT不改写。

Normal为归一化世界坐标有向夹角，GT alpha≥0.5全前景且GT三角面命中；GS要求预测alpha≥0.5且法线有效，Depth Normal额外要求中心差分邻域；Mesh按自身第一射线交点有效，不另要求GS Alpha。Normal GT为原32-bit Blender着色Normal EXR，可能含材质法线贴图；不是纯几何法线真值。三类均使用同一参考，缺失计入覆盖率分母。

Depth为正camera Z与dir自己的Blender逐帧求值三角面射线GT比较，METRIC、meters_per_world_unit=1，无ICP或尺度/偏移拟合。主统计按有效像素汇总，逐帧等权结果另存。

Roughness明确NOT_APPLICABLE：本模型没有粗糙度参数/输出，不能将加载器默认粗糙度当作预测评分。

## 三法线共同有效像素

共同有效像素366157 / 415433，覆盖率88.138641%。

| 方法 | 共同像素MAE |
| --- | ---: |
| GS Render Normal | 20.55943719° |
| Depth Normal | 27.55407872° |
| Mesh Normal | 21.87872001° |

GS缺失法线按180°计入的补充MAE为32.71735420°。Mesh为15个独立时刻的模型预测TSDF，各135训练相机在目标time融合Expected Z；不使用测试相机位姿或GT几何。参数固定voxel0.02、sdf_trunc0.08、depth_trunc20、预测alpha0.5，与env一致，未按GT误差调参。Mesh分数受预测深度与固定提取参数共同影响；冻结几何的Stage2不能据这些分数宣称改善Stage1几何。

## 完整RGB评测

15帧各自相机/time/已知GT方向，独立render_stage2入口：PSNR 17.01860567 dB、SSIM 0.88039081、LPIPS 0.12505545；前景PSNR 10.15766513 dB、遮罩前景包围框SSIM 0.74756655、LPIPS 0.26391009。这些是全15帧指标，不能混同Stage1日志的少量相机子集。

RGB/Alpha/Normal/Albedo视频各150帧，转灯72帧，GoldenBay HDR150帧（2048样本），六视频完整解码；HDR与转灯为响应示例，无对应重光照GT。渲染目录model_mlp/renders_stage2/ours_55000。

## 实际目检与质量限制

本次训练与完整评测执行成功，但RGB重建和材质恢复质量较差：中间帧明显偏暗，albedo总体偏亮、保留照明结构；冻结几何仍有平台突起、肢体碎片和漏覆盖。此结果不能说明GT Lambertian只训练材质能够修复Stage1几何。

- RGB：0010/0150姿态可辨认，但人体颜色斑驳、平台无法重现对应阴影；0080人物和平台明显偏暗、局部呈黑块，15帧PSNR仅17.0186 dB。
- Alpha：人物和平台大体覆盖，手臂/头部/腿脚边缘有遗漏及多余残影；0080与真值轮廓和腿脚位置有偏移。
- Albedo / separation：平台预测接近白色并抹去GT棋盘格；衣服、腿部、面部仍有强照明/阴影/褶皱残留。完整GT内部前景线性RGB平均偏差R=+0.138632、G=+0.109958、B=+0.190370，总体偏亮。静动态分离沿用冻结Stage1，平台红绿混杂未修复。
- Normal与Depth：GS人体和平台法线斑驳，Expected/Median Z显示脚周围平台起伏、突起和残影，0080人物轮廓/腿部深度偏差明显。Depth Normal放大中心差分碎片；Mesh较Depth平滑，但仍有平台凸起、腿脚缺面及轮廓偏移。在共同像素上Mesh误差21.8787°仍高于GS20.5594°、低于Depth27.5541°。GT着色法线与几何差单列诊断，不计作模型自洽误差。
- 重光照：72帧转灯有真实亮度响应，替换光照不改变Alpha；背向光时接近黑色。HDR首中尾可辨认但有橙色偏色及照明/伪几何残留；无对应HDR或转灯GT。

首中尾代表图审核不等于全部150时刻人工逐帧查看；MAE及数值复算覆盖所有15测试帧。在共同有效像素上Mesh法线误差高于GS、低于Depth Normal，不宣称新增Mesh指标改善GS。

![三类法线首中尾](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/three_normals_first_middle_last.png)

- [RGB与GT首中尾](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/rgb_comparison.png)
- [四类首中尾](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/four_categories.png)
- [转灯与HDR](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/relight.png)
- [Albedo首中尾对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/albedo_first_middle_last.png)
- [Albedo真值对照0010](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/albedo/iteration_55000/0010_comparison.png)
- [法线/深度/误差0010](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/normal_depth/comparison_0010.png)

## 审核与归档证据

独立CPU从浮点数组复算Albedo全GT内部前景、三法线各自/共同有效像素和角度、Expected/Median Z与覆盖率；与全量及逐帧结果一致。GT从dir自己blend导出，源blend及模型哈希未改变；15帧相机与PLY/NPZ求交真值审计通过。检查实际checkpoint有限、Adam20,000步、仅albedo优化组、固定GT方向/光强、初末冻结指纹一致。

- [汇总JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/metrics_summary.json)、[CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/metrics_summary.csv)、[三法线逐帧](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/normal_depth/normal_metrics_per_frame.csv)。
- [独立复算](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/independent_audit.json)、[GT审核](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/gt_audit.json)、[最终checkpoint](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/checkpoint_55000.json)、[完整视频与产物审核](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/artifact_checks.json)、[人工目检](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/inspection/visual_review.json)。
- [文件SHA256清单](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/archive_manifest.json)：文档、指标、逐帧表、命令、训练/渲染/评测源码快照、日志与代表图已归档。完整训练权重、GT及预测Mesh、浮点图保留原输出路径，未放入小型文档附件。

## 完整训练、渲染与评测命令

工作目录`/home/lihan/reproduce/LumiMotion-stage2-perlight`，CUDA_VISIBLE_DEVICES=0、OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=4，Python lumimotion；Blender5.2.2用于逐帧真值几何导出。

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

