# 1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源训练计划

服务器：ushas；环境：lumimotion-ushas；批次编号：1011（用户指定）。

原始数据：`/mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2`。训练source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`。MAE source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`。

模型：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp`；输出及日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源`。150帧，135训练／15测试，测试10、20至150；resolution=2（400×400），黑背景。

只读01组35000步Stage1来源，Stage2新增20000步至55000；独立输出，不覆盖原生模型。

输入Alpha沿用既有转换器的albedo Alpha；不以GT albedo RGB、normal、depth、roughness或骨骼作为训练监督。原生组不读取GT光照；05/06使用GT方向，05两盏单位白光×π、06只用训练RGB标定非负光强；不把标定值解释为Blender原能量。

调度按GPU实际空余显存和进程数，每卡最多2个计算任务；05/06的依赖等待不占GPU。原生源相机平移符号仅在输入中预补偿，MAE使用原始世界相机，不改变训练算法。

全15测试帧：Albedo线性RGB MAE、Roughness参数MAE、GS Render／Depth／Mesh Normal角度MAE（度）、Expected／Median Z Depth MAE（世界单位，并记录单位比例）；同时保存覆盖率、共同像素法线分数、逐帧CSV、NPZ、代表图、日志与独立复算。Stage1记录几何MAE，SH无物理Albedo/Roughness。Lambertian无Roughness参数，记N/A。

RGB／Alpha／Normal／分离四类目检与最终PASS/FAILED必须在实际验收后回填，启动前状态，实际验收见下方结果回填；失败产物全部保留。

## stage2

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.train_stage2 --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --stage1_model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/stage1_frozen' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 55000 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode photometric_lambertian --loss_preset lambertian_gt --photometric_light_mode gt_directional --photometric_gt_lights_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/gt_directions.json' --photometric_light_intensity -1 --photometric_albedo_lr 0.001 --lambda_dssim 0.2 --lambda_photometric_albedo_prior 0.001 --lambda_photometric_light_smooth 0 --photometric_log_interval 100
```

日志：`logs/stage2.log`。

## render

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.render_stage2 --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --load_iter 55000 --task all --hdr_samples 2048 --hdr_filepath /home/han.li/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

日志：`logs/render.log`。

## eval55000_albedo

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_albedo_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --iterations 55000 --gt_albedo_dir /mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2/albedo --split test --output_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/albedo' --save_float
```

日志：`logs/eval55000_albedo.log`。

## eval55000_predicted_meshes

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.extract_geometry_mesh_dynamic --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/predicted_meshes'
```

日志：`logs/eval55000_predicted_meshes.log`。

## eval55000_normal_depth

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_geometry_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --gt_root /mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2 --gt_depth_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/gt_depth' --split test --resolution 2 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/normal_depth' --pred_mesh_manifest '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/predicted_meshes/manifest.json' --save_arrays
```

日志：`logs/eval55000_normal_depth.log`。

## eval55000_verify_all_mae

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/audit_1011_mae.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000' 55000
```

日志：`logs/eval55000_verify_all_mae.log`。


## 训练结果回填（2026-10-11 02:42:22 UTC）

最终结论：**FAILED**。训练、15帧全部适用MAE和六段视频解码完成；实际GT材质对比发现投影残留与明显材质偏差，质量验收FAILED。

判定口径：未预先约定数值阈值，MAE不作任意门槛判定。最终FAILED依据必要训练阶段缺失，或GT对比中明确可见的投影残留／材质污染。视频完整解码与首、中、尾图像抽查分别记录，未声称逐帧人工检查全部视频。

- 原始训练／渲染执行结论：`AWAITING_VISUAL_REVIEW`；失败标记与退出码证据：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/review_1011/original_terminal_result.json`；补评测开始时状态快照：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/review_1011/original_state.json`。
- 实际有效检查点：55000；model：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp`；Stage1冻结副本：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/stage1_frozen`。
- 原生输入source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view`；MAE输入source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`；渲染source由模型cfg_args读取。
- 渲染目录：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000`；评测目录：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000`。
- 本次只评测与渲染已有检查点，未重启训练，未删除或覆盖原始失败结果。
- 目检拼图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/review_four_classes.png`；来源与视频帧索引：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/review_four_classes_sources.json`。
- 完整视频解码、检查点有限性：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/artifacts.json`；全15测试帧MAE独立复算：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/mae_summary.json`。

### 定量结果

| 编号／迭代 | Albedo MAE | Roughness MAE | GS Normal ° | Depth Normal ° | Mesh Normal ° | Expected Z | Median Z |
|---|---|---|---|---|---|---|---|
| 06／55000 | 0.20741376 | N/A | 10.50885900 | 18.51896640 | 14.10656756 | 0.03192593 | 0.03039268 |

法线单位为度；深度是相机Z世界单位，单位比例与逐帧统计详见normal_depth/metrics.json。Albedo采用线性RGB前景核心区域，不作拟合后替代分数；Roughness为r参数。

### 覆盖率与共同像素

| 指标 | 有效像素覆盖率 | 共同像素法线MAE ° |
|---|---|---|
| gs_render | 99.84401853% | 10.20114250 |
| depth_normal | 96.63836849% | 18.51738002 |
| mesh | 99.36222959% | 13.93375559 |
| expected | 99.84401853% | N/A |
| median | 99.84401853% | N/A |

### RGB、Alpha、Normal、分离与材质目检

- **RGB**：相机轨迹首／中／尾人物外观完整，但地面偏白、背光细节弱；全图PSNR22.8764，前景16.5746。
- **Alpha**：人物和圆盘轮廓整体闭合，无大面积孔洞；细手指边缘略软。
- **Normal**：01冻结Stage1几何完全继承；地面有明显块状法线起伏，与GT平面不符。所有几何MAE与01的35k一致，不能称为Stage2几何改善。
- **分离**：真实查看01 Stage1分离图作为继承证据；静态人物与地面主体绿色，但腿部和投影区域有红色污染；本组没有额外训练分离模块。
- **材质与重光照**：GT帧10对比中预测Albedo残留黑色投影、地面偏白；HDR改变色调但保留投影。转灯只验证亮度变化与Alpha不变，不能证明获得正确随灯移动的阴影。无遮挡Lambertian积分缺少可见性、镜面和间接光。

目检范围：本组相机轨迹RGB／Alpha／Normal／Albedo视频的索引0、75、149；Albedo GT对比测试帧10和150；继承01固定相机分离视频索引0、75、149。另外查看几何GT对比帧10、方向转灯与HDR视频首／中／尾代表图。

### 原始失败及本次补评测

训练及全部适用MAE已完成；原来的AWAITING_VISUAL_REVIEW已完成实际图像目检。Lambertian没有Roughness参数，且HDR入口是无可见性漫反射球面积分；不能认作完整PBR。分离仅继承01 Stage1，不存在新训练的Stage2分离模块。

### 实际执行命令、退出码与日志

#### stage2

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/logs/stage2.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.train_stage2 --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --stage1_model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/stage1_frozen' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 55000 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode photometric_lambertian --loss_preset lambertian_gt --photometric_light_mode gt_directional --photometric_gt_lights_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/gt_directions.json' --photometric_light_intensity -1 --photometric_albedo_lr 0.001 --lambda_dssim 0.2 --lambda_photometric_albedo_prior 0.001 --lambda_photometric_light_smooth 0 --photometric_log_interval 100
```

#### render

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/logs/render.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.render_stage2 --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --load_iter 55000 --task all --hdr_samples 2048 --hdr_filepath /home/han.li/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

#### eval55000_albedo

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/logs/eval55000_albedo.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_albedo_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --iterations 55000 --gt_albedo_dir /mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2/albedo --split test --output_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/albedo' --save_float
```

#### eval55000_predicted_meshes

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/logs/eval55000_predicted_meshes.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.extract_geometry_mesh_dynamic --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/predicted_meshes'
```

#### eval55000_normal_depth

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/logs/eval55000_normal_depth.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_geometry_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --gt_root /mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2 --gt_depth_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/gt_depth' --split test --resolution 2 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/normal_depth' --pred_mesh_manifest '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/predicted_meshes/manifest.json' --save_arrays
```

#### eval55000_verify_all_mae

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/logs/eval55000_verify_all_mae.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/audit_1011_mae.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000' 55000
```

### RGB测试指标

```json
{
  "experiment": 6,
  "iteration": 55000,
  "protocol": "各帧自身相机/形变/光照；learned 测试光照只由训练方向插值；前景SSIM/LPIPS为遮罩后包围框指标",
  "psnr": 22.8763667308919,
  "ssim": 0.9354594588279724,
  "lpips": 0.0826714778939883,
  "psnr_foreground": 16.574558992493866,
  "ssim_foreground_bbox": 0.8257198452949523,
  "lpips_foreground_bbox": 0.20742960919936498,
  "light_angle_degrees": 0.0
}
```

### 保留产物与代表文件

转灯每帧平均亮度范围：0.00000000—0.21157339；Alpha不变：True。此项只证明有光照响应，不证明投影或材质正确。证据：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000/directional/relight.json`。

- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000/directional/rotating_directional.mp4`；解码帧数：72；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/renders_stage2__ours_55000__directional__rotating_directional.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000/hdr/hdr.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/renders_stage2__ours_55000__hdr__hdr.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000/insights/albedo.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/renders_stage2__ours_55000__insights__albedo.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000/insights/alpha.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/renders_stage2__ours_55000__insights__alpha.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000/insights/normal.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/renders_stage2__ours_55000__insights__normal.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/model_mlp/renders_stage2/ours_55000/insights/rgb.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/inspection/renders_stage2__ours_55000__insights__rgb.jpg`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/albedo/iteration_55000/0010_comparison.png`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/albedo/iteration_55000/0080_comparison.png`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/albedo/iteration_55000/0150_comparison.png`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/normal_depth/comparison_0010.png`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/normal_depth/comparison_0080.png`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-06-LH-static-jumpingjacks_DL_V2-Lambertian与GT方向光源/evaluation_55000/normal_depth/comparison_0150.png`。
