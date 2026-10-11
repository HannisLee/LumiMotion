# 1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段训练计划

服务器：ushas；环境：lumimotion-ushas；批次编号：1011（用户指定）。

原始数据：`/mnt/workspace/users/han.li/lumimotion/data/LH-data/danamic/LH-dynamic-jumpingjacks_DL_V2`。训练source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view`。MAE source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`。

模型：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp`；输出及日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段`。150帧，135训练／15测试，测试10、20至150；resolution=2（400×400），黑背景。

从头随机十万点初始化，原生Stage1 35000步→原生IR Stage2累计55000步；两阶段算法均来自独立原生代码快照。

输入Alpha沿用既有转换器的albedo Alpha；不以GT albedo RGB、normal、depth、roughness或骨骼作为训练监督。原生组不读取GT光照；05/06使用GT方向，05两盏单位白光×π、06只用训练RGB标定非负光强；不把标定值解释为Blender原能量。

调度按GPU实际空余显存和进程数，每卡最多2个计算任务；05/06的依赖等待不占GPU。原生源相机平移符号仅在输入中预补偿，MAE使用原始世界相机，不改变训练算法。

全15测试帧：Albedo线性RGB MAE、Roughness参数MAE、GS Render／Depth／Mesh Normal角度MAE（度）、Expected／Median Z Depth MAE（世界单位，并记录单位比例）；同时保存覆盖率、共同像素法线分数、逐帧CSV、NPZ、代表图、日志与独立复算。Stage1记录几何MAE，SH无物理Albedo/Roughness。Lambertian无Roughness参数，记N/A。

RGB／Alpha／Normal／分离四类目检与最终PASS/FAILED必须在实际验收后回填，启动前状态，实际验收见下方结果回填；失败产物全部保留。

## stage1

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.train_stage1 --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

日志：`logs/stage1.log`。

## stage2

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.train_stage2 --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --load_iter 35000 --iterations 55000 --test_iterations 55000 --save_iterations 40000 50000 55000 --diffuse_sample_num 512 --depth_ratio 0
```

日志：`logs/stage2.log`。

## stage1_insights

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.render_stage1_insights --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 35000 --depth_ratio 0
```

日志：`logs/stage1_insights.log`。

## materials

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.render_materials --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0
```

日志：`logs/materials.log`。

## relight_hdr

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.render_relight_with_hdr --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0 --diffuse_sample_num 2048 --hdr /home/han.li/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

日志：`logs/relight_hdr.log`。

## eval_nvs

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.eval_nvs_dynamic --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_native' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0 --diffuse_sample_num 2048
```

日志：`logs/eval_nvs.log`。

## eval55000_albedo

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_albedo_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --iterations 55000 --gt_albedo_dir /mnt/workspace/users/han.li/lumimotion/data/LH-data/danamic/LH-dynamic-jumpingjacks_DL_V2/albedo --split test --output_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_55000/albedo' --save_float
```

日志：`logs/eval55000_albedo.log`。

## eval55000_predicted_meshes

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.extract_geometry_mesh_dynamic --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_55000/predicted_meshes'
```

日志：`logs/eval55000_predicted_meshes.log`。

## eval55000_normal_depth

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_geometry_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --gt_root /mnt/workspace/users/han.li/lumimotion/data/LH-data/danamic/LH-dynamic-jumpingjacks_DL_V2 --gt_depth_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-dynamic-jumpingjacks_DL_V2/gt_depth' --split test --resolution 2 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_55000/normal_depth' --pred_mesh_manifest '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_55000/predicted_meshes/manifest.json' --save_arrays
```

日志：`logs/eval55000_normal_depth.log`。

## eval55000_roughness

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_roughness_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --iteration 55000 --gt_roughness_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-dynamic-jumpingjacks_DL_V2/roughness' --split test --output_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_55000/roughness'
```

日志：`logs/eval55000_roughness.log`。

## eval55000_verify_roughness

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python -m scripts.verify_roughness_eval '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_55000/roughness'
```

日志：`logs/eval55000_verify_roughness.log`。

## eval55000_verify_all_mae

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/audit_1011_mae.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_55000' 55000
```

日志：`logs/eval55000_verify_all_mae.log`。


## 训练结果回填（2026-10-11 02:42:21 UTC）

最终结论：**FAILED**。原生Stage2首步失败；同时Stage1动态分离和地面几何有可见缺陷，不能验收完整原生管线。

判定口径：未预先约定数值阈值，MAE不作任意门槛判定。最终FAILED依据必要训练阶段缺失，或GT对比中明确可见的投影残留／材质污染。视频完整解码与首、中、尾图像抽查分别记录，未声称逐帧人工检查全部视频。

- 原始训练／渲染执行结论：`FAILED`；失败标记与退出码证据：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/review_1011/original_terminal_result.json`；补评测开始时状态快照：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/review_1011/original_state.json`。
- 实际有效检查点：35000；model：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp`；Stage1冻结副本：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/stage1_frozen`。
- 原生输入source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view`；MAE输入source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`；渲染source由模型cfg_args读取。
- 渲染目录：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000`；评测目录：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000`。
- 本次只评测与渲染已有检查点，未重启训练，未删除或覆盖原始失败结果。
- 目检拼图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/review_four_classes.png`；来源与视频帧索引：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/review_four_classes_sources.json`。
- 完整视频解码、检查点有限性：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/artifacts.json`；全15测试帧MAE独立复算：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000/mae_summary.json`。

### 定量结果

| 编号／迭代 | Albedo MAE | Roughness MAE | GS Normal ° | Depth Normal ° | Mesh Normal ° | Expected Z | Median Z |
|---|---|---|---|---|---|---|---|
| 03／35000 | N/A | N/A | 14.54508423 | 24.55663096 | 18.61244331 | 0.03370737 | 0.03245218 |
| 03／55000 | 未产出 | 未产出 | 未产出 | 未产出 | 未产出 | 未产出 | 未产出 |

法线单位为度；深度是相机Z世界单位，单位比例与逐帧统计详见normal_depth/metrics.json。Albedo采用线性RGB前景核心区域，不作拟合后替代分数；Roughness为r参数。

### 覆盖率与共同像素

| 指标 | 有效像素覆盖率 | 共同像素法线MAE ° |
|---|---|---|
| gs_render | 99.67810148% | 14.17847396 |
| depth_normal | 96.41830186% | 24.55293044 |
| mesh | 99.18352648% | 18.52934500 |
| expected | 99.67791799% | N/A |
| median | 99.67791799% | N/A |

### RGB、Alpha、Normal、分离与材质目检

- **RGB**：固定相机首／中／尾呈现开合跳姿态变化；暗面细节弱，地面投影破碎，局部人物与投影边缘模糊。
- **Alpha**：主要肢体轮廓存在，举臂与下臂姿态均可见；细部边缘有明显锯齿和毛刺，圆盘下缘不平整。
- **Normal**：人物及地面有明显噪声；测试帧80的Depth Normal和Mesh Normal相对GT更粗糙，地面非真实平面起伏显著。
- **分离**：人物大部分呈红色、地面主体绿色；人物胸腹有绿色夹杂，地面投影／周边区域有大量红色污染，分离不纯。
- **材质与重光照**：没有原生55k材质或HDR，Stage1展示不能替代材质／光照分解。

目检范围：本组固定相机四类视频的索引0、75、149；几何GT对比测试帧80。

### 原始失败及本次补评测

原生Stage2在首个材质训练步调用nvdiffrast.texture失败，CUDA 209；没有55k检查点。原环境二进制只含sm_120内核，A40为sm_86。35000的几何分数仅代表Stage1，不能称为完成两阶段。

### 实际执行命令、退出码与日志

#### stage1

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/logs/stage1.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.train_stage1 --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

#### stage2

退出码：1；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/logs/stage2.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.train_stage2 --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --load_iter 35000 --iterations 55000 --test_iterations 55000 --save_iterations 40000 50000 55000 --diffuse_sample_num 512 --depth_ratio 0
```

#### stage1_insights

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/review_1011/stage1_insights.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/native' scripts.render_stage1_insights --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/data_view' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 35000 --depth_ratio 0
```

#### predicted_meshes

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/review_1011/predicted_meshes.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.extract_geometry_mesh_dynamic --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 35000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000/predicted_meshes'
```

#### normal_depth

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/review_1011/normal_depth.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_geometry_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 35000 --gt_root /mnt/workspace/users/han.li/lumimotion/data/LH-data/danamic/LH-dynamic-jumpingjacks_DL_V2 --gt_depth_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-dynamic-jumpingjacks_DL_V2/gt_depth' --split test --resolution 2 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000/normal_depth' --pred_mesh_manifest '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000/predicted_meshes/manifest.json' --save_arrays
```

#### verify_all_mae

退出码：0；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/review_1011/verify_all_mae.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/audit_1011_mae.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000' 35000
```

#### artifact_audit

退出码：1；日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/review_1011/artifact_audit.log`。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/review_1011_artifacts.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段'
```

### RGB测试指标

```json
{
  "experiment": 3,
  "iteration": 35000,
  "protocol": "Stage1原始日志（已四舍五入；不与05/06直接比较）",
  "psnr": 23.18,
  "ssim": 0.95,
  "lpips": 0.07
}
```

### 保留产物与代表文件

- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Albedo_camframe_0010.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/renders_stage1_insights__ours_35000__Albedo_camframe_0010.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Normals_camframe_0010.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/renders_stage1_insights__ours_35000__Normals_camframe_0010.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Separation_camframe_0010.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/renders_stage1_insights__ours_35000__Separation_camframe_0010.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Separation_large_camframe_0010.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/renders_stage1_insights__ours_35000__Separation_large_camframe_0010.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/alpha_camframe_0010.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/renders_stage1_insights__ours_35000__alpha_camframe_0010.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/full_render_camframe_0010.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/renders_stage1_insights__ours_35000__full_render_camframe_0010.jpg`。
- 视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/small_gaussians_camframe_0010.mp4`；解码帧数：150；代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/inspection/renders_stage1_insights__ours_35000__small_gaussians_camframe_0010.jpg`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000/normal_depth/comparison_0010.png`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000/normal_depth/comparison_0080.png`。
- GT对比：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/evaluation_35000/normal_depth/comparison_0150.png`。
