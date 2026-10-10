# 1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源训练计划

服务器：ushas；环境：lumimotion-ushas；批次编号：1011（用户指定）。

原始数据：`/mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2`。训练source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`。MAE source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`。

模型：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/model_mlp`；输出及日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源`。150帧，135训练／15测试，测试10、20至150；resolution=2（400×400），黑背景。

只读01组35000步Stage1来源，Stage2新增20000步至55000；独立输出，不覆盖原生模型。

输入Alpha沿用既有转换器的albedo Alpha；不以GT albedo RGB、normal、depth、roughness或骨骼作为训练监督。原生组不读取GT光照；05/06使用GT方向，05两盏单位白光×π、06只用训练RGB标定非负光强；不把标定值解释为Blender原能量。

调度按GPU实际空余显存和进程数，每卡最多2个计算任务；05/06的依赖等待不占GPU。原生源相机平移符号仅在输入中预补偿，MAE使用原始世界相机，不改变训练算法。

全15测试帧：Albedo线性RGB MAE、Roughness参数MAE、GS Render／Depth／Mesh Normal角度MAE（度）、Expected／Median Z Depth MAE（世界单位，并记录单位比例）；同时保存覆盖率、共同像素法线分数、逐帧CSV、NPZ、代表图、日志与独立复算。Stage1记录几何MAE，SH无物理Albedo/Roughness。Lambertian无Roughness参数，记N/A。

RGB／Alpha／Normal／分离四类目检与最终PASS/FAILED必须在实际验收后回填，当前为PENDING；失败产物全部保留。

## stage2

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.train_stage2 --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --stage1_model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/stage1_frozen' --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 55000 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode original_ir --loss_preset pbr_gt_directional_native --pbr_gt_lights_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793
```

日志：`logs/stage2.log`。

## render

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.render_stage2 --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/model_mlp' --load_iter 55000 --task all --hdr_samples 2048 --hdr_filepath /home/han.li/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

日志：`logs/render.log`。

## eval55000_albedo

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_albedo_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --iterations 55000 --gt_albedo_dir /mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2/albedo --split test --output_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/evaluation_55000/albedo' --save_float
```

日志：`logs/eval55000_albedo.log`。

## eval55000_predicted_meshes

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.extract_geometry_mesh_dynamic --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/evaluation_55000/predicted_meshes'
```

日志：`logs/eval55000_predicted_meshes.log`。

## eval55000_normal_depth

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_geometry_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 55000 --gt_root /mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2 --gt_depth_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/gt_depth' --split test --resolution 2 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/evaluation_55000/normal_depth' --pred_mesh_manifest '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/evaluation_55000/predicted_meshes/manifest.json' --save_arrays
```

日志：`logs/eval55000_normal_depth.log`。

## eval55000_roughness

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_roughness_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --iteration 55000 --gt_roughness_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/roughness' --split test --output_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/evaluation_55000/roughness'
```

日志：`logs/eval55000_roughness.log`。

## eval55000_verify_roughness

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python -m scripts.verify_roughness_eval '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/evaluation_55000/roughness'
```

日志：`logs/eval55000_verify_roughness.log`。

## eval55000_verify_all_mae

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/audit_1011_mae.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-05-LH-static-jumpingjacks_DL_V2-PBR与GT方向光源/evaluation_55000' 55000
```

日志：`logs/eval55000_verify_all_mae.log`。
