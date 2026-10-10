# 1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束 完整Stage2训练计划

状态PREPARED，完整训练未启动，正式七MAE及7度和质量结论待执行。B09完整20k原GS MAE15.94231363度/覆盖95.47845260%，实际质量FAILED，全部失败产物保留。已读GOAL；B10实际226 CPU/20步/GPU/保存加载/共享buffer/4视频功能PASS，画质FAILED_SMOKE_ONLY。

## 本次训练RGB来源局部法线与Preset

继承B09概率gate、仓库原生Direct SUN/GGX/BVH与SH辅助、材质、真实XYZ/q、信赖域、source-only plane15187及原photo/Alpha/ND损失和全部lr。独立Preset pbr_gt_directional_source_rgb_normal，新增pbr_source_local_rotation=True与lambda_pbr_rgb_normal_prior=2；正式warmup2000/ramp1000。source q/raw norm从原35k保存，固定NN32 tensor按当前gate查询来源world旋转，再按R_source_world(p) R_source_canonical^T R_current_canonical运输，随后执行既有plane。初值原d_rotation逐bit保持。实际Gaussian covariance、normal、depth、PBR/BVH共享真实q，绝非评测法线图覆盖。周期预览/FrozenScene/GeometryRun/Mesh/material/HDR/SUN都使用相同入口和顺序。

附加canonical轴loss全99827点平均confidence*(1-dot)，29621非plane点有候选支持（来源static17389/dynamic12232）。来自原135TRAIN RGB线性luminance与自身Alpha、来源预测可见性、源35k qXYZ/固定NN/相机time/白色单位GT方向的独立LS候选；9个全点数组、source6 SHA/135逐图SHA、TRAINkeys/TEST0、finite与CPU全点Gram/解/残差/confidence复算绑定模型。该候选是训练RGB派生，不是GT法线，也不保证GGX/阴影准确；其confidence只约束新增loss，不改变原photo、完整点/帧/难像素或原评分。既有plane点没有重复该轴约束，也没有删除。

模型保存pbr_rgb_normal{,_input}.{npz,json}，source quaternion/target/confidence和原独立input本地复制、各SHA与checkpoint metadataSHA绑定，所有加载严格复算并检查源32NN冻结。新增0 optimizer参数，仍用原canonical rotation Adam raw q lr.001/norm和60度信赖域；probability gate/native sigmoid/有限逆sigmoid原实现不变。canonical .08m不保证gate/plane引起的实际world位移同界。未保存完整Adam状态，不声明无损resume。

226 CPU、独立20步35000→35020实际Trainer/保存加载/GPU五参数梯度/共享buffer/HDR16/SUN/四类150×4和4视频全解码功能PASS，实际画质FAILED_SMOKE_ONLY；RGB约束loss .102662175894、q梯度L2 .00525612570345非零。476核心与双snapshot保持。此完整20k从原35k初始化，不续20步或B09学得checkpoint。GT材质/normal/roughness/depth/mesh/blend/原灯强度颜色全部不参与训练/初始化/约束；仅独立离线评测读取GT标签，原有完整七MAE/覆盖/common/missing180定义保持。

只从原Stage1 source35000→55000独立20000步，仅Stage2，135TRAIN/15TEST、400²/黑背景/原相机time。原RGB/自身Alpha/预测35k/GT direction白色单位，新增轴约束由135TRAIN RGB估计，没有GT normal/albedo/roughness/depth/mesh/blend/原光强颜色。sourceplane15187、全部99827点/原photo/评分保持。正式warmup2000/ramp1000，保存40k/50k/55k。训练初始化没有使用候选替换source q或学得B09权重。

## 完整命令与日志

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_source_rgb_normal --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_rgb_normal_prior_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-17-dir-训练RGB法线约束原型/candidate_assets'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/write_results.log

## 可视化和验收：PENDING

渲染model_mlp/renders_stage2/ours_55000/insights/{rgb,alpha,normal,albedo}各150帧/4视频；HDR150帧×2048samples和SUN72帧/2视频。全15TEST七MAE与coverage/common/missing180原评分；GS有向world shading normal、GTalpha≥.5全前景pixel-weighted，不按物体筛选；Expected/Median正camera Z米不拟合；Mesh15targettime各135TRAIN预测深度重建，不用GT或TEST位姿重建。代表图inspection/{rgb_final_55000_comparison,four_categories_final_55000,relight_final_55000}.png和normal_depth/comparison_0010/0080/0150.png。四类/HDR/SUN实际目检与6视频全解码待执行。失败checkpoint/浮点/图/日志保留。未通过质量不commit/push/新目标分支，不提前声称7度。

## B10完整训练已实际启动（2026-10-10T11:12:19.216698+09:00）

原35k→55k独立20k，TRAIN PID306459/ticks10865815，exec4419实际仍运行；11:11:25.974063 JST启动，warmup2000/ramp1000、lambda2训练RGB轴约束，当前无正式MAE/质量结论。检查点独立共享加载helper先在smoke35020真实exit0核验，无GT标签、0 optimizer、29621支持/全99827点。检查点审计等待exec10940/后处理等待exec65390，精确进程信息见process_handles.json；保存40k/50k/55k后检查，训练完成后9步原全量渲染/七MAE自动顺序执行，人工质量单独验收。源476核心和双snapshot冻结。

训练准备helper首attempt在执行前因字符串引号嵌套语法错误exit1，未创建正式目录/模型、未启动训练；随后shell尝试不存在retry文件exit2，原日志保留。修正helper独立retry实际exit0，正式目录只创建一次，没有训练重启或改生产代码。全部准备脚本、失败JSON/log保留smoke1010-18。

### B10启动只读核对与当前进度（2026-10-10T11:13:35.904698+09:00）

audit_launch实际exit0，配置相比B09只有运行路径/Preset和新增3字段（source_local=True、RGB资产路径、lambda2）共7项差异，原warmup2000/概率gate/lr/源plane15187/全部99827点/ownAlpha150/冻结NN32/direction表byte一致。输入/资产SHA正常，source RGB约束支持29621；0诊断optimizer、0 GT标签读取。真实train PID306459仍运行，已完成1395/20000步，源码476 SHA锁定；exec4419/10940/65390均在运行中，不能提前填exit0。正式七MAE/前后景质量/7度均PENDING。
