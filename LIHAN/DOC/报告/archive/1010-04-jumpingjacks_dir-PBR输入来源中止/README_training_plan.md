# 1010-04 dir原生PBR预测几何与RGB光强完整Stage2计划

状态：B00完整训练和全量MAE已完成，人工图像质量FAILED；本B01作为目标1修复已启动，尚无最终MAE。真实训练PID206152，开始2026-10-10T01:54:52.575485+09:00，来源1009-02/35k，独立20000步Stage2至55k；后处理核对同一PID结束和真实55k后执行，不重启训练。源码快照见commands/code_snapshot、SHA256见code_manifest.json。

来源同1009-02 Stage1/35k，独立20k Stage2至55k；135训练/15测试、400×400、黑背景。只用RGB/输入RGBA、相机/time、Stage1预测、GT方向，原GT颜色/强度丢弃。pbr_gt_directional_geometry开启真实PLY xyz/rotation，2000步warmup、1000步ramp、位移.08/旋转60°、预测GS-depth一致性、来源静态RANSAC平面、轮廓与弱锚定；opacity/scale/feature冻结。白色正光强初值π，仅训练RGB拟合，不读GT能量或测试RGB。GT材质/normal/depth/mesh/blend仅离线评测。

完整MAE：Albedo、Roughness、三类Normal、Expected/Median Z、覆盖率/共同像素和RGB。全150四类可视化、转灯72、HDR150/2048采样；实际四类目检/前后景结论待回填。所有失败结果保留。不能凭GPU单帧或CPU143项通过代替此完整训练/质量验收。

## 训练命令

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-04-jumpingjacks_dir-PBR预测几何与RGB光强/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_geometry --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-04-jumpingjacks_dir-PBR预测几何与RGB光强/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

训练日志：logs/train.log；渲染model_mlp/renders_stage2/ours_55000；所有完整命令planned_commands.json/commands/*.sh。
