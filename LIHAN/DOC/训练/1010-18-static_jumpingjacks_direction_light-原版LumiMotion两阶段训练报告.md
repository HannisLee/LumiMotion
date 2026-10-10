# static_jumpingjacks_direction_light 原版LumiMotion两阶段正式训练

日期：2026-10-10；venus / lumimotion / GPU0。状态：已完成，最终流程与产物验收PASS。

原始数据：`/home/lihan/data/LH-data/static/static_jumpingjacks_direction_light`；转换源：`/home/lihan/data/LH-data/transfer-static/static_jumpingjacks_direction_light`；训练source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/data_view`；评测source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/eval_view`；CLI model：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model`；实际model：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp`。

原版代码：`/home/lihan/reproduce/LumiMotion`，main `65287461075a1a4c0f54845325cc22039bd14e7d`，保留既有FOV修复。SH Stage1 35000 → 原版IR累计55000；不读取GT灯光、albedo RGB、normal、骨骼监督。默认seed0十万点随机方块，每组私有初始化，不载入旧模型。150帧，135/15，resolution2（400×400），黑背景，真实albedo软Alpha。几何冻结原动作第100帧，相机与采集光继续变化；原版形变网络和时间输入保留。静态NVS固定timestep0001形变，读取15测试视角各自采集RGB；运动光照是原版共享环境光建模的限制。NVS包含黑背景，HDR无对应GT。

六步完整命令（每步日志位于本目录logs/<步骤>.log）：

### stage1

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage1 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

### stage1_insights

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage1_insights --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 35000 --depth_ratio 0.0
```

### stage2

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --load_iter 35000 --iterations 55000 --test_iterations 55000 --save_iterations 40000 50000 55000 --diffuse_sample_num 512 --depth_ratio 0.0
```

### materials

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_materials --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0
```

### relight_hdr

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_relight_with_hdr --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048 --hdr /home/lihan/reproduce/LumiMotion/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

### eval_nvs

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_nvs_static --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/eval_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048 --static_source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/eval_view/timestep0001'
```

渲染目录：`model_mlp/renders_stage1_insights/ours_35000`、`model_mlp/trained_materials/ours_55000`、`model_mlp/render_relight_with_hdr_golden_bay_4k_32x16_rot330/ours_55000`、`model_mlp/eval_nvs_static/ours_55000`。

定量指标、代表图片/视频、RGB/Alpha/基色与分离/Normal四类目检、材质/HDR/NVS结论及最终PASS/FAILED将在实际产物验收后回填。失败产物保留，禁止覆盖。

先运行独立1100→1120冒烟及六步流程，人工目检通过后运行正式35k→55k。最多两组并行，所有产物与活动文档保存在当前仓库。

输入兼容：显式--original-loader-translation，预补偿原版reader的局部X平移符号；源相机矩阵单独保留，不修改原版算法。审计分别核查全部150个真实reader相机、455个原始文件SHA256与逐帧RGBA。

输入审计PASS：455源文件SHA256/组；150真实reader相机/组；骨骼与对象世界矩阵差异0；相机中心最大误差2.6645352591e-15米。Alpha3373–6323软边像素/帧，前景覆盖10.791%–21.332%。首中尾实际目检：人物冻结姿态、平台完整，RGB/Alpha轮廓对齐；env原始灰背景转黑，dir保留明显明暗与平台阴影。

## 正式启动

两套完整1100/1120冒烟六步、20视频/3000帧实际解码与首中末代表图目检PASS。正式原版SH35k→IR累计55k两槽流水线已启动，从头随机初始化，所有阶段自动顺序衔接；最终指标和验收待完成。冒烟质量限制见output/smoke_test对应README。

采集光语义：两盏重叠SUN，各Blender energy参数3，表面指向光源的单位方向。这些元数据只用于来源核对；原版IR学习一份共用环境光，静态NVS将固定frame1形变与各测试视角采集GT比较，因此运动光照与相机/时间耦合会影响指标和材质恢复。源EXR为世界坐标着色法线，可能含法线贴图，本轮不作为训练监督。

20000检查点核验PASS：94994个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_20000.json保存证据。

30000检查点核验PASS：94994个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_30000.json保存证据。

## Stage1 35k实际验收

94,994个高斯及形变权重有限，600张诊断图、7视频各150帧完整核验；已实际查看首中末四类与同视角GT对照，阶段流程验收PASS。RGB同视角较接近GT；平台外缘跨时间有明显碎边/拖影，阴影附近法线碎斑，SH基色保留强烈光照/阴影，分离红绿混合。代表图：inspection/stage1_review.png、inspection/stage1_aligned_gt.png；证据inspection/stage1_artifact_review.json。

测试子集（最多4相机）：PSNR29.81893921、SSIM0.98907304、LPIPS0.01909588；这不是完整15视角IR NVS。原版IR已加载35k并开始新增20k至累计55k，最终验收待完成。

40000检查点核验PASS：94994个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_40000.json保存证据。

进度条说明：原版Stage2同时维护两个tqdm，一个按累计iterations=55000显示，另一个按range(35000,55000)显示20,000新增步骤；前者并不会在35k加载时把进度条n设为35000。实际训练循环为新增20k至55k，模型按40000/50000/55000保存。监控脚本优先读取20k条，避免错误的百分比和剩余时间；训练代码保持原版。

50000检查点核验PASS：94994个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_50000.json保存证据。

原版静态NVS限制：固定frame1形变，首测试视角构建BVH后在其他测试视角复用；保留的相机相关形变可能使其他视角的遮挡几何不完全一致。此轮遵循原版实现，不将这些指标直接解释为物理材质恢复准确度。

## 六步流水线实际结果

完成时间：2026-10-10T16:59:26.682125+09:00；checkpoint、15测试帧及有限NVS指标核验通过。完整媒体与真实目检待验收。

{"psnr_nvs_avg": 20.3071470896403, "ssim_nvs_avg": 0.9213215986887614, "lpips_nvs_avg": 0.08277344728509585}

## 最终验收回填

验收时间：2026-10-10T17:03:03.716554+09:00；**PASS（流程与产物验收）**。六步均正常退出，Stage1/Stage2完整checkpoint各属性有限，Stage2环境光权重与HDR存在。

六步用时：7657.7秒；实际起止和阶段时间见STARTED、finish_time.txt与logs/*.start_time.txt/*.finish_time.txt。日志：logs/pipeline.log以及六个步骤对应.log；命令见上方，源码快照与SHA256见19号执行汇总。

NVS：{"psnr_nvs_avg": 20.3071470896403, "ssim_nvs_avg": 0.9213215986887614, "lpips_nvs_avg": 0.08277344728509585}；完整15测试视角，原版静态NVS固定frame1形变，黑背景整图。

Stage1测试子集：{"test/reconstruct - l1_loss": 0.003443296067416668, "test/reconstruct - psnr": 29.818939208984375, "test/reconstruct - ssim": 0.9890730381011963, "test/reconstruct - lpips": 0.019095880910754204, "test/reconstruct - ms-ssim": 0.9927914142608643, "test/reconstruct - alex-lpips": 0.0130283422768116}；最多4测试相机，不能代替15视角完整NVS。

每组10个视频各150帧已全部解码（1500帧）；Stage1 600张、材质300张、HDR150张PNG及15套NVS输出完整。各组Gaussian数量：{"35000": 94994, "55000": 94994}。

代表图：[stage1_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/inspection/stage1_review.png)、[materials_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/inspection/materials_review.png)、[hdr_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/inspection/hdr_review.png)、[stage1_aligned_gt.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/inspection/stage1_aligned_gt.png)、[nvs_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/inspection/nvs_review.png)、[materials_aligned_gt.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/inspection/materials_aligned_gt.png)。

代表视频（固定test camera10与cam0）：

- [Separation_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Separation_camframe_0010.mp4)
- [Separation_large_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Separation_large_camframe_0010.mp4)
- [full_render_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/full_render_camframe_0010.mp4)
- [alpha_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/alpha_camframe_0010.mp4)
- [small_gaussians_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/small_gaussians_camframe_0010.mp4)
- [Normals_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Normals_camframe_0010.mp4)
- [Albedo_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Albedo_camframe_0010.mp4)
- [Albedo_cam0.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/trained_materials/ours_55000/Albedo_cam0.mp4)
- [Roughness_cam0.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/trained_materials/ours_55000/Roughness_cam0.mp4)
- [render_relight_with_hdr2048_view0.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段/model_mlp/render_relight_with_hdr_golden_bay_4k_32x16_rot330/ours_55000/render_relight_with_hdr2048_view0.mp4)

实际首/中/末目检（已打开上述代表图）：

- RGB：Stage1同相机同时间较接近GT；固定相机跨时间仍有位置/形状变化，平台外缘明显碎边与拖影。
- Alpha：人体主体完整；平台边缘跨时间出现碎边、下缘拖影。NVS使用GT Alpha，外缘伪影会被掩膜裁去。
- 基色与分离：SH基色保留强光与阴影，分离红绿混合。最终IR基色与GT差异明显：服装、肤色的强烈明暗和平台黑色投影残留，平台图案局部被斑驳颜色覆盖，未达到可靠的光照/材质分离。
- Normal：人体整体连续，局部服装/关节色块；平台尤其脚下、阴影区域明显斑块。未执行GT角度MAE。
- 材质/HDR：粗糙度偏高且人体、平台存在大块不均匀变化，无本轮GT粗糙度评分。Golden Bay暖色重照明可输出，但沿用基色照明残留与平台伪影；中间时刻平台下缘拖影明显，无目标HDR GT。
- NVS：首中末三个测试视角的衣服、肤色、平台明暗和投影均与GT有明显差异，背面视角也有图案/亮度差异。完整15视角PSNR20.30714709、SSIM0.92132160、LPIPS0.08277345，使用GT Alpha合成并包含黑背景。

PASS表示训练、渲染、评测与产物验收完成；原版共享环境光对运动采集光、保留时间及相机相关形变的建模限制、法线与材质恢复误差按上述实际目检保留。此轮未执行GT法线/深度/Albedo/Roughness MAE。HDR没有目标光照GT。原始EXR仅核对文件与SHA256，不将此记为像素级真值审计。
