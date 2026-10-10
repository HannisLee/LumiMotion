# static_jumpingjacks_environment_light 原版LumiMotion两阶段正式训练

日期：2026-10-10；venus / lumimotion / GPU0。状态：已完成，最终流程与产物验收PASS。

原始数据：`/home/lihan/data/LH-data/static/static_jumpingjacks_environment_light`；转换源：`/home/lihan/data/LH-data/transfer-static/static_jumpingjacks_environment_light`；训练source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/data_view`；评测source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/eval_view`；CLI model：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model`；实际model：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp`。

原版代码：`/home/lihan/reproduce/LumiMotion`，main `65287461075a1a4c0f54845325cc22039bd14e7d`，保留既有FOV修复。SH Stage1 35000 → 原版IR累计55000；不读取GT灯光、albedo RGB、normal、骨骼监督。默认seed0十万点随机方块，每组私有初始化，不载入旧模型。150帧，135/15，resolution2（400×400），黑背景，真实albedo软Alpha。几何冻结原动作第100帧，相机与采集光继续变化；原版形变网络和时间输入保留。静态NVS固定timestep0001形变，读取15测试视角各自采集RGB；运动光照是原版共享环境光建模的限制。NVS包含黑背景，HDR无对应GT。

六步完整命令（每步日志位于本目录logs/<步骤>.log）：

### stage1

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage1 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

### stage1_insights

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage1_insights --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 35000 --depth_ratio 0.0
```

### stage2

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --load_iter 35000 --iterations 55000 --test_iterations 55000 --save_iterations 40000 50000 55000 --diffuse_sample_num 512 --depth_ratio 0.0
```

### materials

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_materials --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0
```

### relight_hdr

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_relight_with_hdr --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048 --hdr /home/lihan/reproduce/LumiMotion/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

### eval_nvs

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_nvs_static --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/eval_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048 --static_source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/eval_view/timestep0001'
```

渲染目录：`model_mlp/renders_stage1_insights/ours_35000`、`model_mlp/trained_materials/ours_55000`、`model_mlp/render_relight_with_hdr_golden_bay_4k_32x16_rot330/ours_55000`、`model_mlp/eval_nvs_static/ours_55000`。

定量指标、代表图片/视频、RGB/Alpha/基色与分离/Normal四类目检、材质/HDR/NVS结论及最终PASS/FAILED将在实际产物验收后回填。失败产物保留，禁止覆盖。

先运行独立1100→1120冒烟及六步流程，人工目检通过后运行正式35k→55k。最多两组并行，所有产物与活动文档保存在当前仓库。

输入兼容：显式--original-loader-translation，预补偿原版reader的局部X平移符号；源相机矩阵单独保留，不修改原版算法。审计分别核查全部150个真实reader相机、455个原始文件SHA256与逐帧RGBA。

输入审计PASS：455源文件SHA256/组；150真实reader相机/组；骨骼与对象世界矩阵差异0；相机中心最大误差2.6645352591e-15米。Alpha3373–6323软边像素/帧，前景覆盖10.791%–21.332%。首中尾实际目检：人物冻结姿态、平台完整，RGB/Alpha轮廓对齐；env原始灰背景转黑，dir保留明显明暗与平台阴影。

## 正式启动

两套完整1100/1120冒烟六步、20视频/3000帧实际解码与首中末代表图目检PASS。正式原版SH35k→IR累计55k两槽流水线已启动，从头随机初始化，所有阶段自动顺序衔接；最终指标和验收待完成。冒烟质量限制见output/smoke_test对应README。

采集光语义：两盏POINT，Blender energy参数800/650。这些元数据只用于来源核对；原版IR学习一份共用环境光，静态NVS将固定frame1形变与各测试视角采集GT比较，因此运动光照与相机/时间耦合会影响指标和材质恢复。源EXR为世界坐标着色法线，可能含法线贴图，本轮不作为训练监督。

20k检查点核验PASS：68000个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_20000.json保存证据。

30000检查点核验PASS：68000个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_30000.json保存证据。

## Stage1 35k实际验收

68,000个高斯及形变权重有限，600张诊断图片、7视频各150帧已完整核验；首中末RGB/Alpha/SH基色与分离/Normal以及同视角GT对照已实际打开检查，阶段流程验收PASS。RGB接近GT，人体法线较冒烟平滑；固定相机跨时间有轻微形变/位置漂移，平台外缘有少量碎边，SH基色保留照明。代表图：inspection/stage1_review.png、inspection/stage1_aligned_gt.png；证据inspection/stage1_artifact_review.json。

测试子集（最多4相机）：PSNR51.74213791、SSIM0.99964571、LPIPS0.00115079；这不是完整15视角IR NVS。原版IR已加载35k并开始新增20k至累计55k，最终验收待完成。

40000检查点核验PASS：68000个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_40000.json保存证据。

50000检查点核验PASS：68000个高斯，高斯属性及形变权重全部有限；inspection/checkpoint_50000.json保存证据。

进度条说明：原版Stage2同时维护两个tqdm，一个按累计iterations=55000显示，另一个按range(35000,55000)显示20,000新增步骤；前者并不会在35k加载时把进度条n设为35000。实际训练循环为新增20k至55k，模型按40000/50000/55000保存。监控脚本优先读取20k条，避免错误的百分比和剩余时间；训练代码保持原版。

原版静态NVS限制：固定frame1形变，首测试视角构建BVH后在其他测试视角复用；保留的相机相关形变可能使其他视角的遮挡几何不完全一致。此轮遵循原版实现，不将这些指标直接解释为物理材质恢复准确度。

55k材质GT初查：已实际打开inspection/materials_aligned_gt.png（同camera10/time9）。GT基色较均匀；预测基色仍保留明显褶皱明暗、肤色阴影和平台投影，脸部/腿部及平台部分偏暗。此为目检，未进行GT MAE定量评测；完整HDR/NVS与最终媒体验收待执行。

## 六步流水线实际结果

完成时间：2026-10-10T16:46:29.181958+09:00；checkpoint、15测试帧及有限NVS指标核验通过。完整媒体与真实目检待验收。

{"psnr_nvs_avg": 39.241677347819014, "ssim_nvs_avg": 0.9951619267463684, "lpips_nvs_avg": 0.004566953284665942}

## env完整流程验收初结论

35k/55k模型与环境光权重有限，600张Stage1/300张材质/150张HDR、15套NVS及10视频各150帧全部核验，首中末四类/材质GT/HDR/NVS已实际目检，流程验收PASS。完整15视角NVS：PSNR39.24167735、SSIM0.99516193、LPIPS0.00456695。基色有明确照明残留，粗糙度总体偏高且片状，固定视角跨时间有轻微几何/平台边缘不稳，以上不由流程PASS消除；总体批次等待dir最终结束后统一回填最终状态。

## 最终验收回填

验收时间：2026-10-10T17:03:03.716554+09:00；**PASS（流程与产物验收）**。六步均正常退出，Stage1/Stage2完整checkpoint各属性有限，Stage2环境光权重与HDR存在。

六步用时：6880.2秒；实际起止和阶段时间见STARTED、finish_time.txt与logs/*.start_time.txt/*.finish_time.txt。日志：logs/pipeline.log以及六个步骤对应.log；命令见上方，源码快照与SHA256见19号执行汇总。

NVS：{"psnr_nvs_avg": 39.241677347819014, "ssim_nvs_avg": 0.9951619267463684, "lpips_nvs_avg": 0.004566953284665942}；完整15测试视角，原版静态NVS固定frame1形变，黑背景整图。

Stage1测试子集：{"test/reconstruct - l1_loss": 0.0007189487805590034, "test/reconstruct - psnr": 51.74213790893555, "test/reconstruct - ssim": 0.9996457099914551, "test/reconstruct - lpips": 0.0011507886229082942, "test/reconstruct - ms-ssim": 0.9992565512657166, "test/reconstruct - alex-lpips": 0.000354735238943249}；最多4测试相机，不能代替15视角完整NVS。

每组10个视频各150帧已全部解码（1500帧）；Stage1 600张、材质300张、HDR150张PNG及15套NVS输出完整。各组Gaussian数量：{"35000": 68000, "55000": 68000}。

代表图：[stage1_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/inspection/stage1_review.png)、[materials_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/inspection/materials_review.png)、[hdr_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/inspection/hdr_review.png)、[stage1_aligned_gt.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/inspection/stage1_aligned_gt.png)、[nvs_review.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/inspection/nvs_review.png)、[materials_aligned_gt.png](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/inspection/materials_aligned_gt.png)。

代表视频（固定test camera10与cam0）：

- [Separation_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Separation_camframe_0010.mp4)
- [Separation_large_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Separation_large_camframe_0010.mp4)
- [full_render_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/full_render_camframe_0010.mp4)
- [alpha_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/alpha_camframe_0010.mp4)
- [small_gaussians_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/small_gaussians_camframe_0010.mp4)
- [Normals_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Normals_camframe_0010.mp4)
- [Albedo_camframe_0010.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/renders_stage1_insights/ours_35000/Albedo_camframe_0010.mp4)
- [Albedo_cam0.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/trained_materials/ours_55000/Albedo_cam0.mp4)
- [Roughness_cam0.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/trained_materials/ours_55000/Roughness_cam0.mp4)
- [render_relight_with_hdr2048_view0.mp4](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段/model_mlp/render_relight_with_hdr_golden_bay_4k_32x16_rot330/ours_55000/render_relight_with_hdr2048_view0.mp4)

实际首/中/末目检（已打开上述代表图）：

- RGB：同相机同时间接近GT，固定视角跨时间仍有轻微位置/形状漂移。
- Alpha：主体完整，平台外缘少量碎边/拖影。
- 基色与分离：SH基色与最终IR基色均保留照明；GT对照显示服装褶皱、肤色明暗和平台投影残留，局部偏暗。分离以绿色为主，平台有点状缺口。
- Normal：人体整体较平滑，服装/关节有局部色块，平台大体平坦；未执行GT角度MAE。
- 材质/HDR：粗糙度总体较高，人体及平台有片状/三角区域变化；没有本轮GT粗糙度评分。Golden Bay暖色重照明正常，有阴影及轻微平台边缘伪影，无目标HDR GT。
- NVS：首中末三个测试视角主体、服装和平台图案较接近GT；局部亮度与细节不同。完整15视角PSNR39.24167735、SSIM0.99516193、LPIPS0.00456695，使用GT Alpha合成并包含黑背景。

PASS表示训练、渲染、评测与产物验收完成；原版共享环境光对运动采集光、保留时间及相机相关形变的建模限制、法线与材质恢复误差按上述实际目检保留。此轮未执行GT法线/深度/Albedo/Roughness MAE。HDR没有目标光照GT。原始EXR仅核对文件与SHA256，不将此记为像素级真值审计。
