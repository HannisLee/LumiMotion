# 1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian训练计划

日期：2026-10-05（Asia/Tokyo）；服务器venus；Conda lumimotion。状态：**训练、渲染与评测完成；流程验收PASS，质量限制见结果记录**。

## 目标、来源和独立输出

本组为**动态原始材质**。使用1003-02原版SH Stage1 iter35000作只读基准，在当前项目已有Stage2 Lambertian管线中使用逐帧GT灯位置，只优化albedo。原版Stage2 iter55000仅用于后续对照，不作为本次初始化。

- Stage1来源：[原实验README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/README.md)；模型`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/model_mlp`。
- Gaussian：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/model_mlp/point_cloud/iteration_35000/point_cloud.ply`，**92,550个点**。
- 形变：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/model_mlp/deform/iteration_35000/deform.pth`，CPU核查32个Tensor均有限。
- 数据：`/home/lihan/data/LH-data/transfer-dynamic/only_clothV4_Dynamic`；与原Stage1 data_view的120张图像哈希和train/test transforms内容一致。
- GT灯位：`/home/lihan/data/LH-data/static/only_clothV4_Dynamic/lights.json`，120条记录，原光为白色AREA。
- 独立正式输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian`；实际模型目录`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model_mlp`。
- 独立冒烟输出预留：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian`，已完成独立200步冒烟与渲染，实际记录见本节末尾。
- [CPU输入核查](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/preflight.json)：文件、数值、划分、RGBA与哈希检查通过；真实Stage2 GPU加载、冻结与梯度检查已逐组通过，见实验inspection/gpu_preflight.json。

## 已核对的GT灯位解释

沿用`gt_directional`：逐帧读取`light_pos_world`，减去固定Stage1 canonical点云中心并归一化，形成世界空间表面指向光源的统一方向。方向表固定，不训练光照方向。

这是**AREA灯位转场景中心方向的近似**，不是逐Gaussian/逐像素的点光位置模型。不使用1/r²、面积光积分、投射阴影、镜面或间接光。元数据1800不直接作为irradiance。如果要真正逐点GT位置着色，需要另拟算法修改与消融计划，当前计划没有引入该改动。

光强使用`--photometric_light_intensity -1`：确认后，每组用自己的105训练帧、SH初始albedo、固定几何法线和GT方向独立标定非负标量，写到`model_mlp/calibration.json`，之后固定。不复用旧1002的6.03245，不用测试帧或GT albedo RGB拟合。本组实际标定光强为6.4058659238，已固定并用于正式训练。

## 统一训练设置

| 项目 | 计划 |
| --- | --- |
| 模式 / preset | `photometric_lambertian` / `lambertian_gt` |
| 来源 / 目标 | 35000→55000，新增20,000步Adam更新 |
| 数据 | 105 train / 15 test，time 0–1 |
| 分辨率 / 背景 | resolution=2，640×360，黑背景 |
| 优化 | 每Gaussian独立sigmoid albedo，时间共享，LR=0.001 |
| 冻结 | Gaussian位置/尺度/旋转/opacity/SH及全部形变网络；法线不优化 |
| 光照 | GT方向与标定后的统一光强固定 |
| 损失 | 0.8前景L1 + 0.2遮罩图DSSIM + 0.001初始albedo先验；方向平滑0 |
| Alpha | 真实RGBA，GT alpha>0.5作RGB监督mask；不训练Alpha，无Alpha loss |
| 保存 | 初始化35000、40000、50000、55000完整状态 |
| 并行 | 最多2组流水线，确认后先03/04，再05/06 |

形变参数冻结，但动态数据仍按每帧真实time计算形变。Stage1形变的颜色输出不参与本Lambertian着色。该Stage2不会改善冻结几何和法线，已有动态边缘缺口可能保留。

## 已授权执行顺序

1. 复核来源哈希、相机/FOV、RGBA和显存，使用真实Stage2加载器检查冻结、梯度与来源只读，完成现有实现CPU/GPU检查。
2. 独立200步冒烟（35000→35200），运行评测、四类渲染、转灯和HDR并目检；冒烟不替代正式验收。
3. 冒烟通过后重新从原Stage1 iter35000初始化正式训练，不从冒烟恢复，新增20,000步至55000。
4. 单独执行`render_stage2 --task all`。新训练器的`test_iterations`不触发在线评测，不能把参数当作已有测试产物。
5. 四组完成后回填各组README与计划执行记录，再整理同协议结果。未执行时不填写训练指标、目检PASS或完成时间。

## 已核对的完整命令

工作目录：`/home/lihan/reproduce/LumiMotion-stage2-perlight`。实际执行状态以本节末尾记录与实验日志为准。

```bash
# 冒烟训练
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage2 --source_path /home/lihan/data/LH-data/transfer-dynamic/only_clothV4_Dynamic --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/model_mlp' --load_iter 35000 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --depth_ratio 0 --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode photometric_lambertian --loss_preset lambertian_gt --photometric_light_mode gt_directional --photometric_gt_lights_path /home/lihan/data/LH-data/static/only_clothV4_Dynamic/lights.json --photometric_light_intensity -1 --photometric_albedo_lr 0.001 --lambda_dssim 0.2 --lambda_photometric_albedo_prior 0.001 --lambda_photometric_light_smooth 0 --photometric_log_interval 100 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model' --iterations 35200 --save_iterations 35200 --test_iterations 35200

# 冒烟完整渲染
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model_mlp' --load_iter 35200 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048

# 正式Stage2训练
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage2 --source_path /home/lihan/data/LH-data/transfer-dynamic/only_clothV4_Dynamic --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段/model_mlp' --load_iter 35000 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --depth_ratio 0 --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode photometric_lambertian --loss_preset lambertian_gt --photometric_light_mode gt_directional --photometric_gt_lights_path /home/lihan/data/LH-data/static/only_clothV4_Dynamic/lights.json --photometric_light_intensity -1 --photometric_albedo_lr 0.001 --lambda_dssim 0.2 --lambda_photometric_albedo_prior 0.001 --lambda_photometric_light_smooth 0 --photometric_log_interval 100 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model' --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 55000

# 正式评测与完整渲染
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

脚本：[commands](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/commands)。训练和渲染日志放对应独立实验的`logs/`；实际启动时记录代码版本/补丁、命令、哈希、时间和显存。禁止覆盖1003来源及任何已有checkpoint/渲染。

## 产物、评测和验收

- 模型：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model_mlp/stage2_photometric/iteration_55000/checkpoint.pth`，含材质、固定方向表/光强、Adam、随机状态、来源哈希和冻结指纹。
- 渲染：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model_mlp/renders_stage2/ours_55000/`。
- `eval/`：15个测试帧，每帧真实相机/time/GT灯方向；整图PSNR/SSIM/LPIPS、前景PSNR和遮罩包围框SSIM/LPIPS。
- `insights/`：120帧RGB/Alpha/normal/albedo和4段视频；`directional/`：固定相机72帧转灯和视频；`hdr/`：GoldenBay 120帧和视频，2048确定性球面样本。
- 检查模型有限值和冻结状态、首中末四类图、6视频完整解码；HDR无独立GT，只验证功能和定性响应。
- 材质使用现有GT albedo工具离线评分，保留raw与仅训练帧拟合scalar的诊断，报告前景/材质均衡误差及Albedo SSIM；GT albedo RGB不作为训练监督。具体评测命令在执行前按当前工具配置补齐。
- 对照1003原版Stage2时，用统一入口重评旧模型；不直接混用旧固定时刻NVS、动态首视角BVH复用与本次真实time指标。10月5日旧55k评测不能作为新35k来源/材质的分数。
- 数据包含AREA光、阴影和原始材质效果，简化模型可能将误差吸收到albedo。流程PASS不代表物理材质恢复正确。失败产物保留，不覆盖。

## 当前执行记录

四组中的本组已完成正式35000→55000、独立光强标定、全量渲染、GT albedo评分、原版同协议RGB重评和真实目检。

- 冒烟：2026-10-05T15:27:43.216455+09:00 → 2026-10-05T15:29:41.788889+09:00，200步，流程PASS。
- 正式：2026-10-05T15:33:33.430904+09:00 → 2026-10-05T16:00:01.329618+09:00，20,000步，流程PASS。
- 标定光强：6.4058659238。
- RGB PSNR/前景PSNR：27.33768495 / 17.96792911。
- raw albedo 前景MAE/bMAE/SSIM：0.12884566 / 0.09581746 / 0.59992866。
- 完整命令、日志、指标、图片/视频和质量限制：[实验README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/README.md)。
