# 1006-04 only_clothV4_Dynamic_Lambertian 原版随机方块初始化重训计划
<!-- 1006-final-record -->

**最终状态：4/4 COMPLETED，流程及产物验收PASS；训练只有1006-01～04。** 原版SH Stage1 35k、IR Stage2累计55k，两阶段共8组结果、120行逐帧指标已完成。

评测完成：2026-10-06 02:44:28（Asia/Tokyo）；最终核验及文档回填：2026-10-06T11:16:08+09:00。

[完整双阶段报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-01-四组Stage1与Stage2真值评测报告.md)；[阶段CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/metrics.csv)；[最终核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/final_audit.json)。

四组的Stage2全图及前景包围框SSIM均下降，depth和normal MAE均增加；共同有效像素比较也显示几何误差上升。只有1006-04的原始albedo MAE下降（0.138899→0.122299，约11.95%），其bMAE也下降；01、02、03的原始albedo MAE增加。Stage1为含光照的SH基色代理，以上数值变化不能解释为Stage1物理材质恢复更准确。

| 阶段 | 步数 | SSIM↑ | 前景包围框SSIM↑ | Albedo MAE | Depth MAE（世界单位） | Normal MAE（°） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage1 | 35000 | 0.995352 | 0.982812 | 0.138899 | 0.087812 | 12.4948 |
| stage2 | 55000 | 0.975389 | 0.879943 | 0.122299 | 0.093827 | 12.8886 |

目检仅覆盖固定相机frame_0008的t0/t60/t119，以及NVS测试索引0/7/14代表图；没有逐帧人工审阅全部视频。PASS表示复现流程、产物与代表帧检查完成，未设材质/几何质量阈值，不表示真值重建准确。

以下核对计划为历史批准记录；当前结果见“当前执行记录”。

<!-- /1006-final-record -->


文档编制：2026-10-05（Asia/Tokyo）；实验编号按用户指定使用1006。状态：**用户已核对并授权正式训练，跳过冒烟**。编号不表示已经安排10月6日自动启动；用户已明确跳过冒烟，直接正式训练。

## 目标与数据

本组：**动态，Lambertian 材质**。对应旧对照：`1003-03-only_clothV4_Dynamic_Lambertian-LumiMotion原版两阶段`。本轮保持1003的算法、图像、相机、划分及训练参数，仅将15万点Blender表面初始化换成原版10万随机方块初始化。

- 原始数据：`/home/lihan/data/LH-data/static/only_clothV4_Dynamic_Lambertian`。
- 转换数据：`/home/lihan/data/LH-data/transfer-dynamic/only_clothV4_Dynamic_Lambertian`。
- 实际 source_path：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view`。
- 正式输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段`；命令 model_path：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model`；实际模型：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model_mlp`。
- 冒烟输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块冒烟`，使用独立 data_view 和独立随机生成点云，正式训练不从冒烟 checkpoint 续训。
- 图像1280×720；训练 resolution=2，即640×360；105训练/15测试；测试帧8/16/…/120，黑背景及真实RGBA alpha。

## 原版随机方块初始化（必须核对）

每组私有data_view只链接images、原版加载名称chapel_day_4k_32x16_rot0以及train/test transforms。**启动Stage1前没有points3d.ply，也没有rgbd目录**。原版readNerfSyntheticInfo因此进入随机分支，自动生成：

```python
num_pts = 100_000
xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
```

三个轴都独立均匀分布在[-1.3,1.3)，以原点为中心；不使用baseline、旧points3d、GT深度/法线或旧训练checkpoint，不作中心平移、场景归一化或范围扩展。原版safe_state保持Python/NumPy seed=0、Torch seed=5000000；高斯基础旋转为单位四元数[1,0,0,0]，尺度由最近邻距离初始化，opacity=0.1，SH/albedo系数按当前原版create_from_pcd初始化。

随机PLY由加载器写到本实验data_view/points3d.ply，模型input.ply保存实际输入。本组准备阶段尚未生成该PLY。启动后核查日志Generating random point cloud (100000)和Number of points at initialisation : 100000；核对input.ply与生成PLY哈希一致、坐标严格符合原版seed=0的随机抽样（浮点存储容差）、法线全零，再将结果保存到inspection/initialization_check.json。四组随机输入是否相同以实际哈希核查，不提前写成已验证。

默认方块不完全覆盖现有场景：旧表面点云部分坐标约±1.45，布料高度约1.93，超出±1.3。本轮保留原版范围，观察优化能否恢复，质量问题如实记录，不以GT几何修正初始化。

## 算法与训练参数

执行代码：`/home/lihan/reproduce/LumiMotion`的main，准备时HEAD `65287461075a1a4c0f54845325cc22039bd14e7d`。保留与1003相同的FOVxy兼容修复；不改训练/渲染算法。代码版本、现有补丁和关键文件哈希已留在1006-05准备目录，实际启动再次记录。

| 项目 | 设置 |
| --- | --- |
| Stage1 | 原版SH几何/形变，35000步，保存10k/20k/30k/35k，测试35k |
| Stage2 | 原版IR，从本组35k开始，累计55000（新增20000），保存40k/50k/55k |
| Stage1显式参数 | densify_until_iter=20000，lambda_separation=0.001，d_xyz_loss_weight=0.001，binarization_warm_up=1000，depth_ratio=1.0，d_color_reg_loss_weight=0.01 |
| Stage2与离线渲染 | depth_ratio=0.0；训练diffuse_sample_num=512；HDR/NVS=2048 |
| 学习率 | 沿用1003和原版参数；Stage1 rotation_lr=0.001，不新增旋转解冻或GT法线监督 |
| 原版Stage2优化范围 | albedo/roughness/opacity、MLP颜色head、环境光；基础旋转和几何形变不继续优化 |
| 冒烟 | Stage1=1100，Stage2累计1120，再执行完整六步流程；仅用于运行检查 |

Lambertian是数据材质标签；本组仍使用原版SH+IR，不使用当前分支GT-directional Stage2。灯位、真值normal、albedo和Blender几何仅用于后续评估。chapel_day加载名称只是images别名。

## 完整正式命令

以下命令已写入本实验commands目录并通过bash -n静态语法检查，**启动前已准备**。各命令在原版代码目录执行，完整环境设置见对应.sh文件。run_pipeline.sh有文档核对门控；已根据本轮用户授权记录DOCUMENT_REVIEW_CONFIRMED。

### stage1

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage1 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

### stage1_insights

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_stage1_insights --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 35000 --depth_ratio 0.0
```

### stage2

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --gt_alpha_mask_as_scene_mask --iterations 55000 --load_iter 35000 --test_iterations 55000 --save_iterations 40000 50000 55000 --diffuse_sample_num 512 --depth_ratio 0.0
```

### materials

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_materials --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 55000 --depth_ratio 0.0
```

### relight_hdr

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_relight_with_hdr --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 55000 --diffuse_sample_num 2048 --depth_ratio 0.0 --hdr /home/lihan/reproduce/LumiMotion/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

### eval_nvs

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.eval_nvs_dynamic --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 55000 --diffuse_sample_num 2048 --depth_ratio 0.0
```

## 原准备冒烟方案（用户已要求跳过）

完整冒烟命令见`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块冒烟/planned_commands.json`及commands/*.sh。仅训练终点和save/test/load iteration对应改为1100/1120，其他参数与正式一致。用户要求跳过冒烟；正式独立从随机初始化开始，不复制任何旧PLY或模型。

正式流水线：Stage1 → Stage1 insights → Stage2 → materials → Golden Bay HDR → 原版NVS。原版固定视角的可视化含120时刻，动态NVS用测试帧实际time、静态NVS固定timestep0001。各步骤日志在logs/；逐步完成不直接标质量PASS。

验收须回填：初始化证据、35k/55k checkpoint、120帧RGB/alpha/normal/albedo及视频、roughness/HDR、15测试NVS、统一实际相机/time的RGB与normal MAE/深度/albedo误差评测；首/中/尾及最差帧目检。动态GT使用已有逐帧Blender几何缓存。与对应1003及1005模型比较时使用同一遮罩、坐标、时间与尺度口径，保留布料单独指标。

原版Stage1训练日志只评最多4测试视角；旧NVS动态BVH复用及静态固定time的限制必须写明。原版NVS用于历史一致对照，统一评测另存，不覆盖或修改旧原版产物。本组指标/代表图/目检目前均为待执行，不沿用1003数字。原版可视化流程已生成normal不等于做了GT法线监督。

服务器venus，环境lumimotion，GPU0 RTX3090Ti 24GB；拟沿用此前最多两组并行，启动前检查空闲显存≥9000MiB。失败保留日志与产物，新目录重试。1003/1005模型、数据原目录和baseline保持只读。


## 用户后续授权：Stage1 / Stage2 真值评测

2026-10-06T00:45:20+09:00：追加两阶段albedo MAE、depth和normal MAE及四组统一报告任务。Stage1=35000、Stage2=55000，同一15测试帧与真值，附覆盖率和共同有效像素。Stage1 albedo为SH DC基色代理，Stage2为IR材质基色；主MAE均未作GT尺度拟合修正。任务状态/命令/日志位于/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task；统一报告：[四组双阶段评测报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-01-四组Stage1与Stage2真值评测报告.md)。此追加只读取checkpoint，不修改或重启训练，不新增实验编号。


### SSIM追加评测（用户确认）

2026-10-06T01:15:06+09:00：03、04完成后，与01、02一起在同一报告分别列Stage1 35k和Stage2 55k的SSIM、albedo MAE、depth MAE；保留先前normal MAE。SSIM为同一15测试帧全图RGB等权平均，同时附前景包围框SSIM。Stage1用原版SH渲染；Stage2用原版IR、真实time和逐时刻更新BVH的现有15帧RGB评测。不是训练日志4视角的舍入SSIM。主报告仍为1006-01四组双阶段报告，不增加实验编号。


## 当前执行记录（实际结果自动回填）

更新：2026-10-06T11:16:30+09:00。状态：**COMPLETED**。用户已确认编号并明确跳过冒烟，直接正式训练。

- 实际步骤：训练、双阶段评测及代表帧目检完成；本步骤进度：None。
- 正式开始：2026-10-06T00:38:30+09:00；结束：2026-10-06T02:40:38+09:00。
- 冒烟：**SKIPPED_BY_USER**，未执行，不作为启动条件。正式独立从默认随机方块开始。
- 实际日志：logs/stage1.log、stage1_insights.log、stage2.log、materials.log、relight_hdr.log、eval_nvs.log、artifact_checks.log、rgb_actual_time.log、geometry_gt.log、albedo_gt.log。
- 完整训练/渲染命令：上方核对计划及commands/*.sh；新增评测实际命令：evaluation_commands.json。实际代码来源：code_commit.txt、code_status.txt、code_diff.patch。
- 默认初始化实证：PASS_DEFAULT_RANDOM_CUBE，100,000点，bbox=[[-1.2999619245529175, -1.2999482154846191, -1.299992322921753], [1.2999730110168457, 1.2999974489212036, 1.2999688386917114]]，seed=0坐标核验通过，零法线；input.ply SHA256 `f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`。

### 原版 NVS（沿用旧协议）

```json
{
  "psnr_nvs_avg": 31.977596791585288,
  "ssim_nvs_avg": 0.9745234529177348,
  "lpips_nvs_avg": 0.030334497801959515
}
```

自动产物检查：PASS。原版NVS动态复用BVH/静态固定time的限制保留；统一actual-time RGB结果另报。

代表图：[四类可视化](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/inspection/four_categories.png)、[材质与HDR](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/inspection/materials_hdr.png)、[NVS对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/inspection/nvs_comparison.png)。完整媒体路径及帧数见inspection/artifact_checks.json。

### RGB（15测试帧实际相机/time，逐时刻更新BVH）

```json
{
  "psnr": 32.43576737220712,
  "ssim": 0.9753887176513671,
  "lpips": 0.031742295374472936,
  "psnr_foreground": 23.152717213386588,
  "ssim_foreground_bbox": 0.8799428145090739,
  "lpips_foreground_bbox": 0.13511917193730671
}
```

### 法线与深度真值评测

```json
{
  "foreground": {
    "frames": 15,
    "gt_pixels": 396456,
    "normal": {
      "angular_deg": {
        "frame_mean": 12.888626231471337,
        "pixel_mean": 12.802741127725366,
        "valid_frames": 15,
        "valid_pixels": 393547
      },
      "axis_deg": {
        "frame_mean": 12.824494740395298,
        "pixel_mean": 12.75189464768561,
        "valid_frames": 15,
        "valid_pixels": 393547
      },
      "component_l1": {
        "frame_mean": 0.10525496854044723,
        "pixel_mean": 0.10390705688020332,
        "valid_frames": 15,
        "valid_pixels": 393547
      },
      "cosine_loss": {
        "frame_mean": 0.0562142419403171,
        "pixel_mean": 0.05372319773984887,
        "valid_frames": 15,
        "valid_pixels": 393547
      },
      "angular_with_missing_deg": {
        "frame_mean": 14.483294726265937,
        "pixel_mean": 14.029552743792337,
        "valid_frames": 15,
        "valid_pixels": 396456
      },
      "depth_derived_angular_deg": {
        "frame_mean": 18.066139202118922,
        "pixel_mean": 17.55873274133037,
        "valid_frames": 15,
        "valid_pixels": 377497
      },
      "depth_derived_axis_deg": {
        "frame_mean": 16.77437759951098,
        "pixel_mean": 16.399797622971285,
        "valid_frames": 15,
        "valid_pixels": 377497
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.09382693985755114,
        "mae_pixel_mean": 0.09741781400963634,
        "abs_rel_frame_mean": 0.011809929301799249,
        "coverage": 0.9926624896583732,
        "valid_frames": 15,
        "valid_pixels": 393547
      },
      "median": {
        "mae_frame_mean": 0.09411387778187467,
        "mae_pixel_mean": 0.09740961257093635,
        "abs_rel_frame_mean": 0.011854355060861275,
        "coverage": 0.9926624896583732,
        "valid_frames": 15,
        "valid_pixels": 393547
      }
    },
    "normal_coverage": 0.9926624896583732
  },
  "interior": {
    "frames": 15,
    "gt_pixels": 354411,
    "normal": {
      "angular_deg": {
        "frame_mean": 12.139750475709992,
        "pixel_mean": 12.151631477170573,
        "valid_frames": 15,
        "valid_pixels": 353823
      },
      "axis_deg": {
        "frame_mean": 12.10875523995423,
        "pixel_mean": 12.125516328855793,
        "valid_frames": 15,
        "valid_pixels": 353823
      },
      "component_l1": {
        "frame_mean": 0.09967499377094034,
        "pixel_mean": 0.09898139184249176,
        "valid_frames": 15,
        "valid_pixels": 353823
      },
      "cosine_loss": {
        "frame_mean": 0.051049609457684955,
        "pixel_mean": 0.04919011128035219,
        "valid_frames": 15,
        "valid_pixels": 353823
      },
      "angular_with_missing_deg": {
        "frame_mean": 12.568440214581482,
        "pixel_mean": 12.430107147201761,
        "valid_frames": 15,
        "valid_pixels": 354411
      },
      "depth_derived_angular_deg": {
        "frame_mean": 18.12530171826825,
        "pixel_mean": 17.629983498112296,
        "valid_frames": 15,
        "valid_pixels": 338168
      },
      "depth_derived_axis_deg": {
        "frame_mean": 16.826349667725008,
        "pixel_mean": 16.477957297928455,
        "valid_frames": 15,
        "valid_pixels": 338168
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.09303592566177882,
        "mae_pixel_mean": 0.09791994472331408,
        "abs_rel_frame_mean": 0.011691152733634518,
        "coverage": 0.9983409092832898,
        "valid_frames": 15,
        "valid_pixels": 353823
      },
      "median": {
        "mae_frame_mean": 0.09321363983212537,
        "mae_pixel_mean": 0.09775723978489065,
        "abs_rel_frame_mean": 0.01171946913606605,
        "coverage": 0.9983409092832898,
        "valid_frames": 15,
        "valid_pixels": 353823
      }
    },
    "normal_coverage": 0.9983409092832898
  },
  "cloth_foreground": {
    "frames": 15,
    "gt_pixels": 149441,
    "normal": {
      "angular_deg": {
        "frame_mean": 25.767123409501046,
        "pixel_mean": 23.603623658088768,
        "valid_frames": 15,
        "valid_pixels": 146964
      },
      "axis_deg": {
        "frame_mean": 25.632172153944413,
        "pixel_mean": 23.515594419134043,
        "valid_frames": 15,
        "valid_pixels": 146964
      },
      "component_l1": {
        "frame_mean": 0.2124701138339971,
        "pixel_mean": 0.19269399878409818,
        "valid_frames": 15,
        "valid_pixels": 146964
      },
      "cosine_loss": {
        "frame_mean": 0.1426104567906411,
        "pixel_mean": 0.11665297551652591,
        "valid_frames": 15,
        "valid_pixels": 146964
      },
      "angular_with_missing_deg": {
        "frame_mean": 38.62909282939036,
        "pixel_mean": 26.19590973887593,
        "valid_frames": 15,
        "valid_pixels": 149441
      },
      "depth_derived_angular_deg": {
        "frame_mean": 33.62736686108812,
        "pixel_mean": 24.589209120246576,
        "valid_frames": 15,
        "valid_pixels": 139203
      },
      "depth_derived_axis_deg": {
        "frame_mean": 28.872804740248217,
        "pixel_mean": 23.24671260974379,
        "valid_frames": 15,
        "valid_pixels": 139203
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.2685183474352393,
        "mae_pixel_mean": 0.2016207411119134,
        "abs_rel_frame_mean": 0.035277793591164765,
        "coverage": 0.9834248967820076,
        "valid_frames": 15,
        "valid_pixels": 146964
      },
      "median": {
        "mae_frame_mean": 0.2817819832934632,
        "mae_pixel_mean": 0.20395090186353546,
        "abs_rel_frame_mean": 0.0370484425148291,
        "coverage": 0.9834248967820076,
        "valid_frames": 15,
        "valid_pixels": 146964
      }
    },
    "normal_coverage": 0.9834248967820076
  },
  "cloth_interior": {
    "frames": 15,
    "gt_pixels": 130265,
    "normal": {
      "angular_deg": {
        "frame_mean": 24.635117225772557,
        "pixel_mean": 22.89812018606787,
        "valid_frames": 15,
        "valid_pixels": 129677
      },
      "axis_deg": {
        "frame_mean": 24.584902273830952,
        "pixel_mean": 22.863311403621086,
        "valid_frames": 15,
        "valid_pixels": 129677
      },
      "component_l1": {
        "frame_mean": 0.20397461154789195,
        "pixel_mean": 0.1869119322240448,
        "valid_frames": 15,
        "valid_pixels": 129677
      },
      "cosine_loss": {
        "frame_mean": 0.13131329174096415,
        "pixel_mean": 0.1095239991350721,
        "valid_frames": 15,
        "valid_pixels": 129677
      },
      "angular_with_missing_deg": {
        "frame_mean": 37.188962180627406,
        "pixel_mean": 23.60725852200302,
        "valid_frames": 15,
        "valid_pixels": 130265
      },
      "depth_derived_angular_deg": {
        "frame_mean": 30.514233600942454,
        "pixel_mean": 23.131327860219645,
        "valid_frames": 14,
        "valid_pixels": 122686
      },
      "depth_derived_axis_deg": {
        "frame_mean": 27.916447166704092,
        "pixel_mean": 22.31746333930973,
        "valid_frames": 14,
        "valid_pixels": 122686
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.2454175151474507,
        "mae_pixel_mean": 0.19999766578325137,
        "abs_rel_frame_mean": 0.031424446541419564,
        "coverage": 0.9954861244386443,
        "valid_frames": 15,
        "valid_pixels": 129677
      },
      "median": {
        "mae_frame_mean": 0.26219998252132687,
        "mae_pixel_mean": 0.20200568860030768,
        "abs_rel_frame_mean": 0.03359342353219149,
        "coverage": 0.9954861244386443,
        "valid_frames": 15,
        "valid_pixels": 129677
      }
    },
    "normal_coverage": 0.9954861244386443
  }
}
```

法线为世界空间角度MAE（度）；深度为相机正Z、原世界单位；均报告覆盖率，详见evaluation/geometry/metrics.json。

### Albedo 真值评测

完整未缩放线性MAE、材质均衡bMAE和训练帧尺度诊断：evaluation/albedo/iteration_55000/metrics.json。真值仅离线评分，没有反馈训练。

### 真实人工目检

- rgb：平展、折叠、下垂的动态变化可见，静态椅子和底盘主体保持；布料局部不均匀。
- alpha：初始及中间时刻边缘毛糙、低alpha局部和薄褶皱锯齿较明显。
- normal：随姿态改变的褶皱颜色可辨，薄边局部不规则；数值角度评分仍有较大误差。
- separation_large：红色运动布料与绿色静态主体总体可辨，局部混合保留。
- albedo：红布比03均匀一些，仍有暗斑及折叠明暗残留，底盘有云状灰斑。
- roughness：局部对比明显；未做roughness真值评分。
- hdr：暖色响应及动态姿态保留，毛糙边界和局部斑驳仍可见。
- nvs：三个代表姿态基本对应GT；布料暗斑、褶皱亮度及底盘模糊阴影与GT有差异，末帧比03均匀但仍有局部误差。

人工结论：PASS；限制：目检仅覆盖固定相机frame_0008的t0/t60/t119，以及NVS测试索引0/7/14代表图；没有逐帧人工审阅全部视频。PASS表示复现流程、产物与代表帧检查完成，未设材质/几何质量阈值，不表示真值重建准确。。

<!-- gt-lambertian-followup -->
## 本组Stage1的后续GT Lambertian Stage2

本组35k另作只读来源，新增独立实验1006-09，统一固定光强6.030241533058244。20k更新至55k及各种MAE评测已完成；原版训练结果保留。

[训练记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian训练记录.md)；[后续报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-09-CV4DL-固定共用光强GT-Lambertian-Stage2评测报告.md)。
