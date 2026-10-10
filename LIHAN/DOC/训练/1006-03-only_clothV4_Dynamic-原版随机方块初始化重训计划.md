# 1006-03 only_clothV4_Dynamic 原版随机方块初始化重训计划
<!-- 1006-final-record -->

**最终状态：4/4 COMPLETED，流程及产物验收PASS；训练只有1006-01～04。** 原版SH Stage1 35k、IR Stage2累计55k，两阶段共8组结果、120行逐帧指标已完成。

评测完成：2026-10-06 02:44:28（Asia/Tokyo）；最终核验及文档回填：2026-10-06T11:16:08+09:00。

[完整双阶段报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-01-四组Stage1与Stage2真值评测报告.md)；[阶段CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/metrics.csv)；[最终核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/final_audit.json)。

四组的Stage2全图及前景包围框SSIM均下降，depth和normal MAE均增加；共同有效像素比较也显示几何误差上升。只有1006-04的原始albedo MAE下降（0.138899→0.122299，约11.95%），其bMAE也下降；01、02、03的原始albedo MAE增加。Stage1为含光照的SH基色代理，以上数值变化不能解释为Stage1物理材质恢复更准确。

| 阶段 | 步数 | SSIM↑ | 前景包围框SSIM↑ | Albedo MAE | Depth MAE（世界单位） | Normal MAE（°） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage1 | 35000 | 0.997447 | 0.989456 | 0.158172 | 0.102184 | 11.5613 |
| stage2 | 55000 | 0.975469 | 0.877510 | 0.161104 | 0.108388 | 11.9397 |

目检仅覆盖固定相机frame_0008的t0/t60/t119，以及NVS测试索引0/7/14代表图；没有逐帧人工审阅全部视频。PASS表示复现流程、产物与代表帧检查完成，未设材质/几何质量阈值，不表示真值重建准确。

以下核对计划为历史批准记录；当前结果见“当前执行记录”。

<!-- /1006-final-record -->


文档编制：2026-10-05（Asia/Tokyo）；实验编号按用户指定使用1006。状态：**用户已核对并授权正式训练，跳过冒烟**。编号不表示已经安排10月6日自动启动；用户已明确跳过冒烟，直接正式训练。

## 目标与数据

本组：**动态，原始材质**。对应旧对照：`1003-02-only_clothV4_Dynamic-LumiMotion原版两阶段`。本轮保持1003的算法、图像、相机、划分及训练参数，仅将15万点Blender表面初始化换成原版10万随机方块初始化。

- 原始数据：`/home/lihan/data/LH-data/static/only_clothV4_Dynamic`。
- 转换数据：`/home/lihan/data/LH-data/transfer-dynamic/only_clothV4_Dynamic`。
- 实际 source_path：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view`。
- 正式输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段`；命令 model_path：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model`；实际模型：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model_mlp`。
- 冒烟输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块冒烟`，使用独立 data_view 和独立随机生成点云，正式训练不从冒烟 checkpoint 续训。
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
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage1 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

### stage1_insights

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_stage1_insights --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 35000 --depth_ratio 0.0
```

### stage2

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --gt_alpha_mask_as_scene_mask --iterations 55000 --load_iter 35000 --test_iterations 55000 --save_iterations 40000 50000 55000 --diffuse_sample_num 512 --depth_ratio 0.0
```

### materials

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_materials --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 55000 --depth_ratio 0.0
```

### relight_hdr

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_relight_with_hdr --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 55000 --diffuse_sample_num 2048 --depth_ratio 0.0 --hdr /home/lihan/reproduce/LumiMotion/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

### eval_nvs

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.eval_nvs_dynamic --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder chapel_day_4k_32x16_rot0 --load_iter 55000 --diffuse_sample_num 2048 --depth_ratio 0.0
```

## 原准备冒烟方案（用户已要求跳过）

完整冒烟命令见`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块冒烟/planned_commands.json`及commands/*.sh。仅训练终点和save/test/load iteration对应改为1100/1120，其他参数与正式一致。用户要求跳过冒烟；正式独立从随机初始化开始，不复制任何旧PLY或模型。

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
- 正式开始：2026-10-06T00:38:19+09:00；结束：2026-10-06T02:41:09+09:00。
- 冒烟：**SKIPPED_BY_USER**，未执行，不作为启动条件。正式独立从默认随机方块开始。
- 实际日志：logs/stage1.log、stage1_insights.log、stage2.log、materials.log、relight_hdr.log、eval_nvs.log、artifact_checks.log、rgb_actual_time.log、geometry_gt.log、albedo_gt.log。
- 完整训练/渲染命令：上方核对计划及commands/*.sh；新增评测实际命令：evaluation_commands.json。实际代码来源：code_commit.txt、code_status.txt、code_diff.patch。
- 默认初始化实证：PASS_DEFAULT_RANDOM_CUBE，100,000点，bbox=[[-1.2999619245529175, -1.2999482154846191, -1.299992322921753], [1.2999730110168457, 1.2999974489212036, 1.2999688386917114]]，seed=0坐标核验通过，零法线；input.ply SHA256 `f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`。

### 原版 NVS（沿用旧协议）

```json
{
  "psnr_nvs_avg": 31.12989959716797,
  "ssim_nvs_avg": 0.972965395450592,
  "lpips_nvs_avg": 0.030412585536638895
}
```

自动产物检查：PASS。原版NVS动态复用BVH/静态固定time的限制保留；统一actual-time RGB结果另报。

代表图：[四类可视化](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/inspection/four_categories.png)、[材质与HDR](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/inspection/materials_hdr.png)、[NVS对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/inspection/nvs_comparison.png)。完整媒体路径及帧数见inspection/artifact_checks.json。

### RGB（15测试帧实际相机/time，逐时刻更新BVH）

```json
{
  "psnr": 32.74313979414034,
  "ssim": 0.975468901793162,
  "lpips": 0.03131094059596459,
  "psnr_foreground": 23.41876532946128,
  "ssim_foreground_bbox": 0.877509864171346,
  "lpips_foreground_bbox": 0.13916294823090236
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
        "frame_mean": 11.939719816694152,
        "pixel_mean": 11.63054562257528,
        "valid_frames": 15,
        "valid_pixels": 395162
      },
      "axis_deg": {
        "frame_mean": 11.858313062575458,
        "pixel_mean": 11.568770160601037,
        "valid_frames": 15,
        "valid_pixels": 395162
      },
      "component_l1": {
        "frame_mean": 0.0975318549857994,
        "pixel_mean": 0.09445154007124072,
        "valid_frames": 15,
        "valid_pixels": 395162
      },
      "cosine_loss": {
        "frame_mean": 0.04858741573504477,
        "pixel_mean": 0.04494880866387819,
        "valid_frames": 15,
        "valid_pixels": 395162
      },
      "angular_with_missing_deg": {
        "frame_mean": 12.541157016173257,
        "pixel_mean": 12.180089768620206,
        "valid_frames": 15,
        "valid_pixels": 396456
      },
      "depth_derived_angular_deg": {
        "frame_mean": 17.75344969527056,
        "pixel_mean": 17.206573211145283,
        "valid_frames": 15,
        "valid_pixels": 378917
      },
      "depth_derived_axis_deg": {
        "frame_mean": 16.55351855289451,
        "pixel_mean": 16.119583580973416,
        "valid_frames": 15,
        "valid_pixels": 378917
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.10838754643351152,
        "mae_pixel_mean": 0.11726101634430987,
        "abs_rel_frame_mean": 0.013761293997250868,
        "coverage": 0.9967360816837177,
        "valid_frames": 15,
        "valid_pixels": 395162
      },
      "median": {
        "mae_frame_mean": 0.10940143518485483,
        "mae_pixel_mean": 0.11801840154750627,
        "abs_rel_frame_mean": 0.013904904670303926,
        "coverage": 0.9967360816837177,
        "valid_frames": 15,
        "valid_pixels": 395162
      }
    },
    "normal_coverage": 0.9967360816837177
  },
  "interior": {
    "frames": 15,
    "gt_pixels": 354411,
    "normal": {
      "angular_deg": {
        "frame_mean": 11.197625656377573,
        "pixel_mean": 11.014475400106464,
        "valid_frames": 15,
        "valid_pixels": 354411
      },
      "axis_deg": {
        "frame_mean": 11.145107580284211,
        "pixel_mean": 10.974538020459473,
        "valid_frames": 15,
        "valid_pixels": 354411
      },
      "component_l1": {
        "frame_mean": 0.091979473022953,
        "pixel_mean": 0.0898033013579634,
        "valid_frames": 15,
        "valid_pixels": 354411
      },
      "cosine_loss": {
        "frame_mean": 0.04327709152147823,
        "pixel_mean": 0.04053702751019614,
        "valid_frames": 15,
        "valid_pixels": 354411
      },
      "angular_with_missing_deg": {
        "frame_mean": 11.197625656377573,
        "pixel_mean": 11.014475400106464,
        "valid_frames": 15,
        "valid_pixels": 354411
      },
      "depth_derived_angular_deg": {
        "frame_mean": 17.85492448487485,
        "pixel_mean": 17.328920592401136,
        "valid_frames": 15,
        "valid_pixels": 338675
      },
      "depth_derived_axis_deg": {
        "frame_mean": 16.658583704619637,
        "pixel_mean": 16.25354780905641,
        "valid_frames": 15,
        "valid_pixels": 338675
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.10697875419086925,
        "mae_pixel_mean": 0.11780330216543108,
        "abs_rel_frame_mean": 0.013589043352532979,
        "coverage": 1.0,
        "valid_frames": 15,
        "valid_pixels": 354411
      },
      "median": {
        "mae_frame_mean": 0.10789215780223442,
        "mae_pixel_mean": 0.11846662781876299,
        "abs_rel_frame_mean": 0.013716826493932447,
        "coverage": 1.0,
        "valid_frames": 15,
        "valid_pixels": 354411
      }
    },
    "normal_coverage": 1.0
  },
  "cloth_foreground": {
    "frames": 15,
    "gt_pixels": 149441,
    "normal": {
      "angular_deg": {
        "frame_mean": 21.89495316642246,
        "pixel_mean": 20.157029091212074,
        "valid_frames": 15,
        "valid_pixels": 148670
      },
      "axis_deg": {
        "frame_mean": 21.700452584328914,
        "pixel_mean": 20.04983100412013,
        "valid_frames": 15,
        "valid_pixels": 148670
      },
      "component_l1": {
        "frame_mean": 0.18192786033473204,
        "pixel_mean": 0.16493454467265234,
        "valid_frames": 15,
        "valid_pixels": 148670
      },
      "cosine_loss": {
        "frame_mean": 0.11228336311199276,
        "pixel_mean": 0.09211192075105908,
        "valid_frames": 15,
        "valid_pixels": 148670
      },
      "angular_with_missing_deg": {
        "frame_mean": 23.74387413132181,
        "pixel_mean": 20.981695217447015,
        "valid_frames": 15,
        "valid_pixels": 149441
      },
      "depth_derived_angular_deg": {
        "frame_mean": 27.756842749033915,
        "pixel_mean": 23.11148484296694,
        "valid_frames": 15,
        "valid_pixels": 140744
      },
      "depth_derived_axis_deg": {
        "frame_mean": 25.204557344375242,
        "pixel_mean": 21.963076244769255,
        "valid_frames": 15,
        "valid_pixels": 140744
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.2465775825121237,
        "mae_pixel_mean": 0.25272415701828366,
        "abs_rel_frame_mean": 0.03172183570750091,
        "coverage": 0.9948407732817633,
        "valid_frames": 15,
        "valid_pixels": 148670
      },
      "median": {
        "mae_frame_mean": 0.25534232903462023,
        "mae_pixel_mean": 0.25743979966062347,
        "abs_rel_frame_mean": 0.03285772294105973,
        "coverage": 0.9948407732817633,
        "valid_frames": 15,
        "valid_pixels": 148670
      }
    },
    "normal_coverage": 0.9948407732817633
  },
  "cloth_interior": {
    "frames": 15,
    "gt_pixels": 130265,
    "normal": {
      "angular_deg": {
        "frame_mean": 20.811039067935024,
        "pixel_mean": 19.548579332041086,
        "valid_frames": 15,
        "valid_pixels": 130265
      },
      "axis_deg": {
        "frame_mean": 20.711973525728954,
        "pixel_mean": 19.48819014820254,
        "valid_frames": 15,
        "valid_pixels": 130265
      },
      "component_l1": {
        "frame_mean": 0.1737319811651041,
        "pixel_mean": 0.160014599201471,
        "valid_frames": 15,
        "valid_pixels": 130265
      },
      "cosine_loss": {
        "frame_mean": 0.10206915986544064,
        "pixel_mean": 0.08530668020146635,
        "valid_frames": 15,
        "valid_pixels": 130265
      },
      "angular_with_missing_deg": {
        "frame_mean": 20.811039067935024,
        "pixel_mean": 19.548579332041086,
        "valid_frames": 15,
        "valid_pixels": 130265
      },
      "depth_derived_angular_deg": {
        "frame_mean": 25.375653503970966,
        "pixel_mean": 21.930108750127904,
        "valid_frames": 15,
        "valid_pixels": 123193
      },
      "depth_derived_axis_deg": {
        "frame_mean": 24.329970068113237,
        "pixel_mean": 21.205881848270263,
        "valid_frames": 15,
        "valid_pixels": 123193
      }
    },
    "depth": {
      "expected": {
        "mae_frame_mean": 0.24947369387413332,
        "mae_pixel_mean": 0.25290732395990756,
        "abs_rel_frame_mean": 0.03195267337475734,
        "coverage": 1.0,
        "valid_frames": 15,
        "valid_pixels": 130265
      },
      "median": {
        "mae_frame_mean": 0.2576281256916126,
        "mae_pixel_mean": 0.2563886287385972,
        "abs_rel_frame_mean": 0.032996873515457874,
        "coverage": 1.0,
        "valid_frames": 15,
        "valid_pixels": 130265
      }
    },
    "normal_coverage": 1.0
  }
}
```

法线为世界空间角度MAE（度）；深度为相机正Z、原世界单位；均报告覆盖率，详见evaluation/geometry/metrics.json。

### Albedo 真值评测

完整未缩放线性MAE、材质均衡bMAE和训练帧尺度诊断：evaluation/albedo/iteration_55000/metrics.json。真值仅离线评分，没有反馈训练。

### 真实人工目检

- rgb：布料由初始平展到中间折叠再到下垂，椅子和底盘主体稳定；中间时刻布料局部较模糊。
- alpha：中间折叠时布料薄边、局部碎片状低alpha和不均匀轮廓可见。
- normal：法线颜色随布料姿态变化，褶皱可辨；薄边局部呈碎片状，不能宣称全时序连续。
- separation_large：运动布料主要呈红色、静态主体呈绿色，近椅腿局部混合。
- albedo：红布有明显灰白斑块，初始平展及折叠阶段均可见；底盘及座面也有斑驳。
- roughness：局部空间变化明显；未做roughness真值评分。
- hdr：暖色响应和姿态变化保留，但布料斑块仍明显。
- nvs：三帧主体姿态基本跟随GT；布料灰白亮斑、暗斑和模糊边界明显，GT红布更均匀，底盘也有斑驳。

人工结论：PASS；限制：目检仅覆盖固定相机frame_0008的t0/t60/t119，以及NVS测试索引0/7/14代表图；没有逐帧人工审阅全部视频。PASS表示复现流程、产物与代表帧检查完成，未设材质/几何质量阈值，不表示真值重建准确。。

<!-- gt-lambertian-followup -->
## 本组Stage1的后续GT Lambertian Stage2

本组35k另作只读来源，新增独立实验1006-08，统一固定光强6.030241533058244。20k更新至55k及各种MAE评测已完成；原版训练结果保留。

[训练记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian训练记录.md)；[后续报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-08-CV4D-固定共用光强GT-Lambertian-Stage2评测报告.md)。
