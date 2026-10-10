# 1006-09 only_clothV4_Dynamic_Lambertian Stage2 GT灯位 Lambertian 训练与验收记录

编制：2026-10-06T16:35:37.981595+09:00（Asia/Tokyo）。**状态：COMPLETED；训练、全量评测及代表图检查完成，流程PASS。**

## 来源与输出

- 统一计划：[1006-10-V4随机方块Stage1来源的Stage2-GT灯位Lambertian训练核对.md](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1006-10-V4随机方块Stage1来源的Stage2-GT灯位Lambertian训练核对.md)。
- 只读 Stage1：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model_mlp`，iter35000；Gaussian 48,672 点。
- PLY：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model_mlp/point_cloud/iteration_35000/point_cloud.ply`；SHA256：`348274da9a6075f8dbd75045d54533f56c9f6b6ade22b9ededd40287137329d4`。
- deformation：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model_mlp/deform/iteration_35000/deform.pth`；SHA256：`09a92c9192c7ed6652b401dd0c1b9d9b429830fa47e2c2bcb9c6255b1ce9835a`。
- source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view`；沿用原 data_view，包括原10万随机输入PLY，图像加载显式使用其 images 链接。
- GT灯位：`/home/lihan/data/LH-data/static/only_clothV4_Dynamic_Lambertian/lights.json`，120帧；GT albedo/normal/深度仅用于离线评分。
- 本组输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian`；命令 model_path=`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model`，入口自动追加 `_mlp`，实际模型=`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp`。
- 服务器 venus；Conda lumimotion；执行代码为当前项目 1001-stage2-perlight，核查时 HEAD `891b7dcf82876a6e3b4ad3a63fb0a3f66fddd674`；原 Stage1 来自原版 LumiMotion main，版本来源见原实验 README。
- [CPU核查](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/cpu_preflight.json)：权重有限、32个形变Tensor、120张RGBA、105/15无交集、GT灯位及15帧深度缓存哈希通过；四组GPU相机/FOV、GT方向、冻结/梯度检查均通过；所有模型及来源哈希核验通过。

## 训练设置

35000→55000（新增20,000次Adam更新）；resolution=2，640×360，黑背景。`photometric_lambertian / lambertian_gt / gt_directional`；每Gaussian三通道sigmoid albedo时间共享，LR=0.001；冻结位置、尺度、旋转、opacity、SH与全部形变网络，动态帧仍按真实time计算形变。固定Stage1 iter35000用于形变调用。

损失：0.8前景L1 + 0.2遮罩图DSSIM + 0.001初始albedo MSE先验；光照平滑0；GT alpha>0.5作监督mask。无Alpha、GT法线、GT深度或GT albedo训练损失。初始化材质取Stage1 SH DC颜色代理；它含有光照影响。

GT灯位减去本组Stage1 canonical点云中心并归一化，每帧使用一个统一方向。按用户要求，四组共用固定光强 **6.030241533058244**。标定参考1006-01 Stage1 iter35000，用其105训练帧RGB、初始SH DC材质及冻结法线拟合一个正标量后固定；四份GT灯光元数据与帧划分完全一致。测试帧和GT albedo不参与标定。本组命令显式传入该正数，不再各组自动标定；[共用标定记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/shared_light_calibration.json)保留来源、有效像素与冻结验证，实际日志位于[成功日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/logs/shared_light_calibration_retry1.log)。**这不是直接使用Blender AREA能量1800的完整GT面积光模型**，不包含面积积分、距离衰减、投射阴影、镜面及间接光。GT方向近似沿用现有实现。

初始化35000和40000/50000/55000保存完整checkpoint；`test_iterations`在该训练器中不触发在线测试，须单独运行渲染命令。实际最多2组并行，先06/07再08/09；本轮冒烟、正式训练和完整评测均已完成。

## 完整命令

工作目录：`/home/lihan/reproduce/LumiMotion-stage2-perlight`。训练/渲染脚本位于commands/，实际执行命令为*_actual.sh；训练、渲染和全部评测日志已生成。已完成独立200步冒烟，正式重新从原Stage1初始化。

### 正式训练（已执行）

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model_mlp' --load_iter 35000 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --depth_ratio 0 --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode photometric_lambertian --loss_preset lambertian_gt --photometric_light_mode gt_directional --photometric_gt_lights_path /home/lihan/data/LH-data/static/only_clothV4_Dynamic_Lambertian/lights.json --photometric_light_intensity 6.030241533058244 --photometric_albedo_lr 0.001 --lambda_dssim 0.2 --lambda_photometric_albedo_prior 0.001 --lambda_photometric_light_smooth 0 --photometric_log_interval 100 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model' --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 55000
```

### 完整评测与渲染

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp' --load_iter 55000 --task all --frame_key 0008 --relight_frames 72 --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048 --hdr_yaw 0 --hdr_exposure 1
```

### Albedo 真值评测

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/static/only_clothV4_Dynamic_Lambertian/albedo --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/evaluation/albedo' --material_palette /home/lihan/reproduce/LumiMotion-stage2-perlight/arguments/albedo_onlyclothv4.json --split test --calibrate_scale --save_float
```

### 法线与深度真值评测

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/static/only_clothV4_Dynamic_Lambertian --gt_depth_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/Archive/1005-01-V4四版本材质与动态几何真值评测/gt/CV4DL' --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --split test --resolution 2 --alpha_threshold 0.5 --interior_radius 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/evaluation/geometry' --save_arrays
```

### 独立200步冒烟（已执行）

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model_mlp' --load_iter 35000 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --depth_ratio 0 --deform-type mlp --sh_degree 3 --hyper_dim 1 --render_mode photometric_lambertian --loss_preset lambertian_gt --photometric_light_mode gt_directional --photometric_gt_lights_path /home/lihan/data/LH-data/static/only_clothV4_Dynamic_Lambertian/lights.json --photometric_light_intensity 6.030241533058244 --photometric_albedo_lr 0.001 --lambda_dssim 0.2 --lambda_photometric_albedo_prior 0.001 --lambda_photometric_light_smooth 0 --photometric_log_interval 100 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model' --iterations 35200 --save_iterations 35200 --test_iterations 35200
```

### 冒烟完整渲染（已执行）

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp' --load_iter 35200 --task all --frame_key 0008 --relight_frames 72 --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

## 产物与验收（已生成）

- 模型：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp/stage2_photometric/iteration_55000/checkpoint.pth`；包括材质、固定光照、Adam、随机状态、来源哈希和冻结指纹。模型依赖上述Stage1路径，不移动来源。
- 配置与训练日志：model_mlp/stage2_config.json、cfg_args、losses.jsonl；共用光强出处为 inspection/shared_light_calibration.json，显式正光强不会自动生成模型目录 calibration.json；进程日志 logs/train.log，渲染及评测日志 logs/render.log、albedo.log、geometry.log。
- 渲染目录：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp/renders_stage2/ours_55000`；eval/15测试帧及RGB指标；insights/120帧 RGB/Alpha/normal/albedo；directional/固定frame0008的72帧转灯；hdr/120帧GoldenBay，2048确定性球面样本，共6段视频。
- 代表路径：insights/{rgb,alpha,normal,albedo}/{0001,0061,0120}.png；eval/{0008,0064,0120}_{render,gt,mask}.png；insights/{rgb,alpha,normal,albedo}.mp4，directional/rotating_directional.mp4，hdr/hdr.mp4（均位于上述渲染目录）。
- RGB：全图PSNR/SSIM/LPIPS及前景PSNR、遮罩包围框SSIM/LPIPS；Albedo：raw主指标、训练帧拟合scalar/RGB尺度诊断、前景MAE/SSIM和四材质bMAE；normal：角度及XYZ分量MAE；depth：expected/median相机Z MAE、覆盖率及布料指标。保留JSON/CSV和float数据。
- 对照：本组Stage1 35k、同来源原版IR Stage2 55k及本轮Lambertian 55k，15测试帧保持相机/time/mask/GT一致；旧Stage1 SH颜色是材质代理，明确标注。冻结几何须重新评测核验，不预填旧55k的几何分数。
- 四类目检：已完成首/中/末代表帧真实检查，逐类结论见最终结果。核查首/中/末帧、测试正面与背面、动态褶皱；转灯/HDR补充目检，6视频完整解码。HDR没有独立GT，仅作定性功能检查。
- 最终结论：**PASS（流程/产物/代表帧检查；未设置质量门槛）**。训练、自动产物核查、人工目检和质量限制分别记录；最终流程结论已填写PASS，质量缺陷单列。失败时保留checkpoint、渲染、统计和日志。

## 最终训练、评测与目检结果

正式：2026-10-06T17:09:40.986267+09:00 → 2026-10-06T17:25:30.551612+09:00。35000→55000，实际20,000次Adam更新；4个完整checkpoint、固定GT方向和共用光强通过核查。

raw Albedo MAE由原版IR的0.122299变为0.120561，下降1.42%；材质均衡bMAE由0.091330变为0.085033。前景SSIM 0.879943→0.828440。

rgb：悬空布料、折叠与最终椅背布料均可辨；0008颜色更红但仍斑驳，0064褶皱内侧明显过暗，0120残留色斑，平台阴影与GT不同。

alpha：首帧布料下方有细碎浮点，主要布料/椅子/底盘轮廓完整；折叠和最终轮廓存在，冻结Stage1的软边/碎点仍保留。

normal：首中末布料、椅子及底盘方向可辨；中段褶皱方向有变化，首帧碎点区存在异常小法线斑。本轮保持Stage1法线。

albedo：红布保留红黑斑与深暗折痕，底盘灰黑阴影斑显著，布下碎点仍带红色；材质未完全脱离光照和几何缺陷。

转灯首尾布料亮、正中全黑，存在明显亮暗响应；HDR首/中/末红布和粉橙底盘可辨，但碎点、深折痕和底盘原阴影仍存在。HDR无独立GT，仅作功能检查。

[本组完整报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-09-CV4DL-固定共用光强GT-Lambertian-Stage2评测报告.md)；[全部指标与核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/final_verification.json)。

实际RGB/Alpha MAE补充命令：

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/commands/rgb_mae.py" "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian"
```

实际Albedo SSIM补充命令（随后调用上述RGB/Alpha评分）：

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/commands/albedo_ssim.py" "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian"
```

补充命令日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/logs/albedo_ssim.log`。正式产物共672张诊断/重光照PNG、6段视频，15测试帧；所有视频完整解码。
