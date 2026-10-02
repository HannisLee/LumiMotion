# 0823-01 CV3s 静态布料:从 iter 1 起 GT 光照 Lambertian + 三法线一致性训练方案与执行记录

日期:2026-08-23
服务器/环境:garuda / `lumimotion-garuda`
状态:完成,**PASS**(2026-08-23;训练 3:21:11,渲染与 GT normal 评估已完成)

## 1. 背景与目标

- 数据集:`data/LH-data/transfer-static/only_clothV3`(别名 **CV3s**,clothV3 的静帧版本:布料静止、相机运动、单盏轨道 key light)。
- 目标:验证 **从 iteration 1 起直接走 Lambertian 管线**(`photometric_start_iter=1`)的可行性,且 **不使用 GT normal 监督**,改由三个 normal consistency loss 约束独立法线(`photometric_normal`)。
- 历史对照:0813 的 `photometric_start_iter=1` 自由几何实验为 `FAILED`,主因是 (a) 当时使用了错误标定强度 2.9(正确值为 5.5043499);(b) 自一致 normal 正则 `start_normal_reg=8000` 启动过晚,albedo 已漂移。本方案针对性修正这两点。
- 与现行标准协议(10001 切换、分阶段 albedo→normal)不同,本实验为探索性配置;标准协议仍保留为对照基线。

## 2. 数据准备(已完成)

- 转换脚本:`scripts/prepare_lh_dynamic.py`(逐帧相机格式;静态场景转换复用动态脚本)。
- 转换命令:

```bash
conda run --no-capture-output -n lumimotion-garuda \
python scripts/prepare_lh_dynamic.py \
  data/LH-data/static/only_clothV3 \
  data/LH-data/transfer-static/only_clothV3 \
  --test-stride 8 --camera-extent 1.0
```

- 结果:120 帧 → train 105 / test 15(test 为第 8,16,…,120 帧),1280×720,albedo soft alpha,相机轨迹半径 2.4084(与 dynamic CV3 完全一致,同一场景同一相机路径)。
- `points3d.ply`:5000 点(仅 xyz),bbox `[-1.424,-1.445,-0.180] ~ [1.447,1.448,1.932]`,由用户手工放置。
- 说明:仓库的 `--eval` 训练模式使用 105 帧训练、15 帧评测;不带 `--eval` 时 reader 会把 test 帧并入训练(`scene/dataset_readers.py` 的 `readNerfSyntheticInfo`)。

## 3. 训练配置决策

| 项目 | 取值 | 理由 |
| --- | --- | --- |
| 光照 | `gt_directional` + `lights.json`(static) | GT light 要求 |
| 强度 | `5.5043499` | 场景与 dynamic CV3 相同,用户确认复用标定值,不重新标定 |
| 光色 | `1.0,1.0,1.0` | 白灯 |
| Lambertian 切换 | `photometric_start_iter=1` | 用户要求 iter 1 起 |
| GT normal 监督 | **禁用**(`lambda_photometric_gt_normal=0`,不传 dir) | 用户要求;GT EXR 仅离线评估 |
| `lambda_photometric_normal_init` | `0.0` | iter 1 时 GS 初始法线无意义,trust-region 先验不启用 |
| albedo / normal LR | `0.001` / `0.0001`,iter 1 起联合训练 | 沿用既有标定;不用分阶段协议 |
| `start_normal_reg` | `500`(原默认 8000) | 0813 失败主因之一;几何自一致正则提前到 warm_up 后 |
| 基础超参 | 指导标准值(35000 iter、densify 1500–5000、max_gaussians 20000 等) | 与基线可比 |
| 训练模式 | `--eval`、`--gt_alpha_mask_as_scene_mask`、`--resolution 2` | 指导标准 |

## 4. 三个 normal consistency loss(本次新增代码)

全部作用于世界系独立法线(`photometric_normal`,经 `photometric_normal_raw` 渲染解码),不消费任何 GT normal:

1. **GS 几何自一致(既有代码,仅提前调度)**
   `1-cos(rend_normal, surf_normal)`,λ=0.02,另含 distortion λ=1000;由 `--start_normal_reg 500` 提前启用(位置:`scripts/loss.py` 的 `compute_stage1_loss()`;2026-08-23 前位于 `scripts/train_stage1.py` 的 `train_step`)。
2. **photometric_normal_live(新增)**
   独立法线渲染图与同帧深度导出法线 `surf_normal`(detach)的 alpha 掩码 cosine,λ=`0.01`,iter `500` 起。作为替代 GT 监督的几何锚:独立法线必须跟随几何演化。
3. **photometric_normal_mv(新增,静帧场景专属)**
   多视角重投影一致:当前帧深度反投影到世界系 → 投影进随机抽取的另一训练相机,在双端 alpha≥0.5 且深度差 ≤ `0.1` 的像素上做世界系法线 cosine,λ=`0.02`,iter `1000` 起、`2000` 步线性升满;配帧渲染在 `torch.no_grad()` 下进行(目标侧不回传梯度)。布料静止 + 相机运动使该约束成立,直接压缩 albedo/normal 模糊。

新增参数(`arguments/__init__.py`,均自动暴露为 CLI):

```text
lambda_photometric_normal_live = 0.0        photometric_normal_live_start_iter = 500
photometric_normal_live_alpha_threshold = 0.5
lambda_photometric_normal_mv = 0.0          photometric_normal_mv_start_iter = 1000
photometric_normal_mv_ramp_iters = 2000     photometric_normal_mv_alpha_threshold = 0.5
photometric_normal_mv_depth_tol = 0.1        photometric_normal_mv_interval = 1
```

实现位置:`scripts/loss.py` 的 `render_world_normal_map()`、`normal_consistency_terms()`(2026-08-23 前为 `scripts/train_stage1.py` 的 `render_world_normal_map()`、`normal_consistency_losses()`,行为不变);挂载在 `photometric_normal_init` 之后,新 loss 进入 `audit_loss_terms` 与 TensorBoard(`photometric_normal_consistency/*`)。该组合现也可用 `--loss_preset lambertian_normal3` 一键启用(等价于本文 live/mv 两个 λ 的默认取值)。

注意:渲染器输出的 `photometric_normal` 是 camera-facing 的,跨视角比较必须用世界系 `photometric_normal_raw`,实现已按此处理。

## 5. 冒烟测试

目录:`output/smoke_test/0823-01-CV3s-lambert-iter1-3ncons/`(命令见同目录 `run.sh`)。

- 1200 iter;live iter 500 起、mv iter 1000 起,全部新代码路径在冒烟内被触发;
- `--gradient_audit_interval 50`,核对 `gradient_audit.csv` 中 `photometric_normal_live`、`photometric_normal_mv` 两项的 loss 值与梯度量级;
- 验收:数据装载无报错、三项 loss 非 NaN 且量级合理、显存稳定无 OOM、checkpoint 正常写出。

冒烟结论:**PASS**(2026-08-23)

- 60 步调试冒烟 + 1200 步完整冒烟均正常退出(`EXIT=0`),三项新代码路径全部触发;
- 收敛:PSNR 14.6(iter 1)→ 31.12(iter 1000,SSIM 0.978 / LPIPS 0.033),loss 0.118 → 0.045,无 NaN;
- 梯度审计(`gradient_audit.csv`):
  - `photometric_normal_live`(加权)0.0047 → 0.0033~0.0038 稳定,normal 参数组梯度量级 ~2.5e-4,albedo 组梯度为 0(符合预期,该 loss 不含 albedo);
  - `photometric_normal_mv`(加权)iter 1000 为 3.5e-6(法线近一致),iter 1050~1100 升至 7e-4~9e-4——法线开始空间分化后约束自动生效,符合设计;
- 数据装载修正:用户提供的 `points3d.ply` 仅含 xyz,仓库 `fetchPly` 需要颜色/法线属性,裸 `except` 会静默置 `pcd=None` 导致进程退出。已补全属性(灰色 128 + 零法线),原始文件备份为 `points3d_xyz_only.ply.bak`;
- 迭代速度 ~3.2 it/s(含 mv 配帧渲染),35000 iter 预计 ~3.2 小时;显存稳定,未 OOM。

## 6. 正式训练

- 输出目录:`output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/`(实验 README 已建立)
- 启动方式:`tmux new-session -d -s cv3s_0823 "bash output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/run.sh"`;监控 `tmux attach -t cv3s_0823` 或 `tail -f output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/train_stage1.log`。
- 命令:见该目录 `run.sh`;日志 `train_stage1.log`;模型目录自动加 `_mlp` 后缀。
- 完整训练命令:

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n lumimotion-garuda \
python -u -m scripts.train_stage1 \
  --source_path data/LH-data/transfer-static/only_clothV3 \
  --model_path output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/CV3s_stage1 \
  --train_light_folder images --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 --iterations 35000 --warm_up 500 \
  --densify_from_iter 1500 --densify_until_iter 5000 --densification_interval 200 \
  --densify_grad_threshold 0.0004 --opacity_reset_interval 3000 --min_opacity 0.01 \
  --prune_from_iter -1 --max_gaussians 20000 --binarization_warm_up 1000 \
  --lambda_separation 0.005 --d_xyz_loss_weight 0.001 --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 --start_normal_reg 500 \
  --render_mode photometric_lambertian --photometric_start_iter 1 \
  --photometric_light_mode gt_directional \
  --photometric_gt_lights_path data/LH-data/static/only_clothV3/lights.json \
  --photometric_gt_light_intensity 5.5043499 --photometric_gt_light_color 1.0,1.0,1.0 \
  --photometric_albedo_lr 0.001 --photometric_normal_lr 0.0001 \
  --photometric_normal_start_iter 1 --lambda_photometric_normal_init 0.0 \
  --lambda_photometric_normal_live 0.01 --photometric_normal_live_start_iter 500 \
  --lambda_photometric_normal_mv 0.02 --photometric_normal_mv_start_iter 1000 \
  --photometric_normal_mv_ramp_iters 2000 \
  --gradient_audit_interval 25 \
  --test_iterations 500 1000 5000 10000 20000 30000 35000 \
  --save_iterations 499 500 1000 5000 10000 20000 30000 35000 \
  --quiet \
  2>&1 | tee output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/train_stage1.log
```

## 7. 验收计划

1. 训练中:梯度审计核对三个 loss 项;Gaussian 数量曲线(初始≈5k,5000 后≤20k);`d_xyz` 应接近 0(静帧)。
2. 训练后:
   - `scripts.render_stage1_insights --render_mode photometric_lambertian` 全时序渲染(120 帧,RGB/alpha/albedo/normal/separation);
   - GT normal 离线角度误差(`scripts/eval_stage1_normals_gt.py` 等价流程,GT EXR 仅评估不入损失);
   - light direction 角度误差;
   - 四类可视化目检 + 定量指标写入实验 README,给出 PASS/FAILED。

## 8. 风险与说明

- `photometric_start_iter=1` 属探索配置,指导文档中该模式仅作为历史 baseline 保留;本实验结论不改变标准协议。
- 静帧场景为仓库首次训练,deformation 应收敛到近零;若 `d_xyz` 异常或点数异常,优先检查数据/PLY 对齐。
- garuda 四张 A40 当前被 `relight_svd` 任务占满(100% 利用率),本次训练挤用剩余显存最大的 GPU,若 OOM 需等卡。
- 影响面说明:新增代码仅在对应 λ>0 时激活,默认值全 0,不影响任何既有实验与训练结果。

## 9. 训练结果与验收结论(2026-08-23)

训练在 garuda GPU 2 完成 35000 iter(3:21:11;tmux 会话 `cv3s_0823`,训练后 watcher 会话被系统回收,渲染与评估改为手动执行,结果不受影响)。

### 9.1 定量指标

| 指标 | 数值 |
| --- | --- |
| Best PSNR(iter 30000) | 47.65 |
| SSIM / MS-SSIM | 0.9987 / 0.9997 |
| LPIPS / Alex-LPIPS | 0.0045 / 0.0012 |
| GT normal 角度误差(120 帧平均,仅离线评估) | mean 19.23° / median 15.60° / p95 59.90° |
| GT normal 逐帧 mean 范围 | 17.18° ~ 21.79°(train 19.23° / test 19.19°,无过拟合差) |
| alpha 覆盖率 | mean 0.1717(min 0.1678 / max 0.1804),时序差 2.6e-4 |

对照:dynamic CV3 标准分阶段协议(0817-04,无 GT 监督)为 mean 28.69° / median 22.61° / p95 68.81°;本实验三项全面更优(场景不同,仅作量级参照)。

### 9.2 渲染与评估产物

- 全时序渲染(120 帧 × RGB/alpha/normals/separation_large/separation_small + 7 个视频 + contact sheets):
  `output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/CV3s_stage1_mlp/renders_stage1_insights/ours_35000/`
- GT normal 离线评估:`output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/normal_gt_eval_independent/ours_35000/`
  (`normal_metrics.json` + gs/gt/error contact sheets)
- 日志:`train_stage1.log`、`render_insights_35000.log`、`eval_normals_35000.log`;梯度审计 `CV3s_stage1_mlp/gradient_audit.csv`

### 9.3 四类可视化目检

1. **RGB**:`eval_rgb_contact_sheet.png` 中渲染与 GT 几乎一致,差异列近黑;
2. **alpha**:覆盖率逐帧稳定(0.168~0.180),时序差 2.6e-4,静帧场景无抖动;
3. **albedo/separation**:布料 albedo 均匀白色、跨帧恒定,shading 未泄漏进 albedo;
4. **normal**:误差图主体低误差,高误差集中在褶皱与轮廓边缘(高频区域,符合预期)。

### 9.4 结论

**PASS**。从 iteration 1 直接走 Lambertian + GT directional light(5.5043499 复用)、三个 normal consistency loss(无 GT normal 监督)的管线在静帧 CV3s 上成立:RGB 重建接近完美,独立法线恢复到 mean 19.23°(优于 dynamic 基线),albedo/normal 分离干净。iter-1 管线 + 一致性 loss 可作为后续静帧场景的候选默认配置;dynamic 场景是否同样受益需另行实验。
