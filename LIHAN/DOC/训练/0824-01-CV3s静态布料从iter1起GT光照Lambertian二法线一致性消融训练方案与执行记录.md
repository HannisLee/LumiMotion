# 0824-01 CV3s 静态布料:从 iter 1 起 GT 光照 Lambertian + 二法线一致性消融训练方案与执行记录

日期:2026-08-24
服务器/环境:garuda / `lumimotion-garuda`
状态:完成,**PASS**(2026-08-24)

## 1. 背景与目标

- 0823-01(`--loss_preset lambertian_normal3`)验证了 iter 1 起 Lambertian + **三**法线一致性(① GS 几何自一致法线、② `photometric_normal_live`、③ `photometric_normal_mv`)可行且 PASS。
- 本实验为其**单变量消融**:在完全相同的训练设置下**只去除①**(GS 几何自一致法线),保留②③,回答“GS 几何自一致法线对几何/法线恢复是否有贡献”。
- 不使用 GT normal 监督(与 0823-01 一致,`lambda_photometric_gt_normal=0`,GT EXR 仅离线评估)。

## 2. 配置决策(相对 0823-01 只改一行)

| 项目 | 0823-01 | 0824-01(本实验) |
| --- | --- | --- |
| ① GS 几何自一致法线 | λ=0.02,`start_normal_reg=500` | **`--lambda_gs_normal 0`(去除)** |
| ② `photometric_normal_live` | λ=0.01,iter 500 起 | 保持 |
| ③ `photometric_normal_mv` | λ=0.02,iter 1000 起、2000 步升满 | 保持 |
| distortion | λ=1000,`start_normal_reg=500` 门控 | **保持**(用户确认;为此新增 `--lambda_gs_normal` 开关,见 `DOC/修改/0824-01-*`) |
| 其余全部超参 | 见 0823-01 文档 §3 | 逐项相同(光照 `gt_directional`、强度 5.5043499、白灯、`photometric_start_iter=1`、albedo/normal LR 0.001/0.0001、35000 iter、`--eval`、resolution 2、densify 1500–5000、max_gaussians 20000 等) |

技术前提核对(2026-08-24):

- `photometric_lambertian` 着色使用**独立法线**(`gaussian_renderer/__init__.py` `deform_independent_normal` 分支),去除①不会直接解除着色法线约束;①的作用是约束 2DGS 板朝向贴合 `surf_normal`(几何自一致),并间接影响深度/`surf_normal` 质量,而 `surf_normal` 正是②的 detach 目标。
- ①与 distortion 原共享 `start_normal_reg` 门控且①权重硬编码,故先做代码改动 `DOC/修改/0824-01-lambda_gs_normal开关与lambertian_normal2预设.md`(默认值不变,既有实验零漂移),再跑本实验。

## 3. 训练命令

输出目录:`output/0824-01-CV3s-GTlight_lambert_iter1_2ncons/`;启动 `tmux new-session -d -s cv3s_0824_01 "bash output/0824-01-CV3s-GTlight_lambert_iter1_2ncons/run.sh"`;日志同目录 `train_stage1.log`;模型目录自动加 `_mlp` 后缀。

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n lumimotion-garuda \
python -u -m scripts.train_stage1 \
  --source_path data/LH-data/transfer-static/only_clothV3 \
  --model_path output/0824-01-CV3s-GTlight_lambert_iter1_2ncons/CV3s_stage1 \
  --train_light_folder images --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 --iterations 35000 --warm_up 500 \
  --densify_from_iter 1500 --densify_until_iter 5000 --densification_interval 200 \
  --densify_grad_threshold 0.0004 --opacity_reset_interval 3000 --min_opacity 0.01 \
  --prune_from_iter -1 --max_gaussians 20000 --binarization_warm_up 1000 \
  --lambda_separation 0.005 --d_xyz_loss_weight 0.001 --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 --start_normal_reg 500 --lambda_gs_normal 0 \
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
  --quiet
```

等价一键方式:`--loss_preset lambertian_normal2`(覆盖 live=0.01、mv=0.02、`lambda_gs_normal=0`;本实验 run.sh 采用全显式写法,便于与 0823-01 逐行 diff)。

## 4. 冒烟测试

目录:`output/smoke_test/0824-01-CV3s-lambert-iter1-2ncons/`(命令见同目录 `run.sh`,1200 iter;live 500 起、mv 1000 起、ramp 压缩到 200)。

验收标准:

- `gradient_audit.csv` 中 `normal` 项恒为 0(`lambda_gs_normal=0`),`distortion` 非 0;
- `photometric_normal_live`、`photometric_normal_mv` 正常触发、无 NaN;
- 数据装载/收敛/显存/ckpt 正常。

冒烟结论:**PASS**(2026-08-24)

- 1200 iter 正常退出(`EXIT=0`),用时 3:45(~3.1 it/s;当日 GPU 与 santo 的 4 卡任务共享,未见明显降速);显存占用 ~1.8 GiB,无 OOM;
- 收敛:PSNR 31.57(iter 1000 best,SSIM 0.98014 / LPIPS 0.03304 / MS-SSIM 0.98806),与 0823-01 冒烟(31.12 @1000)同档;
- 梯度审计(`gradient_audit.csv`,1404 行)逐项核验:
  - `normal`(GS 几何自一致):iter 1–1200 共 150 行**恒为 0**——`--lambda_gs_normal 0` 生效;
  - `distortion`:iter 550 起非零(门控 `iter>start_normal_reg=500`),4.4e-5~7.0e-5——按用户要求保留;
  - `photometric_normal_live`:iter 500 起,加权 0.00469→0.00453,无 NaN(0823-01 冒烟 0.0047→0.0033~0.0038,同量级);
  - `photometric_normal_mv`:iter 1000 起,加权 3.95e-6→0.00227(200 步 ramp 在 1200 升满),无 NaN,趋势与 0823-01 冒烟一致;
- checkpoint 正常写出(`smoke_mlp/point_cloud` 等)。

## 5. 正式训练

- 训练用时 3:24:21(35000 iter,GPU 2;当日与 santo 4 卡任务共享,未见明显降速);
- Best PSNR 51.95(iter 30000,SSIM 0.99931 / LPIPS 0.00278 / MS-SSIM 0.99983 / Alex-LPIPS 0.00042);0823-01 同期为 47.65;
- 梯度审计(全周期,91980 行):`normal` 8406 行**恒 0**(`--lambda_gs_normal 0` 生效);`distortion` iter>500 起非零(30000+ 均值 1.99e-5,保留);`photometric_normal_live` iter 500 起(30000+ 均值 3.5e-3);`photometric_normal_mv` iter 1000 起(30000+ 均值 2.2e-4);全程无 NaN;
- 注意(既有行为,与 0823-01 相同):`--quiet` 时 `utils/general_utils.safe_state` 会吞掉全部 stdout,`Training complete.` 不进日志(进度条走 stderr 保留)。`post_train.sh` 等待条件已改为“日志标记 或 (iteration_35000 checkpoint 已落盘 且 训练进程已退出)”,见同目录脚本注释。

## 6. 渲染与评估

- 渲染:`scripts.render_stage1_insights`(source `data/LH-data/transfer-static/only_clothV3`,model `CV3s_stage1_mlp`,load_iter 35000,`--render_mode photometric_lambertian --depth_ratio 1.0`),输出 `CV3s_stage1_mlp/renders_stage1_insights/ours_35000/`,日志 `render_insights_35000.log`,exit 0;
- GT normal 评估:`scripts.eval_stage1_normals_gt`(gt_normal_dir `data/LH-data/static/only_clothV3/normal_exr`),输出 `normal_gt_eval_independent/ours_35000/`,日志 `eval_normals_35000.log`,exit 0;
- 完整命令见实验目录 `README.md` 与 `post_train.sh`;
- 定量指标(与 0823-01 对比,iteration 35000):

| 指标 | 0823-01(3n) | 0824-01(2n) |
| --- | --- | --- |
| Best PSNR(iter 30000) | 47.65 | 51.95 |
| SSIM / MS-SSIM | 0.9987 / 0.9997 | 0.9993 / 0.9998 |
| LPIPS / Alex-LPIPS | 0.0045 / 0.0012 | 0.0028 / 0.0004 |
| GT normal mean / median / p95 | 19.23° / 15.60° / 59.90° | 21.89° / 19.36° / 49.17° |
| 逐帧 mean 范围 | 17.18°~21.79° | 19.38°~23.94° |
| train / test mean | 19.23° / 19.19° | 21.90° / 21.84° |
| alpha coverage / 时序差 | 0.1717 / 2.6e-4 | 0.1689 / 7.3e-5 |

- 代表产物:视频 `full_render_camframe_0008.mp4`、`Normals_camframe_0008.mp4`、`Albedo_camframe_0008.mp4`、`alpha_camframe_0008.mp4`、`Separation_large_camframe_0008.mp4`;图片 `eval_rgb_contact_sheet.png`、`normals_contact_sheet.png`、`alpha_render_contact_sheet.png`、`separation_large_contact_sheet.png`;法线评估 `normal_metrics.json` 与三张 contact sheet。
- 四类可视化目检:① RGB 与 GT 几乎一致、差异近黑;② normals 平滑连续、褶皱梯度合理、逐帧一致;③ alpha 轮廓稳定、边缘干净、无浮点,时序差 7.3e-5 优于 0823-01;④ separation 前景近白/背景全黑、无散斑。normal 误差图主体近黑,褶皱处少量暖色,与 0823-01 同型。

## 7. 结论

**PASS**(2026-08-24)。消融有效且完整:去除 GS 几何自一致法线后 RGB/alpha 不降反升,GT normal mean 21.89° 仍显著优于 dynamic 分阶段基线(28.69°),但 mean/median 较 0823-01 变差 +2.66°/+3.76°(p95 改善 59.9°→49.2°)。**结论:GS 几何自一致法线对法线整体恢复有正向贡献,默认协议仍推荐三法线组合(`lambertian_normal3`)**;`lambertian_normal2` 作为可用替代配置保留。失败产物(如有)一律保留不删。
