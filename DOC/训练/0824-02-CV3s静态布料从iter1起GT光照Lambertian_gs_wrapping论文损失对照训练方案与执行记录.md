# 0824-02 CV3s 静态布料:从 iter 1 起 GT 光照 Lambertian + gs_wrapping 论文损失对照训练方案与执行记录

- 日期:2026-08-24
- 服务器/环境:garuda / `lumimotion-garuda`
- 实验目录:`output/0824-02-CV3s-GTlight_lambert_iter1_gswrapping/`
- 冒烟目录:`output/smoke_test/0824-02-CV3s-lambert-gs-wrapping/`
- 代码改动:`DOC/修改/0824-03-gs_wrapping预设.md`(新增 `gs_wrapping` 预设,纯新增,零漂移)
- 对照实验:`output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/`(三法线一致性)、`output/0824-01-CV3s-GTlight_lambert_iter1_2ncons/`(二法线一致性)

## 1. 背景与目标

`lambertian_normal2`/`normal3` 的两个独立法线一致性项与 Gaussian Wrapping(From Blobs to Spokes,arXiv:2604.07337,2026)论文 §4.1 的法线对齐损失 `L_N=Σ_p 1−N(p)·∇D(p)`、多视角几何一致性 `L_gc` 在形式上同类(均为 1−cos 余弦一致性)。本实验检验:**按论文权重方案配置损失组合**(`L_DN` 0.05 + `L_N` 0.05 + `L_gc` 0.02,保留 GS 几何自一致法线项)相对现有消融组合(0823-01 三法线 / 0824-01 二法线)在 CV3s 静态布料上的表现。

## 2. 配置决策(相对 0824-01 的差异即实验变量)

| 项 | 0823-01(3n) | 0824-01(2n) | 0824-02(gs_wrapping) | 依据 |
| --- | --- | --- | --- | --- |
| `lambda_gs_normal`(≈论文 L_DN) | 0.02(默认) | 0 | **0.05** | 论文 λ_DN=0.05 |
| `lambda_photometric_normal_live`(≈论文 L_N) | 0.01 | 0.01 | **0.05** | 论文 λ_N=0.05 |
| `lambda_photometric_normal_mv`(≈论文 L_gc) | 0.02 | 0.02 | 0.02 | 论文 λ_gc=0.02 |
| 论文 L_pc(多视角光度一致性,λ=0.6) | — | — | 无对应项,不实现 | 本仓库无跨视角光度项 |

- 调度为控制变量,沿用本仓库:live@500 起、mv@1000 起 + 2000 步线性 ramp、gs_normal 受 `start_normal_reg=500` 门控(论文为从头施加,差异注明)。
- 机制层面不可复现项(结论限定用):论文深度为 0.5-等值面、L_N 双向可微、L_N 驱动翻转法线稠密化;本仓库分别为 `surf_depth`、live 目标 detach 单向、无稠密化耦合。
- 传参方式:用 `--loss_preset gs_wrapping` 注入三项权重(不再显式传三个 lambda),验证预设路径;start_iter/ramp 仍显式传(与 0824-01 同值)。
- 其余超参与 0824-01/0823-01 完全一致(数据、35000 iter、光强 5.5043499、photometric_start_iter=1、normal_lr=1e-4 等)。

## 3. 训练命令

见实验目录 `run.sh`。与 0824-01 `run.sh` 的唯一差异:

```text
- --lambda_gs_normal 0
- --lambda_photometric_normal_live 0.01 --lambda_photometric_normal_mv 0.02
+ --loss_preset gs_wrapping
```

(`--loss_preset gs_wrapping` 注入 `lambda_gs_normal=0.05`、`lambda_photometric_normal_live=0.05`、`lambda_photometric_normal_mv=0.02`;训练启动日志应打印 `[loss_preset=gs_wrapping] applied defaults: ...`。)

## 4. 冒烟测试

目录:`output/smoke_test/0824-02-CV3s-lambert-gs-wrapping/`(命令见同目录 `run.sh`,1200 iter;live 500 起、mv 1000 起、ramp 压缩到 200)。

验收项:
1. 启动日志出现 `[loss_preset=gs_wrapping] applied defaults: lambda_gs_normal=0.05, lambda_photometric_normal_live=0.05, lambda_photometric_normal_mv=0.02`;
2. gradient audit 无 NaN;`normal`(gs_normal)项 500 后非零、`photometric_normal_live` 500 起、`photometric_normal_mv` 1000 起;
3. 收敛与 0824-01 冒烟同档(31.57 @1000);
4. checkpoint 正常写出。

冒烟结论:**PASS**(2026-08-24)

- 启动日志确认 `[loss_preset=gs_wrapping] applied defaults: lambda_gs_normal=0.05, lambda_photometric_normal_live=0.05, lambda_photometric_normal_mv=0.02`,预设传参路径生效;
- 损失触发时序(gradient audit,audit interval=50):`normal`(gs_normal)550 起、`photometric_normal_live` 500 起、`photometric_normal_mv` 1000 起、`distortion` 550 起,与门控设计一致;1200 步加权值:gs_normal 0.0448、live 0.0158、mv 0.00084(含 ramp);
- 数值:全程无 NaN/Inf;
- 收敛:PSNR 30.77(iter 1000 best,SSIM 0.97492 / LPIPS 0.03887 / MS-SSIM 0.98580),与 0824-01 冒烟(31.57 @1000)同档,略低与几何约束加权增大(×5)方向一致;
- checkpoint 完整写出(`smoke_mlp/point_cloud|deform|photometric` 均含 iteration_1200);
- 环境既有行为注记:与 0823-01/0824-01 相同,进程在全部训练与保存完成后静默退出,未打印 "Training complete."(无数据损失);正式实验 `post_train.sh` 完成判据已改为"35000 三处 checkpoint 完整 + 训练进程退出"或标记出现(二者其一)。

## 5. 正式训练

- 目录:`output/0824-02-CV3s-GTlight_lambert_iter1_gswrapping/CV3s_stage1`(自动 `_mlp` 后缀),35000 iter,日志 `train_stage1.log`。
- GPU:2(A40,当日独占,~10-14 it/s)
- 启动确认:`[loss_preset=gs_wrapping] applied defaults: lambda_gs_normal=0.05, lambda_photometric_normal_live=0.05, lambda_photometric_normal_mv=0.02`。

### 5.1 中间观察:早期位置梯度爆炸(训练期记录,已完结)

- iter 1~600:与冒烟、0824-01 几乎重合(@500 测试 PSNR 20.986 vs 0824-01 20.986);
- **iter ~650 起位置(xyz)梯度爆炸**:总位置梯度 0.13→0.71(650)→3.96(700)→9.16(750)→20.5(900),同期冒烟为 0.07~0.21;主要来自 `rgb_l1`(700 处 1.80)与 `photometric_normal_live`(700 处 0.88,0824-01 同处 0.0058),`normal`(gs_normal)项本身仅 0.15;
- 伴随 `deformation_xyz` 正则从 ~0 升至加权 0.012~0.023(变形场大幅偏移),训练侧 `rgb_l1` 由 0.006 档退化到 0.014~0.025 档并震荡;
- 测试侧:Best PSNR 停留在 23.35 @1000(0824-01 同处 31.66,5000 处 42.31);@5000 评估未超过该值;
- 冒烟(同权重)未触发该不稳定:500~600 两者轨迹同量级、650 起分叉,判断为**权重配置相关的随机不稳定性**——live λ=0.05(×5)经光栅化几何路径反传位置梯度,与深度导出法线目标构成正反馈(几何偏移→法线/深度失配→梯度更大),本种子在 650 附近进入该回路;
- 处理:不停训练,跑满 35000 后如实评估;若最终显著劣于对照,结论按 FAILED 记录并保留全部产物(不覆盖不删除)。

## 6. 渲染与评估

- `post_train.sh` 与 0824-01 同构:`render_stage1_insights`(全时序,`--load_iter 35000`)+ `eval_stage1_normals_gt`(GT EXR normal,`data/LH-data/static/only_clothV3/normal_exr`);完成判据已加固(见 §4 注记),本次自动执行成功(render exit=0、eval exit=0)。
- 产物:`CV3s_stage1_mlp/renders_stage1_insights/ours_35000/`(视频 full_render/Normals/Albedo/alpha + 四类 contact sheet)、`normal_gt_eval_independent/ours_35000/`(normal_error/gs_normal/gt_normal contact sheet + `normal_metrics.json`)。
- 四类可视化目检:① RGB 布面形状保持但斑驳、暗斑/小洞;② Normals 大体平滑但多视角椒盐噪声与成片噪声区;③ Alpha 轮廓完整、部分帧内部小孔;④ Separation 整体可用、部分视角斑块噪声。整体明显劣于 0823-01/0824-01。

## 7. 定量指标对照

| 指标 | 0823-01(3n) | 0824-01(2n) | 0824-02(gs_wrapping) |
| --- | --- | --- | --- |
| Best PSNR / SSIM / LPIPS | 47.65 / 0.9987 / 0.0045 | 51.95 / 0.99931 / 0.00278 | 33.66 / 0.98085 / 0.03248 |
| MS-SSIM / Alex-LPIPS | 0.9997 / 0.0012 | 0.99983 / 0.00042 | 0.98980 / 0.02908 |
| GT normal mean / median / p95 | 19.23° / 15.60° / 59.90° | 21.89° / 19.36° / 49.17° | 27.58° / 20.22° / 68.29° |
| alpha 覆盖率 / 时序差 | 0.1717 / 2.6e-4 | 0.1689 / 7.3e-5 | 0.1644 / 1.7e-3 |

注:0824-01 法线/alpha 指标已记录于其 README;2026-08-24 补跑 render+eval 重新生成对应产物文件,数值与 README 一致。

## 8. 结论

**FAILED**(2026-08-24)。

1. 论文 §4.1 权重方案在本管线全面劣于既有消融:RGB −18.3 dB(对 0824-01)、法线 mean +5.7°、alpha 时序差 ×23;
2. 根因:iter ~650 起位置梯度爆炸(live λ=0.05 经光栅化几何路径反传位置,与深度导出法线目标正反馈),变形场大幅偏移;1000~20000 长期劣化后部分恢复;
3. 冒烟同权重未触发,属权重配置相关的随机不稳定性;
4. 建议:论文方案若再试,先降权(live ≤0.01 档)或延后施加(如 5000 iter 后),并做多 seed 复核;本仓库既有标定(live 0.01 / mv 0.02 / gs_normal 0~0.02)更稳;
5. 全部产物保留,不覆盖不删除。
