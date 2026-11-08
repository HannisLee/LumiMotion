# 1108-01 T01 环境光建模与 RGB 损失新 Stage 2：实现记录

日期：2026-11-08（Asia/Tokyo）。服务器 ushas，conda 环境 lumimotion-ushas。分支 `1108-env-light-stage2`（自 `1107-pipeline-nobg` 派生）。
依据：规划 `LIHAN/DOC/规划/1108-01-T01环境光建模与RGB损失新Stage2规划.md`（下称"规划"）；GOAL `LIHAN/DOC/GOAL/1108-01-环境光建模与RGB损失的新Stage2.md` 的 T01 裁定（全部采纳规划默认：$s\equiv1$、ρ 由 Stage 2 闭式求解、L2 环境、通道等权、不改正则、不跑 Stage 3）。
本记录只含实现、静态审查与仅合成数据的数值自检；没有读取任何 GT，没有跑正式实验。

## 1. 新增文件（均在 `LIHAN/scripts/`，r3/r5b 与其他冻结代码未改动）

| 文件 | 内容 |
|---|---|
| `static_rgb_light_fit_r6_env.py` | R6-ENV 拟合、K1–K8 诊断、图 01–05、合成自检 S1–S10、`--smoke`、`--e4-regress-only`、`--k8` |
| `static_rgb_light_evaluate_r6_env.py` | 冻结后唯一一次 GT 评测（R6、E4、E3），E4 复现冻结评测（差 ≤1e-9°），PASS/FAILED 规则 |
| `freeze_r6_env.py` | 冻结脚本：登记规划/修改记录/训练计划/代码/输入/拟合产物 SHA256，状态 `R6_FROZEN_READY` |

## 2. 规划条目与代码对应

| 规划条目 | 实现 |
|---|---|
| 3.2 SH 基 `real_SH_L2_v1`（27 参数） | `sh9`；系数 `eps`(3,9)；`A_HAT` 仅用于 S2 自检 |
| 2.3 线性化与一致性断言 | `srgb_to_linear`；`support_and_obs` 断言 `isfinite(intensity)==isfinite(rgb).all(-1)`、$\max|\kappa^\mathsf T\mathrm{lin}(\mathrm{rgb})-\text{intensity}|\le10^{-5}$、法线范数误差 ≤1e-4 |
| 3.1/3.5 观测模型与数据项（$\omega_c=1/3$） | `env_terms`：$A=E+s\cdot x$，$r=\rho A-y$，data $=\sum w\,\omega_c\,\tfrac12 r^2/\sum w$ |
| 3.8(a) ρ 闭式（一维 NNLS，分母 +1e-6） | `rho_closed`（autograd 穿过闭式解） |
| 3.8(b)(d) ε 线性子问题与 ALS 初始化 | `solve_eps`（岭项 1e-12·tr/9）、`als_init`（至多 500 轮，判据 max\|Δε\|<1e-10 且 data 相对变化 <1e-12） |
| 3.6 目标 | `total_objective`：data + 1e-5·mean(ρ²) + λa(a1+a2)+λe(e1+e2)+0.1·rev；惩罚项直接调用 `r5b.penalties`；ε 无正则（Q1） |
| 3.9 优化器与证书 | `fit_traj_env`：T-A（ε lr 1e-3，同样两次衰减）、T-B（ε 用 Jacobi 预条件 `eps_precond`，C2 加 max\|Δε\|<1e-6）、C3 探针（ε lr 1e-5，加 max\|Δε\|<1e-3，结束后恢复全部参数）、T-C（q 空间含 qε）、Adam 回退（ε lr 1e-3） |
| 4 第 9–12 行 支撑集/E1–E3 初始化/参数化 | `support_and_obs`、`main` 逐字复用 r3/r5b，并断言与冻结 `support.npz` 逐元素相等、bootstrap 索引与冻结相等、E1–E3 对照 R3_PRED ≤1e-6 |
| I2 依赖代码哈希 | `check_inputs`：对照 1031-12 `freeze_manifest.json` 的 r3/r5b 哈希，不一致即停止 |
| K1（Fisher 保留率，含 E4 对照） | `fisher_env`（解析雅可比 + 对角 ρ 的 Schur 补）、`fisher_e4`（(ρ,b) 2×2 块）、`prior_hessian`（autograd.hessian）、`fisher_report`（22 个模式：u_e0、u_phi0、DCT-II k=1..10 分别作用于 e 与 φ；纯数据版岭项 τ=1e-10·tr/300；典型相关 top5；旗标 F1） |
| K2、K3、K4、K5、K6、K7、K8 | `energy_share`、`irradiance_check`（含 Fibonacci 10000 点）、`profile_zmode`（Δ∈{−4..4}°，三种模式，二次曲率）、`channel_residuals`、`repeatability`、`heldout`、`stage1_chroma`（`--k8`） |
| O1–O8 产物 | `predictions.npz`(frames,E0–E3,R6)、`env.npz`、`runs/R6_<flow>/`（direction/eps/als_init/history/certificate/config）、`runs/E4regress_full/`、`diagnostics/*.json`+`irradiance_sphere.npz`、`method_config.json`、图 01–05 |
| 8.3 CLI | `--output --observations --e4-root --plan [--self-test|--smoke] [--k8]`；另增 `--e4-regress-only`、`--eps-lr`；输出目录已存在则拒绝运行；`launch.json` 写入规划 SHA256 |

## 3. 与 r5b 的代码级差异（对应规划第 4 节）

| 位置 | r5b | R6 | 归属 |
|---|---|---|---|
| 观测 | `d['intensity']`（luminance） | `d['rgb']` 逆 sRGB 后的线性 RGB（C=3） | Δ2 |
| 内层 | `kkt()`：(ρ,b) 三分支 | `rho_closed`：ρ 一维两分支；b 删除；环境 $E_c(\mathbf n_{tg})$ 乘在 ρ 上 | Δ1 |
| 外层变量 | φ、a | φ、a、ε(3×9)；ALS 初始化 ε | Δ1 连带 |
| 系数正则 | $10^{-5}(\overline{\rho^2}+\overline{b^2})$ | $10^{-5}\overline{\rho^2}$ | Δ1 连带（Q1） |
| 数据项归一 | $\sum w\tfrac12 r^2/\sum w$ | 再乘 $\omega_c=1/3$ 并对通道求和 | Δ2 连带 |
| 预条件/证书 | φ、a | 增加 ε 的 Jacobi 预条件与 ε 判据（C2、C3） | Δ1 连带 |
| 选择 | 12 候选 × G0–G7 × J | 无选择；A/B/hi/lo/boot 只作 K6 诊断 | GOAL 规定的单配置 |
| 其余（支撑集、E1–E3、仰角带宽、λ、优化器步数、种子、float64、线程数） | — | 逐字沿用 | 一致 |

## 4. 静态审查要点

1. ρ=0 的 (g,c) 在 K1 中从干扰参数剔除（规划 6.4 要求）；其对 θ、ε 的雅可比项本来为 0，代码中由 ρ 因子自然给出。
2. 命题 2/3 的 1e-12 级验证必须关闭 $10^{-6}$ 分母稳定项（`damp=0`），否则 ρ 不再恰好等价于 $k\rho$；该关闭只出现在自检，生产路径始终使用 $10^{-6}$。S4/S5 另加阴性对照（只平移光向、不修正 $\beta$），确认检验有区分力。
3. ALS 在真实数据上（开发期用 E4 冻结方向试验，仅为测试代码路径与耗时，没有解读数值）500 轮内未满足 $10^{-10}$ 判据（末轮 max\|Δε\|≈2.7e-8），按规划只记录 `converged=false`、不阻塞；这是 ALS 线性收敛尾部的预期行为。
4. 证书 C2 的方向变化使用 arccos 形式，其数值下限约 1.2e-6°，远小于阈值 0.005°，不影响判定。
5. `chord_angle_deg` 仅用于 E4 回归判定（见第 6 节）。
6. 诊断中的 K4 在可用时使用 cuda（float64）以缩短耗时；其余诊断在 CPU。
7. 冻结前代码不 import 评测模块，也不读取 `lights.json`；评测脚本中读取 1031-12 `evaluation/results.json` 的位置在 GT 读取之后，仅用于验证 E4 复现。

## 5. 合成数据数值自检（`--self-test`）

输出：`output/smoke_test/1108-01-SYNTH-T01环境光RGB自检/numeric_tests.json`（合成真值不是数据集 GT；`gt_read=false`）。结果（以该目录内文件为准，下表为终版代码的运行值）：

| # | 通过条件 | 结果 |
|---|---|---|
| S1 | Gram 偏差 <1e-3 | 2.76e-06 |
| S2 | 辐照度公式相对误差 <1e-3 | 2.45e-08 |
| S3 | ρ 对照 `scipy.nnls` 最大差 <2e-5 | 1.15e-07（含 ρ=0 分支与零权重） |
| S4 | 命题 3：data 差 <1e-12 | 2.2e-19；阴性对照 8.9e-05 |
| S5 | 命题 2：data 差 <1e-12 | 0.0；阴性对照 5.0e-04 |
| S6 | 方向 Mean <1°、converged、逐位重复 | Mean 0.053°；converged；重复最大差 0.0；T-C 路径与 Adam 回退路径均 converged 且与主路径差 <0.1°；ε 最大绝对误差 3.6e-3 |
| S7 | gradcheck（φ,a,ε） | PASS |
| S8 | 惩罚项恒等 | 0.0（且 `penalties is r5b.penalties`） |
| S9 | K1 Schur 补对照稠密求逆 | 相对差 4.9e-16 |
| S10 | ε≡0、luminance 路径与 b≡0 的 E4 目标一致 | 0.0 |
| 附加 | K1–K7 诊断函数在合成数据上空跑，数值有限 | PASS |

注意：S6 的合成场景较易，主要验证实现与收敛路径，不代表真实数据上的可辨识性。

## 6. E4 回归复现（规划 I7/D12）：需主会话确认的偏离

在真实观测上运行 `--e4-regress-only`（支撑集 → E1–E3 复现 → 调用 `r5b.fit_traj`，λ=(1,10)），不读 GT。

- 首次运行（保留：`output/smoke_test/1108-01-LH-static-ELDL-V3-T01环境光RGB-E4回归检查-v1-arccos判据超差保留/`）：按规划写法用 `ang_deg`（arccos 点积）度量，最大角差 1.2074e-06° >1e-6°，脚本按 D12 停止。
- 排查：重现方向与冻结 `direction.npy` **逐分量完全相同**（最大绝对差 0.0），证书（路径 lbfgs、extended、206 次外迭代、各 C 值）逐位相同。1.2074e-06° 恰好等于冻结证书里的 `C2_max_deg`，即 arccos(点积) 对**相同**单位向量也会给出的浮点下限（点积舍入 1−1.1e-16 → 约 1.2e-6°）。因此 1e-6° 的容差无法用 arccos 形式度量，超差是度量问题而不是复现失败。
- 处理：门槛 1e-6° 不变，度量改为弦长角 $2\arcsin(\|a-b\|/2)$（数值精确），并在 `e4_regression.json` 同时记录 arccos 形式和逐分量最大差。改动已单独 commit，可随时回退。第二次运行（`output/smoke_test/1108-01-LH-static-ELDL-V3-T01环境光RGB-E4回归检查/`）：弦长角最大差 0.0，逐分量最大差 0.0，`ok=true`。
- **请主会话裁定**：是否认可以弦长角作为 D12 的度量；若不认可，只能放宽或换度量，我没有再做其他改动。

## 7. 正式运行命令（供 exp-run 使用，本任务未执行）

```
cd /home/han.li/reproduce/LumiMotion-stage2-perlight; CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python -u -m LIHAN.scripts.static_rgb_light_fit_r6_env \
  --observations output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz \
  --e4-root output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角 \
  --plan LIHAN/DOC/规划/1108-01-T01环境光建模与RGB损失新Stage2规划.md \
  --output "output/1108-NN-LH-static-ELDL-V3-T01环境光RGB新Stage2/fitting" > "output/1108-NN-LH-static-ELDL-V3-T01环境光RGB新Stage2/run.log" 2>&1
```

- 日志目录需先 `mkdir -p output/1108-NN-…/`（`--output` 本身要求不存在）。
- 冒烟：加 `--smoke`，输出到 `output/smoke_test/1108-NN-LH-static-ELDL-V3-T01环境光RGB冒烟`；未收敛时唯一允许的调整是 `--eps-lr 3e-4` 重做一次。
- 自检：`--self-test`，输出到 `output/smoke_test/1108-NN-SYNTH-T01环境光RGB自检`。
- 目检 5 张图后写 `<OUT>/visual_inspection.json`（5 项，`gt_read=false`），再运行 `python -m LIHAN.scripts.freeze_r6_env --run <OUT> --selftest-json <numeric_tests.json> --modify-doc <本文> --train-plan <执行计划> [--smoke-json …]`，之后由 analysis 运行 `python -m LIHAN.scripts.static_rgb_light_evaluate_r6_env --run <OUT> --allow-gt-evaluation`（先不带该开关做冻结核验）。
- 预计耗时：7 条流程各约 1–3 分钟 GPU 优化加 ALS 初始化；K4 约 0.5 分钟（GPU）。

## 8. 分支与同步

- 本任务所有改动已按改动逐次 commit 并 push 到 `origin/1108-env-light-stage2`。
- AGENTS 要求的 LIHAN 全分支同步：本任务**没有**把这些文件同步到其他分支（`1107-pipeline-nobg`、`main`、`perlight` 等均未更新），也没有处理工作区里先前就存在、与本任务无关的未提交改动（它们没有被纳入我的 commit）。同步由主会话安排。
