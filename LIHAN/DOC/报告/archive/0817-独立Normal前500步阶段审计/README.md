# GT Light 5.5 独立 Normal 双数据集前 500 步阶段审计报告

日期：2026-08-17  
服务器：garuda  
环境：`lumimotion-garuda`  
实验：CV3 与 CV3L，iteration 10001～10500  
阶段结论：`FAILED`

> 2026-08-17 再审更新：旧实验已归档到 `output/不重要`；调度缺陷与 CV3L 光强配置错误已修复，并已启动两套替代训练。下文保留旧实验的失败证据，新增的再审结论不改变其 `FAILED` 判定。

## 结论先行

独立 normal 参数已经成功接入训练，并且本轮实验严格冻结了原 GS geometry、deformation 和 GT directional light；因此这次实验不再存在“固定 GS 导致 normal 根本不能变化”的旧问题。然而，RGB Lambertian loss 在 500 步内把 independent normal 系统性推离 GT：CV3 的平均角误差由 23.72° 增至 27.25°，CV3L 由 25.39° 增至 28.15°，两个数据集均为 **0/120 帧改善、120/120 帧恶化**。

这说明当前主要问题不是“normal 没有梯度”或“几何仍在代偿”，而是 **上一轮材料调度与光强配置没有形成预期的可控实验**。代码再审确认，名为 `albedo_only` 的阶段实际上同时打开了 albedo 与 independent normal；此外 CV3L 沿用了只适用于 CV3 的 5.5 光强。即使排除这两个错误，单个全局方向光对 Blender 近场面光源仍只有约 16.3～16.5 dB 的 oracle foreground PSNR，因此 normal–albedo 可辨识性和光照模型误差仍是剩余风险。

## 2026-08-17 再审结论与替代训练

### 已确认的两个实验配置问题

1. **材料阶段调度语义错误**：旧实现进入 photometric 阶段时已经为 albedo 和 independent normal 都设置了非零学习率；随后名为 `albedo_only` 的分支只冻结 GS geometry，并未关闭 normal。旧实验前 500 步实际上是 `albedo + normal` 联合优化，不能用于判断“先拟合 albedo、再单独恢复 normal”是否有效。
2. **CV3L 光强错误复用**：重新用 GT RGB、GT albedo、GT normal 和 GT light direction 做方向光 oracle 标定后，CV3 推荐全局 irradiance 为 **5.50435**，CV3L 为 **7.84349**。旧 CV3L 也使用 5.5，导致系统性欠照，normal/albedo 必须额外吸收光强误差。

### 已量化但尚未消除的模型误差

| 数据集 | 推荐全局 irradiance | oracle foreground PSNR | GS 点方向光近似平均误差 | P95 |
| --- | ---: | ---: | ---: | ---: |
| CV3 | 5.50435 | 16.318 dB | 6.19°～7.58° | 10.46°～12.89° |
| CV3L | 7.84349 | 16.524 dB | 6.27°～7.65° | 10.41°～12.74° |

这里的 oracle 已使用 GT normal 与 GT albedo；PSNR 仍只有约 16.5 dB，证明单个全局方向光不能完全复现 Blender 面光源。方向近似在椅头/椅脚等空间跨度大的部位可达到更大误差（CV3 最大 14.39°，CV3L 最大 19.74°）。因此即使新调度正确，也不能把 RGB loss 的最优点直接等同于 GT normal。

### 新训练的单变量设计

- iteration 1～10000：重跑原 SH Stage-1，保持 baseline 可追踪。
- iteration 10001～10500：只训练 photometric albedo，independent normal 学习率严格为 0。
- iteration 10501 起：冻结 albedo，只训练 independent normal；normal LR 从 `1e-3` 降为 `1e-4`。
- position、rotation、scale、opacity、deformation 与 GT light 全程冻结到 35000 之后，本轮不允许它们代偿。
- 增加相对初始 GS normal 的非 GT 信赖先验，权重 `0.01`；**仍未使用 GT normal loss**。
- 每 25 步记录材料参数与各几何参数梯度，重点验收 10501 后 normal 梯度及离线 GT 角误差。

新实验目录：

- [CV3：0817-04-CV3-GTlight_i5p5_A500_Nonly_lr1e4](../../../output/0817-04-CV3-GTlight_i5p5_A500_Nonly_lr1e4/README.md)
- [CV3L：0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4](../../../output/0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4/README.md)

## 实验设计核对

- iteration 1～10000：原 SH Stage-1，学习 geometry/deformation。
- iteration 10001：从当时 GS rotation 初始化独立 canonical normal。
- iteration 10001 起：固定 position、rotation、scale、opacity、deformation 与 light；只训练 independent normal 和 photometric albedo。
- 光照：GT directional per-light，irradiance 5.5，RGB color = (1, 1, 1)。
- independent normal learning rate：`1e-3`。
- **没有使用 GT normal loss**；GT normal 只参与本次离线审计。
- 评估：120 帧全部参与，alpha threshold = 0.5；GT Blender world-space normal 与 renderer world-space normal 直接比较。

## 定量结果

| 数据集 | 指标 | iter 10001 | iter 10500 | 变化 | 相对变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV3 | mean | 23.719° | 27.245° | +3.526° | +14.87% |
| CV3 | median | 15.652° | 20.130° | +4.478° | +28.61% |
| CV3 | P95 | 67.632° | 69.469° | +1.837° | +2.72% |
| CV3L | mean | 25.385° | 28.148° | +2.762° | +10.88% |
| CV3L | median | 19.018° | 24.231° | +5.213° | +27.41% |
| CV3L | P95 | 68.901° | 68.491° | -0.410° | -0.59% |

CV3L 的 P95 微降不能解释为恢复成功：它的 mean、median 均显著变差，而且 120 个逐帧 mean 全部变差。这更像误差分布重新分配，而非整体法线改善。

### 逐帧一致性

| 数据集 | 总帧数 | mean 改善 | mean 恶化 | 最差变化帧 | 最大恶化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV3 | 120 | 0 | 120 | source frame 109 | +5.519° |
| CV3L | 120 | 0 | 120 | source frame 99 | +4.342° |

## 冻结审计

iteration 10001 与 10500 的 checkpoint 逐项比较如下：

| 数据集 | GS 数量 | position max | rotation max | scale max | opacity max | deformation | light dir max | normal 平均漂移 | normal P95 漂移 | albedo 平均绝对变化 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| CV3 | 16059 | 0 | 0 | 0 | 0 | SHA 相同 | 0 | 15.653° | 35.182° | 0.1720 |
| CV3L | 16404 | 0 | 0 | 0 | 0 | SHA 相同 | 0 | 14.896° | 33.175° | 0.1636 |

因此：

1. geometry/deformation/light 确实冻结，没有隐藏的几何补偿；
2. independent normal 并非“约束太弱导致不动”，500 步平均已漂移约 15°；
3. albedo 同时产生很大变化，normal 与 albedo 正在共同吸收 RGB 误差；
4. 当前 photometric 梯度能够快速改变 normal，但方向没有朝向 GT。

## RGB baseline 参照

训练日志中的最佳 test 指标仍出现在 iteration 10000 的 SH baseline：

| 数据集 | PSNR | SSIM | LPIPS | MS-SSIM | Alex-LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV3 | 36.13052 | 0.98804 | 0.02326 | 0.99490 | 0.01818 |
| CV3L | 36.43810 | 0.98963 | 0.02492 | 0.99579 | 0.02290 |

这些数字只能说明 SH baseline 很容易拟合 RGB，不能说明 normal 正确；本次结果再次证明 RGB 高保真与 normal 恢复之间存在明显解耦。

## 可视化目检

### 1. independent normal

10001 到 10500 之间，椅背布料、圆台和轮廓处出现一致的颜色方向偏移，CV3/CV3L 两组都可见。变化不是随机噪声，而是跨视角的系统性漂移。

- CV3：[10001 normal](../../../output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10001/gs_normal_contact_sheet.png) / [10500 normal](../../../output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/gs_normal_contact_sheet.png)
- CV3L：[10001 normal](../../../output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10001/gs_normal_contact_sheet.png) / [10500 normal](../../../output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/gs_normal_contact_sheet.png)

### 2. GT normal

GT contact sheet 可正常读取，三处代表帧都有连续世界空间法线；本轮未发现 EXR 缺帧或读取失败。

- [CV3 GT normal](../../../output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/gt_normal_contact_sheet.png)
- [CV3L GT normal](../../../output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/gt_normal_contact_sheet.png)

### 3. normal error

误差图与数值方向一致，10500 的主体表面与边界误差没有出现整体收缩。

- CV3：[10001 error](../../../output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10001/normal_error_contact_sheet.png) / [10500 error](../../../output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/normal_error_contact_sheet.png)
- CV3L：[10001 error](../../../output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10001/normal_error_contact_sheet.png) / [10500 error](../../../output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/normal_error_contact_sheet.png)

### 4. RGB / 训练状态

训练日志的最佳 RGB 指标仍停留在 10000；没有证据表明 independent-normal 阶段同时刷新 RGB baseline 与 normal 指标。两条训练均异常停止，CV3 日志停在约 13063，CV3L 停在约 12320，未发现 Python traceback；最新持久化 checkpoint 均为 10500。

## 运行与产物状态

| 数据集 | 目标 iteration | 日志停止位置 | 最新 checkpoint | 训练进程 | 阶段结论 |
| --- | ---: | ---: | ---: | --- | --- |
| CV3 | 35000 | ≈13063 | 10500 | 已退出，无 traceback | FAILED |
| CV3L | 35000 | ≈12320 | 10500 | 已退出，无 traceback | FAILED |

失败结果、checkpoint、评估图片、统计和日志均已保留，未覆盖或删除。

## 原因判断

### 已排除或显著降低的可能性

- **“GS geometry 在替 normal 代偿”**：本轮 position/rotation/scale/opacity 全为零变化，已排除。
- **“normal 根本没有梯度”**：500 步平均漂移约 15°，已排除。
- **“GT light 在被错误学习”**：light_dirs 逐元素零变化，已排除参数更新层面的漂移。
- **“只在 CV3 特定纹理上失败”**：CV3L 同样 120/120 帧恶化，说明问题具有管线结构一致性。

### 当前优先怀疑

1. **normal–albedo 不可辨识性**：albedo 与 normal 同时自由学习时，RGB loss 存在大量等价或近似等价方向；优化器选择了能降 RGB、但偏离 GT normal 的解。
2. **directional light 与真实 Blender 灯的模型不完全一致**：固定 5.5 只能固定强度标量，不能弥补近场位置光导致的椅背/椅脚方向差异。
3. **渲染方程或 light-direction 符号/空间约定仍可能有局部不一致**：本次证明参数固定，不等于证明传入 renderer 的向量与 GT normal 的点积符号必然正确。
4. **normal LR 偏大或参数化缺少信赖域**：`1e-3` 在 500 步造成约 15° 平均漂移；即使梯度符号正确，步长也可能跨过局部合理区域。

## 下一步建议：先做最小可证伪审计，不直接续训

建议按以下顺序进行，每一步只改变一个因素：

1. **单帧、单灯、固定 albedo，只训练 independent normal**：从 10001 checkpoint 开始跑 50～100 步；如果角误差仍上升，问题集中到 light/normal 坐标、符号、渲染方程或 LR，而不是 albedo 歧义。
2. **做数值梯度方向检查**：选 1 个 Gaussian / 1 个像素，比较 autograd 与 finite difference，并直接记录 `grad · (n_gt - n)`；若该值长期为负，说明 RGB 梯度与 GT normal 恢复方向相反。
3. **验证 Lambertian 解析闭环**：使用已知 `n`、`l`、albedo、5.5 合成单点 RGB，再从同一实现反传；该测试应能在无相机/GS/坐标变换干扰下恢复 normal。
4. **固定 GT 或标定 albedo 后再做多灯共享 normal**：至少使用三条不共面的 light directions；单灯下 normal 只有 `n·l` 投影约束，天然不充分。
5. **通过 1～4 后，再用小 LR 与信赖域**：normal LR 从 `1e-4` 或 `1e-5` 起，增加相对初始化 normal 的角度正则/最大更新角度，并以每 25～50 步 GT 离线角误差做 early stop。
6. **若位置光效应显著**：无法优化 light position 时，至少离线从 Blender light position 与物体 bbox 计算逐点方向变化上界，量化 directional approximation 在椅头/椅脚可能造成的角度误差。

当前不建议直接重启相同配置到 35000。最有信息量的下一轮是“固定 albedo、normal-only、50～100 步 + finite-difference 梯度审计”。

## 复现评估命令模板

CV3（将 `LOAD_ITER` 替换为 10001 或 10500）：

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n lumimotion-garuda \
python -u scripts/eval_stage1_normals_gt.py \
  --model_path output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/CV3_explicit_normal_mlp \
  --source_path data/LH-data/transfer-dynamic/only_clothV3 \
  --gt_normal_dir data/LH-data/danamic/only_clothV3/normal_exr \
  --load_iter LOAD_ITER --is_blender --eval --resolution 2 \
  --output_dir output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_LOAD_ITER \
  --quiet
```

CV3L 同理替换 source、GT normal 和 output/model 路径为 `only_clothV3_lambertian` 与 `0817-05-CV3L...`。

## 最终判定

`FAILED`：独立 normal 参数与冻结机制工作正常，但前 500 步的 normal 恢复目标失败，且在两个数据集上呈现 120/120 帧一致恶化；训练还异常中断，未达到 35000。
