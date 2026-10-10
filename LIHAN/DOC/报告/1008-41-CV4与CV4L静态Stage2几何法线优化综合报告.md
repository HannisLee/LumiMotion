# 1008-41 CV4 与 CV4L 静态 Stage2 几何法线优化综合报告

## 已完成停止，结果仍未达到 5°

以 **1006-01 / 1006-02 的真实 Stage1 35k** 为起点，本轮已完成 Lambertian Stage2 的小学习率几何优化、预测法线约束、训练 RGB 法线先验和静态来源对照。用户要求停止后，确认所有本轮训练及配套验收已结束，记录停止状态，取消未执行的下一项工作；不再启动训练或新诊断，恢复需要用户新指令。

同一组 11 个内部验证视角上，CV4 的 GS 法线 MAE 从 **7.45087° → 6.52671°，降低 12.40%**；CV4L 从 **7.60637° → 5.88048°，降低 22.69%**。没有使用 GT 法线进行训练、初始化、伪目标构造或点资格筛选。GT 法线和深度仅进入独立离线评测，GT 材质不用于训练。

**两套数据均未达到 GS MAE <5°。** 本报告的成绩是内部验证结果；Stage1 已见过这 11 个视角，因此不能作为独立测试成绩。原 15 个测试视角未用于本轮选参，也未在停止后追加评测。CV4L 的最佳 58k 恰在旋转预热终点，此段 XYZ/尺度/材质/光照及对应 Adam 状态冻结；不能将这一新最佳归因于位置学习率。

执行和产物完整性为 **PASS**，最佳模型自身目检为 **COMPLETED**，法线质量目标为 **FAILED_NORMAL_TARGET**，实验执行状态为 **USER_STOPPED**。这些状态分别描述流程、目检、质量及用户停止决定。

## 比较口径与数据边界

原 105 个训练视角中固定取 11 个作 Stage2 内部验证，其余 **94 个视角用于优化**；分辨率统一为 **640×360**。内部帧键为 `0001 / 0012 / 0023 / 0035 / 0046 / 0058 / 0069 / 0081 / 0092 / 0103 / 0115`。本报告直接读取已完成实验的 `metrics.json`，用 SHA256 锁定输入，并核对运行汇总和最佳模型自身的独立 Mesh 复评。

GS 法线是实际面片协方差法线经透射率/alpha 混合后归一化的渲染结果；Depth 法线来自模型预测 expected Z 深度反投影及有限差分；Mesh 法线来自模型 105 个训练相机的预测深度 TSDF 网格，独立渲染后评测。三者均与同一 GT 世界坐标单位法线比较，采用有向 `acos(dot)` 角度，不把相反方向当作等价轴。主指标先算每帧前景均值，再平均 11 帧。

常规 MAE 仅统计有效预测法线，同时报告覆盖率，并另报缺失/无效预测按 **180°** 计入的 MAE，防止丢失困难像素得到假改善。GT 合格前景由 alpha≥0.5、有效法线或射线命中定义。深度使用源场景的米制世界单位（`meters_per_world_unit=1`），未做尺度、平移、ICP 或逐帧对齐。布料区域标签仅用于离线分区统计。

审计包含 **88 个具有 initial_command 的实验目录、85 个有指标的正式训练实验目录，以及 327 份评测记录**。327 份中有 2 份 Stage1 基线、283 份训练模型评测、42 份零优化诊断评测；重复检查点、常规/Mesh 复评都单独留存，327 不是训练次数。诊断资产禁止进入初始化或恢复，也不参与“正式训练最佳”的排名。

## 最佳模型的多指标效果

下表以同数据集、同视角 Stage1 35k 为基线。Mesh 基线没有提取评测，记为“未评测”；不能据当前 Mesh 数值推断相对 Stage1 的网格收益。

CV4 的 GS、Depth 法线和深度误差均有改善，但 GS 覆盖率下降 **0.00383 个百分点（9 个像素）**；缺失惩罚指标仍从 7.76422° 降到 6.84286°。CV4L 的 GS 与布料法线改善明显，GS 覆盖率提高 **0.02976 个百分点（70 个像素）**，Depth 法线只改善约 0.20554°、深度误差几乎不变。因此，CV4L 的主要收益是渲染法线，整体几何精度的收益较小。

CV4 当前 52k 是 **原始 GS 指标最低的正式训练模型**。相对其合格 49k 来源，GS 由 6.53484° 小幅降至 6.52671°，但 Mesh 由 **8.34005° 上升至 8.37069°**，且覆盖率下降，不能认定综合几何验收优于来源。保留 49k 作为几何质量参考。

| 指标 | CV4 Stage1 35k | CV4 本轮 52k | CV4L Stage1 35k | CV4L 本轮 58k |
|---|---:|---:|---:|---:|
| GS 法线 MAE（°） | 7.45087 | 6.52671 | 7.60637 | 5.88048 |
| Depth 法线 MAE（°） | 9.69680 | 9.00628 | 9.85184 | 9.64630 |
| 布料区域 GS MAE（°） | 13.54970 | 11.43424 | 13.60112 | 10.79703 |
| 深度 MAE（米／原世界单位） | 0.026040 | 0.024796 | 0.027196 | 0.027140 |
| GS 有效法线覆盖率（%） | 99.82144 | 99.81762 | 99.82399 | 99.85375 |
| GS MAE：缺失按 180°计（°） | 7.76422 | 6.84286 | 7.91421 | 6.13613 |
| Mesh 法线 MAE（°） | 未评测 | 8.37069 | 未评测 | 8.28848 |

## 实际做了哪些改进

| 改进 | 实现方式与作用 | 结果与使用范围 |
|---|---|---|
| 小 LR 几何联合优化 | 冻结 Stage1 PLY/变形网络；Stage2 学习共享世界坐标 XYZ 残差、旋转和可选尺度。位置冻结、渐入、衰减，累计梯度；限制位移、旋转角、尺度比例；对照中 opacity 固定。 | 参数与 checkpoint 审计证明几何确有更新。允许光度信息反传几何，但收益必须逐组验证。 |
| 预测法线—深度一致性 | 比较模型 GS 法线与模型预测深度导出的法线，加入 Stage1 预测深度锚定、alpha 保持及表面邻域约束。 | 不使用 GT 法线/深度。控制位置梯度和约束权重，减少模型相互拖动。 |
| Lambertian 成像校准 | 校准已知面积光/点光、距离衰减、发光面余弦、环境光、模型预测可见性和 Filmic 映射；按源 Blender **5.2** 能量约定修正面积光功率。 | 合成白平面与 Blender 线性 EXR 校验误差约 0.092%。只证明该数值校验，不等于整个场景模型完全正确。旧错误能量分支保留并退出候选。 |
| 训练 RGB 材质与面片着色 | RGB 聚类得到 K=4 的共享调色板；真实面片先进行物理着色，再 alpha 合成；不给 GT albedo 或材质标签。 | 两套四色材质节点 GS 均改善；仍可见接触区阴影和灰碎纹，材质分离未完全解决。 |
| 表面细分 | 可选父片映射及 4/16 子片，每个子片有真实位置/旋转/尺度；CV4 使用 172560 子片，CV4L 使用 682464 子片。 | 提高表示能力；CV4L 密集节点 GS 改善，但 Depth 法线没有同步改善，不能用密度代替几何精度。 |
| 稳定深度畸变与固定邻域 | 独立可选 CUDA 数值分支修复负畸变抵消；以 Stage1 预测时间稳定性过滤邻域边、按真实子片建立图，保持点参与渲染；对邻域位置梯度预条件。 | 数值稳定与图损失检查通过，部分正式节点小幅改善。默认关闭新数值分支，旧管线保留。 |
| 光度支持区域减权 | 用预测可见性≥0.98、预测亮度≥0.2定义支持区；阴影/低照度区域法线一致性可减至 0.1。 | CV4L 节点改善至 6.14437°；CV4 采用同来源、同预算控制确认后的小改善。不是 GT 筛选。 |
| 训练 RGB 法线先验 | 用 94 训练视角 RGB、已知灯光、模型预测可见性/几何作光度法线估计。先按训练 fit/check 门槛确定点与资产，锁定后才做离线 GT 评测。 | CV4L 先验与预热显著改善 GS；CV4 弱先验/残差诊断多次回退，未直接照搬。 |
| 旋转预热与先验覆盖扩展 | 先仅训练旋转 2000 步，再联合；父面片汇聚增加 5259 个 RGB 先验点，同表面传播再增加 2402 个点，原目标逐位保持。 | 真正同预算对照证明预热有效；新增覆盖带来小改善，λ20 相比 λ2 的额外差异极小。 |
| 静态来源可选 preset | 固定 Stage1 的 fid 和 source camera_center 为 `0002`；实际优化视角/灯光仍遍历 94 帧。原默认 dynamic 保持。 | 完成 CV4 与 CV4L 同预算动态/静态对照，GS 上静态均未超过动态，不采用静态默认。 |
| 恢复、兼容与验收 | 真 checkpoint 来源初始化/新 Adam 与完整恢复分开；缓存/图/外部 RGB 资产可携带恢复；分开记录 GS/Depth/Mesh、覆盖、缺失惩罚和全部 CLI/SHA。 | 已有 119 项 CPU 测试通过，真实 GPU 4/16 子片旧前向五类图逐位一致。严格恢复失败保留，不扩大门槛或声称跨进程训练轨迹逐位。 |

源码入口是 `scene/stage2_geometry.py`、`scene/stage2_photometric.py`、`scene/stage2_rgb_prior.py`、`scene/stage2_source_deformation.py`、`gaussian_renderer/render_stage2_lambertian.py`、`scripts/train_stage2.py`、`scripts/loss_stage2.py` 和 `arguments/stage2.py`。改进均以可选参数/preset 接入，旧 IR 与冻结 Lambertian 路径保留。表中“校准/兼容通过”和“质量改善”依据不同，不互相替代。

## 整体路线与最有信息量的对照

节点表反映实际完成的探索路线。来源 checkpoint、训练预算及多个配置会变化，因此节点差值**不能相加作为各改动的独立因果贡献**。下面三个同来源、同预算对照比节点路线更适合判断效果。

**旋转预热：**真实 52k 来源、新 Adam、同样 2000 步；54k 联合控制 GS 为 6.136071307°，旋转预热为 5.907615405°，降低 **0.228455901°**。这一阶段 XYZ、尺度、材质、光照及其 Adam 状态实际冻结。布料 GS 反而由控制的 10.632436108° 升到 10.785250854°，整体均值收益没有覆盖每个区域。

**父面片汇聚：**真实 54k→59k 同预算双组，仅改变 RGB 资产；各自最佳 56k 控制为 5.910495892°、汇聚先验为 5.891952102°，降低 **0.018543791°**。增加 5259 个先验点，同时布料和 Depth/Mesh 并非同步改善；联合尾段两组回退。

**局部传播与强度：**真实 56k→61k 三组，2000 步预热+3000 步联合，其余 LR/约束一致。局部资产由 108725 增至 111127 点（约占全部 682464 子片的 16.28%；这与 99.85% 的渲染覆盖率不是同一量）。λ20 相比同资产 λ2 的 58k 差异仅 **0.000032228°**，没有独立重复/置信区间支持其统计显著性。61k 全部劣于各自 58k，保留真实最佳 checkpoint，不拿最后一步覆盖最好结果。

| 数据集 | 已验证节点 | 检查点 | GS MAE（°） |
|---|---|---:|---:|
| CV4 | Stage1基线 | 35000 | 7.450867 |
| CV4 | 训练RGB四色材质 | 43000 | 6.900614 |
| CV4 | 四子片/受限尺度 | 43000 | 6.576723 |
| CV4 | 稳定深度畸变分支 | 46000 | 6.551023 |
| CV4 | 邻域梯度与约束平衡 | 49000 | 6.538231 |
| CV4 | 光度支持区域同预算控制 | 49000 | 6.534844 |
| CV4 | 49→52k微小几何LR动态控制 | 52000 | 6.526709 |
| CV4L | Stage1基线 | 35000 | 7.606374 |
| CV4L | 训练RGB四色材质 | 43000 | 7.045689 |
| CV4L | 四子片/受限尺度 | 43000 | 6.450415 |
| CV4L | 十六子片密集表面 | 43000 | 6.364189 |
| CV4L | 训练RGB法线先验λ2 | 46000 | 6.203471 |
| CV4L | 光度支持区域减权 | 52000 | 6.144371 |
| CV4L | RGB旋转预热2000步 | 54000 | 5.907615 |
| CV4L | 父面片汇聚先验 | 56000 | 5.891952 |
| CV4L | 局部传播先验λ20 | 58000 | 5.880480 |

| 56→61k 同预算配置 | RGB 目标点数 | 58k GS（°） | 61k GS（°） |
|---|---:|---:|---:|
| 原先验 λ2 | 108725 | 5.890350154 | 6.069677 |
| 局部传播 λ2 | 111127 | 5.880511848 | 6.060844 |
| 局部传播 λ20 | 111127 | 5.880479620 | 5.977870 |

## 静态来源与极小几何 LR：本轮最终答案

“静态”在这组实验中指固定冻结 Stage1 变形的来源 fid 与相机位置，仍使用不同优化视角和对应灯光；没有丢弃多视角光度观测。两组从同一真实 checkpoint、新 Adam、相同 3000 步预算启动，只改变 dynamic/anchor0002 来源模式。

| 配置 | CV4 49→52k | CV4L 58→61k |
|---|---:|---:|
| XYZ LR 配置上限 | 3e-6 | 1e-7 |
| 旋转 LR | 1e-5 | 1e-5 |
| 尺度 LR | 5e-5 | 5e-6 |
| 位置冻结 / 渐入 / 梯度累计 | 500 / 250 / 2 | 500 / 250 / 2 |
| 位移上限（场景 extent 比例） | 0.005 | 0.01 |
| 旋转上限 / 尺度比例范围 | 20° / [0.8,1.25] | 20° / [0.8,1.25] |

CV4 在每个评测点静态 GS 都略差于动态；静态的 Depth/布料部分指标稍好，但没有证据支持静态成为 GS 优化默认。动态 52k 达到本轮最低 GS 6.52671°，代价是相对来源 Mesh 和覆盖率略差。

CV4L 从最佳 58k 的 5.88048°出发，将 XYZ LR 压低至 1e-7后，两组各自最好 59k 为 **5.90220° / 5.90485°**，后续 60/61k 继续回退。第一条非零位置更新日志在 **58600，实际 XYZ LR=3.6514e-8**；58→59k 动态/静态 checkpoint 最大 XYZ 分量变化为 **6.8177e-5 / 6.9579e-5**（世界单位），轴角参数分量最大变化约 0.00576 rad、对数尺度参数分量最大变化约 0.00121。这些是参数分量的最大绝对差，不是向量范数。位置确已开启，问题不是配置写了 LR 却仍完全冻结。

因此，本轮的实证结论是：**小 LR 位置/尺度可以安全受控地更新，但当前损失与成像模型下，继续联合更新没有稳定超过旋转预热最佳；固定静态来源也没有 GS 收益。** 两组自身 59k 四类三帧与视频完整性已验收，没有新全局最佳，按事前规则不新增 HDR/Mesh，保留真实 58k 自身的完整产物。

| 数据集 | 迭代 | 动态来源 GS（°） | 静态来源 GS（°） | 静态−动态（°，越小越好） |
|---|---:|---:|---:|---:|
| CV4 | 50000 | 6.528487 | 6.531844 | +0.003357 |
| CV4 | 51000 | 6.527266 | 6.532645 | +0.005378 |
| CV4 | 52000 | 6.526709 | 6.533647 | +0.006938 |
| CV4L | 59000 | 5.902198 | 5.904847 | +0.002648 |
| CV4L | 60000 | 5.913578 | 5.917497 | +0.003919 |
| CV4L | 61000 | 5.918095 | 5.921517 | +0.003422 |

## 未采用的尝试及得到的约束

| 尝试 | 已观察到的结果 | 处理 |
|---|---|---|
| 几何先行、Depth-only/position-only 一致性、单独改 LR | 部分 GS 小幅变化，多组无法超过各自来源；后期联合仍回退。 | 保存全部正式结果，不称为有效默认。 |
| 学习 opacity、屏幕过滤、PCA/大平面平滑 | 没有稳定同时改善 GS、覆盖和区域质量；部分恢复门控失败。 | 不依靠删点/透明化压低 MAE，不采用。 |
| 正向 RGB 残差减权 | CV4 原始 GS 一度约 6.52727°，但覆盖、缺失惩罚及布料回退。 | 不设为默认，不把单一均值最低称为综合更好。 |
| CV4 原始/固定反照率 RGB 光度法线 | 训练留出 RGBA 回退超过预声明门槛，在 GT 前拒绝；残差版本虽通过 RGBA 门槛，锁定后 GS 6.61421°回退。 | 拒绝整资产，未用 GT 重选点。 |
| 固定全部或不稳定 Stage1 点来源诊断 | 两套优化 RGBA 误差约 +2.42%～+2.61%，超过事前 0.5%门槛。 | 不进入训练；该只读诊断无 GT、无训练 checkpoint。 |
| CV4L RGB 旋转梯度拆分 | full/shading/footprint 六个零优化候选均先完成 RGB 资格并锁定；GT 离线 GS 最好约 5.89220°，仍高于来源 5.88048°。 | 分量分解数值检查通过，不能当质量改善；六候选全部拒绝。 |
| 累计 RGB 先验风险约束 | full_guard 留出光度−0.85565%、RGBA−0.18722%、覆盖不变，但 GS 5.88217°仍差于来源；footprint_guard 在 GT 前 RGB 门槛失败。 | 不进入训练，原数值证明/API 失败保留；最终 v3 修复检查并维持原门槛。 |
| 严格 CUDA 恢复/旧 anchor0059 | 曾出现 strict1e-6、own-repeat 或原 0.001 图像界限失败；anchor0059 法线差约 0.0014374。 | 保留失败与 checkpoint，使用分项记录，不以放宽容差掩盖。 |

旋转梯度分解的 76 个拟合训练帧前向五类图逐位一致，`full = shading + footprint` 最大差约 2.33e-10，主要旋转光度梯度来自 shading；这是当前参数上的只读证据，不证明后续任意模型均如此。累计先验风险检查按逐点公式≤2e-7，float32 实际差包含约 6.01e-7 数值量，使用原定 1e-6 数值预算；不声称实际误差严格≤2e-7。

所有零优化诊断均不计为训练最佳，也未通过 GT 分数局部挑点后重新注入训练。面积光逐采样可见性与照度相关性审计只形成了未执行计划，用户停止后保持**未开始**。

## 最佳模型自身的视觉、HDR 与 Mesh 验收

两套当前最佳都已生成并查看自身 `0001 / 0058 / 0115` 的 RGB、法线、albedo、alpha 四类静帧及 HDR 代表帧，Mesh 来自各自 105 相机预测深度，独立评测。四类视频通过 640×360、55 帧、12fps 的完整性检查；HDR 视频为 11 帧/24fps。实际视觉结论依据代表静帧，未声称完整视频逐帧动态目检。

共同剩余问题是布料褶皱/薄边、椅腿和底盘法线残差，接触区与盘沿碎纹、0058 椅腿投影过重及 0115 盘面阴影偏重。albedo 四色分区保留，但灰黑颗粒和阴影残留仍在；alpha 主体完整、腿部孔隙可见，薄边局部锯齿。HDR 暖色响应完整且线性 RGB 有限/非负，盘面斑纹没有完全消失。不能凭宏观相似的静帧断言万分之一度的 MAE 差异可见。

CV4 52k 的 Mesh 为 **45779 顶点/89393 三角**；CV4L 58k 为 **46947 顶点/92028 三角**。提取与拟合都未使用 GT 网格。CV4 的 Mesh 相对来源回退；CV4L 的 Mesh 相对 56k 仅约 0.00011°变化，不能描述成显著网格改善。

## 保留的最佳 checkpoint 与复核入口

**CV4：CV4_z27_real49_dynamic_control / iteration 52000**

- 模型：[model_mlp](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/model_mlp)；[checkpoint](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/model_mlp/stage2_photometric/iteration_52000/checkpoint.pth)。
- checkpoint SHA256：`c33f5744b69991ef88c08cd20e40caf6356d7597d7c99521cbeb8b60898e13a3`。
- [同口径常规指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/validation_52000/metrics.json)；[自身 Mesh 独立指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/validation_mesh_52000_v2/metrics.json)。
- [自身四类目检/命令/日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/visual_validation_52000/README.md)；[自身 HDR 验收](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/hdr_validation_52000/README.md)。

**CV4L：z44_real56_local_rgbprior_weight20 / iteration 58000**

- 模型：[model_mlp](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/model_mlp)；[checkpoint](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/model_mlp/stage2_photometric/iteration_58000/checkpoint.pth)。
- checkpoint SHA256：`965f6465ab20211262334b0d784ed143ae1bdadf399abc9a48120ea3c46414cb`。
- [同口径常规指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/validation_58000/metrics.json)；[自身 Mesh 独立指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/validation_mesh_58000/metrics.json)。
- [自身四类目检/命令/日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/visual_validation_58000/README.md)；[自身 HDR 验收](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/hdr_validation_58000/README.md)。

下列是**已经执行的完整渲染命令**的只读转录，仅供复核；报告生成不执行这些命令。模型、迭代、输出及日志还可从上述自身 README 核对。完整正式训练命令保存在各实验 `initial_command.json`，完整配置保存在各模型 `stage2_config.json`。

CV4 四类，来源 [实际命令文件](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/visual_validation_52000/execution_command.json)：

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m output.1007-03-static-geometry-stage2.render_validation_insights --model /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/model_mlp --iteration 52000 --output /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/visual_validation_52000
```

CV4 HDR，来源 [实际命令文件](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/hdr_validation_52000/execution_command.json)：

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m output.1007-03-static-geometry-stage2.render_validation_hdr --model /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/model_mlp --iteration 52000 --output /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/CV4_z27_real49_dynamic_control/hdr_validation_52000 --hdr example_envmaps/golden_bay_4k_32x16_rot330.hdr --samples 2048
```

CV4L 四类，来源 [实际命令文件](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/visual_validation_58000/execution_command.json)：

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m output.1007-03-static-geometry-stage2.render_validation_insights --model /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/model_mlp --iteration 58000 --output /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/visual_validation_58000
```

CV4L HDR，来源 [实际命令文件](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/hdr_validation_58000/execution_command.json)：

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m output.1007-03-static-geometry-stage2.render_validation_hdr --model /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/model_mlp --iteration 58000 --output /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/z44_real56_local_rgbprior_weight20/hdr_validation_58000 --hdr example_envmaps/golden_bay_4k_32x16_rot330.hdr --samples 2048
```



## 报告与完整数据

- [327 份全部评测 CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/all_evaluations.csv)：保留训练/诊断/基线分类、全部检查点、常规/Mesh复评和指标原文件路径。
- [85 个正式实验各自最佳 CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/formal_experiment_best.csv)：按各实验原始 GS 最低检查点选取，不能当成 85 个全部通过验收的模型。
- [基线至最佳 CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/baseline_to_best.csv)；[动态/静态同预算 CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/static_comparison.csv)。
- [审核证据与 SHA 清单](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/reviewed_evidence.json)；[原始证据生成脚本](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/producer_snapshot.py)；[报告生成脚本](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/author_report.py)。
- [用户停止记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/STOP_REQUESTED.json)；[本轮最新状态](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/status.json)；[停止续作状态](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-03-static-geometry-stage2/continuation.json)。
- [可搜索并离线打开的交互报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/1008-41-CV4-CV4L-Stage2综合报告.html)支持完整实验表搜索、排序、翻页及来源查看；[报告输出索引](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-41-CV4-CV4L-Stage2综合报告/README.md)汇总附件。文档按仓库规则同步到两个现有工作树。

报告数据截点为 **2026-10-08T02:21:34.757198+00:00 UTC**，停止请求为 **2026-10-08T02:29:52.249666+00:00 UTC**。生成报告不会改变证据日期或训练/模型资产。两份可变运行汇总保留读取时的原字节快照；其他输入均以原路径和 SHA256 复核。停止前训练及诊断源码/辅助程序的 **154 项 SHA** 已核对，报告与停止状态整理的后续变化单独记录。


