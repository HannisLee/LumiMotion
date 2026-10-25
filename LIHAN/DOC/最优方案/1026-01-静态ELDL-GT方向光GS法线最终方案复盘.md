# 静态ELDL-GT方向光GS法线最终方案复盘

日期：2026-10-26（JST）；服务器 ushas / lumimotion-ushas；所属GOAL：`LIHAN/DOC/GOAL/1024-02-静态ELDL从头Stage1与GT方向光GS法线突破.md`。撰写：规划与实现Agent（Claude Opus 5.5，high）。

文中所有数值都引自独立评测报告：`LIHAN/DOC/报告/1024-05-静态ELDL评测协议冻结与EL基准复核.md`（第八至十六节，DEV）与 `LIHAN/DOC/报告/1025-05-静态ELDL-G11里程碑TEST独立验收.md`（G11 TEST）。本文不自行计算或推断任何精度。最终候选G15的TEST由独立评测方填写（第六节留空）。

## 一、结论摘要

- **DEV选定的最终候选**：G15@8100（S0@35000 → A0@1080 → G15，新增7020次）。DEV固定分母P为3.2707，是全部合格保存点中最低的；相对父模型A0（3.8231）下降0.5524，相对G11（3.3905）下降0.1198，18/18帧改善。G15的种子复验（G15-s2/s3）正在进行（训练/1026-04），只用于确认，不参与选点。
- **基本目标里程碑**：G11@8100 TEST 条件MAE 3.3512，低于历史EL 3.50676425与V3重评EL 3.5096，也低于本轮S0 3.8113；固定分母3.5145 < 3.7037；coverage 0.99908。判定为 **基本目标PASS**，**1°进阶目标FAILED**（还差0.8444°），**几何完整性PASS（有保留）**（报告1025-05）。
- **未达成**：1°进阶目标（TEST条件≤2.50676425）。

## 二、最终谱系与复现

公共输入（全部阶段相同，仅训练输入）：

| 项 | 路径 | SHA256 |
|---|---|---|
| 训练源（117 TRAIN，16位RGB＋原Alpha） | `output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_fit_v2`（no_gt_manifest.json） | `beebcbb9d194cac0c533e6124ab89aebb48fb7717b5c10cbc92b075f960ef237` |
| transforms_train.json | 同上 | `993315428542212e4f787185f1b76b302d3a25ce8496492f107b7179e3b978eb` |
| 划分（117 TRAIN / 18 DEV 61–69,71–79 / 15 TEST） | `output/1024-17-LH-static-jumpingjacks-ELDL-V3-P0协议与净化输入/split_train117_dev18_test15.json` | `6ce991aeff528743740dbaf1a1ab2bf320c8a3797f24922f967b7be2b2659e67` |
| GT主光方向（只含117 TRAIN帧号＋单位方向） | 同目录 `train117_sun_directions.json` | `4a35fa33ba34d1f0b020768a8ef360821577384262f4e2664b848243053a2583` |

除TRAIN帧主光方向外，训练不读取任何GT（法线、深度、材质、mesh、光强、点光、环境、语义）。

### 2.1 S0：静态原生Stage1（随机初始化，seed0）

- **目录**：`output/1024-14-LH-static-jumpingjacks-ELDL-V3-S0原生Stage1`；源码快照 `snapshot_code`（manifest SHA256 `d9c78debe2aa6f8670439c94665caa9ec7c4b6cb96c024cf4bb5d35b65024510`）；计划为训练/1024-06。
- **完整命令**：见 `train_command.sh`，原文如下：

```bash
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_EXTENSIONS_DIR=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/native_cache CUDA_HOME=/home/han.li/miniconda3/envs/lumimotion-ushas MAX_JOBS=4 PYTHONUNBUFFERED=1 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1024-14-LH-static-jumpingjacks-ELDL-V3-S0原生Stage1/snapshot_code scripts.train_static_eldl_stage1 --source_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_fit_v2 --model_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1024-14-LH-static-jumpingjacks-ELDL-V3-S0原生Stage1/model_mlp --split /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1024-17-LH-static-jumpingjacks-ELDL-V3-P0协议与净化输入/split_train117_dev18_test15.json --iterations 35000 --save_iterations 10000 20000 35000 --deform-type mlp --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1 --d_color_reg_loss_weight 0.01 --data_device cpu --start_normal_reg 8000
```

- **静态配置**：几何形变头训练前置零并冻结，`pred_color=False`。
- **真实更新数**：名义35000，实际34999次optimizer更新（原生Trainer在末步少一次）；最终171167点。
- **父检查点**：`model_mlp/point_cloud/iteration_35000/point_cloud.ply`（SHA256 `7f5cb128a0768a6a969b6e13d3521cf2ff9469f34fe64b0eb4bd39bd82ec098c`），`deform/iteration_35000/deform.pth`（`acc34b48ac49444ed02d5e7f1a9a57a3e7e9793404cb1850b04213a6a5b7d80f`，零几何头，此后各阶段原样复制）。

### 2.2 A0：物理照明适配（几何冻结）

- **目录**：`output/1024-20-LH-static-jumpingjacks-ELDL-V3-A0照明适配`；快照manifest `cd48027c8b1e21d2c61ee09a04ff1442f44d9b601edf21b3ae0afba5b42e762c`；计划为训练/1024-07，实现为修改/1024-07。
- **命令**（`train_command.sh`，GPU前缀同上）：`… bootstrap_1011.py <A0>/snapshot_code scripts.train_static_eldl_physical --stage A0 --parent <S0>/model_mlp --parent_iteration 35000 --source <data_fit_v2> --split <split> --directions <directions> --output <A0目录> --steps 1080`。
- **物理前向**：C_lin = ρ_lin·[(1−m)·S(n) + m·V·max(n·l,0)]。其中S为共享l≤2 normal-SH环境；固定幅度规范为“环境平均亮度辐照＋直射峰值辐照≡π”；V为预测几何BVH可见度（no_grad）。
- **日程**：1–540步只学albedo与环境SH，541–1080步加入直射比例m。
- **真实更新**：1080次；门禁PASS（m=0.503）；几何与S0@35000逐位相同。
- **检查点**：`model_mlp/point_cloud/iteration_1080/point_cloud.ply`（`ea4f5cd1017388348f02c573e78594ba310ddcc9df264aea7342d62d51c957f5`），`lighting/iteration_1080/lighting.json`（`419f94b3057b67fed5e7c7e18a29e8beb64d0d1360c6c5c38d41140708ef02cc`）。

### 2.3 G15（最终DEV候选）：物理几何优化

- **目录**：`output/1026-09-LH-static-jumpingjacks-ELDL-V3-G15-full-部分覆盖photo排除-xyz与scaling-LR30倍-rotation0.3倍`。
- **源码快照**：与H谱系共用，位于 `output/1026-06-LH-static-jumpingjacks-ELDL-V3-H-A0-S0at10000照明适配/snapshot_code`（manifest `f7971adc0ca1aee257350c262294722b2852e5a8273b3e48a46a6cc4cd46a623`）。
  - 入口 `scripts/train_static_eldl_physical.py` `e5638b24e25f17c47fec8053efb3a2759bf7f5bdbe77c7d89cef5d1de5f14013`
  - `scene/static_eldl_lighting.py` `227dca035815601fc8b7be84ddd968e3620c6a59a688170037c21ecf16d76cff`
  - `scene/gaussian_model.py` `23361f3878fc…`；`gaussian_renderer/__init__.py` `c9c72097…`（全部来源SHA见 `model_mlp/physical_plan.json` 的sources）。
- **完整命令**（`train_command.sh` 原文）：

```bash
env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_EXTENSIONS_DIR=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/native_cache CUDA_HOME=/home/han.li/miniconda3/envs/lumimotion-ushas MAX_JOBS=4 PYTHONUNBUFFERED=1 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1026-06-LH-static-jumpingjacks-ELDL-V3-H-A0-S0at10000照明适配/snapshot_code scripts.train_static_eldl_physical --stage G --route full --parent /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1024-20-LH-static-jumpingjacks-ELDL-V3-A0照明适配/model_mlp --parent_iteration 1080 --source /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_fit_v2 --split /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1024-17-LH-static-jumpingjacks-ELDL-V3-P0协议与净化输入/split_train117_dev18_test15.json --directions /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1024-17-LH-static-jumpingjacks-ELDL-V3-P0协议与净化输入/train117_sun_directions.json --photo_partial_alpha exclude --geometry_lr_multiplier 30 --rotation_lr_multiplier 0.3 --output /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1026-09-LH-static-jumpingjacks-ELDL-V3-G15-full-部分覆盖photo排除-xyz与scaling-LR30倍-rotation0.3倍 --steps 7020
```

- **配置**：
  - **loss**：censored鲁棒photo（阈值0.08；正残差权重0.25、负残差0.5；满码单边；**GT Alpha∈(0,0.99)的部分覆盖像素photo权重为0**，分母不变）＋ 0.1·Alpha BCE ＋ 0.1·expected-depth法线一致性 ＋ 1000·distortion。
  - **路由**：normal→opacity为full。
  - **首步LR**：xyz 3e−5（×30）、scaling 3e−4（×30）、rotation 1.5e−5（×0.3）、opacity 1e−3，四组按0.1^((g−1)/7019)对数线性衰减；albedo 2e−4、照明1e−5恒定。
  - **其余**：新建零动量Adam；seed1012不放回相机序列（首相机image_0048）；固定拓扑，无增密、裁剪、reset。每步顺序为 forward/backward→optimizer→保存；保存的是更新后参数。
- **真实更新**：本阶段7020次。累计为S0 34999 + A0 1080 + G15 7020；保存相对总步4050/8100，最终171167点。
- **可重载检查点**（`model_mlp`，@8100）：

| 文件 | SHA256 |
|---|---|
| `point_cloud/iteration_8100/point_cloud.ply` | `ea9c7e583c272e9e5566c208f322601cdac5618059fa05d4f64bd9a03e3e2e0b` |
| `deform/iteration_8100/deform.pth` | `acc34b48ac49444ed02d5e7f1a9a57a3e7e9793404cb1850b04213a6a5b7d80f` |
| `cfg_args` | `613807c1f1f7fbb8bcab7dd98444c9b29044670dd515f089fa596ca96a717ac6` |
| `lighting/iteration_8100/lighting.json` | `2b5f294b260c21a993bc90cd3088c528c2df2719f3141366fb969e6e83e45386` |
| `eldl_v3_training.json`（评测marker） | `6e4d719cb7c240e208f380c63784759ebbce69257db1eca5f84bd052f0612fa2` |
| `physical_plan.json` / `physical_finished.json` | `54d26ec900fa…71a2` / `94e3e3a4993d…cca7` |

### 2.4 G11（基本目标TEST里程碑）

- 目录：`output/1025-21-LH-static-jumpingjacks-ELDL-V3-G11-full-部分覆盖photo排除-xyz与scaling-LR10倍-rotation1倍`。
- 快照manifest：`9339d062f53f87625b74a0152bd7f3a0cb29fa902f3729a4b205bbacb8ac0a36`。
- 命令：与G15相同，只是 `--geometry_lr_multiplier 10 --rotation_lr_multiplier 1`，快照为自身目录下的snapshot_code。
- 检查点SHA（报告1025-05）：PLY `c082ca00f75f88eee19b1d194b4616ca4282868fa22e616c7922ef5ae0f47471`，lighting `906cbf369b27…e020`，marker `9d65c5e8a0a0…1bbc`。

### 2.5 复现性说明

- 光栅反向使用原子加，存在非确定性：同命令复跑时第1步loss逐位相同，100步后相对差约1e−5（冒烟smoke_test/1025-07、1025-23）。
- G4配置三种子的DEV P标准差为σ_seed=0.0011（报告1024-05第13.2节）。该σ只包含相机采样顺序噪声，不含父模型与DEV 18帧的抽样不确定性。
- 正式训练一律通过 `LIHAN/scripts/bootstrap_1011.py <快照> <模块>` 从冻结快照运行，不受工作区后续改动影响。

## 三、评测命令（独立评测Agent使用）

冻结入口 `LIHAN/scripts/eldl_1024_eval.py`（`5821cadb847bf99495daea4956d8e5b16140c9aabc68194a9c6a0892a3d6336d`）未改。混合光模型的RGB adapter在 `LIHAN/scripts/eldl_1024_eval_static_gt.py`（`54091f32ed9daf64c34eda5e0f608812130d81079cc666657541a36ed12f265a`）。由于1025起lighting文件新增了两个训练损失函数，评测方使用compat入口 `LIHAN/scripts/eldl_1024_eval_static_gt_compat.py`（`17e73fe59e01ca7f16528bd153311a386762f1d7e98c89bd8659bc2c741398eb`）：它对lighting文件按AST比对，只排除 `censored_loss` 与 `partial_alpha_keep_mask` 两个训练函数，渲染代码必须完全一致；数学、掩膜、分母不变。

```bash
cd /home/han.li/reproduce/LumiMotion-stage2-perlight
PY=/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python
# DEV（18帧）
$PY LIHAN/scripts/eldl_1024_eval_static_gt_compat.py --action run \
  --model output/1026-09-LH-static-jumpingjacks-ELDL-V3-G15-full-部分覆盖photo排除-xyz与scaling-LR30倍-rotation0.3倍/model_mlp \
  --iteration 8100 --output output/<新目录>/G15_8100_DEV --split held --gpu <G>
# TEST（15帧，含逐帧TSDF Mesh与四类视频；冻结checkpoint后只跑一次）
$PY LIHAN/scripts/eldl_1024_eval_static_gt_compat.py --action run --model <同上> --iteration 8100 \
  --output output/<新目录>/G15_8100_TEST --split test --gpu <G>
```

协议（报告1024-05第四、六节）：
- GS法线为世界坐标、有向角；前景为GT Alpha≥0.5 ∩ GT有效；预测alpha<0.5或法线无效即为缺失。
- 主指标为像素加权条件MAE，以及固定分母缺失罚180°的P。
- DEV选择规则：P最低者胜，持平容差0.03°，并须满足coverage/人体/鞋足/Depth/RGB/IoU门槛。
- DEV/TEST的RGB渲染使用评测专用GT主光方向（只影响RGB）。

## 四、各轮配对实验结论（数值均引自报告1024-05对应节；P为DEV固定分母GS）

| 杠杆 | 配对（单因素） | 关键DEV数值 | 结论 |
|---|---|---|---|
| 物理照明适配 | S0@35000 → A0@1080 | GS逐位相同（P 3.8231），PSNR 30.51→32.78 | 继承无损；只改善外观（第八节） |
| 几何优化＋normal路由 | G0 full vs G1 detach（LR×1） | G0@8100 P 3.8738，G1 3.9637；条件MAE 3.41/3.49 | 条件MAE改善，但边缘覆盖丢失，使P劣于父模型；LR×1下detach更差（第九节） |
| 几何LR×10（含rotation） | G2/G3 vs G0/G1 | G2@8100 P 4.0919；平台1.70、足18.68 | 人体改善，平台和足退化（第九节） |
| **photo部分覆盖像素排除**（成因：target为a·sRGB(含opaque世界背景)，与线性合成的物理前向不一致，诊断见报告1025-02） | G4 vs G0 | G4@8100 P 3.5701（−0.30对G0），coverage恢复到0.99906 | **有效**，是首个通过全部门槛的点（第十节） |
| opacity冻结 | G5 vs G0 | G5@8100 P 3.9777 | 无效（第十节） |
| normal权重0.3 | G6 vs G4 | G6@8100 P 3.6050 | 无收益，鞋变差（第十一节） |
| 几何LR×3（含rotation） | G7 vs G4 | G7@8100 P 3.6241；人体13.77，平台1.60 | 平台变差抵消人体收益（第十一节）；审计显示平台退化是rotation带来的无方向法线噪声（报告1025-04） |
| sRGB域合成（另一种物理转换修正） | G8 vs G4 | G8@8100 P 3.5778（持平），PSNR高1.6 dB | 法线持平，只改善RGB（第十一节） |
| 原增密（成熟父） | F0 vs F0c（父G4） | F0 P 3.68，比F0c高约0.11，0/18帧更优 | **无效**：裁掉低opacity点后平台和足变差；增密在成熟父上几乎不再触发（第十二节） |
| 续训步数 | F0c vs G4；G12 vs G9 | ΔP −0.006；G12 −0.004/−0.010 | 已饱和（第十二、十四节） |
| **xyz/scaling剂量，rotation×1** | G9（×3）vs G4 | G9@8100 P 3.4769（−0.093，18/18帧） | **有效**：人体收益来自位置和尺度，平台不退化（第十三节） |
| 鲁棒阈值0.08→0.036 | G10 vs G4 | G10@8100 P 3.5746 | 无效（第十三节） |
| **xyz/scaling剂量×10** | G11 vs G9 | G11@8100 P 3.3905（−0.086，16/18帧） | **有效**，之后送TEST作为里程碑（第十四节） |
| **xyz/scaling剂量×30** | G13 vs G11 | G13@8100 P 3.3401（−0.050，10/18帧）；足16.89 | 有效但已递减，足部上升（第十五节） |
| **rotation×0.3** | G14 vs G11 | G14@8100 P 3.3207（−0.070，16/18帧）；平台1.25，人体13.68 | 有效：平台改善、人体变差，二者存在权衡（第十五节） |
| **剂量×30＋rotation×0.3叠加** | G15 vs G11 | **G15@8100 P 3.2707**（−0.120，18/18帧），条件3.0811，coverage 0.99893，IoU 0.99766，平台1.07/1.25，人体13.15/13.40，足16.97，鞋12.24，Depth 5.539 | **有效且近似可加**（0.120≈0.050+0.070）；判据P<3.2731通过，余量0.0024（第十六节） |
| opacity lr 0.05 | Ra vs G11 | Ra@9000 P 3.3356（−0.055，12/18帧），@10100 −0.029 | 收益与噪声同量级，并随训练减弱（第15.2节） |
| reset（full） | Rb vs Ra | Rb@9200 P 5.1817；腿38°、足和鞋约59° | **失败**：腿脚消失（第15.2节） |
| reset后detach（DL T06机制） | Rc vs Rb | Rc@9300 P 3.3031，@10100时Depth 8.04未过门槛 | detach阻止了崩坏，但收益是不稳定的瞬时窗口（第15.2节） |
| 早期切入新谱系H（S0@10000，原生LR＋增密＋reset＋detach） | H1/H2 vs G11 | 最好的H1@4080 P 3.9320，0/18帧优于G11；随步数单调恶化 | **否定**：“成熟几何是人体误差瓶颈”的假设不成立（第十六节） |
| 照明容量扩展（可学习点光或空间照明） | 只读审计，未训练 | 平台残差能被平滑空间函数或GT点光辐照解释的R²只有0.05/0.025；法线偏转与残差的相关为0.005 | **证据不支持**，未实现（报告1025-04） |

## 五、已知局限

1. **1°进阶目标未达成**：G11 TEST条件3.3512，比历史EL只低0.1556°，距2.50676425还差0.8444°（报告1025-05）。
   - 误差主要来自人体（TEST人体条件13.107°）；平台1.280°已优于EL（1.393）。
   - 主线上每个单项杠杆只带来0.03–0.12的P收益，剂量收益在递减；H谱系也已否定。
   - 在本数据和本前向下，没有找到能让人体降到约8°的机制。
2. **鞋足的Mesh/Depth精度仍差**：G11 TEST中footwear的Mesh为22.15°、Depth为21.37°，人体Depth coverage只有0.940。GS的鞋足（12.97°）优于EL，但不能宣称鞋足被完整精确恢复（报告1025-05“几何完整性PASS（有保留）”）。
3. **TEST非全新盲测**：旧TEST此前已被历史实验查看；本轮先对G11做了一次里程碑TEST，若再对G15做最终TEST，则本轮共使用TEST两次。两次都只针对DEV选出的单一checkpoint，没有用TEST结果回头选点。
4. **G15判据余量很小**：G15@8100的判据余量只有0.0024，约为σ_seed（0.0011）的2倍；coverage与IoU的余量也只有约0.0004/0.0005。因此需要种子复验确认（G15-s2/s3，训练/1026-04，进行中）。
5. **共享低阶照明的近似边界**：两盏固定点光的近场与互反射没有被显式建模。审计显示它不是平台法线退化的主因，但RGB残差仍有约9%的高频成分（报告1025-04）。
6. **Separation**：静态配置下为常量/继承门控，没有学到分离（各轮目检）。

## 六、最终TEST结果（独立评测方填写）

TEST第2次使用，非盲测：G15是按DEV冻结规则选出，不是按TEST挑出；G11的第1次TEST并未用于选G15。独立评测报告：；输出 。

| 项 | G15@8100 TEST | G11@8100 TEST（里程碑） | S0@35000 | EL（V3重评 / 历史） |
|---|---:|---:|---:|---:|
| GS条件MAE | **3.2412** | 3.3512 | 3.8113 | 3.5096 / 3.50676425 |
| GS固定分母 | **3.4052** | 3.5145 | 3.9801 | 3.7037 / 3.6534 |
| coverage | 0.99907 | 0.99908 | 0.99904 | 0.99890 / 0.99917 |
| Depth Normal | 5.536 | 5.645 | 8.002 | 7.3555 |
| Mesh Normal | 4.223 | 4.281 | 5.6075 | 5.4595 |
| 人体 / 平台 / footwear（GS条件） | 13.187 / 1.129 / 12.575 | 13.107 / 1.280 / 12.966 | 15.016 / 1.435 / 15.442 | 13.501 / 1.393 / 15.257 |
| RGB PSNR / Alpha IoU | 31.99 / 0.99737 | 32.08 / 0.99749 | 30.32 / 0.99798 | – |
| 基本目标 / 1°目标 / 几何完整性 | PASS / FAILED（3.2412>2.50676425，差0.7344°） / PASS（有保留） | PASS / FAILED / PASS（有保留） | – | – |

相对历史EL，G15条件MAE低0.2656°；相对G11低0.1100°，其中平台改善0.151°、人体反而略差0.080°。

## 七、文档索引

- 诊断与审计：报告/1025-02（边缘覆盖）、报告/1025-04（平台与照明容量）
- 训练计划：训练/1024-06、1024-07、1024-10、1025-02、1025-03、1025-05～1025-10、1026-03、1026-04
- 实现：修改/1024-07、1025-02、1025-04、1026-08
- 评测：报告/1024-05、报告/1025-05
