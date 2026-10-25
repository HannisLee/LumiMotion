# 静态ELDL从头Stage1与GT方向混合光续训入口

日期：2026-10-24；服务器 ushas / lumimotion-ushas；所属GOAL：`LIHAN/DOC/GOAL/1024-02-静态ELDL从头Stage1与GT方向光GS法线突破.md`（P0、S0、A0、G0/G1）。

全部为新增文件，未修改任何共享模块（scene/static_dl_*、gaussian_renderer、scene/gaussian_model.py 等均保持原样，供动态GT方向会话继续使用）。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `scripts/prepare_static_eldl_p0.py` | P0：核验 data_fit_v2（117 TRAIN）manifest/逐文件SHA/划分；逐像素核对 RGB（Alpha>0处）与原 `image/` 逐位相同、Alpha=0处已清零、Alpha与 `albedo/` Alpha逐位相同；相机与 camera.json 一致；150帧 `light_dir_world` 与SUN局部+Z轴一致并单位化；只导出117 TRAIN帧号与单位方向；写 split 与 p0_audit。唯一读取原始lights/albedo/camera的环节。 |
| `scripts/snapshot_static_eldl_code.py` | 把 arguments/assets/gaussian_renderer/scene/scripts/utils 冻结复制到实验目录并写逐文件SHA清单；正式训练用 `LIHAN/scripts/bootstrap_1011.py <快照> <模块>` 从快照运行，避免其他会话改动共享代码。 |
| `scripts/train_static_eldl_stage1.py` | S0：原生 `scripts/train_stage1.Trainer` 包装。训练前形变几何头置零冻结（复用 `scripts/train_static_dl.freeze_geometry_heads`）且 `pred_color=False`（无时变颜色头）；117帧/DEV与TEST不可见断言；审计钩子禁止读取白名单外图像/模型/GT；逐次计数optimizer更新并记录每个保存点；抽查形变输出恒0；结束核对几何头全零与形变网络state未变。 |
| `scene/static_eldl_lighting.py` | 混合光物理前向与审计（见第2节）。 |
| `scripts/train_static_eldl_physical.py` | A0照明适配与G（G0 full / G1 detach_opacity）固定拓扑几何优化入口（见第3节）。 |

## 2. 混合光模型（scene/static_eldl_lighting.py）

逐Gaussian在线性域着色后用原生光栅合成，再转sRGB与训练RGB比较：

```
C_lin = rho_lin · [ (1−m)·S(n) + m·V·max(n·l,0) ]
S(n)  = max(0, Σ_{l≤2} a_ck Y_k(n)) / L00,  L00 = Σ_c w_c a_c0 · Y00,  w=(0.2126,0.7152,0.0722)
m     = sigmoid(beta)
```

- n：rotation第三列，按相机方向翻面（符号detach，与静态DL一致）；l：P0净化表中对应TRAIN帧的GT主光单位方向，固定不训练。
- 共享低阶normal-SH（l≤2，RGB×9）环境辐照，无空间、相机、时间或逐帧自由度；环境项不乘直射可见度V。
- **幅度规范（固定）**：环境平均亮度辐照(1−m)π + 直射峰值辐照mπ ≡ π。全局rho/光强尺度被消除，可学的只有环境/直射比例m、环境SH色度（log_dc）与形状（l=1,2）、反照率。初值m=0.5、SH为常数灰（不依据GT）。
- V：复用 `scene/static_dl_photometric.StaticVisibility`（预测Gaussian二十面体BVH，bias 0.03，no_grad，阴影梯度不经BVH）。
- 鲁棒RGB：`censored_loss`，权重规则逐式沿用 `scene/static_dl_censored.censored_photometric_loss`（阈值0.08，正残差0.25、负残差0.5，Alpha≤0.5背景权重1，√(r²+1e−6)）。**差别**：满码集合改用 `camera.observed_saturation_mask`（原分辨率任一子像素RGB==65535且Alpha>0，再AREA缩放）∧Alpha>0.5，而非旧的“缩放后target==1”。16位AREA缩小后部分满码像素的值仍是真实均值的下界，旧规则会把它们当精确观测；新集合是旧集合的超集，日志同时记录两者比例。
- 几何损失 `geometry_losses`：逐式复制 `scene/static_dl_ambient.geometry_losses` 与 `scene/static_dl_normal_opacity.geometry_losses`（Alpha BCE、depth_ratio=0的expected-depth法线一致性、GT Alpha>0.99腐蚀5×5∧预测Alpha>0.5掩膜、distortion），route=detach_opacity时只在normal副本的原生光栅调用 `detach_opacity=True`。
- 审计：`audit_physical_forward`（环境+直射分量加和、法线全部朝向相机、可见度缓存=重建、前向签名不含目标、数值有限）；`audit_normal_routes`（full/detach前向逐元素一致；normal→opacity有限非零→严格0；normal→xyz/rotation/scaling在atol1e−6/rtol1e−4一致；RGB/Alpha/distortion→opacity保留；总opacity梯度差恰为normal项；总xyz/rot/scale梯度相同；参数与RNG不变）。

与历史冻结实现比对：`scene/static_dl_{normal_opacity,ambient,photometric,censored,shadow_split,foot_recovery}.py`、`gaussian_renderer/__init__.py` 与F-T04冻结SHA逐文件一致；`scene/gaussian_model.py`（新增默认关闭的屏幕半径事件开关）、`scene/dataset_readers.py`（新增rgba16分支）、`utils/camera_utils.py`（新增满码掩膜属性）相对冻结版本只有不影响本入口固定拓扑路径的增量。

## 3. 训练入口（scripts/train_static_eldl_physical.py）

- 冻结常量 `FROZEN`：A0 albedo/照明lr 1e−3；G xyz 1e−6、rotation 5e−5、scaling 1e−5、opacity 1e−3（几何四组按0.1^((g−1)/7019)对数线性衰减到0.1倍），albedo 2e−4、照明1e−5恒定；loss系数0.1/0.1/1000；seed 1012；正式预算A0=1080（phase1=540）、G=7020，正式运行不能改步数/保存点/父模型。
- A0：父须为S0@35000（核对training_finished：34999次更新、几何头全零、形变state未变、pred_color=False）；几何（xyz/rotation/scaling/opacity）与albedo_rest等全部requires_grad=False；1–540学albedo DC与环境SH，541起把beta作为新param group（零动量）加入；保存0/540/1080；每个保存点断言几何五项哈希与父逐位一致。A0中Alpha/normal/distortion只在no_grad下记录。
- G：父须为A0@1080（physical_finished：A0、1080次、非冒烟）；时钟从相对总步1080开始，保存相对总步4050/8100（新增2970/7020）；新Adam零动量、增密统计清零；固定拓扑（无增密/裁剪/reset，逐步断言点数不变）；BVH每步重建。G0/G1仅 `--route` 不同。
- 公共：独立 `random.Random(1012)` 不放回遍历117相机（G0/G1相同序列；入口预推导首相机并断言）；顺序 forward/backward→optimizer→保存；每步都更新（含末步），保存的是更新后参数；保存PLY、lighting.json（含照明状态、方向SHA、累计更新数）、Adam状态+相机RNG/剩余栈、形变（S0零头副本）、cfg_args；审计钩子同S0。
- 未实现：增密统计“两个normal副本screen梯度先相加再累计”只在后续F阶段原增密时需要，本入口固定拓扑不涉及。

## 4. 冒烟（仅工程）

- `output/smoke_test/1024-04-LH-static-jumpingjacks-ELDL-V3-S0原生Stage1冒烟`：PASS（200步，更新199次，形变恒0，拓扑/reset执行）。
- `output/smoke_test/1024-05-LH-static-jumpingjacks-ELDL-V3-A0照明适配冒烟`：PASS（父S0冒烟@200，60步）。
- `output/smoke_test/1024-06-LH-static-jumpingjacks-ELDL-V3-G冒烟-full`、`…-G冒烟-detach_opacity`：PASS（父A0冒烟@60，各40步，零更新路线审计全项通过，两路初始化哈希与首相机一致）。
- 另见训练计划1024-07中以S0中间检查点进行的真实规模冒烟。

## 5. 1024冻结评测入口的adapter/marker登记（2026-10-24 补充）

冻结评测入口 `LIHAN/scripts/eldl_1024_eval.py` 未做任何修改（当前SHA256 `5821cadb847bf99495daea4956d8e5b16140c9aabc68194a9c6a0892a3d6336d`，与冻结值5821cadb…一致）；不改变任何评测数学、掩膜、聚合或门槛。

- 新增 `LIHAN/scripts/register_static_eldl_eval_marker.py`（SHA256 `5e600ff9f280aad66a2a1d1f548cd6a77fd448017bfd4e31a5d0b574e8683f2c`）：为模型目录补写入口需要的 `eldl_v3_training.json` marker（不触碰checkpoint，已存在则拒绝覆盖）。
  - S0：baseline=static_sh；args由S0实际 train_command.sh 按快照中原argparse重新解析得到带类型完整参数；source_code_root=S0快照；rasterizer_provenance取自训练记录。语义说明：入口static_sh只要求几何头严格零；S0另有pred_color=False，形变step返回d_color=None，原生SH着色无时变颜色。
  - A0/G：baseline=static_eldl_mixed，gt_supervision='TRAIN_GT_direction_only'（唯一允许的GT例外），source_code_root=A0快照。
- 新增包装器 `LIHAN/scripts/eldl_1024_eval_static_gt.py`（SHA256 `e66752467216430e1c0e8d40fa45c1426ea9f0025da614371c281014f6ce17d3`）：导入冻结入口后仅替换 model_record/adapter/make_plan 三个查找点。static_eldl_mixed的几何通过冻结入口原static_sh adapter（全部原校验照常执行）；RGB改用 `scene/static_eldl_lighting.render_mixed` + checkpoint的lighting.json + 评测专用GT主光方向（lights.json对应帧light_dir_world，与SUN轴及训练净化方向逐帧核对）；separation等override颜色渲染仍走原生render。make_plan把GPU步骤模块改为包装器并在plan.json登记包装器SHA；CPU指标步骤仍调用冻结入口原文。其他baseline直接回退原函数。
- 验证：S0@35000 DEV（冻结入口原文）exit0；A0评测adapter冒烟 `output/smoke_test/1024-09-LH-static-jumpingjacks-ELDL-V3-A0评测adapter冒烟`（3帧，几何parity全0）；A0@1080 DEV完整执行，GS/Depth与S0@35000逐数值一致（几何未变的预期），RGB由混合光渲染。

## 6. 几何学习率倍率参数（2026-10-24 补充）

`scripts/train_static_eldl_physical.py` 新增 `--geometry_lr_multiplier`（choices 1/10，默认1；A0禁用）。只作用于G阶段xyz/rotation/scaling三组，衰减日程不变；opacity、albedo、照明不变。默认1时与G0/G1数值完全相同（G0/G1从A0快照运行，不受影响）。physical_plan.json记录倍率与首步lr；`LIHAN/scripts/register_static_eldl_eval_marker.py` 把倍率写进marker的args。G2/G3快照 `output/1024-25-…/snapshot_code`（入口SHA256 c4532d61…）。
