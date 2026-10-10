# 原生LumiMotion、GT Lambertian与方向光PBR训练完整总结

日期：2026-10-10T11:54:52.820177+09:00（Asia/Tokyo）。用户要求完成当前训练后停止迭代；本报告生成时B10全部20,000步及9步后处理完成，不启动后续实验。

## 1. 最终结论

- B10：原Stage1 35000→55000独立Stage2 20,000步完成，训练和9步后处理均实际exit0且会话关闭；完整15测试帧七项MAE、独立CPU复算、实际代表图检查完成。
- 原有向、全前景像素加权GS Normal MAE **15.54006647°**，覆盖率 **95.48687755%**；目标<7° **FAILED**。人工画质 **FAILED**。
- 相比B09同55k，GS误差变化-0.40224716°；相比B00历史基线变化-5.68868943°。最新实验不能仅凭约束loss下降认定法线更准确。
- 停止迭代，保留全部成功/失败checkpoint、渲染、逐帧数组、日志和来源审计。不执行新的训练、commit、push或建分支；原目标中质量验收及后续分支条件尚未实现。

## 2. 已完成的前置任务

两套jumpingjacks_environment_light/direction_light均转换完毕，各150帧800²、135 TRAIN/15 TEST，time=frame/150、原相机/尺度不变。原版训练resolution2为400²、黑背景。environment_light文件名沿用来源，实际为双POINT；direction_light是两盏重叠SUN。

env已用原版LumiMotion完整Stage1 35k+Stage2新增20k，并完成材质/HDR/NVS与MAE；dir已完成原版Stage1 35k、99827点和32形变tensor，随后完成GT Lambertian Stage2及离线评测。早期GT Lambertian兼容灯表保留两SUN的energy总和6，实际标量按135 TRAIN RGB标定；该阶段不套用后续严格PBR方向白色单位表的来源说明。原版转换借albedo文件Alpha，后续严格PBR转为原RGB自身Alpha并验证全部150帧Alpha逐位一致；没有重新训练Stage1或宣称最初转换就仅使用原RGB文件。

| 前置实验 | GS Normal MAE° | GS覆盖% | Depth Normal° | Mesh Normal° | Albedo MAE | Roughness MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| env原版LumiMotion | 13.88986765 | 92.93556362 | 18.92264049 | 14.95726286 | 0.19101779 | 0.19992533 |
| dir GT Lambertian | 21.52504346 | 92.93748932 | 27.60512758 | 22.88577862 | 0.23996675 | N/A（Lambertian无输出） |

这些是独立前置实验，不将其与最新方向光PBR当作同一配置的消融；历史报告PASS表示对应流程/复算验收，并非满足本次<7°或PBR画质门槛。前置完整命令与图像见训练1009-01、1009-02、1009-04及报告1009-03。

## 3. B00–B10迭代总表

| 实验 | 主要变化 | GS MAE° | GS覆盖% | Depth° | Mesh° | PSNR dB | 质量结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B00 | 原生PBR方向光初始基线 | 21.22875590 | 93.078788 | 27.802115 | 22.792036 | 18.384562 | FAILED |
| B02 | 原RGB自身Alpha、GT仅方向、RGB估计强度 | 19.40743705 | 93.694049 | 27.601540 | 22.243801 | 18.912290 | FAILED |
| B03 | 材质梯度初始化与范围投影修复 | 18.89548239 | 93.718121 | 27.506921 | 22.193614 | 19.353545 | FAILED |
| B04 | 固定来源形变；使用更正后几何评分 | 18.77743450 | 93.850031 | 27.219899 | 22.151338 | 19.475974 | FAILED |
| B05 | 来源预测平面真实XYZ/q投影 | 17.73081984 | 93.884935 | 26.212828 | 20.414000 | 19.375063 | FAILED |
| B06 | 绑定仓库原生CUDA光栅及正确法线朝向 | 17.29779705 | 94.101335 | 26.041745 | 20.344906 | 19.682203 | FAILED |
| B07 | 直接SUN Lambert+GGX、真实BVH光线预算 | 16.81871120 | 95.003286 | 25.836374 | 20.065406 | 19.235614 | FAILED |
| B08 | 来源输出raw gate微调 | 16.84133987 | 95.012674 | 25.820591 | 20.077078 | 19.249383 | FAILED |
| B09 | 直接概率优化、原生sigmoid保存与有限编码 | 15.94231363 | 95.478453 | 25.188075 | 19.886489 | 19.810345 | FAILED |
| B10 | TRAIN RGB来源局部轴约束与source-local旋转运输 | 15.54006647 | 95.486878 | 25.163986 | 19.759042 | 19.754227 | FAILED |

B01（1010-04预测几何与RGB光强）因Alpha来源审计不符合严格输入要求中止，未完成正式55k/七项评测，保留现场，不填作完整结果。B04表采用1010-11更正后的几何重评，不采用旧1010-09错误runtime评分；材质/RGB沿用原实际产物。B00–B05沿用当时安装光栅，B06起绑定仓库原生CUDA内核，因此跨该界线差异包含实现修复，不能全归因单一优化项。B07之后直接SUN/BVH预算一致；B09→B10是同源、同20k/同评分的最新对比。全部完整实验均未达质量/<7°条件。

### 历次完整实验七项误差

| 实验 | Albedo | Roughness | GS° | Depth Normal° | Mesh° | Expected Z m | Median Z m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B00 | 0.19659954 | 0.29154181 | 21.22875590 | 27.80211481 | 22.79203567 | 0.08852290 | 0.09119155 |
| B02 | 0.18603245 | 0.24401021 | 19.40743705 | 27.60153971 | 22.24380070 | 0.08647351 | 0.09010209 |
| B03 | 0.15765579 | 0.20939310 | 18.89548239 | 27.50692138 | 22.19361409 | 0.08631321 | 0.09000833 |
| B04 | 0.15600631 | 0.21155915 | 18.77743450 | 27.21989900 | 22.15133832 | 0.08569735 | 0.08932931 |
| B05 | 0.15432221 | 0.21168068 | 17.73081984 | 26.21282757 | 20.41400002 | 0.08031497 | 0.08444845 |
| B06 | 0.16231946 | 0.21780360 | 17.29779705 | 26.04174483 | 20.34490562 | 0.08001558 | 0.08431691 |
| B07 | 0.20195435 | 0.20060596 | 16.81871120 | 25.83637410 | 20.06540566 | 0.07846366 | 0.08282073 |
| B08 | 0.20137736 | 0.20119953 | 16.84133987 | 25.82059126 | 20.07707837 | 0.07851869 | 0.08287710 |
| B09 | 0.19121691 | 0.19018926 | 15.94231363 | 25.18807492 | 19.88648893 | 0.07691848 | 0.08110909 |
| B10 | 0.19107403 | 0.18083939 | 15.54006647 | 25.16398643 | 19.75904152 | 0.07666492 | 0.08065818 |

每次的逐项覆盖、共同像素和missing180保留在各自metrics_summary.json；本总结档案复制所有完整实验实际指标JSON和来源SHA，原始误差数组留在各实验output。

## 4. 当前B10完整七项MAE

| 指标 | B09 MAE | B10 MAE | B10−B09 | 单位 | B10覆盖% |
| --- | ---: | ---: | ---: | --- | ---: |
| Albedo MAE | 0.19121691 | 0.19107403 | -0.00014289 | linear RGB | 97.56697185 |
| Roughness MAE | 0.19018926 | 0.18083939 | -0.00934987 | r | 97.76863691 |
| GS Render Normal MAE | 15.94231363 | 15.54006647 | -0.40224716 | degree | 95.48687755 |
| Depth Normal MAE | 25.18807492 | 25.16398643 | -0.02408849 | degree | 90.80116409 |
| Mesh Normal MAE | 19.88648893 | 19.75904152 | -0.12744741 | degree | 95.38313037 |
| Expected Z Depth MAE | 0.07691848 | 0.07666492 | -0.00025355 | m | 95.48687755 |
| Median Z Depth MAE | 0.08110909 | 0.08065818 | -0.00045091 | m | 95.48687755 |

共同有效GS/Depth/Mesh像素 **376203/415433（90.55684069%）**，共同口径MAE依次 **14.51173016° / 25.10148879° / 18.71977060°**。GS缺失像素按180°惩罚 **22.96234465°**。共同口径和惩罚项是附加诊断；目标仍使用原GS完整前景pixel-weighted定义。

### RGB重建

| 指标 | B09 | B10 |
| --- | ---: | ---: |
| psnr | 19.81034533 | 19.75422671 |
| ssim | 0.92021880 | 0.91935445 |
| lpips | 0.10036608 | 0.10111726 |
| psnr_foreground | 12.31944750 | 12.27867602 |
| ssim_foreground_bbox | 0.81921756 | 0.81865975 |
| lpips_foreground_bbox | 0.20633367 | 0.20679339 |

黑色背景占比较大，整体PSNR/SSIM须同时看前景指标和实际人物/平台；不以全图RGB较好替代几何验收。

## 5. 当前严格PBR评分定义与数据边界

- 全15 TEST：0010至0150每10帧，共415433个GT Alpha≥.5前景像素；保留人物、平台、难像素、全部99827 Gaussian。法线为有向world shading normal角度、像素加权，未转为无向/仅人物/仅共同像素的目标。
- Albedo：linear RGB，GT Alpha≥.99+erode2，预测缺失补0，无颜色拟合。Roughness：独立原blend EEVEE透明Alpha合成GT来源，GT Alpha≥.99+erode2、预测Alpha≥.9后去预乘；不是首命中粗糙度。
- Depth：独立原blend三角形camera-positive-Z、单位米，未对齐或缩放拟合；Expected/Median Z及Depth Normal分别报告。Mesh：每个目标时间独立融合同时间135 TRAIN预测视图，voxel.02/sdf.08/depth20/Alpha.5，不融合其他时间、GT或TEST。
- 训练/初始化/约束只使用原RGB、自身Alpha、相机time、原预测35k/固定NN、GT光方向。白色单位灯表仅保存方向；全局正强度由RGB拟合，不读取GT灯能量/颜色。GT albedo/normal/roughness/depth/mesh/blend仅供独立离线评分。

## 6. 管线修改及Preset落地

原生PBR保留Lambert+GGX F0=.04、真实Gaussian/BVH可见性、SH辅助重建。Direct SUN使用已知方向及RGB学习的全局强度；trace origin沿incident方向偏移.1，实际L个直接光线及启用时N个间接光线与分配预算一致。Direct默认关闭间接PBR照明但仍保留SH辅助训练，HDR/SUN和训练共享几何/材质。光栅按独立namespace绑定仓库原生source与.so SHA，未替换系统pip供其他训练使用。

后续修复材质初始化梯度与范围投影、源35k固定形变、真实XYZ/q平面投影、概率gate优化与native sigmoid有限保存。冻结源NN32/opacity/scale；几何warmup2000/ramp1000，canonical XYZ trust .08m、q60°，不声称gate/plane改变后的world位移仍受该canonical界限。每个完整实验从原35k独立初始化。

当前Preset **pbr_gt_directional_source_rgb_normal** 继承B09参数，新增 **pbr_source_local_rotation=True**、**lambda_pbr_rgb_normal_prior=2** 与资产路径。source-local运输为R_source_world(p)·R_source_canonicalᵀ·R_current_canonical，作用于真实Gaussian quaternion；初始source==current保持原d_rotation逐bit一致，运输后应用既有source-only plane15187。FrozenScene/周期预览/GeometryRun/Mesh/材质/HDR/SUN使用同一入口；不覆写评测法线图。

原135 TRAIN RGB线性亮度+自身Alpha/源预测可见性/源qXYZ/单位方向形成独立LS轴候选；新增约束29621非plane点（static17389/dynamic12232）。loss=sum(confidence·(1−dot))/sum(confidence)，按29621归一化，全部99827点仍参与原训练/评分。资产与全部source/135逐图SHA和checkpoint绑定；输入confidence/Gram/解/残差由CPU独立复算。它是RGB估计轴，未引入GT normal，也未保证GGX/阴影情况下成为准确法线。新Preset不增加optimizer参数；legacy默认False/0。完整Adam状态未保存，不宣称无损续训。

具体实现：[stage2参数](/home/lihan/reproduce/LumiMotion-stage2-perlight/arguments/stage2.py)、[Preset](/home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/loss_stage2.py)、[RGB约束与运输](/home/lihan/reproduce/LumiMotion-stage2-perlight/scene/pbr_rgb_normal.py)、[训练入口](/home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/train_stage2.py)；完整修改历史在LIHAN/DOC/修改/1010系列。

## 7. B10实际目检

- **RGB与背景**：实际GT/B09/B10同55k TEST10/80/150对比：黑背景正常，人物和椭圆平台可辨认；人物上衣/手臂/裤腿有明显黑色碎斑，脚周及平台有大块黑灰伪几何/阴影混杂，最新外观与B09近似。背光中帧GT本身较暗，预测仍更暗，不把黑色RGB直接解释为Alpha缺失。RGB质量FAILED。
- **Alpha**：insights1/75/150均保留人物和完整平台Alpha，手臂/腿/脚轮廓与孤立飞点、平台边缘有模糊碎片。中帧RGB近黑但Alpha人物完整；未删平台/难点。
- **Albedo与静动态**：人物首中尾均可见，反照率保留照明方向、阴影和碎斑，手臂及裤腿脏乱；平台有颜色斑块和脚周污染。没有干净材质分解或静动态分离的证据。
- **法线/深度**：实际查看normal_depth/comparison_0010/0080/0150完整三行GT/GS/Depth/Z/Mesh对照：GS平台主体紫蓝，但脚周和人物表面有杂色、伪起伏；Depth Normal高频波纹更严重。Expected/Median Z在脚周平台有明显假起伏，与GT平面相比偏差集中在人物下方；Mesh也保留平台凸起，中帧身体和腿部有孔洞/缺片，末帧脚部不完整，三种法线人物误差明显。未删难像素或使用共同像素替代目标；整体几何质量FAILED。
- **HDR/SUN**：实际查看HDR1/76/150和SUN0/36/71，并单独查看SUN0036原PNG：HDR有橙色人物及平台色彩响应，但人物/腿部明显碎斑、反照率照明污染和平台脚周假起伏仍保留。SUN0/71人物与平台受光，0036几乎全黑（仅小点），对应背/下半球直接照明且关闭间接的情况，不把该单帧黑色当独立渲染错误。重光照功能有响应，未提供重光照GT精度或质量达标证明。

实际查看TEST10/80/150、insights1/75/150、HDR1/76/150、SUN0/36/71及normal-depth代表图；六视频均全帧解码，但未宣称人工逐帧观看全部视频。重光照没有对应GT，响应存在只说明功能可用。

![rgb_final_55000_comparison](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/inspection/rgb_final_55000_comparison.png)

![four_categories_final_55000](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/inspection/four_categories_final_55000.png)

![relight_final_55000](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/inspection/relight_final_55000.png)

## 8. 已有诊断证据与限制

warmup前2000步B09/B10 PhotoL1最大差2.68e−7，说明初始外观保持。新增轴loss从.0992468下降至55k .00371028，确实优化；它不是GT MAE。40k中间GS15.71225125°/覆盖93.93837%、Depth25.68376882°，中间Mesh未生成，不当正式七项或三法线共同评分。source confident q raw norm中位1.029/P991.805，没有支持“普遍巨大的q norm导致失效”的证据。

源RGB候选29621点在TRAIN1/75/149只贡献前景Alpha合成质量38.47%/28.52%/39.15%，plane贡献12.07%/16.43%/11.85%；点数不等于像素约束覆盖。近半贡献来自二者之外，单纯增大loss权重没有充分覆盖证据。source-only放松静态/axis条件后稳定近平台点仅15187→16110且无新增动态，简单放宽条件未解决主要缺口。此分组只用于诊断，未删评分像素。

停止要求之前完成的六TRAIN来源Jacobian诊断：动态近平台约37–44%、其余动态约44–48%det≤0；可逆动态轴与native旋转轴差约35–52°。源位置MLP不保证物理可逆体积形变，直接以J逆转置替换旋转候选未接受、未修改生产。有限差分相对最大差约.9–3.4%，近零角有float32舍入局限；不将该轴相容性诊断当GT法线准确度。全部现有诊断归档，不启动新候选实验。

## 9. 执行、审计和复现入口

训练 2026-10-10T11:11:25.974063+09:00 → 2026-10-10T11:41:41.517407+09:00，source35000/target55000，正式warmup2000/ramp1000/λ2。训练会话4419、checkpoint审计10940、后处理65390均真实终止；40k/50k/55k审计exit0。226 CPU、20步Preset、真实五参数梯度/Direct预算与共享buffer检查已实际完成；本次总结未启动额外测试或训练。

| 后处理步骤 | 实际退出码 | 秒 |
| --- | ---: | ---: |
| render | 0 | 533.11 |
| albedo | 0 | 17.74 |
| predicted_meshes | 0 | 127.42 |
| normal_depth | 0 | 24.17 |
| roughness | 0 | 17.37 |
| verify_roughness | 0 | 0.61 |
| verify_all_mae | 0 | 1.65 |
| audit_artifacts | 0 | 42.59 |
| write_results | 0 | 0.40 |

全量四类150×4、HDR150×2048采样、SUN72、六视频、15 albedo/roughness/normal/depth及15独立Mesh完成；CPU七MAE、roughness来源和artifact审计PASS。476核心+两source快照SHA冻结并核对，native kernel及来源资产绑定。自动数值PASS不等于人工质量PASS。完整train_argv、planned_commands、所有shell/封装命令和日志保留在本次输出，训练/评测文档均为最终状态。

### 完整实际训练命令

```bash
#!/bin/bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_source_rgb_normal --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_rgb_normal_prior_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-17-dir-训练RGB法线约束原型/candidate_assets'

```

复现渲染/评测时按 [planned_commands.json](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/planned_commands.json) 的九个完整argv顺序，逐步命令、环境、source/model/iteration、输出和日志见 [本次README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/README.md)；本报告不执行这些命令。

### 主要产物

- [最终模型与55000 PLY](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp)
- [七项指标与全部覆盖](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/metrics_summary.json)
- [人工目检记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/inspection/visual_review.json)
- [独立MAE复算](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/inspection)
- [训练文档](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束训练报告.md)
- [评测报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束各项MAE评测报告.md)
- [逐次GOAL](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md)
- [B10完整归档](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束)
- [RGB原型/20步冒烟](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test)
- [已有Jacobian诊断](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/archive/1010-19-dir-来源形变Jacobian法线运输诊断)

完整模型、大规模渲染和原评分数组留在output；归档保存报告、核心源码双快照、脚本、日志、指标、来源SHA及代表图。没有删除失败产物；没有把大型权重描述为已复制进文档归档。

### 最终checkpoint与来源核验

全部99827点finite；opacity/scale与32来源NN冻结；canonical最大位移0.080000013m、最大q变化60.00005341°、概率gate最大变化1.00000000；RGB拟合全局强度7.73176718，不是GT能量。材质投影界限通道比例41.508142%，硬clamp比例0.0；大量通道到界提示仍有材质/照明拟合限制，不能据此认定物理分解成功。

原35k PLY SHA256：8ebd88b6f847a20178458777d51aa191d5f67ba4bbc490fd1e99ef62f66d5529；原35k MLP与最终冻结MLP：e9ddf76b59a1ef8a85caa16891473e3a60663d903a5ad3928ac3e6ca46892bf5。最终55k PLY与RGB来源资产SHA见下表。

| 文件 | SHA256 |
| --- | --- |
| point_cloud.ply | af58022cd20c3ae2b1aa400a765d4ef4a22474944a6af808c96e3104150984a5 |
| deform.pth | e9ddf76b59a1ef8a85caa16891473e3a60663d903a5ad3928ac3e6ca46892bf5 |
| pbr_rgb_normal.npz | 8f2987d0af3f0554ab16891c62ee99cdc04e3a7e1ca0f418046144aabceeada2 |
| pbr_rgb_normal.json | e36b993b4d4cf3f3cb8d1fb1d43be397597a3157af2e85dc338e1e45bc39aaef |
| pbr_rgb_normal_input.npz | 87752217f37e9fe9b96835f892ee2289c3740560d2f4592029652db52d03bc15 |
| pbr_rgb_normal_input.json | 40d6b721ae4082522394c446270f1d37484c26832dfb4261aafe6b5e6b6e171e |

原生光栅source SHA256：82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0；.so SHA256：35bb1fd0a6f891df6e1e52082413b06d24946d5762b9157ade415b50bab7a3c0。模型pbr_rasterizer.json保留namespace/实际路径。

## 10. 代码与停止状态

分支 **1001-stage2-perlight**，HEAD **a9e4d1061fec9986d82362f78b42c0c269e6146e**；修改留在工作区，未进行本轮commit/push，未创建1010-goal-opt-pbr-jacks。最初“画质通过后提交，再建分支达<7°”的条件未达成。用户已明确停止后续迭代，持续目标暂停；只完成当前训练的评测、归档与本总结。报告/索引/GOAL与档案同步当前、LumiMotion-perlight、LumiMotion三个checkout，其他checkout生产源码没有同步修改。
