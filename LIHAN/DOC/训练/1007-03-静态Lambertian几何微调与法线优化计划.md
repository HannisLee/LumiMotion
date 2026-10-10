# 1007-03 静态 Lambertian 几何微调与法线优化计划

用户授权：以1006-01/02为来源开展Stage2实验，目标Normal MAE低于5°；保留旧管线，通过preset选择。状态：多轮筛选进行中，尚未达到质量目标；当前结果见文末及输出validation_summary.csv。

主目标采用原报告相同的15测试视角全前景GS Render Normal有向角度MAE <5°；同时报告Depth Normal、Mesh Normal、深度和覆盖率，另列布料区域。GT法线、深度、albedo不参与反向传播。选择超参数使用原105训练视角中每10个抽1个的11个内部验证视角；这不是Stage1未见视角，最终测试集仍为原15视角。

来源：1006-01及1006-02的model_mlp/iteration_35000；不从原版IR55k加载。优先CV4L，后CV4。沿用640×360、GT灯光方向、共用光强6.030241533058244。数据包含面积光、世界光、阴影，方向光近似的模型误差需检查。

新preset：`--render_mode photometric_lambertian --loss_preset lambertian_geometry --photometric_light_mode gt_directional`。原irgs_baseline、lambertian_gt及lambertian_default保留。

几何使用共享世界坐标translation和轴角rotation残差，不修改Stage1基础参数和MLP输入。默认xyz实际Adam LR=1e-6、rotation=1e-5；材质预热2k、旋转2k起、位置4k起，1k渐增，20k衰减到0.1倍。固定scale/opacity/MLP/光照，不增密裁剪。每步累计4个相机的平均梯度。位移上限为稳健包围盒对角线0.2%，旋转5°；这些为初始待检验值，不能当作已验证最优。

损失：原光度及材质项，加normal-depth一致性、软alpha轮廓、位置/旋转锚定、弱参考深度。使用Stage1预测作为锚定，无GT几何监督。先检查100步冒烟中的梯度、checkpoint重载与新旧渲染，再跑冻结对照、rotation-only、xyz+rotation及半LR。每个变体单独目录记录完整config/命令/日志/checkpoint，不覆盖来源和历史结果。

输出根目录：`output/1007-03-static-geometry-stage2`；冒烟：`output/smoke_test/1007-03-static-geometry`。服务器venus，环境lumimotion。最终README需回填实际source/model/iteration、渲染命令、定量指标、RGB/alpha/normal/albedo代表图及目检、视频和PASS/FAILED。未完成的项目明确标记，不以流程通过代替Normal<5°。

## 实际进展（未达成质量目标）

内部验证11帧：0001/0012/0023/0035/0046/0058/0069/0081/0092/0103/0115。已独立从Blender导出这些相机的GT几何并核验，没有将其用于训练。

第一轮p01/p02/p03均已完成新增3k：只旋转7.56512°、联合7.56049°、半LR7.57648°；Stage1基线7.60637°。该轮为缩短的筛选调度：500步材质预热、1000步位置起步、500步ramp、两相机累计、5000步LR调度周期，详细命令在各目录initial_command.json。

第二轮q01/q02/q03采用旋转LR5e-5、normal-depth权重0.2，比较方向光、近场点光和近场加0.15环境辐照度。新增5k检查时全前景MAE分别8.16452°、8.24411°、8.14760°，深度MAE增加；已主动早停，保留40k检查点、日志和early_stop.json。这轮不作为成功结果。

额外只读诊断：局部法线平滑8.49572°、局部PCA7.60502°、大平面投影8.16743°，均不采用，不能标作训练优化结果。

成像审计：源blend的Filmic / Medium High Contrast与旧训练sRGB变换不同。同一帧实际重渲染，Filmic相对数据PNG前景MAE约0.024，Standard约0.074。已实现独立Filmic preset，65³ log域LUT由OCIO生成，与Blender线性EXR到PNG转换均值误差约0.001；它是近似，不保证全部导出帧完全匹配。

第三轮r01/r02/r03正在运行：Filmic近场光、固定光强6.030241533058244、环境辐照度0.15，比较xyz LR1e-6、关闭normal-depth项、xyz LR5e-6。旋转LR3e-5，材质预热2k、位置3k起、1k ramp；目标新增12k，检查点37k/39k/41k/44k/47k。代码在每组initial_code中快照；所有命令、日志及内部验证CSV见输出README/status.json。不能把尚未生成的检查点当作已完成。

参考遮挡诊断已通过GPU验证：仅对Stage1预测点云构建BVH，四点近似面积光，缓存源几何可见度，不使用GT网格。shadow preset已启动s01_filmic_shadow（同r03设置加参考阴影），已完成前100步且损失有限，法线质量待验证；缓存几何假设只适用于小位移，HDR/替换方向光不复用捕获光照的阴影。

第三轮39k内部验证：r01=7.64752°、r02=7.65488°、r03=7.62509°；r02关闭normal-depth未见收益，已早停保留39k。r01/r03继续，s01补入空闲槽位。r系列及s01检查点由独立只读评测队列监控；实时状态见输出validation_queue.json及shadow_validation_queue.json。

## 后续筛选与失败记录

r01/r03到41k分别8.00653°/7.97352°，均早停；s01含参考阴影到44k为8.34068°，深度误差也增大，已早停保留全部产物。修正成像变换及加入阴影尚未改善几何目标。

t01/t02新增共享颜色材质对照（palette K=4/8）：只对Stage1颜色的对数线性色度做确定性聚类，不使用GT材质标签或GT albedo。39k分别7.59281°/7.55391°，K=8布料13.31178°，仍未达目标，继续观察41k。参数和代码快照在各initial_command.json及initial_code目录。

训练RGB独立鲁棒光度立体诊断（diag_ps_*）：仅94个优化相机、已知灯位、Stage1预测可见度与几何，逆Filmic后求解法线。三个先验强度的验证误差8.29119°/8.26512°/7.96835°，不采用为训练监督。GT仅在后续评测使用；诊断检查点禁止恢复训练。

u01_geometry_regularized加强法线—深度一致性到2.0、轮廓到1.0，光度权重0.1、参考深度0.01；xyz LR1e-5、rotation1e-4，位移信赖域0.5%、角度15°。这是检验几何协调性的对照，不是已验证最优；目标43k，37k/39k/41k/43k验证。

新增surfel前向着色preset待冒烟及筛选：每个高斯先按真实平面法线计算Lambertian辐照度，再光栅化累积。与旧屏幕空间着色共享完全相同的几何和评测口径。单帧GPU检查确认位置、旋转、材质梯度有限且非零，albedo/normal/depth/alpha输出与screen路径完全一致，训练质量待验证。

u01在37k验证GS法线7.52468°、Depth Normal9.59070°、深度MAE0.027271、覆盖率0.998367，比基线法线和深度法线均有小幅改善，尚不能证明长训有效。

已准备CV4独立11视角GT缓存（只离线评测使用），基线GS法线7.45087°、Depth Normal9.69680°、深度MAE0.026040、布料13.54970°。来源1006-01 Stage1 35k；尚未开展本轮CV4训练。固定CV4L Stage1形变时间为0的只读诊断7.64048°，没有优于原逐帧形变，因此不切换这项行为。

目前全量49项单元/回归测试通过（tests_v6_surfel.log）。目标仍未达成，原15测试集继续保留。

t01/t02在41k全前景法线8.10988°/8.01458°、深度约0.031、覆盖率约0.9964，均早停；虽然t02布料降到12.99487°，整体护栏失败，不能称成功。u01到39k为7.53612°，Depth Normal9.40022°、深度0.027118、覆盖率0.998746，继续到后续检查点。

新增正式训练CV4_u01_geometry_regularized：相同u01参数迁移1006-01，目标43k，CV4独立评测队列。新增v01_surfel_fullres_palette：CV4L原始1280×720训练、surfel前向着色、8色共享材质，几何一致性2、光度1、深度锚0.01，xyz LR1e-5 / rotation1e-4，位移0.5% / 旋转20°；1k预热、1.5k起位置、500 ramp、8k调度，检查37/39/41/43k。此为细节恢复的组合候选，不用于单因素因果归因；评测仍640×360，与基线一致。

surfel原始分辨率40步冒烟与从35020恢复到35040完全状态对比通过（geometry/material/light/optimizer/相机序列/RNG），见surfel_resume.json；原管线默认screen不变。

## 光强与面积光进一步审计

仅94个优化相机的线性RGB和Stage1法线估计，亮色簇在I=6.03024下反照率中位数约1.173（无GT材质输入），说明此光强和Filmic组合可能不足。旧EEVEE矩形光源码定义辐亮度为P/(4A)，而非直接复用旧sRGB标定标量；本项目渲染器显式除以π，因此对应参考距离8的标量为1800/(4×8²)=7.03125，同时需要发光面余弦和面积积分。源码依据：https://github.com/blender/blender/blob/blender-v3.6-release/source/blender/draw/engines/eevee/eevee_lights.c 及其shaders/lights_lib.glsl、ltc_lib.glsl。这是针对旧EEVEE的对齐近似，尚未证明与全部源PNG完全一致；尝试获取3.6官方目录被HTTP403阻止，未能做该版本二进制对照。

只从源blend灯对象读取120帧世界朝向和两条面内轴，增强lights JSON；没有读取表面网格、GT法线或GT材质来训练。原文件哈希保持不变。新area preset继承lit，使用四子区域中点积分、真实光源轴，以及已知功率推导固定光强7.03125；阴影射线也使用这些实际采样点。GPU有限梯度、原几何/材质通道保持一致检查通过；52项单元/回归通过，正式冒烟待槽位。

当前结果：CV4_u01 41k为7.32587°，Depth Normal9.15883°；v01原分辨率37k为7.39716°；w01受光区筛选37k为7.54628°且深度MAE0.028450，尚未改善到目标。CV4L u01主指标37/39/41k停滞，已早停保存41k。

## 面积光三组正式筛选

CV4_u01完成43k：GS法线7.31741°，Depth Normal9.13526°，深度0.0257743，覆盖率0.998797；相对其Stage1的7.45087°有小幅改善，仍未达5°。v01到39k为7.57869°，虽然布料12.90691°及深度改善，主指标比37k的7.39716°恶化，早停保存39k。w01到39k为8.12836°、深度0.029518，早停保存39k。

area 40步冒烟已完成，自动光强7.03125落盘calibration.json，损失和位移有限。x01_area_palette（640×360、8色共享材质）、x02_area_fullres（1280×720、8色）、x03_area_pergaussian（640×360、逐点材质）正式启动；其余几何/损失调度同w01，目标43k，检查37/39/41/43k。统一640×360内部验证，原15测试集未用来调参。source/model/参数/代码快照均在各initial_command.json及initial_code中；面积光metadata为lights_with_orientation_CV4L.json，checkpoint内固定光强已记录。

前向surfel着色跳过未使用的屏幕辐照度计算，避免面积光和HDR重复计算；GPU检查确认优化前后结果与梯度一致。仍只用真实高斯平面旋转输出GS Normal指标。

## 重要校正：以源blend保存版本5.2为准

前述旧EEVEE功率约定不适用于本数据，相关推断已撤回。只读检查bpy.data.version得到[5,2,44]，引擎名BLENDER_EEVEE不能用于判定旧实现。5.2源码shape_radiance_get明确使用P/(πA)：https://github.com/blender/blender/blob/blender-v5.2-release/source/blender/draw/engines/eevee/eevee_light.cc 。因此参考距离8、已知功率1800的固定标量为8.952465548919113。额外读取灯光normalize=True、exposure=0、diffuse_factor=1、世界尺度和场景light_threshold≈0.01，得到影响距离42.4264；按同版本着色器公式补入影响距离衰减。metadata为lights_blender5_full_CV4L.json，仅读取灯光数据。

x01/x02/x03在启动早期中止，只保留35000初始检查点及所有命令/代码/训练日志；没有非零几何训练检查点评测，不计入质量排名。legacy功率约定仅为旧实验可重载而保留，新area默认blender5。

独立合成测试（无数据集GT输入）：单位漫反射白平面、400功率矩形光、距离10，四点积分与Blender 5.2线性EXR中心亮度相对差约0.096%；加入默认影响距离衰减后约0.092%。世界背景线性0.05的增量实测0.049805，验证ambient irradiance=π×world radiance的量纲。文件在smoke_test/.../blender5_area_power[_v2]和blender5_area_world。初次忽略截断产生13.9%差异的失败比较也保留，后续已定位并修正。

正确版本的40步冒烟、GPU梯度、几何评测入口和独立albedo renderer一致性均通过；52项回归通过（tests_v10_area_cutoff.log）。y01_blender5_palette（640×360、8色）、y02_blender5_fullres（1280×720、8色）、y03_blender5_pergaussian（640×360、逐点材质）已启动，使用强度8.952465548919113和源世界光对应环境辐照度0.15983194568780254；其余几何/损失调度同原x系列。完整参数保存在blender5_experiment_configs.json及各initial_command.json。检查点37/39/41/43k，统一640×360内部验证，目标尚未达成。

## 从训练RGB初始化共享材质

仅按Stage1色度聚类的初始化把暗椅面与亮底盘混在一起，已在albedo图中确认。使用94优化视角的逆Filmic线性RGB、已知Blender5面积光和Stage1预测法线，鲁棒估计每点反照率；再以估计反照率的对数色度聚4类，弱受光/无观测点按空间与原色度近邻传播标签。未读取GT normal/depth/albedo或GT材质标签。57.4%点有至少4个受光观测，四组点数27647/7423/5469/2115；初始化线性颜色约[0.822,0.822,0.822]、[0.407,0.017,0.027]、[0.206,0.061,0.023]、[0.050,0.064,0.077]。这只是训练RGB估计，不是真值材质。

8色和4色色度初始化可视化在smoke_test/.../palette_init_gpu与palette_init4_gpu；已目检0002/0057，4色版本能分开椅面和底盘，并去除albedo中的大部分投影阴影。新`photometric_palette_init`输入标签和初始颜色，校验Stage1路径/迭代/两份文件SHA及尺寸；checkpoint保存完整标签/颜色，恢复不依赖初始化文件。53项测试及GPU重载/渲染通过。初始化输入和光照JSON从本轮z实验起也复制进initial_inputs并记录SHA。

y01在37k：GS7.28484°、Depth Normal9.56989°、深度0.027151、覆盖率0.997534；y02在37k：GS7.36172°、Depth Normal9.54755°、深度0.026773、覆盖率0.998861。y03逐点材质37k为7.44260°，为资源优先筛选早停，不能称其长训失败。z01_rgb_palette4已启动（与y01几何/光照/损失调度相同，唯一主要差异为4色训练RGB初始化），37/39/41/43k检查。原15测试视角仍未用于此次调参。

面积光修正后的独立光度立体法线诊断diag_ps5_*仍为8.26580°/8.26506°/7.99964°，不采用作监督。光照正确不代表源几何投影能提供准确的逐点跨视角对应。

## 训练RGB初始化后的几何强度对照

z01 37k GS法线7.27466°、Depth Normal9.57813°、深度0.026937、覆盖率0.997543、布料13.22238°。目前仅有小幅改善，仍不达标。y01/y02分别在39k回升到7.54450°/7.50309°，已早停保留39k。

新增z02_rgb_palette4_freer_xyz：相对z01把位置LR从1e-5调整到3e-5，最大位移从场景尺度0.5%放宽至1%，参考深度权重从0.01降到0.001。新增z03_rgb_palette4_weak_nd：相对z02仅把法线/深度一致性权重2降到0.2。均从1006-02 Stage1 35k开始，目标43k，37/39/41/43k内部验证，原15测试视角保留。三组并发符合单卡最多三训练限制。

Stage1高斯的光栅保守包围半径中位数约6–10像素（640×360）；这是包围界，不能直接解释为高斯标准差或证明表示能力不足。准备可选尺寸微调实验：两个面内轴使用log-scale残差，尺寸倍率限制在0.8–1.25，初始为零，保留alpha/深度约束及尺寸锚点。新增preset的质量尚未验证，暂未用于正式训练。

## 39k验证与尺寸实验排队

z01 39k GS7.08452°、布料12.81599°；z02 39k GS7.14413°、Depth Normal9.29631°、深度0.026624、布料12.78443°。保守位置更新的主指标较好，较大LR改善深度，仍需后续检查点。z03减弱法线/深度耦合的37k为7.16946°、布料12.97798°，相同步数好于前两组，继续观察。

z01 39k已生成11内部验证视角的RGB/alpha/normal/albedo静帧及视频，路径z01_rgb_palette4/visual_validation_39000；55帧视频ffprobe通过。0001/0058目检：材质去除大部分阴影污染，轮廓完整，RGB投影阴影过硬/错位，布料细法线仍不足。质量FAILED_NORMAL_TARGET，非最终候选。

由训练RGB四组材质和源点云重新推导的平面诊断diag_material_planes为8.37318°，拒绝使用。该诊断没有使用GT，平面仅作为一次失败假设保留。8个训练视角梯度审计显示，布料组旋转的几何项梯度中位范数约0.00204，光度项约0.00114；两者整体方向接近正交。它支持继续评估减弱几何项，但不证明是唯一瓶颈。

run_scale_queue.py等待三个正式训练中至少一个结束，顺序执行40步尺寸冒烟与20+20恢复一致性检查。通过后自动启动z04_rgb_palette4_scale：以z03为对照，仅增加有界尺寸优化（LR2e-4、1500步开启、倍率0.8–1.25）；目标43k，37/39/41/43k验证。所有步骤保持最多三训练。当前仅排队，不声称已完成。

## RGB材质初始化对照完成及后续实验

CV4L z01保守位置组43k：GS7.045689°、Depth Normal9.298025°、深度0.0265979、覆盖率0.997483、布料12.584125°，目前内部验证主指标最佳，但未达5°。z02较大位置组43k：GS7.112424°、Depth Normal9.162193°、深度0.0267733、覆盖率0.997411、布料12.485845°。z03弱一致性组43k：GS7.243424°、Depth Normal9.510401°、深度0.0279471、覆盖率0.997602、布料11.989532°；布料更好但主指标回退，三组均正常完成。

尺寸冒烟连续40步与20+20恢复，在原1e-6绝对容差下一个尺寸分量相差1.0692e-6，初次失败记录和scale_queue.log原样保留。独立连续40步重复运行的最大尺寸差为2.7021e-6，RNG、相机顺序、优化器状态匹配；据此仅log_scale采用rtol1e-5/atol3e-6，其余几何分量仍1e-6，恢复检查通过。正式z04_rgb_palette4_scale已手动启动；37k GS7.143915°、Depth Normal9.610476°、深度0.0268004、布料12.959517°。这只是首个检查点，未证明尺寸收益。

新增z05_rgb_palette4_fast_rotation：相对z03仅把旋转LR从1e-4改为5e-4，位置仍3e-5、位移上限1%、旋转上限20°，其余调度相同；37k GS7.342314°、Depth Normal9.635720°、布料12.223940°，全局不优于保守组，继续到39k检查。

CV4光照元数据仅读取灯光与世界背景，源blend SHA未改变；导出lights_blender5_full_CV4.json与CV4L逐项一致，环境辐照度0.1598319457。94训练RGB和各自Stage1估计反照率，纯色度4类不能分开灰椅面/底盘；加入0.2对数亮度拟合6类后，再在同一特征空间逐次合并最近的两组至4类。合并仅使用训练估计的颜色/计数，未使用GT材质标签。最终资产training_rgb_blender5_palette4_merged_CV4.npz，经SHA校验和0002/0057 albedo目检，椅面与底盘分开，红布阴影子类消除，底盘边缘仍有少量串色。失败/中间4类和6类资产、图片均保留。

CV4_z01_rgb_palette4从1006-01 Stage1 35k启动，采用CV4L z01的保守参数，唯一数据相关差异为各自source、灯光元数据和材质初始化；37k GS7.119534°、Depth Normal9.415518°、深度0.0259004、覆盖率0.997462、布料13.131958°。统一11内部验证，原15测试视角保留。

## 联合平面诊断及法线一致性容差

先前平面诊断仅旋转法线。新增diag_material_plane_xyz（只投影位置）和diag_material_plane_joint（位置与法线均投影），仅由Stage1和训练RGB分组推导，约25.2%点受影响，位移P95约1.04%、最大2.50%场景尺度。前者GS7.86715°，后者8.36002°，均失败，不用于初始化或监督，诊断checkpoint禁止续训。

离线GT审计在11内部验证的物体掩码内腐蚀2像素后，GT深度导出法线与Normal EXR：底盘3.58466°、布料3.61597°；GT角点插值法线与EXR分别0.08552°、0.71838°。差异包含平滑法线、像素采样、深度差分等贡献，不能据此把全部差异归因于某一来源。文件gt_normal_depth_agreement.json仅用于解释评测，训练没有读取这些目标。

新增lambertian_geometry_area_margin，默认允许3°法线/深度法线角度差，超出部分才计算一致性损失；原preset仍为0°，旧公式保留。56项测试与真实GPU损失/梯度检查通过。run_margin_queue.py已排队等待训练数少于3，先20步冒烟，再启动z06_rgb_palette4_normal_margin。z06以z01为匹配对照，仅改变该损失形式，目标43k、37/39/41/43k评测。此刻尚未训练，不能宣称效果。

## 最新41k检查及角度容差正式启动

CV4_z01 39/41k GS6.920386°/6.902447°，41k Depth Normal9.171524°、布料12.479961°。z04尺寸组39/41k GS7.079079°/7.048519°，41k Depth Normal9.316159°、布料11.850079°。尺寸组39k及较快旋转组39k的11视角四类PNG/55帧视频已生成，代表0001/0058目检及ffprobe完成；README/manifest质量均为FAILED_NORMAL_TARGET，细褶皱不足和阴影错位仍可见。

z05 39k GS7.335419°、深度0.0279620、布料11.471640°；37->39k主指标仅改善0.0069°，相比保守组较差，按资源优先早停保留39k及全部产物，不断言其长训必然失败。

角度容差20步冒烟已通过（normal_margin_smoke.json），配置落盘margin=3°、损失有限，源模型保持只读。z06_rgb_palette4_normal_margin正式启动，从1006-02 Stage1 35k、与z01相同保守位置参数，检查37/39/41/43k。当前并发为CV4_z01、z04、z06。未达到Normal MAE<5°，原15测试视角和最终mesh验收仍保留到候选确定后。


## 43k结果与表面细分对照

CV4_z01正常完成43k：GS6.900614°、Depth Normal9.158549°、深度0.0256189、覆盖率0.997313、布料12.391541°。CV4L尺寸组z04正常完成43k：GS7.026097°、Depth Normal9.289106°、深度0.0269781、覆盖率0.997551、布料11.729603°，略好于z01主指标。z06角度容差组43k：GS7.085072°、Depth Normal9.329102°、布料12.573852°，没有优于匹配的z01。仍未达到5°，不能据此宣称目标完成。

新增可选lambertian_geometry_area_refine，把每个冻结父高斯展开为4个面内子表面片，仅新分支的残差可优化。父Stage1文件、形变输入、材质标签及源可见度仍保持原N点；子位置/旋转/尺寸分别优化，四个子点共享父材质，不读取GT几何或材质标签。子尺寸为父尺寸的一半，局部偏移为两个轴上±sqrt(0.75)倍父尺寸，复制父opacity。保持中心、二阶矩及线性opacity积分，但不能保证实际alpha/RGB逐位相等。

细分初始化11内部验证：GS7.804270°、Depth Normal10.698714°、深度0.0264227、覆盖率0.999945，布料13.885711°；比原Stage1 GS7.606374°更差。因此只作为容量实验，不声称初始化已改善。0002/0057 RGB和alpha初始改变较小；0002目检轮廓完整，但腿/布边存在细碎伪影，需要持续检查。

59项单元/回归、旧分支GPU前向/梯度兼容、子位置/旋转/尺寸有限非零梯度、三个渲染入口几何一致、独立albedo零差检查通过。smoke_refine40与smoke_refine_resume正在执行连续40步及20+20恢复；通过后才启动正式对照。计划z07_refine_margin采用保守位置LR1e-5、位移上限0.5%、参考深度0.01；z08_refine_free_xyz仅改为LR3e-5、上限1%、深度0.001。两者旋转LR1e-4、上限20°，normal权重2、margin3°，尺寸LR2e-4、倍率0.8–1.25，warmup1000、位置/尺寸1500步开启、ramp500、accumulate2、目标43k及37/39/41/43k评测。完整参数在refine_formal_configs.json；此处仅为计划，尚未正式训练。


细分40步训练及20+20恢复完成。原1e-6严格比较失败保留在refine_smokes.log/refine_resume_initial_audit.json；独立40步重复的位置/轴角/log-scale最大差1.389e-5/3.789e-5/8.287e-5，均大于续训差4.737e-6/1.967e-5/5.829e-5。按独立重复范围，仅细分分支核验使用位置2e-5、轴角5e-5、log-scale1e-4绝对容差和rtol1e-5；位置Adam一阶矩1e-5，其余参数/动量仍1e-6。RNG、相机顺序、整数映射及缓冲严格一致，真实参数重载逐位一致。两代表帧续训渲染最大RGB差7.99e-6、alpha差6.20e-6、法线分量差4.48e-5、深度差9.54e-5，未出现实质性续训偏移，结果refine_resume.json。首次核验脚本CPU/GPU设备不匹配的失败日志也保留，修正后通过；不影响训练。

增加第三组z09_refine_scale_weak_nd，以此前z04尺寸组为匹配对照，只增加细分；采用位置3e-5、上限1%、深度0.001、normal权重0.2、margin0。三个正式细分训练将同时启动，保持单卡最多三训练。此时无正式细分训练质量结论。


## 细分正式运行与43k三类法线补测

z07/z08/z09已正式启动，初始checkpoint35000和完整命令/代码/输入SHA均保存，各运行README记录尚未完成的质量验收。当前三训练，约170616子点/训练，材质热身结束后按既定调度开始更新旋转，位置和尺寸仍按1500步开启。watch_refine_validation.py自动离线检查37/39/41/43k。

CV4_z01和CV4L z04的43k均补齐11内部验证视角RGB/alpha/normal/albedo PNG及55帧12fps视频，ffprobe通过。代表CV4 0001目检：轮廓完整、主要颜色接近参考，布料细节仍过平滑；albedo四种主要材质分开，边缘有少量串色。代表CV4L 0058目检：alpha主体完整，RGB阴影偏硬/错位、腿部有伪影，albedo大部分阴影去除但边缘串色仍有，normal细褶皱不足。两组README/manifest均FAILED_NORMAL_TARGET，不是最终选定模型。

z04 43k只从105训练相机的预测expected-Z融合TSDF（体素0.02、截断0.08），得到45387顶点/88816三角形，没有读取GT几何，完整命令mesh_train_43000.command.json、日志同名.log及manifest保存。在同11内部验证的GT Normal EXR上：GS7.026097°（覆盖0.997551）、Depth Normal9.289106°（覆盖0.955637）、Mesh8.611450°（覆盖0.980397）；三者共同有效像素GS6.460202°、Depth9.285221°、Mesh8.405630°（共同覆盖0.955310）。共同像素值不能取代主指标。结果validation_43000_mesh/metrics.json。

离线物体分项（像素加权，不等于帧等权主指标）：z04布料11.678845°、椅面7.205962°、底盘4.578250°；Stage1对应13.521265°/9.006384°/4.352746°。底盘误差略回退，几何与法线仍需同步改善，不能只报告旋转输出。所有GT对象标签只用于该离线诊断，未进入训练或初始化。


## 细分37k首轮评测

z07_refine_margin：GS7.077345°、Depth Normal10.162094°、深度0.0259516、覆盖率0.999923、布料13.083255°，实际高斯170616。

z08_refine_free_xyz：GS6.962952°、Depth Normal9.913083°、深度0.0258759、覆盖率0.999545、布料12.906612°，实际高斯170616。

z09_refine_scale_weak_nd：GS6.897257°、Depth Normal9.958921°、深度0.0262563、覆盖率0.999328、布料12.740020°，实际高斯170616。

z09比匹配z04同37k（GS7.143915°）改善约0.247°，也好于此前z04 43k GS7.026097°；但Depth Normal比z04 37k 9.610476°更差。容量收益尚不能视为完整几何收益，继续39/41/43k及四通道视觉检查。原15测试视角仍保留，目标<5°未达成。CV4_refine_candidate_config.json仅准备了各自光照/训练RGB材质的迁移参数，未启动，需先选择候选并空出训练槽位。

只读严格平面内部联合投影diag_interior_plane_joint仅影响2.96%点，足迹完全位于源推导平面凸包内部、3MAD深度及10°夹角筛选；起点z04 43k。GS6.947965°略好于原7.026097°，Depth Normal9.421945°回退，未采用作初始化/监督，诊断checkpoint禁止续训。阈值来自源统计、未使用GT筛选，GT只在独立离线评测读取。所有产物保留。


细分三组37k四通道各11静帧及55帧12fps视频已生成，ffprobe通过。代表z07 0001、z08 0115、z09 0058目检：整体轮廓保留、主要材质分开，腿部/椅面/布料边缘有细碎伪影，RGB阴影过硬/错位仍可见，normal细褶皱仍不足。各visual_validation_37000/README.md及manifest为FAILED_NORMAL_TARGET，训练仍继续，不能宣称最终视觉通过。


## 1007 用户再次确认的训练约束与邻域正则计划

用户明确要求继续，但禁止GT法线训练。Stage1、Stage2训练入口及dataset reader均不读取normal_exr；RGB、alpha及已知光照用于监督，geometry_normal比较当前预测法线和预测深度差分法线，reference depth来自冻结Stage1。GT Normal只在独立离线评测中读取，可用于内部验证选参，绝不进入loss、梯度、初始化或邻域图。原15测试视角继续保留，未用于选参。

细分三组41k内部验证：z07 GS6.504565°/Depth Normal9.268143°/深度0.026112981/覆盖0.998436；z08 GS6.535972°/Depth Normal9.079721°/深度0.026216284/覆盖0.998227；z09 GS6.473118°/Depth Normal9.257917°/深度0.026536106/覆盖0.998070。三组继续43k后再选CV4迁移参数，不以微小差异提前中止。此处只报告内部验证，尚未达到5°目标。

新增邻域一致性方案lambertian_geometry_area_coherent：继承细分分支，但仅用初始Stage1预测子片、source opacity/static掩码和训练RGB材质标签，建立最多6邻居的固定图。同标签、法线轴夹角小于45°、平均切平面偏离比例不超过0.35、距离不超过两端高斯半径3倍时才连边。约47.74%的图槽位有效；不从GT获取邻接、位置、法线、深度或材质。

邻域损失为平均双端法线与边位移的点积平方，除以固定参考边长平方，即 ((x_j-x_i)·(n_i+s_ij n_j)/2/d_ij)^2。s_ij由初始来源法线符号确定；球面可为零损失，因此不强制所有点共面。固定邻域保存在checkpoint，续训不得重建；训练时位置与旋转均获得梯度。

62项单元/回归通过，日志tests_v15_coherence.log；coherence_gpu/result.json检查旧分支前向逐位一致、反向数值一致，新增项位置/轴角梯度范数0.10114/0.000472，预测几何评测和独立albedo与主入口一致，checkpoint图逐位重载、源文件与参数保持冻结。这些是流程检查，不能宣称新正则已经改善法线。

run_coherence_smokes.py准备连续40步、20+20恢复及独立40步重复，并逐次检查GPU最多3个训练。若1e-6严格比较失败，保留记录，必须核对固定图/RNG/相机严格一致，并用独立CUDA重复及真实两帧渲染误差验证，不能仅调大容差。冒烟检查通过才可启动z10_coherent_005和z11_coherent_020；两组以z09为匹配对照，只改变coherence权重0.05/0.2，其余参数及94训练RGB相同。目标43k，37/39/41/43k内部验证。CV4迁移使用各自1006-01 Stage1、光照元数据和训练RGB材质资产，不共用CV4L标签。

离线depth_choice_offline_audit.json比较同像素expected/median预测深度导出法线：Stage1 beta=0/0.25/0.5/1为9.860/9.984/10.468/11.909°，z04为9.244/11.116/14.205/20.919°。median混合更差，未引入训练或改动评测深度定义；该GT审计只用于离线分析。


## 细分43k完成与CV4迁移

z07/z08/z09均正常完成43k。内部验证GS分别6.470417°/6.497089°/6.450415°，Depth Normal9.207049°/9.007948°/9.210445°，深度0.026140290/0.026180476/0.026509911，覆盖0.998355/0.998202/0.998112，布料11.721994°/11.449638°/11.026079°。以主GS指标选择z09参数迁移至CV4_z02_refine_scale_weak_nd；不使用原15测试选参。CV4输入是1006-01 Stage1及CV4自有light/palette。此为继续改善的对照，非质量验收通过。

z09 43k已生成visual_validation_43000四通道静帧/视频。0001全通道、0058 RGB/normal及0115 albedo目检：主体轮廓与四种主要材质完整，细分比37k少一些碎片，但腿/边缘伪影、偏硬阴影和布料细法线不足仍在；结果FAILED_NORMAL_TARGET。完整命令在visual_validation_43000/command.json，日志同目录父级同名.log。

邻域正则40连续/20+20恢复/40独立重复已完成，首次1e-6严格检查失败，所有结果保留。恢复XYZ/轴角/log-scale最大差为3.142e-6/9.727e-6/8.006e-6，第一次连续重复0.773e-6/0.933e-6/1.314e-6；单次新重复不足以解释恢复误差，暂未启动coherent正式训练。下一步再做两个预先指定独立重复（其中一组保存20步中间状态），核对中间相机/RNG、固定图与真实两帧渲染；既有细分CUDA重复范围作为明确保留的数值基线，不能盲目扩大绝对阈值。


## 邻域正则实际恢复核验与有限精度正式训练门槛

额外两次独立重复已全部完成。新分支连续重复最大XYZ/轴角/log-scale差约0.773e-6/1.610e-6/3.531e-6，仍未覆盖原恢复3.142e-6/9.727e-6/8.006e-6。因此最初1e-6严格检查和“恢复差不超过该分支连续重复最大值”检查均记为失败，日志coherence_smokes.log、coherence_resume_verified_v1.log及两个审计JSON保留，不能声称被新分支重复范围充分解释。

另外在独立新输出coherence_restore_probe实际实例化PhotometricTrainer恢复20步checkpoint，位置/旋转/尺寸、全部固定邻域缓冲、材质/光照、全部Adam张量和组参数、相机栈及Python/NumPy/torch/CUDA RNG均逐位相等，冻结源指纹通过，未执行训练；该probe只作诊断，不得续训。说明保存/载入未丢状态，但不证明跨进程重新训练逐位一致。

check_coherence_bounded_resume.py作为独立有限精度核验保留原失败条件，不修改原检查器或绝对上限。恢复差小于既有同rasterizer细分独立CUDA重复实测范围1.389e-5/3.789e-5/8.287e-5，且小于预先记录XYZ2e-5/轴角5e-5/log-scale1e-4界限；其余参数与优化器规则未扩大。5个结果在13视角重新渲染，恢复最大RGB1.508e-5/法线分量2.701e-4/深度6.866e-5/alpha7.570e-5，均小于原先图像差上限1e-3；图、RNG和相机精确一致。该有限精度检查通过(coherence_resume_bounded.json)，strict_1e6_passed=false、own_repeat_range_covers_resume=false明确记录。

据实际重载精确、预先存在界限及13视角实际误差，接受有限精度恢复并启动z10_coherent_005/z11_coherent_020正式对照；不能将有限精度通过写成严格或新重复范围支持的通过。两组权重0.05/0.2；同z09的XYZ LR3e-5、位移上限1%场景尺度、旋转LR1e-4和20°上限、尺寸LR2e-4及倍率0.8–1.25，geometry_normal=0.2，normal_margin=0，其余配置保持。CV4_z02正运行，与两组共最多3训练。run_coherence_formal.py记录coherence_training_gate.json和所有证据SHA，完整argv/代码/输入快照在各实验initial_*。

z09 43k仅融合105训练相机的模型预测expected-Z，TSDF0.02/截断0.08得到45814顶点/89669三角形，未读取GT几何。新mesh_train_43000命令/日志/代码SHA和manifest保留；内部验证GS6.450415°(覆盖0.998112)、Depth Normal9.210445°(覆盖0.956241)、Mesh8.495420°(覆盖0.982757)，三者分开报告，仍未达到质量目标。原15测试保持未选参。


## 只读静态源时间变化审计

CV4L：source opacity>0.2共24842点，其中feature>=0.5有217点(0.874%)；全部高opacity点随94优化帧时间的法线RMS1.349800°，位移RMS占场景尺度0.007413。source-static组最大位移变化的点P95为1.576e-07、法线变化P95为5.317e-05°，但最大值可达11.870%场景尺度/25.510°，极少数离群点不可忽略。

CV4：source opacity>0.2共25186点，其中feature>=0.5有261点(1.036%)；全部高opacity点随94优化帧时间的法线RMS1.892305°，位移RMS占场景尺度0.011189。source-static组最大位移变化的点P95为1.114e-06、法线变化P95为3.340e-04°，但最大值可达20.137%场景尺度/37.400°，极少数离群点不可忽略。

audit_static_source_time.py只对94优化帧的冻结Stage1预测位置和世界法线做相对首帧统计，不读取GT法线或GT几何、不修改源。结果static_source_time_audit_CV4L.json/CV4.json及各自日志保存。总体P95稳定，点统计也不等于像素误差贡献，因此不能认定少量动态点已解释法线瓶颈。下一步只读检查这些点的实际alpha/法线像素贡献，再考虑是否做独立静态约束preset；当前不改正在运行的三组配置。


## 变化点像素贡献、邻域41k与密集细分计划

上一目标回合为progress：完成邻域代码/检查、CV4迁移和两组coherent正式训练及静态源审计；当前继续实际实验，目标仍为两数据集原15测试GS Normal MAE<5°。

variable_source_coverage_CV4L/CV4仅根据94训练Stage1预测的binary feature、时间位移>0.5%场景尺度或法线变化>5°定义变化点，不读取GT法线。11内部验证full与移除变化点的反事实渲染：CV4L402点移除后平均预测法线变化0.239382°，>5°变化占0.788%前景，预测coverage0.998096→0.997844，RGB L1 0.023540→0.023870；CV4608点对应0.353870°、1.282%、0.997821→0.997489、0.030458→0.030615。此处法线变化是预测间差异而非GT误差。点比例小且RGB略差，不作为主要瓶颈，不启动全局canonical/移除训练；来源、各点指标npz、完整命令、四通道PNG和统计全部保存，阴影固定为完整源预测，须注意反事实的限制。

CV4_z02已正常43k，GS6.978188°/Depth Normal9.364767°/深度0.025766146/覆盖0.997862，布料11.513361°。主指标较39k最佳6.854905°回退，不能将细分迁移声称为CV4正式质量改善。coherent 0.05/0.2的41k GS6.448505°/6.420456°，Depth Normal9.187146°/9.197221°，布料11.304697°/11.338005°；略好于同期z09主指标，但尚未达5°，继续43k观察。

refinement_footprint_and_graph_audit.json只读核对：coherent图89.83%有效边跨父高斯，非仅同父约束；三优化视角各训练RGB材质组的可见frustum radii中位约3–6像素、P95约6–17像素。radii代表支持足迹，含遮挡点，不能等价于实际像素贡献；但结合4子片使CV4L GS7.026→6.450的容量收益，支持测试更细表面采样作为独立容量假设，不保证效果。

新增lambertian_geometry_area_dense，源N保持冻结，使用4×4=16个子片，42654→682464；材质/MLP仍N。geometry_subdivision_side默认2保持已有4子片。等距轴为(2i-(s-1))*sqrt(3)/s，子尺度1/s，保持源中心、协方差及线性opacity积分，非实际alpha/RGB等价。GPU682464子片前向约0.106秒（并发条件下，仅0002单帧，非完整训练速度），位置/旋转/尺寸光度与几何梯度有限非零，GeometryRun及独立albedo/真实重载通过，旧4子片前向逐位一致、反向数值一致。64测试通过(tests_v16_dense.log)。dense_gpu为诊断checkpoint，不得续训。

run_dense_smokes.py计划连续40、20+20恢复和独立40重复，最多3训练；恢复检查实验前固定XYZ2e-5/轴角5e-5/log-scale1e-4，其他参数1e-6，XYZ Adam一阶矩1e-5，13视角实际渲染最大差<1e-3；严格1e-6及新重复范围覆盖情况独立记录，不盲目扩大界限。通过后才启动z12_dense_scale_weak_nd：以z09为控制，仅改变密度与对应preset，coherence=0，XYZ3e-5/旋转1e-4/scale2e-4，1500步位置/尺寸开启、1000步旋转开启、500步ramp、原信赖域和弱normal/depth权重保留。目标43k，37/39/41/43k内部验证，原15测试保留。此刻只有GPU/单元通过，训练恢复与质量目标尚未验证。


## 邻域43k完成、密集初始化与恢复实测

coherent 0.05/0.2均正常完成43k：GS6.417136°/6.392736°，Depth Normal9.135025°/9.139947°，覆盖0.998023/0.997951，布料11.146844°/11.211632°。最佳主GS为0.2组，但相对z09的6.450415°改善仅0.05768°，布料比z09的11.026079°略差，不能认为主要瓶颈已解决。原15测试仍未用于选择，目标未达成。

Dense GPU初始11视角GS9.295075°、Depth Normal12.325849°、覆盖1.0，布料15.432772°；明显差于4子片初始化7.804270°，证明矩/积分匹配不能保证真实渲染或法线初值不退化。只作为容量假设继续训练，不能声称密度增加已有质量收益。

Dense连续40、20+20恢复及独立40重复完成。1e-6严格比较失败保留dense_resume_initial_audit.json。恢复XYZ/轴角/log-scale差1.177511e-5/1.646666e-5/5.965907e-6，独立重复1.177511e-5/1.646663e-5/5.930604e-6；恢复轴角/scale略超过单次重复，own_repeat_range_covers_resume=false如实记录，未以此宣称重复充分解释。所有差异处于实验前既定2e-5/5e-5/1e-4上限，13视角恢复最大RGB2.779365e-4/法线分量7.447302e-4/深度3.147125e-5/alpha6.991625e-5均<1e-3，固定子片映射/缓冲/相机/RNG严格相同；有限精度检查PASS，未扩大上限。

独立dense_restore_probe实际PhotometricTrainer恢复参数、材质/光照、全部Adam张量与组参数、相机栈及全部RNG逐位相等，源冻结；只诊断，不得续训。基于该实测及GPU回归通过，启动z12_dense_scale_weak_nd，完整门槛dense_training_gate.json、初始代码与输入SHA以及argv均保存，源1006-02 Stage1未改；37/39/41/43k队列只作独立验证，不将GT法线用于loss、初始化或图。


## CV4两组受控约束对照

CV4_z02弱normal/depth及较自由XYZ在43k主指标回退，39k仍为最佳6.854905°，不能把CV4L配置直接当作CV4最终选择。安排CV4_z03_refine_margin：迁移已经完成GPU/恢复核验的z07保守参数束，XYZ1e-5、位移0.5%、normal2、depth0.01、margin3°，四子片与其余光照/材质设置相同。该组测试整个保守参数束，不声称能分离各超参数因果效应。

第二组CV4_z04_refine_coherent_020以CV4_z02为匹配对照，仅开启源预测邻域一致性权重0.2。保持XYZ3e-5、位移1%、normal0.2、depth0.001和margin0；有界scale仍2e-4、倍率0.8–1.25。该源图来自CV4自己的Stage1和训练RGB标签，绝不用CV4L标签或GT法线。两组目标43k，37/39/41/43k内部验证；先用自己的43140源点完成40步实际冒烟和有限梯度/残差检查，已有对应分支恢复机制已在先前独立检查验证。程序run_cv4_controls.py逐次核对<=3训练，并且等两组实际出现后启动CV4验证队列，和CV4L密集组共最多3正式训练。

已核对全部最近CV4L z系列本来就是surfel着色，不能把“从平均法线改成逐surfel Lambertian”当成未做的新方案。密集试验继承当前surfel光照实现，唯一容量变化为16子片；不重复已有分支。


## 当前三组实际启动、最佳可视化及固定图时间异常

run_cv4_controls.py两个各自40步冒烟正常完成并有有限非零位置/旋转残差、coherent图ready；两正式CV4对照已启动，与z12密集组共3训练。密集37k内部验证GS7.588852°/Depth Normal11.093141°/覆盖0.999996，较其初始化9.295075°改善，但仍明显弱于4片同期z09的6.897257°，不可预先宣称容量效果好。观察39k再判断是否继续，不掩盖深度法线退化。

最佳z11_coherent_02043k已补齐visual_validation_43000的11内部验证四通道PNG和55帧12fpsMP4，ffprobe通过，0001 overview/0058 normal/0115 albedo已目检。RGB主体接近参考但阴影偏硬/腿部伪影；alpha主体完整/边缘碎片；normal细褶皱不足；albedo主要材质分开/少量边缘串色，FAILED_NORMAL_TARGET。完整渲染argv/源模型/iteration/输出/统计和目检见该目录command.json、manifest.json、README.md及父级同名.log。预测mesh仍只已有z09，不能把它冒充z11 Mesh成绩。

cv4_coherence_temporal_edge_audit.json只读分解CV4_z04初始化固定图：变化点(仍仅由94训练Stage1预测定义)涉及0.6753%有效边，在0002占0.75%loss，但0105占78.63%loss；总coherence0.032177→0.150118，而不涉及这些点的稳定边平均loss约0.03215–0.03230。少量时间变化点的像素影响小，不代表对归一化图正则影响小；图在单帧初始化、源MLP轻微/极少数大形变可能造成跨帧约束矛盾。

下一可执行动作：继续监测当前三组，不改已有配置或覆盖checkpoint；若C4邻域回退或梯度被异常边支配，用新的可选preset及初始化掩码排除训练源预测时间变化点的图边，或独立鲁棒损失对照。掩码只可由94优化帧Stage1预测建立，必须保存source/iteration/PLY+MLP SHA、验证点数/布尔dtype，图完整存checkpoint，恢复不依赖外部资产；禁止使用GT法线/GT几何或11验证的像素错误挑点。当前该筛选尚未实现，不能写成已修复。


## 源时间稳定图对照与GT法线约束再次确认

用户再次要求GT法线不能用于训练。核对后继续执行：GT法线只进入独立离线内部验证/最终评测，不进入训练损失、初始化、图或掩码。训练的normal一致性使用模型预测surf_depth；source参考深度/法线由冻结Stage1预测。11内部验证的GT指标用于选择候选，不宣称选参完全不使用GT；原15测试仍保留最终验收。

目前CV4L最佳内部GS为z11 43k的6.392736°，CV4最佳为z02 39k的6.854905°，仍没有低于5°的结果。密集16片37k为7.588852°/Depth11.093141°，明显弱于4片控制，不把更多采样当成有效改进。CV4保守束37k为GS7.012445°/Depth9.934759°；原邻域0.2对照37k为GS6.857466°/Depth9.805431°，只比无邻域同迭代6.872629°略好。

新的稳定图仅修正固定邻域跨时间失效问题，质量收益须正式对照验证。计划CV4_z05_static_coherent_020匹配CV4_z04，z13_static_coherent_020匹配CV4L z11；唯一算法变化是训练来源稳定掩码，preset名随之变化。完整配置static_coherence_formal_configs.json：XYZ LR3e-5、旋转1e-4、尺寸2e-4；旋转warmup1000，位置/尺寸1500开始，ramp500、schedule8000，累积2；位置上限extent1%，旋转20°、尺寸倍率1.25，normal0.2/depth0.001/margin0/coherence0.2。保持4子片、Blender5面积光、各自训练RGB palette与来源，不以GT法线初始化。检查点37/39/41/43k分别独立离线验证。

完整准备命令：
```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.prepare_stage2_static_mask --model_path output/1007-03-static-geometry-stage2/CV4_z02_refine_scale_weak_nd/model_mlp --iteration 35000 --output output/1007-03-static-geometry-stage2/source_static_mask_CV4
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.prepare_stage2_static_mask --model_path output/1007-03-static-geometry-stage2/z09_refine_scale_weak_nd/model_mlp --iteration 35000 --output output/1007-03-static-geometry-stage2/source_static_mask_CV4L
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m output.1007-03-static-geometry-stage2.run_static_coherence_experiments
```
日志分别source_static_mask_CV4.log、source_static_mask_CV4L.log、static_coherence_experiments.log。run_static_coherence_experiments每次训练前核对/proc真实scripts.train_stage2数量，最多3；当前先排队做冒烟，再按恢复证据门槛启动两配对正式对照。不得以GPU预检或67单元测试通过替代正式法线质量验收。程序状态与FAIL保留见continuation/status及独立冒烟报告；目标仍PENDING。


密集16子片39k离线内部验证更新：GS6.506686°，同迭代4片z09为6.560072°；Depth9.987192°差于4片9.387115°，coverage0.998831，布料11.537195°。37k差的结果不能直接外推后续，保留密集训练至后续检查点观察；尚未达到5°。新稳定图冒烟仍按三槽位限制排队，不为启动新实验而忽略已有改善。


## 后续完成结果、实际恢复失败及LR调控

三旧正式训练均正常完成43k。CV4_z03保守设置43k GS6.576723°/Depth9.084047°/depth0.024807/coverage0.998355/cloth11.678212°成为CV4内部最佳；其39k四通道11静帧+55帧12fps视频已目检/ffprobe通过，质量FAILED_NORMAL_TARGET。CV4_z04邻域39k最好6.723726°，43k回升到6.808217°（Depth9.200661°），不能只报告该组较好迭代。密集z12 43k GS6.364189°成为CV4L内部最佳，但Depth9.737270°比旧4片coherent 9.139947°差，coverage0.998618，目标仍未达5°。

只读审计：reference_shading_audit_CV4以94训练Stage1预测比较方向光筛选与面积光筛选，差异约0.405%前景/0.643%原选择，不据此新增正式分支。static_coherence_gradient_audit七训练帧仅邻域项的加权几何梯度最大CV4旧0.06943/筛选0.02040，CV4L旧0.02743/筛选0.02035，未达到clip阈值1，不能声称已证实全局clip。normal_concentration_audit不读取GT法线：alpha>0.99内部验证预测中，低集中度<0.95比例约10.5%/11.5%/14.8%（CV4保守/L4片coherent/L16片），其RGB残差约高集中度的2–3倍。提示混合方向与渲染残差关联，尚不能证明因果；后续可考虑有界opacity残差，不能用GT误差定义点掩码。

首次静态图40步/恢复失败及加入源预测缓存后的失败完整保留。真实Trainer逐位恢复通过，缓存自身逐位恢复也通过；但旋转轨迹最大差约5.56e-5超预设5e-5，0046前景normal分量最大差0.001098超1e-3。正式训练未启动，不更改阈值或伪报恢复通过。将rotation LR由1e-4调到8e-5，并把正式调度由8k延为10k、目标43k延至45k，保留20°上限，补足累计更新；XYZ/scale及其上限保持。重新40+20/20+repeat+CV4L40预检，所有容差同原声明。

计划成对配置static_coherence_formal_configs_rotation8e5.json：CV4_z05_static_coherent_020对CV4_z06_coherent_020_rotation8e5，z13_static_coherent_020对z14_coherent_020_rotation8e5；每对LR/长度/缓存/来源/训练RGB palette相同，只有源时间稳定图筛选不同。每卡最多三训练，第四个按空槽排队。全部GPU/实际Trainer/固定界限恢复门槛通过前不启动；原1e-4方案记录不覆盖。

完整新入口：
```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m output.1007-03-static-geometry-stage2.run_static_coherence_rotation8e5
```
日志static_coherence_rotation8e5.log，预声明上限位于output/smoke_test/1007-03-static-geometry/static_coherence_rotation8e5_predeclared_limits.json；实际状态须核对/proc及初始日志。原15测试仍未用于选参，GT法线仅独立离线评测。


8e-5调控预检实际通过：XYZ恢复最大6.5108e-7、axis-angle8.4980e-7、log-scale2.4786e-6，均小于原2e-5/5e-5/1e-4；13帧最大RGB7.0367e-4、normal8.1116e-4、depth6.1035e-5、alpha7.4565e-5均<1e-3。严格1e-6仍FAILED（log-scale等），自身重复包络仍false（log-scale轻微超出），不误报严格恢复。实际Trainer全部参数/Adam/图/掩码/源cache/RNG/相机逐位重载，外部mask不可用也通过。static_coherence_training_gate_rotation8e5.json记录证据SHA后启动C4 z05/z06及L z13三个正式训练；L z14配对无筛选控制等待空槽。完整argv/代码/输入SHA位于各initial记录，45k独立评测计划已写。GT法线不进入训练和初始化。

密集z12最佳43k四通道11静帧+55帧12fps视频已补齐、目检及ffprobe通过；RGB主体接近参考，阴影偏硬/腿部伪影；alpha主体完整/边缘细碎；normal细褶皱不足、底盘轻微斑块；albedo四色分开/少量串色。visual_validation_43000/README.md及manifest完整记录FAILED_NORMAL_TARGET；该新最佳尚无Mesh成绩，旧z09 Mesh8.495不归给它。


## 有界opacity衰减的单因素试验

先前只读审计在完整前景alpha>0.99区域发现，约10–15%像素的预测法线混合集中度低于0.95，RGB残差约为高集中度区域的2–3倍；仅为关联，不证明它是主要误差来源。审计只用冻结/当前模型预测与观察RGB/alpha，没有使用GT法线筛点或逐像素目标。

新增geometry_optimize_opacity默认False，以及geometry_opacity_lr=1e-4、geometry_opacity_start=4000、geometry_max_opacity_ratio=2、lambda_geometry_opacity_anchor=0.001。新preset lambertian_geometry_area_opacity继承refine并开启该项，显式CLI优先。每个实际子片保存log_opacity∈[-ln2,0]；alpha=alpha_source*exp(log_opacity)，初值0精确保留source，包括接近1的alpha仍有梯度。关闭时直接返回原张量，不增加旧checkpoint状态、优化器组、loss项或浮点运算。project裁剪信赖域并清除越界Adam一阶/二阶矩。共享prepare使主Lambertian、GeometryRun及独立albedo一致应用残差。新增可选锚定项与实际衰减比/下限触达率诊断，opacity独立学习率与延迟调度。

4个新单元测试核对关闭状态、零初值、饱和来源梯度、只衰减界限、Adam动量及独立子片重载，总72项在tests_v20_opacity.log通过。tests_v19_opacity.log中的一次CPU夹具失败是preset启用4片后仍传父opacity，修正测试展开后通过；失败日志保留，不包装为训练数值失败。真实GPU的CV4四片和CV4L十六片检查分别保存在output/smoke_test/1007-03-static-geometry/opacity_gpu/result.json和opacity_dense_gpu/result.json：旧前向与零初值逐位一致、旧反向数值一致，opacity梯度有限非零、评测/主渲染一致、albedo零差、checkpoint重载准确、Stage1源未变。两个诊断checkpoint禁止续训。

正式配置在opacity_formal_configs.json：CV4_z07_bounded_opacity只相对CV4_z03_refine_margin增加opacity自由度；CV4L z15_dense_bounded_opacity只相对z12_dense_scale_weak_nd增加opacity自由度。两者保持各自4/16子片、来源35k、几何/法线LR与损失权重、8000步调度/43k目标、种子与94优化视角；opacity开始步显式1500。当前三个稳定图对照与排队L无筛选控制先完成，run_opacity_experiments.py后台等待其终态后执行40步、20+20恢复、独立40步重复及CV4L40步。实际恢复未执行，不能预报PASS。

opacity_predeclared_limits.json提前记录XYZ2e-5、axis-angle5e-5、log-scale1e-4、log-opacity1e-5，XYZ Adam一阶矩1e-5，其他Adam atol1e-6/rtol1e-5以及13训练/内部验证视角四通道最大分量差1e-3；严格1e-6和独立重复包络单独报告。另做实际PhotometricTrainer全部参数/Adam/RNG/相机逐位恢复probe。所有检查通过才写opacity_training_gate.json并启动双正式实验；失败日志、checkpoint与审计必须保留，不扩大上限来报通过。

GT法线不用于训练、初始化、筛点、掩码或权重。normal项仍由预测surf_depth和预测法线计算，参考深度仍为冻结Stage1预测；GT仅独立离线评测与内部验证选参，原15测试继续保留最终评价。opacity会同时报告GS/DN误差、观察RGB、alpha及前景覆盖，不能只报丢失像素后的误差降低。

稳定图新37k内部对照：CV4筛选GS 6.867451°/DN 9.801259°，同LR同长度无筛选GS 6.880526°/DN 9.805230°，覆盖率两者0.999553609。当前单因素GS改善仅0.013075°，尚不足证明最终收益，继续到45k；CV4L筛选GS 6.921509°/DN 10.003311°，无筛选配对尚排队。全数未达5°，不是原15测试成绩。

CV4当前最佳保守43k的visual_validation_43000已补齐，11内部视角四通道PNG、55帧12fps640×360视频及完整执行argv/日志齐备。已目检0001_overview.png、normal/0058.png、albedo/0115.png：主体RGB/alpha稳定，布料法线仍平滑与局部斑块，albedo主要四色分开、细边少量串色。GS6.576723°/DN9.084047°，最终FAILED_NORMAL_TARGET；不以图像/程序通过代替指标。


## 已微调几何上的训练RGB光度立体诊断及弱先验

此前Stage1投影上的独立光度立体结果约8°，未采用。新diagnose_refined_photometric_normals.py在当前已训练43k几何上只采集94优化视角，排除11内部验证和15测试，使用观察RGB/alpha、已知Blender5面积光及world、模型预测可见度/深度/颜色。通过逆Filmic得到线性亮度，拒绝预测alpha≤0.99、参考可见度≤0.98、法线混合集中度≤0.98、材质色度差≥0.1、深度偏离≥0.06、阴影/掠射/高亮投影。以当前预测法线和学习的RGB调色板为初值，鲁棒岭回归估计ρn，先验0.01，至少10观测、条件数<30、相对残差<0.15及预测方向变化<20°。修正幅度限制5°，并保持源20°旋转信赖域；XYZ与尺度固定，旋转会改变表面片，因此预测depth有小幅变化。全过程没有GT法线、GT深度、GT材质输入；GT只用于随后的独立验证。

CV4L 16片仅15.0176%点符合可靠观测，符合者观测中位37、条件数10.84、相对亮度残差0.00807。基线GS6.364189°，半幅修正6.127496°、全幅5.994869°，覆盖均0.998618315；全幅DN9.714561°、布料10.941626°，布料没有明确改善。主要改善来自白色/灰色面：训练RGB分组的可靠比例白色18.70%、红色10.25%、棕色1.08%、灰色19.61%，不使用GT材质标签筛点。CV4相同方法可靠17.4183%、相对残差中位0.04996，半幅GS6.685996°/全幅6.971571°均差于保守6.576723°，因此不迁移该先验到CV4。具体refined_photometric_comparison.json。

上述checkpoint均diagnostic标记，原优化器与新旋转不一致，禁止直接续训或报告为训练结果。CV4L全幅诊断已输出11内部视角四类PNG、55帧12fps视频，完整执行argv/日志与source/model/iteration在visual_validation_43000，已目检0001总览、0058法线及0115 albedo：底盘斑块稍减弱，布料仍平滑；颜色/alpha主体接近观察值，细边局部瑕疵。结论FAILED_NORMAL_TARGET_DIAGNOSTIC_ONLY，仍未到5°，不是原15测试。

新增可选geometry_initial_stage2_model/iteration：首次初始化仅接受相同Stage1源SHA与指纹、相同几何/材质/94视角划分的真实训练检查点，拒绝diagnostic状态；复制几何/材质参数，已知灯光须精确一致，使用新Adam，不声称继承原训练轨迹。保持Stage1来源35k只读，completed与独立geometry_schedule_origin从初始化43k开始，新训练43→46k调度不会错误地沿35k计数。来源checkpoint完整路径/SHA/iteration与新优化器约定写stage2_initialization.json。旧默认关闭，不改变旧checkpoint字段、参数/优化器或调度；新初始化分支保存独立调度起点，恢复必须匹配。

新增lambda_geometry_rgb_normal_prior默认0、geometry_rgb_normal_prior_path默认空、新preset lambertian_geometry_area_rgb_prior默认权重0.2。prepare_refined_rgb_normal_prior.py从训练RGB诊断的预测旋转提取source方向的世界法线，仅可靠点参与；不是GT法线。资产绑定真实已训练z12 43k初始化SHA、原Stage1文件SHA、94优化帧、RGB生产脚本SHA与无GT协议。load_rgb_normal_prior拒绝错误shape/dtype/非单位/空mask/不同初始化或划分/任何GT标志，allow_pickle=False。register_buffer保存目标法线与bool有效点，自包含恢复，无外部先验读取。关闭时None不进入旧state/loss/renderer返回字段。开启时主renderer返回未相机翻面的raw世界法线，geometry_losses加平均1−cos弱约束，位置仍由观察光度、预测深度一致性和其他几何项小LR优化。法线输出和GS评价定义不变。

76项单元测试通过tests_v21_rgb_prior.log，新增无GT协议/源SHA/训练划分/单位法线拒绝、默认关闭、buffer独立重载及真实训练初始化拒绝诊断和不匹配来源。rgb_prior_gpu实际PhotometricTrainer复制所有43k几何/材质参数逐位相等，空新优化器、43k调度起点，初始四通道等原模型逐位；先验旋转梯度非零有限，GeometryRun/albedo一致，源冻结。rgb_prior_legacy_gpu对CV4四片及CV4L十六片最新关闭分支与正式initial_code核对前向/loss项逐位、反向数值与状态键一致，通过。

run_rgb_prior_experiments.py后台等待当前三个稳定图对照结束，并优先让已排队L无筛选控制启动，之后使用剩余槽位，实际训练最多3。执行40步、20+20恢复、独立40重复及关闭先验warm control40；固定XYZ2e-5/axis-angle5e-5/log-scale1e-4、XYZ Adam一阶矩1e-5、其他Adam atol1e-6/rtol1e-5、13视角四通道分量最大差1e-3，严格1e-6与自身重复包络另报。实际Trainer重载probe在先验资产和Stage2初始化路径都不存在时核对全部参数/Adam/RNG/相机/先验buffers/调度起点逐位恢复。此刻尚未执行，不能预报恢复PASS。

通过恢复门槛才启动z16_dense_rgb_normal_prior与z17_dense_warm_control：同一真实z12 43k初始化、16片、94RGB、XYZ LR1e-6、normal8e-5、scale5e-5、albedo1e-4、3000步独立调度、44/45/46k保存；唯一变量RGB先验0.2/0。正式配置rgb_prior_formal_configs.json、日志rgb_prior_experiments.log；GT法线不进入训练或初始化。流程通过与只读改善不代表目标达成，仍须真实训练和双数据集原15测试GS<5°及三类法线/四通道/HDR验收。


RGB弱先验恢复已经实际完成：40连续、20+20、独立40重复及关闭先验warm control40均正常结束；固定参数界限、buffers/RNG/相机精确、13视角渲染通过。严格1e-6恢复比较本轮通过；自身重复包络（含1e-6数值底限）True。XYZ/axis/log-scale恢复差 9.779e-08/6.130e-07/4.168e-07，13视角最大RGB/normal/depth/alpha差 2.503e-06/3.754e-05/2.575e-05/2.980e-06。rgb_prior_restore_probe在外部RGB资产及初始化Stage2目录均不存在时，实际Trainer全部参数/Adam/RNG/相机/目标buffers/43k调度起点逐位重载通过，source35k不变；probe诊断标记禁止续训。根据实际证据SHA门槛，z16_dense_rgb_normal_prior与z17_dense_warm_control已经启动；尚无44k正式成绩，不把5.994869°只读诊断当作训练结果。

前三组稳定图对照45k正常结束：C4筛选/无筛选最终GS6.827628°/6.847074°，各最佳在39k为6.706789°/6.727035°，小幅单因素改善但均差于保守43k6.576723°；L筛选45k GS6.384211°/DN9.083381°，GS仍略差于密集43k6.364189°，DN较好。L无筛选配对z14已启动；完整三组记录static_coherence_completed_three.json。所有质量仍未达5°，原15测试继续保留。opacity监督器仍等待L配对终态与最多3训练槽位，未预报恢复通过。

正式44k已评测：z16 RGB弱先验GS6.303238°/DN9.724911°，z17同配置无先验GS6.328830°/DN9.726700°，两者coverage0.998614063。布料10.779045°/10.801840°；with-missing6.545010°/6.570567°。单因素GS改善0.025592°，prior较旧dense43k6.364189°成为新的训练内部最佳，仍FAILED_NORMAL_TARGET；45/46k继续，5.994869°只读诊断仍独立，不当训练成绩。


## 1007 晚间：稳定深度畸变试验与46k回填

继续严格禁止GT法线进入训练、初始化、RGB先验与像素筛选。GT法线/深度只在独立内部验证中使用，原15测试视角仍不用于选参。当前真实训练CV4L最佳为z16 RGB弱先验46k：GS6.277772°、Depth9.709195°、布料10.705582°、覆盖0.998609812；匹配无先验z17为GS6.310963°，布料10.723049°。仍未达5°；只读PS5.994869°不视作训练成绩。CV4保守43k仍为GS6.576723°。

只读审计发现原raster深度畸变约48%前景为负，接近1的NDC深度二阶矩相减有float32抵消。基于当前安装包精确commit的独立稳定核在两模型各7优化帧上，常规RGB/alpha/normal/depth/radii逐位相等，负畸变比例降为0；双精度成对参考、XYZ/opacity有限差分及旧分支梯度回归通过。所有审计只读观察RGBA和预测，不读GT法线/深度/几何。

新preset `lambertian_geometry_area_distortion`恢复预测ray-splat成对深度畸变自监督，权重默认0/preset1e6，不改其他preset。1e6在CV4L审计的XYZ梯度为既有photo+geometry的11.72–45.68%，旋转为1.31–4.56%，作为首次小权重试验，不宣称它是质量瓶颈。

正式训练须通过79测试、4/16子片真实Trainer初始化/梯度/GeometryRun/Albedo及旧分支回归、40步/20+20恢复/独立重复/13帧输出/实际Trainer重载。固定恢复界限：XYZ2e-5、旋转5e-5、scale1e-4、图像1e-3；严格1e-6及独立重复范围单独报告，禁止放宽失败界限。冒烟正在执行，正式质量收益未得到。

预注册配置 `output/1007-03-static-geometry-stage2/distortion_formal_configs.json`：CV4_z08_stable_depth_distortion与CV4_z09_warm_distortion_control同从真实CV4 z03 43k初始化，XYZ LR1e-6、旋转5e-5、scale5e-5、albedo1e-4，保留源累计XYZ0.005/旋转20°/scale1.25上限；畸变唯一变量1e6/0。CV4L z18从真实z12密集43k初始化，与已完成z17同XYZ1e-6、旋转8e-5、scale5e-5、albedo1e-4，仅增加畸变1e6。均43→46k独立调度，XYZ/scale延迟500步，ramp250，保存/独立评测44/45/46k。

监督器 `run_distortion_experiments.py`先完成恢复门控再执行；run_experiment在文件锁下检查并启动实际训练，防止多个监督器同时抢到第3槽。新队列不重启已有opacity或静态队列。流程PASS不能替代质量PASS；保留所有失败产物。


### 稳定畸变正式执行与opacity失败后调控

稳定畸变40/20+20/独立重复/13帧输出的预注册分量界限通过；严格1e-6及独立重复范围未通过，保留失败审计。恢复XYZ最大差6.71e-8、轴角3.53e-6、scale1.10e-6；图像最大差RGB2.92e-6、normal3.16e-5、depth5.72e-6、alpha1.67e-6。实际Trainer参数/Adam/RNG/相机及origin43逐位精确重载，通过且无需外部Stage2初始化目录。14视角以及额外真实核零畸变回归均通过，新核常规反向最大绝对差<5e-10。三正式组CV4_z08/CV4_z09/z18均实际启动，每卡最多3。CV4 45k GS6.552866°/关闭对照6.555984°，Depth9.037778°/9.051871°，覆盖0.998324972/0.998316470；收益很小，仍未达5°。

旧静态四组均45k正常结束且全部watcher退出0；CV4L无筛选z14 45k GS6.383824°，与筛选z13 6.384211°几乎相同，均不及当前RGB弱先验6.277772°。该源时间变化筛选未显示稳定质量收益，不能继续声称其有效。

原opacity40/20+20/独立重复及16子片40全部执行，但恢复门槛失败：独立scale最大差1.4744e-4超过1e-4；opacity恢复/重复差3.1789e-5/3.3790e-5超过1e-5。失败日志、检查点和预注册界限全部保留，旧CV4_z07/z15正式训练未启动，实际Trainer重载因前序门槛失败未执行。

新的降LR队列先重做相同恢复检查，界限逐位复制而不扩大：opacity LR1e-4→2e-5、scale2e-4→5e-5、rotation1e-4→5e-5，XYZ统一1e-5。新配置opacity_low_lr_formal_configs.json：CV4_z10/CV4_z11及L z19/z20从35k训练到43k，各自启用/关闭opacity配对且其他参数相同，避免把LR变化当作opacity单因素收益。监督器run_opacity_low_lr_experiments.py等待当前稳定畸变槽位，仅在恢复与实际Trainer检查通过后启动四组正式。当前重试仅排队，恢复/质量均未证明通过；再次失败亦不能放宽界限。

当前最佳L z16 46k四通道11验证视角与HDR已补齐；normal/0058和albedo/0115及0001总览已目检，布料法线细节仍不足，质量FAILED_NORMAL_TARGET。HDR仅11内部验证、2048样本，保存3个线性NPZ并目检/视频检查；不替代法线指标，也未渲染原15测试。当前最佳C4 45k的新四通道待该批结束择优后补齐；两数据集最优模型Mesh与原15最终验收仍未完成。


### 本轮终态回填：稳定畸变已完成，降LR opacity继续

稳定畸变三组43→46k全部完成：CV4 z08 GS6.551023°/Depth9.029719°，匹配关闭z09 GS6.553736°/Depth9.046351°；L z18 GS6.309418°/Depth9.689956°、布料10.726679°、覆盖0.998605561，对照z17 GS6.310963°/Depth9.710552°。畸变唯一变量的GS收益仅约0.0027°/0.0015°，不能作为下一步主要提升方向；较小的深度法线改善仍分别报告。CV4当前最佳z08 46k四通道11视角已目检、视频检查、质量FAILED_NORMAL_TARGET；首渲染误用系统Python缺imageio的命令/日志保留，正确环境输出visual_validation_46000_v2。C4本模型HDR和Mesh尚未生成，L最佳z16的四通道/HDR已经完备，二者均未达5°。

降LR opacity40/20+20/独立重复/16片及13帧界限通过，原界限SHA一致，没有扩大。恢复/重复XYZ2.3285e-6/2.3280e-6、axis1.0652e-5/1.0649e-5、scale8.0141e-6/8.0106e-6、opacity1.9396e-6/1.6537e-6；13帧最大RGB8.75e-6、normal6.0333e-4、depth9.25e-5、alpha1.0729e-5，均小于预先的1e-3。严格1e-6与独立重复范围仍FAILED，实际Trainer参数/Adam/RNG/相机精确重载PASS。新的训练门控opacity_low_lr_training_gate.json绑定全部证据，CV4 z10/z11及L z19已正式启动，L关闭z20等待槽位；每卡最多3。它们从35k→43k，GT法线/深度不作监督，质量结果仍待产出。


### RGB先验修正幅度与屏幕滤波只读诊断（10月7日23时）

继续严格禁止GT法线进入损失、初始化、先验或eligibility。GT法线/深度仅在独立评测器中计算11内部验证误差并用于试验选择；原15测试尚未参与选参。现有94训练RGB拟合的raw normals保持不变、总源旋转20°约束不变，仅把先验单次修正5°扩大到10/15/20°做只读诊断。5°状态逐位复现原诊断；GS分别5.994869/6.023543/6.044861/6.050665°，布料10.941626/11.077382/11.154858/11.176811°，覆盖完全相同0.998618315。扩大幅度有害，保留5°。这些诊断checkpoint优化器不匹配且禁止初始化/恢复，不是训练成绩。

独立只读屏幕滤波核记录全部贡献门控之后rho2d<rho3d的alpha*T。两模型各7个优化帧、常规RGB/albedo/normal/alpha/depth/radii与原/稳定核逐位一致，稳定前8个map逐位一致；原安装SHA未变，诊断核梯度模式直接拒绝调用。CV4平均滤波权重16.16%、观测RGB红色区域25.63%；CV4L密集平均44.67%、红色区域61.10%。统计是实际贡献，不是因果误差证明。

固定当前真实46k最佳模型，只在独立评测进程比较默认0.70710678/0.5/0.35像素范围。默认完整评测指标逐位复现旧结果；C4 GS6.551023/6.726130/6.926524°、覆盖0.998312218/0.987080180/0.979500043；L GS6.277772/6.759447/7.326857°、覆盖0.998609812/0.986769832/0.975580308。布料也变差、缺失惩罚变差，不采用缩小滤波范围。诊断结果放diag_screen_filter_*，不进入真实训练最佳排序；所有失败日志/图像/编译源码保留。贡献总览0002/0059/0119与两数据集0.35的comparison0058已目检，细碎盘面和边缘缺口仍明显。完整命令/来源/指标见screen_filter_audit/README.md。

### 下一组：仅将RGB先验权重0.2调整为2

真实z12 43k初始化的7优化视角梯度审计，不执行优化器步骤：权重0.2的轴角梯度约既有photo+geometry的1.99–4.84%，权重2为19.93–48.37%，权重20为199.34–483.72%。选择较保守的2，保留训练RGB-only的5°先验修正和eligibility，不增加几何LR。z21_dense_rgb_normal_prior_weight2除lambda_geometry_rgb_normal_prior=2之外，完整配置逐项等于已完成z16权重0.2，且复用z17权重0对照；从真实z12 43k独立初始化、fresh Adam，43→46k，XYZ1e-6/rotation8e-5/scale5e-5/albedo1e-4，保存及离线评测44/45/46k。

监督器run_rgb_prior_weight2.py已启动，先等待已承诺的z20 opacity关闭对照实际启动，再经原子训练槽执行40步有限性冒烟后启动正式，单卡实际训练始终最多3。当前仅排队，无权重2训练质量结果。已有同分支权重0.2恢复证据仍适用其原范围；本轮不会声称权重2已单独通过严格恢复或重复界限。只调整实验配置、不改现有训练实现或默认管线。配置rgb_prior_weight2_formal_config.json、队列rgb_prior_weight2_queue.json、日志rgb_prior_weight2_experiments.log绑定GPU审计与原79测试SHA。


补充回填：CV4 z08真实46k的HDR内部11预览已完成，0001/0058/0115已目检，3份线性NPZ有限且非负、11PNG和640×360/24fps/11帧视频通过；自身四通道和HDR均已补齐，质量仍FAILED_NORMAL_TARGET。两个当前最佳模型的Mesh和原15最终评测仍待完成。滤波诊断的初始method标签错误把tag050/035记为0.05/0.035，编译源码始终为0.5/0.35；初始method/README/summary保留screen_filter_audit/initial_radius_label_metadata，依据编译manifest修正标签，数值指标和实际评测命令未变。


10月8日00时更新：当前真实46k两模型各自Mesh已提取与评测，C4 Mesh8.330507°/L8.284224°，不替代GS主成绩；自身四通道/HDR/Mesh来源、完整命令、指标与失败结论均已补齐。新默认关闭的预测深度跟随分支已实现并通过83测试和4/16旧、新GPU检查，恢复/质量尚待门控。最新完整方案见1008-01-预测深度跟随光度法线的Stage2对照.md（训练目录）与1008-01-法线深度一致性可选梯度分支.md（修改目录）。原GT法线禁训、原15不选参及最多3训练限制保持。
