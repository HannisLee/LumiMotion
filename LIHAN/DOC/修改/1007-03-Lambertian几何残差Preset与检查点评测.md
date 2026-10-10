# 1007-03 Lambertian 几何残差 Preset 与检查点评测

新增`lambertian_geometry`，原IR和冻结Lambertian行为保留。几何参数作为附加GeometryResidual存在，不修改原Stage1参数与形变网络；xyz和轴角独立学习率、局部步数调度、信赖域投影和越界动量清理。几何模式下打通rasterizer法线/alpha/depth梯度，添加内部normal-depth一致性及参考几何约束。

检查点v2保存几何残差、优化器、随机状态和采样栈，加载仍验证原Stage1哈希和冻结指纹；兼容旧v1。渲染、几何评测及其下游mesh提取使用同一几何残差，防止读取旧Stage1几何产生错误成绩。

训练、测试与真实GPU验证进行中，结果后续回填。原用户工作区未提交的评测代码在原文件上扩展，不覆盖已有内容。

## 扩展与验证

- 新增`lambertian_geometry_point`（逐像素灯位方向、距离平方衰减）、`lambertian_geometry_filmic`（近场光及Filmic显示LUT）、`lambertian_geometry_shadow`（额外只读源几何面积光可见度）。原有preset默认行为保留；新增开关和损失权重均在Stage2参数组。
- Albedo评测支持v2几何残差，已与主渲染器逐像素核对，误差为0。Mesh提取复用GeometryRun，因此读取更新后的几何。
- 45项单元/回归测试通过；真实GPU材质/位置/旋转梯度、零残差渲染、检查点重载及恢复状态通过。Filmic颜色映射、阴影缓存和GPU梯度另有数值验证。shadow新preset已开始s01实验，尚无质量结论，不能把前向/梯度验证写成训练质量验收。
- 自动旧式光强标定不适用于Filmic或非零环境项，这些模式要求显式固定光强。显示LUT包含在几何checkpoint内；重载仍依赖随代码提供的assets初始化模块。
- 测试证据：output/1007-03-static-geometry-stage2/tests_v4_shadow.log；output/smoke_test/1007-03-static-geometry/{gpu_checks,albedo_v2_check,filmic_gradient,shadow_gpu}.json。

## 共享材质与前向着色对照

`lambertian_geometry_palette`继承shadow参数，新增`photometric_material_mode=palette`和`photometric_palette_size`。标签由Stage1颜色计算并存入checkpoint，恢复直接使用保存标签；默认仍为per_gaussian。GPU重载及独立albedo renderer核对通过。

`lambertian_geometry_surfel`继承filmic参数，新增`geometry_shading=surfel`，默认仍screen。surfel分支在高斯中心计算近场辐照度/方向光/HDR，以可见面朝向法线着色，再累积RGB；独立albedo累积保持材质输出正确。几何normal评测仍取光栅化真实旋转法线，不使用另一个着色法线替换指标。阴影若开启仍是只读Stage1屏幕可见度近似。新增GPU反向传播检查记录在smoke_test/.../surfel_gpu.json。

`lambertian_geometry_lit`进一步加入可选直接受光区域筛选：只读Stage1可见度≥0.98、原始参考方向光辐照度≥0.2时计算RGB/SSIM光度项，共享palette材质；全表面几何一致性、轮廓和锚定不受此筛选影响。阈值来自预测几何和已知光照，不读取GT normal/depth/albedo；参考掩码不依赖当前可优化法线，避免通过旋转逃避监督。默认阈值为0，旧管线行为不变。此分支尚待GPU训练冒烟，不预先宣称质量改善。全量50项测试通过，日志tests_v7_lit.log。

## 已知面积光模式

新增`geometry_area_emitter`及`lambertian_geometry_area`，默认关闭。`AreaLightTable`保存真实矩形光世界坐标采样点和发射方向，计算受光面余弦×发光面余弦/距离平方；通过四子区域中点平均近似积分。共用固定功率从灯光metadata读取，自动强度采用旧EEVEE的P/(4×reference_distance²)，显式给定强度仍优先。各帧功率不同或非矩形灯会明确拒绝。`geometry_shadow_size`不覆盖area模式已知灯尺寸；源几何遮挡射线使用相同采样点。HDR/替换方向光继续走各自重光照入口。源码量纲依据和有限近似见训练文档，不宣称是完整EEVEE复现。

area_gpu.json确认位置/旋转/材质光度梯度有限且非零，alpha/depth/normal/albedo通道与screen着色一致；tests_v8_area.log的52项测试包括旧管线回归、远场功率、发射方向背面截止及位置梯度。

## 面积光版本校正

上节P/(4A)只对应旧实验，不适用于保存于5.2的数据。`geometry_area_power_convention`新配置默认blender5，P/(πA)；旧area checkpoint缺少该字段时按legacy重载，保持历史结果含义。实际读取normalize、exposure、diffuse_factor、世界尺寸，并支持已知influence_radius的平滑衰减。所有增强metadata均新文件，未覆盖旧输入。旧x系列早期中止，正式正确版本实验改为y系列。

合成白平面对照在Blender 5.2实际渲染线性EXR下相对亮度差约0.1%，含截断与不含截断两种都验证通过；世界光增量误差约0.0002。52项回归通过，area_blender5_full_gpu.json核对新checkpoint位置/旋转光度梯度、几何评测以及albedo与主renderer零差。该数值检查不代表完整数据集模型误差或法线质量已通过。

## 训练RGB共享材质初始化文件

新增可选`photometric_palette_init`（默认空，保留旧色度初始化）。仅palette模式使用；加载npz中的整数标签、K×3有限sRGB颜色、Stage1路径/迭代/点云和MLP SHA。严格检查来源、数量和范围；初始化后完整写入material state，恢复时不再读取该文件。测试覆盖来源被修改时拒绝、初始化文件缺失仍可从checkpoint准确恢复；53项测试通过，GPU初始化/渲染/重载通过。实际训练用的4色文件由94优化相机RGB估计，不使用GT材质标签。输出脚本保存新实验的初始化文件与灯光JSON快照，便于复现。

## 可选的有界尺寸残差（验证中）

新增lambertian_geometry_area_scale preset，仅该preset默认开启geometry_optimize_scale。GeometryResidual保存两个面内轴的log_scale，零初始化、独立LR/启动步数、倍率信赖域以及尺寸锚点；越界分量投影后清除对应Adam动量。默认geometry_scale_lr=1e-4、geometry_scale_start=4000、geometry_max_scale_ratio=1.25、lambda_geometry_scale_anchor=0.001。关闭时apply_scales直接返回原张量，旧checkpoint不增加状态。

主Lambertian、独立albedo、几何评测/mesh所用GeometryRun三条路径统一应用尺寸残差；光照可见度仍采用只读Stage1参考。新增零残差/梯度/信赖域/重载单元检查，GPU与训练恢复检查进行中。尚未宣称法线收益。

尺寸分支55项单元/回归通过，日志tests_v12_scale.log。scale_gpu/result.json验证真实GPU位置/旋转/尺寸梯度有限非零、尺寸零残差逐位一致、非零尺寸确实改变渲染几何，以及GeometryRun/独立albedo与主渲染一致；旧checkpoint正常重载。训练断点恢复由排队冒烟进一步验证，尚未完成。

## 尺寸恢复的CUDA重复误差与法线容差preset

尺寸恢复的初次严格比较失败保留在scale_resume_initial_comparison.json及scale_queue.log。独立连续重复证明尺寸参数本身存在最大2.7021e-6的CUDA数值差异，恢复差1.0692e-6；仅log_scale比较使用atol3e-6，其余几何保持1e-6，RNG/相机严格一致。结果在smoke_test/.../scale_resume.json，重新核验日志scale_resume_verified.log。正式z04在该核验通过后启动。

新增geometry_normal_margin=0.0及lambertian_geometry_area_margin preset（3°）。normal_depth_penalty保留默认1-dot公式；margin>0时计算max(1-dot-(1-cos(margin)),0)，要求角度在[0,90)。不增加模型参数，不改变GS Normal输出或评测定义，checkpoint配置保存容差；旧checkpoint缺失该字段按0处理。56项单元/回归通过（tests_v13_normal_margin.log），包括默认逐位等价、容差内零梯度、超限梯度和显式CLI优先。真实GPU normal_margin_gpu.json验证损失与位置/旋转梯度有限，GT几何未输入损失。

实验launcher新增写入前检查，已有initial/resume命令、代码、输入或日志时直接拒绝，避免训练器最终拒绝覆盖之前先覆写实验记录。对已有smoke_scale40实测退出且所有记录SHA不变，结果launcher_existing_guard.json。


## 可选的源表面细分残差

新增geometry_subdivide=False及lambertian_geometry_area_refine preset；refine继承area_margin并开启有界尺寸残差。GeometryResidual保存parent_index和child_offsets缓冲，关闭时两者为None，不进入旧checkpoint。开启时N→4N，prepare统一展开冻结位置/四元数/尺寸/opacity，用父四元数旋转对称局部偏移，再应用每个子片的世界位置/旋转/尺寸残差。expand_values仅重复父材质输出；材质和形变网络仍使用原N点。源文件和参数保持冻结。

主Lambertian、独立albedo、GeometryRun（及下游mesh）共用prepare。实际GS Normal仍来自可优化子片的四元数与原rasterizer，不替换评测法线。源参考alpha/depth/阴影仍采用Stage1，细分与小幅运动下的近似须通过覆盖率、轮廓和深度监控。四子混合只匹配均值/协方差/线性积分，不等于原非线性alpha渲染。

59项单元/回归通过（tests_v14_refine.log）。refine_gpu/result.json验证42654→170616子片，位置/旋转/尺寸梯度有限非零、三个几何入口一致、独立albedo与主通道零差、源SHA/冻结指纹不变。refine_legacy_gradients.json核对关闭细分时的旧渲染器前向及材质/位置/旋转反向，最大梯度差3.49e-10。refine_gpu/validation_initial记录实际初始质量，不能以API检查代替训练验收。连续40步与20+20恢复正在检查，所有日志和失败产物保留。


细分恢复采用独立40步CUDA重复支持的分量容差，具体实测值见训练文档及refine_resume.json；父映射/固定缓冲/RNG/相机严格一致，GPU重载参数逐位一致。旧分支容差和默认行为保持。真实两帧渲染恢复误差远小于图像量化精度；这只验证恢复，不宣称法线目标通过。初次严格失败、核验脚本设备错误及修正后日志均保留。

几何评测在细分分支将gaussian_count记录为实际4N渲染数量，同时保存source_gaussian_count和rendered_gaussian_count，避免把父点数量误报为当前容量；不改变指标计算。旧分支报表行为保留。


细分三正式训练已启动，完整代码快照在各initial_code。当前最佳z04 43k的预测训练视角TSDF网格已离线评测，三类法线均分别报告，不替换主GS指标；详见训练文档及validation_43000_mesh/metrics.json。


## 可选邻域位置与法线联合一致性

新增lambda_geometry_coherence默认0，以及邻域数6、切平面阈值0.35、法线轴阈值45°、半径倍数3参数；新preset lambertian_geometry_area_coherent默认权重0.05，显式CLI可覆盖为0。关闭时所有新缓冲为None，不写旧checkpoint，不增加renderer辅助字段或旧loss运算。旧前向逐位回归和反向数值回归已通过。

GeometryResidual.initialize_coherence在首次初始化、开始优化前，用冻结source经prepare得到的实际4N子片构建cKDTree；拒绝低opacity、动态、不同训练RGB材质标签及不满足距离/法线/平均切平面条件的候选。距离与索引确定排序，不消耗RNG。固定邻接索引、符号、参考边长、有效权重和ready标志保存到checkpoint；恢复及离线渲染直接使用，不按相机重建。

主renderer仅在该分支额外返回实际geometry_means和未经相机翻面的单位geometry_normals。geometry_losses加入平均双端切平面正则，两端位置和四元数旋转共同优化；按参考边长无量纲归一化，源权重固定。GeometryRun、独立albedo继续共用prepare，不替换法线输出或评价定义。

新增3个回归测试验证来源分组/static限制、图精确重载、球面曲率零损失及位置/法线恢复方向梯度、关闭默认和CLI覆盖；总62项通过。真实GPU预检coherence_gpu/result.json通过，约47.74%图槽位有效，位置与旋转梯度有限非零，源冻结。训练/恢复冒烟由run_coherence_smokes.py和check_coherence_resume.py执行；此刻准备启动，不能预先声称训练恢复或质量目标通过。

训练约束再次核对：Stage1与Stage2入口、scene/dataset_readers.py、scene/cameras.py没有GT normal_exr读取。geometry_normal仅使用预测surf_depth和normal；source参考深度为Stage1预测；新增图只使用Stage1与训练RGB标签。GT法线仅进入独立离线评测，不作为loss、梯度或初始化输入。


## 邻域正则恢复检查的实际结果及正式训练门槛

最初1e-6严格恢复检查失败，新增两次独立连续重复仍未覆盖恢复参数最大差，原检查器与全部失败日志保留。独立实际PhotometricTrainer恢复probe证明全部参数、Adam状态、固定图、相机和RNG逐位重载，源冻结；不声称跨进程逐位训练等价。另一个独立检查器check_coherence_bounded_resume.py采用预先已有绝对界限、同rasterizer细分历史实测误差及13视角真实渲染，有限精度通过；明确strict=false、own_repeat_range_covers_resume=false，未盲目扩大容差。

run_coherence_formal.py据GPU梯度/旧分支回归、实际训练器精确重载、有限精度恢复证据启动两组权重对照，coherence_training_gate.json保存证据SHA和失败条件说明。各运行通过preset/CLI启用，没有增加GT法线监督。新增watch_selected_validation.py区分CV4/CV4L独立离线验证，启动后等待实际训练进程再判断终态，避免早期启动竞争导致缺失评测。


## 可选16子片密集表面采样preset

geometry_subdivision_side默认2；仅geometry_subdivide开启时使用，支持2/4。新lambertian_geometry_area_dense继承原refine并将side设为4，显式CLI可覆盖回2。旧4片保持原偏移构造顺序与sqrt(0.75)常量；关闭细分状态无新增张量。每轴4片时均匀对称偏移(2i-(s-1))*sqrt(3)/s、子尺度1/s，理论中心/协方差/线性opacity积分相同，但实际非线性alpha渲染不同。

GeometryResidual.prepare按side²验证来源数量；diagnostics用构造时源N记录source_gaussians，避免将682464仍除4误报。parent_index/child_offsets和几何残差已有checkpoint保存机制继续使用；config保存side，新密度checkpoint与旧4片参数形状不兼容会明确拒绝，旧缺少side字段按2恢复。主renderer、GeometryRun、独立albedo共享prepare，无额外改动法线输出定义。

64项单元/回归包括16片矩/积分、重复颜色梯度、精确重载、数量统计、形状不兼容拒绝以及新preset CLI优先，tests_v16_dense.log通过。dense_gpu/result.json核对旧4片前向逐位/反向数值一致、682464片位置/旋转/尺寸梯度有限非零、独立albedo与主通道零差、GeometryRun一致、源冻结。训练恢复尚需run_dense_smokes.py完成；无GT法线损失或初始化。


密集分支40步及20+20恢复完成：严格1e-6失败保留；有限精度上限保持实验前值，13视角最大RGB2.779e-4、normal分量7.447e-4，均<1e-3，映射/固定缓冲/RNG/相机严格一致。dense_restore_probe实际Trainer全部参数及Adam/RNG逐位重载，源冻结。独立重复范围不能严格覆盖全部恢复差，标志false保留，不误报严格恢复。run_dense_formal.py读取三项实际证据并记录SHA门槛后才启动；密集初始化质量更差，不能用程序检查替代目标验收。


## 固定邻域的源时间稳定性筛选

CV4源预测审计发现：固定图中仅0.6753%的边涉及训练时间变化点，却在0105帧占78.6291%的初始图损失，损失由首帧0.03218升到0.15012。这些点的像素贡献虽小，不能据此认定它们不会影响正则。继续保留原coherent分支，新增可选geometry_coherence_source_mask路径及lambertian_geometry_area_static_coherent preset。新preset在coherence>0时要求该资产；显式权重0关闭，不要求掩码文件、不增加掩码状态。原preset未设置路径时默认逻辑不变。

scripts/prepare_stage2_static_mask.py只使用94个Stage2优化视角的冻结Stage1位置/四元数预测，以首个优化帧为参考，筛除移动超过场景extent的0.5%或预测法线时间变化超过5°的源点；这些“法线”是Stage1预测，不是GT法线。不读取11内部验证/15测试的预测或误差来生成掩码，也不读取GT几何或材质。CV4 43140点中42532稳定；CV4L 42654点中42252稳定。源点不删除，仍参与渲染、光度损失和其他几何损失，只禁止其固定邻域边。

资产内保存bool源点掩码、变化统计与JSON协议。load_coherence_source_mask检查dtype、N、源目录、iteration、PLY/deform SHA、优化帧顺序及无GT法线协议；不使用pickle，不读取资产元数据指向的文件。runtime在首次建图时与binary_feature<0.5合取，展开到实际子片，拒绝两个端点之一不稳定的边。coherence_source_static与既有图缓冲存入checkpoint；恢复/渲染不再次读取外部资产。初始化输出geometry_coherence_source_mask.json，launcher快照输入与SHA，并包含新资产准备脚本。

67项单元测试通过（tests_v17_static_mask.log），新增真实来源/划分不匹配拒绝、bool/N、关闭兼容、新preset及自包含图恢复。static_coherence_gpu/result.json验证旧preset前向逐位/反向数值回归、掩码与此前94训练来源审计一致、双端筛选、有限非零位置/旋转/尺寸光度梯度、GeometryRun一致、独立albedo零差、Stage1冻结。七训练帧的图损失为0.03214664–0.03229417，初始时间异常已消除。GPU诊断checkpoint有diagnostic标记，禁止续训。

训练/恢复尚未完成时不得报PASS；run_static_coherence_experiments.py执行实际40步、20+20恢复、独立40步重复、CV4L40步及真实Trainer重载。预先保留XYZ2e-5、axis-angle5e-5、log-scale1e-4、XYZ Adam一阶矩1e-5、其他Adam1e-6/rtol1e-5、13视角渲染最大差1e-3界限；严格1e-6和自身重复范围另行报告，不因失败扩大阈值。恢复probe使用外部掩码不存在的自有检查点副本，核对全部Adam/参数/图/掩码/RNG/相机，诊断副本禁止后续训练。正式训练须通过GPU、有限精度恢复与实际Trainer精确恢复门槛。


## 可选冻结来源预测缓存的检查点保存与恢复

原静态图40步/20+20恢复在axis-angle最大差5.55586e-5上超过预设5e-5，原失败完整保留，正式训练被拦截。新增geometry_checkpoint_source_cache默认False；static_coherent preset默认True，显式CLI优先。保存geometry_reference_cache（冻结Stage1预测depth/alpha/参考shading）和geometry_visibility_cache（冻结Stage1预测可见度）。没有GT法线、GT深度或目标RGB缓存。关闭时旧checkpoint字段和旧恢复路径不变。

恢复只允许94优化相机键、准确depth/alpha/(shading)字段、相机像素形状、float32参考/float16可见度及有限张量；拒绝验证/测试键和额外字段。配置开启却缺少缓存则明确拒绝，不静默重算。68单元测试通过tests_v18_source_cache.log。原和缓存版本实际PhotometricTrainer全部参数/Adam/图/掩码/RNG/相机逐位重载；缓存版额外核对缓存逐位一致，缺失外部掩码仍可恢复。两个probe均诊断标记，禁止续训。

缓存版重复检查仍失败，且独立连续重复也出现axis-angle约5.55632e-5；不能宣称缓存修复轨迹差异。单独13帧输出检查在0046一个真实前景像素normal分量差0.00109816超过预设0.001（alpha约0.999885、全图RGB最大差0.000209987），不以背景豁免或放宽阈值报PASS。全部失败记录保留。下一轮以rotation LR8e-5保持原数值界限重新检查；对应正式训练延长至45k/10k调度，并用同LR同长度的无筛选对照控制变量。

watch_selected_validation.py新增可选--iterations列表，旧默认37/39/41/43不变，用于新的45k检查点。仅改变独立离线评测队列，不进入训练。


新8e-5旋转LR的固定参数与13帧图像界限通过，原失败不覆盖、阈值未扩大；实际Trainer源缓存逐位恢复通过。严格1e-6及自身重复范围仍失败，单独报告。原GT-free训练约束和旧preset关闭路径保持。


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


## 1007 晚间：可选稳定深度畸变核

增加默认关闭的 `lambda_geometry_distortion=0` 与preset `lambertian_geometry_area_distortion`（1e6）。正权重才加载独立 `diff_surfel_rasterization_stable`，返回geometry_distortion并加入全图预测畸变均值，不引入GT法线或深度输入；旧渲染字典、几何state字段和旧loss保持不变。79测试通过；两真实模型旧分支状态字段、forward/loss逐位、backward1e-6/1e-5回归通过。真实新分支4/16子片初始化状态/常规输出逐位，XYZ/旋转/scale畸变梯度有限且非零，材质梯度为零，GeometryRun/Albedo一致，冻结来源检查通过。实际续训检查仍在执行，不能提前宣称恢复或质量通过。

独立CUDA目录 `submodules/diff-surfel-rasterization-stable`来源为安装包direct_url精确commit f7e2b68a0f17d6d7bca3fa564a96c7678581ce5a，由本地git对象读取。前向以首个实际贡献的ray深度为参照，计算C*(d-d_ref)/(d*d_ref)再累积二阶矩；保存第4个像素辅助槽以供反向使用，参考深度导数因成对差平移不变性抵消。原安装模块Python和二进制SHA完全不变，新模块0.2.0独立安装。

首个基于仓库Lumimotion源码的构建未通过真实RGB回归（最大差0.002334），发现与安装版本不一致，已完整保留到stable_rasterizer_build/failed_lumimotion_source和stable_depth_distortion_audit，不用于训练。首轮GPU有限差分在完全同深度时改变排序而失败，保留stable_distortion_gpu；改为同深度共同平移零梯度检查，非同深度仍保持原有限差分界限。首轮真实Trainer测试错误地要求材质grad为None，实际custom raster返回全零Tensor，保留distortion_gpu失败目录并禁止其检查点续训；正确零梯度检查通过distortion_gpu_v2，未放宽容差。

证据：stable_distortion_hbb_gpu/result.json、distortion_gpu_v2/result.json、distortion_legacy_gpu/result.json位于output/smoke_test/1007-03-static-geometry；14真实视角审计stable_hbb_depth_distortion_audit/audit.json，安装SHA stable_rasterizer_installed_manifest_hbb.json。run_experiment新增可选CUDA来源/二进制散列记录及文件锁原子训练槽门控，单GPU最多3个实际train_stage2；所有旧和新入口共享此门控。


### 真实恢复与新增核回归结果

稳定畸变新增核在真实CV4/CV4L几何上，畸变上游为零时常规forward/loss逐位、backward1e-6/1e-5通过，实际最大绝对梯度差<5e-10；证据distortion_backend_equivalence/result.json。实际Trainer精确重载与预注册有界40/20+20/重复及13帧输出通过；严格1e-6和独立重复范围失败分别保留，绝不称训练轨迹逐位可复现。原opacity实际恢复门槛失败，旧正式组未启动，保留opacity_resume_initial_audit.json及opacity_queue_failed.json；新增降LR监督脚本，不修改几何/opacity公式或扩大恢复界限，采用各数据集匹配关闭对照。

L z16 46k四通道输出visual_validation_46000/README.md及HDR hdr_validation_46000/README.md包含完整命令、日志、来源、指标、图片/视频、目检与失败结论；HDR只用内部验证相机，原15测试不用于这轮选择。代码来源、失败源码构建与失败probe均保留。


### 最终回填与环境要求

稳定畸变三正式43→46k全完成，GS唯一变量收益约C4 0.0027°/L0.0015°，未显著改善；原管线默认off不变。降LR opacity门控通过但严格1e-6/独立重复范围失败，如实保留，正式配对继续且尚无质量结论。所有命令必须使用venus的/home/lihan/miniconda3/envs/lumimotion/bin/python；C4可视化误用系统Python缺imageio的失败命令/日志保留，已在该环境生成独立v2结果并目检与ffprobe检查。训练文档记录实际46k指标和低LR参数/13帧最大差；仍不可把流程PASS当作normal<5质量通过。


### 输出目录中的只读滤波审计与权重2队列

新增output/1007-03-static-geometry-stage2下的prepare_filter_audit.py、audit_screen_filter.py、prepare_filter_variants.py、evaluate_filter_variant.py、summarize_screen_filter.py，复制已验证的稳定核到独立输出包，仅增加第9个low-pass可见权重map。0.5/0.35范围只改输出包auxiliary.h，不修改原模块、安装或任何训练渲染器。全部独立wrapper在梯度模式拒绝forward；原安装Python/CUDA SHA复核通过，默认完整指标与两真实46k原评测逐位一致。审计PASS不等于法线目标PASS；较小范围均质量FAILED，拒绝迁移到训练。

新增audit_rgb_prior_strength.py：真实Stage2初始化的7优化帧只读权重梯度审计，初始状态立刻标记diagnostic并禁止恢复，优化器没有step。新增run_rgb_prior_weight2.py只使用原preset/CLI，把lambda从0.2改为2；其余与z16完全匹配。监督器尊重先前z20队列、共享flock最多3训练、先40步检查有限性和几何学习率后正式。没有新增训练参数或修改core模块；权重2恢复尚未单独验证，不将权重0.2的严格恢复证据冒充权重2。

update_status.py增加动态读取RGB修正幅度审计、滤波贡献与只读范围记录；diag_*永远不计入训练最佳。各诊断的源码、命令、环境、binary SHA、失败统计和图像均保留。两自有中文文档按先前同步SHA检查后更新另一工作树，不复制无关修改。


只读滤波诊断的初始半径标签解析错误已修正：实际编译0.5/0.35未变，初始0.05/0.035标签及README/summary完整存档，按编译manifest回填，数值误差/覆盖未改写；只读拒绝梯度的复核另存screen_filter_readonly_guard_v2。C4当前真实最佳46k的HDR预览、线性结果和视频检查已完成，README包含完整命令、来源、日志、目检与FAILED_NORMAL_TARGET结论。当前新增helper仅在输出目录，未更改core训练实现；syntax检查通过，原79回归证据保持。


10月8日00时更新：当前真实46k两模型各自Mesh已提取与评测，C4 Mesh8.330507°/L8.284224°，不替代GS主成绩；自身四通道/HDR/Mesh来源、完整命令、指标与失败结论均已补齐。新默认关闭的预测深度跟随分支已实现并通过83测试和4/16旧、新GPU检查，恢复/质量尚待门控。最新完整方案见1008-01-预测深度跟随光度法线的Stage2对照.md（训练目录）与1008-01-法线深度一致性可选梯度分支.md（修改目录）。原GT法线禁训、原15不选参及最多3训练限制保持。
