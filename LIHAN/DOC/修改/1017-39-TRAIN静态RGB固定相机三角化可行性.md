# TRAIN静态RGB固定相机三角化可行性终审

2026-10-17，ushas，Conda `lumimotion-ushas`。T13失败后的一次CPU数据检查，无新训练。最终：**工程PASS；有限的局部稀疏位置证据PASS；直接作为全身初始化或联合未知光训练准入FAILED**。保留115条轨迹，不将其视为稠密表面或可靠法线。总目标仍ACTIVE，T08继续作为训练控制。

## 数据、代码和执行

仅原135 TRAIN RGBA（800×800）及transforms_train.json，共136数据输入。原来源总控/data_train_T04_temporal08，历史no_gt_manifest只作SHA清单校验，不加载灯表/点云/旧Stage1身份或权重，不读取GT。两相机函数通过原T08 96项manifest校验；Blender原转换/getWorld2View2先float32后float64。原manifest SHA093a6ac43a432b4b3154612751976946f74708201980978e880f1e3e30c3e3a3。全部136输入、源码、清单、相机函数前后SHA一致。

唯一科学源码LIHAN/scripts/diagnose_static_rgb_triangulation.py，SHA59ccd67adfdee5a222f17dc4210a30e82bfbc5ae354a5671f4421eb01035bf27；输出内frozen_source.py完整保留。原生产前登记从03b00fc提交原样导出到1017-39-TRAIN静态RGB固定相机三角化可行性-生产前登记.md，未按结果改配置。

cwd：`/home/han.li/reproduce/LumiMotion-stage2-perlight`。完整实际监督命令：

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python output/1017-01-静态DL未知光总控/run_static_RGB_feasibility_v1.py
```

唯一真实子命令：

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python -u /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/diagnose_static_rgb_triangulation.py
```

监督覆盖环境CUDA_VISIBLE_DEVICES空、OMP_NUM_THREADS4、OPENBLAS_NUM_THREADS1。实际UTC16:34:17.579434—16:34:45.812728，监督546530/子550809，root原session77421收真实exit0。没有GPU、相机更新、optimizer或checkpoint；model/iteration为N/A。渲染命令N/A（本项没有渲染）。

输出/渲染输出规定：本项所有实际产物在`output/1017-01-静态DL未知光总控/1017-05-TRAIN静态RGB多视角三角化`；无新增模型渲染。日志logs/construction.log、logs/independent_result_integrity_CPU.log、logs/independent_result_finalizer_CPU.log、logs/coverage_helper_v2_stdout_excerpt.log。launch.json/supervisor_completed.json保存完整argv、env、起止和真实退出。

## 生产前锁定的方法

原135帧按source编号排序，零基i%9==4的15帧为本次内部保留，其余120拟合。这15帧是原TRAIN的source5/15/...145，**不是原15 GT test**，也不称整个研究流程未见。相机metadata与输入SHA可提前读取；只有120fit先提取特征，轨迹/XYZ冻结并写held_isolation_boundary后才解码和提取15held特征。

SIFT nfeatures3000、contrastThreshold .02、edgeThreshold10、sigma1.6、nOctaveLayers3；唯一描述子RootSIFT=L1归一化后逐元素开方，L2双向Lowe比率<.75且互选。中心位于Alpha>252的5×5侵蚀区，不保证整个描述子支持域在前景。0.1像素位置去重，最高response优先、原索引打破平局；npz保存全部raw位置/尺度/方向/响应和去重原index映射。

SIFT坐标+.5配原native主点400；拟合相机取光轴夹角≤40°内最近8邻居、无向并集、source编号打破平局，无时间限制/不按匹配数补pair。F由Eb@inv(Ea)原相机直接产生，双向点线距离≤1px，不估计相机或另一个F。匹配连通分量同帧多关键点整组拒绝，至少4视角。确定性枚举双视图DLT候选，选择正深度且2px内最大一致集，仅固定相机优化XYZ，至少4内点且最大视差≥3°。完整轨迹观测和排除项保留，不以世界高度/GT bbox/语义筛点。

最终内点逐一留出重新三角化/优化，LOO仅为全拟合资格筛选后的条件敏感性。120fit轨迹/XYZ冻结后，15held只向固定拟合邻居匹配，同保留关键点须由同轨至少两个拟合内点视角关联。关联阶段无held F、预测XYZ邻域或预测误差过滤；多轨/多位置歧义另计。全部无歧义预测误差、负深度、越界原样报告，不回流拟合。

## 实际定量与覆盖

- 原raw关键点34378，去重29564；598fit pairs，5041双向描述子匹配，2025原F通过边。
- 1234连通分量：1106少于4视角、9无四视角一致集、3优化后未保持阈值、1同帧冲突；最终115轨迹，600拟合内点，607全轨迹观测（7排除）。
- 每轨内点视角中位4、最大16；视差中位21.921198°、范围5.764578—76.764689°。
- 拟合内点像素误差中位.325423、P95 1.080340；全轨迹含排除观测最大8.892253px，不隐藏尾部。
- LOO位移中位.803406mm、P95 9.372675mm、最大69.250766mm；LOO重投影中位.481652px。600次均有限；不是独立深度准确度证明，不按LOO删点。
- 120held pairs，0歧义；13观测/13唯一轨迹，误差中位.503681px、P90 2.105238px、最大3.018464px、均值.824633px。0负深度/越界。11/13≤2px仅作报告，所有13都保留。
- held只有5/15帧有观测：frame5有7，45有1，55有1，85有2，115有2；另10帧0。fit只有92/120帧有观测，28帧0。成功匹配子集不能代表全表面。
- fit相机/轨迹二部图三组件：81视角/111轨迹、6/2、5/2。固定4×4格中fit199个“帧×格”、held10个“帧×格”含观测；这不是表面面积覆盖率。

全部115 XYZ包含在triangulated_xyz.npy/frozen_tracks_before_held.json和三正交图；Z范围.163972—1.824413m、中位1.443913m，没有按高度筛选。定量JSON result.json、coverage.json、coverage_review_retry1/coverage_details.json、fit_bipartite_support_graph.json完整保留。

## 实际目检和限制

root实际看完整120fit观察拼图、15held观察拼图、全部115 XYZ及frame1/5全图；独立Agent另完成固定8张1/5/41/75/81/121/145/149全图，全部135生产overlay实际PIL解码。代表图片：coverage_review_retry1/fit120_observations_only.png、held15_observations_only.png、all115_XYZ.png及各frame观察图；全部原overlay在overlays。视频N/A。

主要观测在头发/面侧、手腕/手指及少量衣物，鞋/踝非零但非常稀疏，腿、衣褶大面和平台几乎无支持；不是稠密全身覆盖。这只是人工描述，没有产生语义mask或用于筛点。观察拼图红圈仅实际关联；原生产overlay黄十字是全部正深度图内XYZ投影，不证明可见或观测支持。空白视角不代表物体缺失。

RGB/Alpha/Normal/separation四类渲染与150帧视频均N/A，因为没有新模型渲染；GS/Depth/Mesh对GT法线指标N/A。本项没有训练收益度量，不宣称已经改善法线/鞋形或完成未知光辨识。

## 独立终审、来源与故障历史

独立完整数值复算/135特征与overlay读取、来源/隔离、600fit/13held投影/覆盖、实际图审全部通过：independent_CPU_review_v1/review.json SHAd3f9a6087ab121fb4dccb3109e37d4f9a0770252ffe741015a4acde198c358d1；quantitative_integrity.json SHA58ae4dbce2bf1829bf659e54b9f4f4e6125c8146792c25588bd8921f994931df。独立脚本/命令/冻结manifest及真实exit0见该收据。root_actual_coverage_review_v1.json单列真实范围。

136输入/config冻结SHA72c4a59319235b0627dc1016d8ae83a97ac02ab442293ba11774ff427ed27b3c；held前轨迹冻结1f73f399bc0fce74762a0ced282d8c4283b0efaed9172122f1389a27901936ca。实际启动独立工程准入为869a4e4ef6989719508418e7533291f8cb723d8141418c4e21b01ca89f3bc678，原字节在independent_engineering_actual_admission.json。后来仅收据status/schema规范化5fd6...未回填启动；原history完整，engineering_receipt_history_binding.json明示并发历史。

工程自检含原相机AST轴、实际native_uv半像素、已知F/DLT/LOO；独立6非共轴float32相机/不同焦距与1离群测试通过。审查修复了held提取顺序、raw映射保存、半像素恒等式测试不足和失败state仍RUNNING问题，均在生产前完成，没有改科学配置。辅助input preflight v1变量s/ns笔误发生在输入核验前，失败收据保留，独立helper v2来源核验exit0。后处理图表v1缺失系统字体，在读图/绘图前失败，coverage_review/FAILED.json及原helper保留；v2仅用包内字体fallback与新目录coverage_review_retry1，真实exit0，不重跑匹配或改生产结果。全部失败、模型既有结果和原日志均保留。

## 结论与下一步

**PASS：原RGB可提供少量多视角可重复的稀疏位置观测。FAILED：这些覆盖与13个保留观测不足以直接宣布全身初始化或联合未知光训练准入。** 不扫SIFT/阈值/邻居/描述子，不按LOO或held误差重选115轨迹；保留局部观测，不创建新16k计划。

下一步只考虑固定这115轨迹/600fit+13held观测，与T08既有全图native Expected/Median/Alpha缓存作CPU残差诊断，确认是否存在模型尚未满足的独立位置约束。原缓存只读取baseline全图，不使用旧plane会员/Stage1身份，不执行N/H，不改模型、灯或点云。具体诊断另行预登记；是否新增训练约束由真实残差与空间限制决定。Goal保持ACTIVE。
