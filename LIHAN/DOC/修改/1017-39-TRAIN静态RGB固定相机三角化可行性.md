# TRAIN静态RGB固定相机三角化可行性

2026-10-17，ushas，环境`lumimotion-ushas`。T13完整科学终审FAILED后登记本项。目标仍是未知光静态GS/Depth/Mesh法线恢复；本项仅回答原始图像是否提供可信的稀疏位置观测，不是新正式训练，不保证稀疏点能改善法线。

## 一次锁定的CPU问题与输入

只读取原135 TRAIN RGBA和`transforms_train.json`。不加载GT、旧Stage1身份/可见性、训练权重、灯表或点云，不调整相机。沿用冻结T08的原Blender相机转换与float32世界到相机矩阵，再转float64计算。135帧按source编号排序，零基`i%9==4`的15帧保留，其余120帧拟合。该保留只针对本次三角化；不称整个研究流程未见数据。

旧1015 T16在动态120帧/400分辨率做过相机兼容SIFT循环，已发现阴影及平台重复纹理误配；12边/8Gaussian是旧模型关联的共可见无饱和高gate子集，不是全部SIFT数量。本项新增可证伪内容是固定相机静态XYZ和未拟合视角预测，不依赖旧模型身份。

唯一配置：原800×800，不缩放；SIFT nfeatures3000、contrastThreshold .02、edgeThreshold10、sigma1.6、nOctaveLayers3。描述子只用L1归一化后逐元素开方的RootSIFT，L2双向Lowe比率<.75且互选。特征中心在Alpha>252的5×5侵蚀区；不保证整个尺度相关描述子支持域都在前景内，记录尺度和边界距离。同位置按0.1像素去重，最高response优先、原索引打破平局。保留原关键点与去重统计，不另跑其他描述子挑优。

OpenCV关键点坐标加.5后配合原native K主点400；相机数学自检验证投影和已知F。拟合视角按原光轴夹角，每帧取≤40°内最近8邻居，按source编号破同角平局，无向并集。静态场景不限制时间，不按匹配数补pair。已知相机直接产生F，拟合匹配要求双向点到极线距离均≤1像素，不估计F/RANSAC相机。

## 轨迹、预测和稳定性

拟合匹配边的连通分量出现同帧多个不同关键点则整个分量记冲突并排除。至少4拟合视角才三角化。确定性枚举轨迹视图对、DLT候选，以正深度且重投影≤2像素的最大一致集选候选，平局按内点误差和source编号；至少4内点，固定相机只优化XYZ。最终内点须保持正深度、≤2像素，最大视线夹角≥3°。记录全部轨迹观测、排除观测和拒绝原因。没有基于GT或世界坐标范围的过滤。

对最终拟合内点逐一留出，重新DLT和优化XYZ，报告位移毫米/相对相机距离以及留出重投影误差；这是经过全拟合轨迹资格筛选后的敏感性诊断，不冒充独立验证。误差低本身不证明深度准确。

先冻结120帧轨迹、XYZ及资格，再对保留帧向同规则固定8个拟合邻居匹配。保留关键点须被同轨至少两个不同拟合内点视角关联。此关联阶段禁止使用保留帧F、预测XYZ邻域或重投影误差；多个轨迹关联同保留关键点或同轨在同保留帧关联多个位置的情况计歧义并单列，不用误差选择。对无歧义关联的全部保留观测原样报告预测误差、负深度、越界，不按误差删点，不回流拟合。这只代表描述子可匹配子集，不代表全表面无偏验证；未匹配不等于几何缺失，Alpha内投影不证明可见。

## 覆盖、停止规则和产物

报告全部关键点/匹配/冲突/轨迹/内点/保留关联计数、每轨视角数、视差、稳定性及预测分布。每帧固定4×4格报告轨迹中心数和Alpha像素数，保存135帧叠图和世界XYZ坐标分布，不套GT bbox、语义掩膜或高度阈值。人工可指出腿鞋少，不能将其转换为筛点标签。平台密集不能代替人体支持，不以总点数单独宣布可训练。

代码`LIHAN/scripts/diagnose_static_rgb_triangulation.py`；输出统一`output/1017-01-静态DL未知光总控/1017-05-TRAIN静态RGB多视角三角化`，完整日志同目录。生产运行前冻结源码、配置、136输入SHA，CPU工程自检和独立审查；失败现场保留，任何工程修复单列重试而不调科学参数。无GPU、optimizer、checkpoint、渲染，四类视频和法线指标N/A。本项真实结果另追加；当前尚未运行匹配/三角化。

一次结果若轨迹稀少、集中、视差小、位置不稳或保留预测失败，则结论是支持不足，不扫参数，不由本项自动安排16k。若支持可信且空间有用，另论证如何把新观测引入训练并更新训练计划；保留T08控制。


## 生产前工程检查

root实际执行`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/diagnose_static_rgb_triangulation.py --selftest`，首次与补强后两次真实exit0；自检收据总控/static_RGB_engineering_selftest_v1.json、v2.json及各自.log完整保留。最终当前源码SHA 4d10ef6f928ca1dcc61bd4b4ee7ebb0580cdfc272c8225a430a8062cfe37fad6，DLT最大误差8.881784197001252e-16，已知F精确/构造错配拒绝/LOO精确。补强只锁定原T08 manifest SHA及两相机函数来源、结束复核和异常日志恢复，未改科学配置。独立审查待结束；尚未实际匹配。


独立生产前审查发现并已补强三处记录/隔离边界：相机135只先转换metadata，先提取120fit特征，轨迹/XYZ和held_isolation_boundary冻结后才提取15held；npz保存全部raw位置/尺度/方向/响应及kept_raw_indices；native_uv实际函数及原T08 Blender identity轴自检替代恒等式。root v3自检真实exit0，收据static_RGB_engineering_selftest_v3.json/source73fc93dd...，新增actual_native_uv_center和original_Blender_identity_axes均PASS。独立额外6非共轴相机float32/不等焦距测试F最大1.63546e-13px、DLT1.94289e-16m，加入一离群正确留5内点；漏+.5可产生4.35434mm偏差，说明补强有必要。最后仅在异常分支同步写state=FAILED，避免失败仍显示RUNNING，科学配置完全不变。

来源工程核验C/static_RGB_engineering_input_preflight_v2.json真实exit0：136数据SHA与原manifest一致、原96manifest/两相机AST来源通过，未读取GT/模型/灯表/点云值。v1 inline辅助命令变量s/ns拼写错误发生于输入核验前，非生产代码/非匹配；失败收据static_RGB_engineering_input_preflight_v1_FAILED.json保留，仅纠正独立helper后完成v2。

最终待执行源码SHA：59ccd67adfdee5a222f17dc4210a30e82bfbc5ae354a5671f4421eb01035bf27（仅异常状态写法较v3受测数学版本变化，compile真实通过）。执行监督总控/run_static_RGB_feasibility_v1.py会强制绑定该SHA和独立工程PASS收据，设置CUDA_VISIBLE_DEVICES空，并保存真实child PID/退出码、起止UTC、同目录README/启动/日志；不从工程自检结果推断数据可用性。
