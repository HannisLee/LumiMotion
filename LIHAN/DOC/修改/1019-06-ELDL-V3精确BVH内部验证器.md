# ELDL-V3 精确BVH内部验证器

2026-10-19（Asia/Tokyo启动日；主机日志2026-10-18 UTC）。新增`scripts/validate_eldl_v3.py`，不修改并发的`cache_eldl_v3.py`或`fit_eldl_v3.py`。固定规则见`LIHAN/DOC/报告/1019-05-ELDL-V3精确BVH内部验证预登记.md`。

验证器以fit选定初值、135个预测方向和冻结rho/EL重算真实预测Gaussian BVH的接收平面透射率，bias固定0.5/1/2倍。保留原valid分母和逐通道满码删失，报告fit/held等帧及像素加权MSE、已有constant_held及偏移退化。中心须优于constant；两端相对中心退化≤25%且仍优于constant，支持与三初值门槛同时保留。

入口使用`utils.no_gt_guard.install_no_gt_guard`，仅读白名单缓存、fit结果、I00模型与TRAIN输入。正式脚本不导出灯光资产，`internal_validation_passed=false`；所有数值通过仍待主任务实际目检，不写精度PASS。CPU计划/检查不构建CUDA，精确trace需显式GPU环境并经原生bootstrap加载。

修改与运行验证结果将在下方追加。GPU使用与主任务协调；I00@16000的held18完整验证使用独立目录，与8000步一致，不能阻塞GPU2缓存/拟合。

已实现实际surfel可选`receiver_normals`余弦；旧plane分支保留。核查候选去重、原支持、单位法线、plane朝向及fit可见相机朝向零冲突。缓存生产源码绑定manifest中不可变快照，不执行可能并发修改的live cache/fit入口；本地固化生产nlerp，CPU边界自检通过。

旧plane `validation_v1`首次CPU检查错误拒绝合法快照，保留CPU_FAILED；修正后`validation_v1_retry1`精确执行exit0但数值FAILED。bias半/中/双held删失MSE为0.02163027/0.00694637/0.01913803，constant为0.01039043，两端退化211.39%/175.51%，且三初值一致性原本失败，绝不导出。

实际接收点`validation_v3_surfel`精确执行exit0，使用cache_v2_surfel/fit_v3_surfel selected3及真实法线。半/中/双held删失MSE为0.00764773/0.00423650/0.00769719；虽均优于constant0.01031050，两端退化80.52%/81.69%，超原25%门槛，最终FAILED。固定fit1/81/149与held61/71/79中心六图均实际查看，灰洞是原支持，阴影位置/形状仍有残差、饱和删失影响显著。实际逐图SHA与失败观察见输出`manual_review.json`，禁止放行灯光资产；internal_validation_passed=false。

I00内部held18在8000与16000完成同协议完整RGB/Alpha及四类可视化。PSNR由32.52508升至32.72565dB，AlphaIoU由0.9978143升至0.9979550；最终16000训练exit0。只用于内部收敛报告，不引入外部GT，也不以8000模型替代预登记最终16000几何。

唯一几何尺度新bias相同五轮重拟合后，`validation_v4_geometry_bias`在GPU2以冻结原验证器执行exit0。半/中/双fit等帧MSE0.005013428470/0.002996438222/0.003027154279，held0.005697366979/0.003805303602/0.003949279514，constantheld0.010310498066。半档退化49.7217%超25%，双档3.78356%通过；其他支持/初值一致性/能量边界检查全通过，最终仍FAILED。原235025fit、36019held有效观测不变，真实法线fit朝向冲突0。六张固定中心图全部实际查看，中央阴影边界仍见局部残差，饱和删失影响广泛。逐图SHA、失败目检收据及精简完整指标在输出manual_review.json/compact_summary.json，不导出。
