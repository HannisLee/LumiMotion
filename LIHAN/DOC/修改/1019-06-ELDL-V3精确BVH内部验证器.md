# ELDL-V3 精确BVH内部验证器

2026-10-19（Asia/Tokyo启动日；主机日志2026-10-18 UTC）。新增`scripts/validate_eldl_v3.py`，不修改并发的`cache_eldl_v3.py`或`fit_eldl_v3.py`。固定规则见`LIHAN/DOC/报告/1019-05-ELDL-V3精确BVH内部验证预登记.md`。

验证器以fit选定初值、135个预测方向和冻结rho/EL重算真实预测Gaussian BVH的接收平面透射率，bias固定0.5/1/2倍。保留原valid分母和逐通道满码删失，报告fit/held等帧及像素加权MSE、已有constant_held及偏移退化。中心须优于constant；两端相对中心退化≤25%且仍优于constant，支持与三初值门槛同时保留。

入口使用`utils.no_gt_guard.install_no_gt_guard`，仅读白名单缓存、fit结果、I00模型与TRAIN输入。正式脚本不导出灯光资产，`internal_validation_passed=false`；所有数值通过仍待主任务实际目检，不写精度PASS。CPU计划/检查不构建CUDA，精确trace需显式GPU环境并经原生bootstrap加载。

修改与运行验证结果将在下方追加。GPU使用与主任务协调；I00@16000的held18完整验证使用独立目录，与8000步一致，不能阻塞GPU2缓存/拟合。
