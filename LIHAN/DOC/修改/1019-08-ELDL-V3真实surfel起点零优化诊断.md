# ELDL-V3 真实surfel起点零优化诊断

2026-10-19（Asia/Tokyo启动日）。新增`scripts/diagnose_eldl_v3_origins.py`，仅读白名单旧cache、fit_v2、I00和TRAIN相机。不改cache/fit，不拟合rho/EL/方向，不迁移旧rho做RGB判定，不导出光照资产。

4096原格点按主平面XY映射高opacity、近面且法线10°内的真实surfel，距离限制为较大原网格轴步长，按格点序号保留首个有效映射，随后限定原fit支持≥10。三档bias下分别比较共同点旧plane origin与实际中心的全BVH透射率，记录映射/位置/V分位数与固定代表图。另核查平面统一法线与有原valid支持的fit相机朝向翻转同号性。

科学范围和固定规则见`LIHAN/DOC/报告/1019-07-ELDL-V3真实surfel起点诊断预登记.md`。无新的通过阈值，internal_validation_passed始终false，实际执行将在下方追加。
