# 已见TRAIN诊断的RGB边界合成修正

评测Agent对完整135原始TRAIN进行了纯CPU核查：原始fractional Alpha共604120像素。原训练Scene使用 `no_bg=True`，即AREA(RGB×Alpha)，既有 `diagnose_eldl_v3_train_stability.py` 没有显式该参数，采用AREA(RGB)。400分辨率目标RGB最大差0.40768、全图平均绝对差0.00174868，99407个前景像素受影响；Alpha与饱和mask逐位相同。收据在总控 `train_rgb_boundary_cpu_audit_v1/receipt.json`。

新增独立 `scripts/diagnose_eldl_v3_train_stability_v2.py`，只将相机读取显式设为no_bg=True，并在summary标注预处理。原V1脚本和所有既有结果保留，不覆盖，不把旧photo重新描述成训练loss复现。所有几何预测、Alpha支持、Expected Z、有向GS变化和固定分母公式不变；新M06/M07使用V2，双方同口径。

原M04/M05的photo是对同一未合成RGB固定目标的诊断，而非原训练目标复现。其光度升降不能直接表述为训练光度项升降；已报告的几何稳定性变化和离线GT评测不受此目标RGB差异影响。充分预算、短长初始化假设和控制变量不变，不按此差异更改物理训练损失、初始化或灯光。
