-- 报告图表使用的已复核快照。
-- 原始来源是两个数据集 iteration 10001/10500 的 normal_metrics.json，
-- 以及对应 PLY、deform、photometric 和 light checkpoint 的差分审计。
-- 2026-08-17 再审补充：
--   CV3 directional oracle: irradiance 5.5043498758, foreground PSNR 16.31834 dB；
--   CV3L directional oracle: irradiance 7.8434866773, foreground PSNR 16.52403 dB。
-- 来源：output/smoke_test/0817-06-CV3-directional-oracle-recheck/calibration.json
--       output/smoke_test/0817-07-CV3L-directional-oracle-recheck/calibration.json
-- 方向光相对逐点近场方向误差是在 iteration 10001 GS 点上按 frame 1/60 复核：
--   CV3 mean 6.19°–7.58°, P95 10.46°–12.89°, max 14.39°；
--   CV3L mean 6.27°–7.65°, P95 10.41°–12.74°, max 19.74°。

WITH normal_metrics(dataset_metric, iteration_10001_deg, iteration_10500_deg) AS (
    VALUES
        ('CV3 · mean',   23.719318, 27.245435),
        ('CV3 · median', 15.652022, 20.130081),
        ('CV3 · P95',    67.631985, 69.469210),
        ('CV3L · mean',  25.385322, 28.147734),
        ('CV3L · median',19.017694, 24.230737),
        ('CV3L · P95',   68.900562, 68.490627)
)
SELECT dataset_metric, iteration_10001_deg, iteration_10500_deg
FROM normal_metrics;
