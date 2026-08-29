-- CV3s 0823 后实验与双法线真值审计
-- 构建器中的实际数据快照为同目录 analysis_snapshot.json。
-- 以下查询对应报告组件使用的六个规范化数据集；加载快照时将
-- datasets 下的同名数组注册为同名只读视图。

-- 审计范围与最佳指标卡
SELECT *
FROM summary_cards
WHERE scope = '0823后审计';

-- 双法线分组柱形图
SELECT experiment, independent_mean_deg, gs_mean_deg
FROM normal_comparison
ORDER BY experiment;

-- 六条正式实验总览
SELECT *
FROM experiment_overview
ORDER BY experiment;

-- 11 组法线源详细指标
SELECT *
FROM normal_detail
ORDER BY mean_deg ASC;

-- 训练损失配置
SELECT *
FROM loss_config
ORDER BY experiment;

-- 14 份原始文档与 1 份本轮新增记录
SELECT *
FROM document_audit
ORDER BY document;
