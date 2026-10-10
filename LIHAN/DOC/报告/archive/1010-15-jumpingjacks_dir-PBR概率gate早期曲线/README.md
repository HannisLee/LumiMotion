
## B09早期曲线实际快照（2026-10-10T10:16:35.143803+09:00）

exec53315实际exit0并关闭，固定B08/B09事件字节snapshot至3327步，100步均值实际view inspection/early_training/training_curves.png。前2000 warmup photo/total平均delta −2.16e−9/−2.41e−9、最大绝对delta2.68e−7/3.28e−7；geometry阶段1327步photo均值delta −.00135368、total −.00165965，比raw gate下降。相同Direct SUN/native/all-eligible alpha>.9；total包含不同坐标弱anchor，不能替代GT MAE/质量/7°，光强仍仅RGB拟合。0额外优化/无GT标签，生产474核心不改。CSV/原事件字节及metadata来源SHA保留；命令commands/plot_early_training.py，日志logs/plot_early_training.log。
