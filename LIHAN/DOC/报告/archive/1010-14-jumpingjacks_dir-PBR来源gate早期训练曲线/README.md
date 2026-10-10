
## B08早期训练曲线实际快照（2026-10-10T09:16:37.128262+09:00）

固定事件字节快照到3663个Stage2步，100步均值photo L1/total loss及训练RGB拟合正E与B07曲线早期几乎重合。两轮Direct SUN/GGX/BVH/alpha>.9全可用photo一致，B08仅开放输出gate且total含新弱anchor；前2k仍warmup，早期曲线不能替代全15法线MAE或质量结论。实际view inspection/early_training/training_curves.png，原CSV、B07/B08事件字节snapshot、来源SHA与时刻保留metadata.json。作图exec59372实际exit0并关闭，命令commands/plot_early_training.py，日志logs/plot_early_training.log；0额外优化，无GT标签读取，不更改live训练。完整B08仍运行，40k预览与最终七MAE待完成。
