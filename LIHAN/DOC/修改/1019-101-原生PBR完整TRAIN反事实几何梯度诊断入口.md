# 原生PBR完整TRAIN反事实几何梯度诊断入口

2026-10-19，ushas／lumimotion-ushas。新增总控diagnose_pbr_geometry_train_v1.py，只读计划1019-34预先指定的小残差主模型A42@4050及其完整135TRAIN诊断，不读取新A41/A42 TEST/GT或ELV2验收元数据，不修改冻结V12/ROOT物理代码，不启动任何新优化。

动机来自TRAIN：同一S25追加4050后内部一致性仅约改善1.08%，光度略退；当前PBR协议固定几何，因而无法由该协议验证几何更新是否有用。原目标允许有依据时另做小学习率几何对照，但必须先检查原生PBR几何导数的可用性及其与Expected先验的竞争；不以GT的哪项差值决定本诊断或参数。

入口必须等A42完整135TRAIN的0/1350/2700/4050收据齐全，核查实际Stage3 optimizer仅材质/环境/幅度，actual geometry lr=0、actual normal loss weight=0。仅在独立内存中开放XYZ/rotation/scale/opacity求导，冻结其余参数；不构建或调用optimizer。保存参数及checkpoint前后SHA，要求内存几何和磁盘checkpoint不变。原训练仍是冻结几何，反事实导数不能冒充当时实际训练的几何更新。

完整135个TRAIN相机分别使用原生training=True的2048像素/256射线一次采样、原support和原删失/完整光度辅助项，加全图0.1Alpha；环境/幅度anchor对几何恒常，不计入所报loss值。采样种子0；报告实际采样前景和normal全图内缩像素数。相同随机状态核验只改变requires_grad不改变首个TRAIN光度/Alpha/mask前向。不能把完整135相机简称为每帧全部像素的光度梯度。

反事实normal沿用父Stage2的0.5权重作机制参照，独立复制pipe.depth_ratio=0获取Expected深度法线；原生PBR的追踪起点仍用native_half，不混淆两个接口。要求四参数组photo/normal预期路径真实连通且有限，不将None补零冒充成功；分别报范数比、余弦、冲突帧比例、跨视角均值，以及每点跨视角相干||sum g||/sum||g||，保存可检查的NPZ聚合梯度。

BVH可见度及MLP输入查询仍按原实现停止梯度，导数不含完整阴影边界/MLP输入Jacobian；不含Adam自适应和曲率。非零或相干导数不能证明有限训练、GT三法线或足鞋精度改善。须等真实诊断及完整TRAIN/实际目检之后，再决定是否有必要预登记新的有限对照；本文不授权跳过计划去修改当前训练预算或按GT调参。

输出预留`output/1019-50-LH-static-jumpingjacks-ELDL-V3-A42原生PBR几何梯度零更新诊断`，日志同目录，GPU3。当前为诊断准备/等待完整TRAIN，不预先宣称执行PASS或科学通过。四类正式可视化继续按既有模型流程交付，梯度诊断本身不替代图像、视频及最终验收。
