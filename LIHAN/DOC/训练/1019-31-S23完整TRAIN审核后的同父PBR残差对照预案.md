# S23完整TRAIN审核后的同父PBR残差对照预案

2026-10-19，ushas／lumimotion-ushas。在新Stage3优化前登记；当前S23仍运行，本文不代表已选择新父或启动训练。

依据既有S22父的完整135TRAIN对照：A36纯环境PBR的终点PSNR30.57516、前景删失.05149043，主要DL基础A37为35.54000/.02483464；加入受限小残差A38为35.54741/.02480243。因此继续采用已验证的环境原生PBR＋一次继承主要DL，比较额外残差0/.1，不再扫描基础权重/分位/采样/光强。小残差收益很小，不能称显著；保留此消融是为了完整交付。

父资产审核必须发生在读取新S23 TEST数字或图像用于任何选择之前。待S23固定8100真实正常退出、全部节点与V10来源冻结，并完成0/4050/8100全135TRAIN后，检查内部GS/Expected一致性是否低于同设置S22的9.0824207871°，报告Alpha覆盖与IoU变化、前景/主体/亮区误差，以及三个TRAIN人体腿足鞋是否连续。仅据该TRAIN证据决定是否采用S23作为外观父，写独立parent_train_only收据，记录当时未读取新TEST结果。若TRAIN证据不支持，暂不启动本文的新Stage3，继续TRAIN机制诊断，不按GT寻找参数。

通过父审核时，两组从同一S23@8100独立转换，不载入父优化器，不从旧A37/A38材质续训。A39为主要DL基础、额外残差lambda_residual=0；A40同基础、小残差lambda_residual=.1。输出分别output/1019-41-LH-static-jumpingjacks-ELDL-V3-A39强法线父主要DL基础、output/1019-42-LH-static-jumpingjacks-ELDL-V3-A40强法线父主要DL小残差，GPU1/2各一训练；显存允许可与只读评测共存，每卡最多3训练。

固定V10源码、相同135TRAIN相机/16位颜色/原Alpha/饱和删失/共同统计与方向资产；继承lambda_mix=.5和normal_weight=.5，但Stage3不计算几何一致性损失，元数据仅记真实父配置。原生PBR环境16×32 exp、完整BRDF和SH间接项、256半球光线、每步2048采样像素、全BVH、native_half=.5 Expected+.5 Median、depth_ratio1、normal_existing_tmin起点偏置9.9785986mm保持。

基础=(1-.5)×原生环境PBR+.5×完整DL，完整DL按rho/pi×2π×V×max(n·L,0)，主要DL只加一次；总色=基础+lambda_residual×(.25 sigmoid(raw_amp))×完整DL，初始amp=.01、非负上限.25，不新增强度或逐像素门控。环境亮度初始化复用TRAIN均值/主体统计除以平均线性反照率，范围[.05,4]；优化约束在初值.5～2倍，anchor .01；残差幅度anchor .01。完整RGB/Alpha及base辅助.25、res辅助.1、Alpha .1保持。

每组固定新增4050真实更新、累计47149，保存0/1350/2700/4050。albedo/roughness lr2e-4，environment lr1e-3，A40 residual_amplitude lr1e-3；全部线性衰减至.1倍。四几何/opacity、完整MLP、sharedSH、方向全部冻结，无增密/裁剪/opacity重置。两组种子0及相机序列相同；CUDA并行累加不保证训练后逐位一致，冻结几何和MLP必须逐位一致。

正式前分别在output/smoke_test/1019-60-ELDL-V3-强法线父主要DL基础门禁、1019-61-ELDL-V3-强法线父主要DL小残差门禁做20步，保存0/4/8/20。检查真实父记录继承、四几何/MLP/SH/方向冻结、启用材质组实际变化、无目标推理、native混合深度、实际PBR起点、独立环境与主要DL参考、两层加和、重载；实际查看三个TRAIN四类与全部物理分量。通过才启动正式4050预算。

固定终点先完整135TRAIN四节点，再同一最终模型隔离15TEST全几何/覆盖/固定分母距离、RGB/亮区和四类150帧视频及分量；至少小残差模型完成三个固定TRAIN相机时序。用户唯一数值门槛仍为三类法线严格低于ELV2精确原值，不增加RGB或覆盖阈值；TEST不用于预算、停止、初始化、权重或父选择。不能把两不同几何父的差异归因残差。所有失败资产、checkpoint、渲染、统计与日志保留。完整实际命令/PID/GPU/来源SHA和目检范围写各README；未经执行不得称PASS。
