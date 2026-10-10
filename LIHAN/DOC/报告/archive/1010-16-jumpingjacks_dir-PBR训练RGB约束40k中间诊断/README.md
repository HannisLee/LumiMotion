
## B10同一40k检查点中间诊断与实际目检（2026-10-10T11:30:39.834769+09:00）

真实40k保存审核child exit0，99827点finite/opacity-scale/NN32冻结，canonical位移max.03547572m/qmax60.00006485度，native pmaxdelta.61573005，RGB global E5.50377035（非GT）；实际加载新RGB资产/源图CPU复算/marker PASS，canonical轴loss.003838724。检查点父exec10940还在等待50k，不提前称父会话退出。

独立40k预览exec86565、15帧GS/Depth/Z exec27161、CPU复算exec21859均真实exit0并关闭。RGB PSNR18.29660847/SSIM.90320820/LPIPS.10866528（foreground10.95702402/bboxSSIM.78457834/bboxLPIPS.22543707）。GS原有向pixel-weighted15.71225125度/覆盖93.93837273%；Depth Normal25.68376882度/覆盖89.29189544%；ExpectedZ.07940795m/MedianZ.08330994m（93.93837%）。missing180=25.67076223度；共同GS+Depth370948/415433、MAE14.73601245/25.68376882度，此中间未生成Mesh，不能当三法线共同口径或正式全七MAE。独立CPU逐数组/帧/聚合和506来源SHA PASS。原GT评分全415433前景保持，无GT标签参与训练/源RGB候选。

实际首中尾对比B09同40k：黑背景正常、人物/平台出现；上衣/裤腿/手臂碎黑斑，脚周与地面黑灰伪形变，当前40k质量FAILED_EARLY。不把较暗的背光帧直接称缺失，GT中帧本身较暗。 首帧GS平台紫蓝、足周杂色和人物大块混色；Depth/Mesh列此时Mesh明确不可用，Depth高频噪声更明显，预测Z/GTZ对比仍有偏差。中间图不作为全量质量通过证据。

另source-only native alpha合成贡献诊断exec37263实际exit0/关闭，三TRAIN0001/0075/0149、0 optimizer/0 GT标签，source RGBconf只贡献自身前景alpha质量38.47498%/28.52117%/39.15403%，既有plane12.06668%/16.42836%/11.84679%。可靠RGB支持29621源opacity中位.012234、mean.253954，数量不等于像素覆盖；近半像素合成仍来自两个约束外的source点。这给出继续加lambda不足的证据，不把子组贡献当正式Normal覆盖或评分子集。

仅source XYZ/原NN全135TRAIN位置稳定/opacity>.05诊断：去掉source feature和轴cos筛选、保留motion-span≤.01，.2m band也只得到16110点（原15187外多923，全部source_static，无新增dynamic）；.05m band12647（原投影重合12051），仅52点重合RGBconf。说明简单去掉轴cos或source静态gate不会解决主要覆盖缺口，不据此改生产plane支持。现有资产/core/所有原点/帧/评分保持，不生成新训练prior/mask。等待本次55k完整20k/15七MAE与最终画质后决定下一迭代，不commit/push/新分支。
