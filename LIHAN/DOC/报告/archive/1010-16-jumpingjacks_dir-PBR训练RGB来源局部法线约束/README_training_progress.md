
## B10完整训练已实际启动（2026-10-10T11:12:19.216698+09:00）

原35k→55k独立20k，TRAIN PID306459/ticks10865815，exec4419实际仍运行；11:11:25.974063 JST启动，warmup2000/ramp1000、lambda2训练RGB轴约束，当前无正式MAE/质量结论。检查点独立共享加载helper先在smoke35020真实exit0核验，无GT标签、0 optimizer、29621支持/全99827点。检查点审计等待exec10940/后处理等待exec65390，精确进程信息见process_handles.json；保存40k/50k/55k后检查，训练完成后9步原全量渲染/七MAE自动顺序执行，人工质量单独验收。源476核心和双snapshot冻结。

训练准备helper首attempt在执行前因字符串引号嵌套语法错误exit1，未创建正式目录/模型、未启动训练；随后shell尝试不存在retry文件exit2，原日志保留。修正helper独立retry实际exit0，正式目录只创建一次，没有训练重启或改生产代码。全部准备脚本、失败JSON/log保留smoke1010-18。

### B10启动只读核对与当前进度（2026-10-10T11:13:35.904698+09:00）

audit_launch实际exit0，配置相比B09只有运行路径/Preset和新增3字段（source_local=True、RGB资产路径、lambda2）共7项差异，原warmup2000/概率gate/lr/源plane15187/全部99827点/ownAlpha150/冻结NN32/direction表byte一致。输入/资产SHA正常，source RGB约束支持29621；0诊断optimizer、0 GT标签读取。真实train PID306459仍运行，已完成1395/20000步，源码476 SHA锁定；exec4419/10940/65390均在运行中，不能提前填exit0。正式七MAE/前后景质量/7度均PENDING。

## B10训练曲线与RGB约束生命周期只读证据（2026-10-10T11:18:58.872501+09:00）

两个诊断均实际exit0/关闭（curve exec27142、RGB lifecycle exec92303），固定事件字节快照SHA，0 optimizer、0 GT标签读取。实际查看两张图：warmup2000 B09/B10曲线重合，逐step PhotoL1最大差2.68e−7/均差2.39e−9，total最大3.13e−7；2001–2464 RGB L1均差−.00016006，新增total均差+.03294333，total含附加RGB loss不能直接比较成画质退化。全局RGB拟合强度几乎一致，此说明新几何phase前初始外观保持，不当最终MAE。

RGB轴约束首记录37100（2100步）loss .0992467552，39400（4400步）.0038797222，lambda2*ramp后当前 .0077594444；实际在优化但不是GT normal误差。29621 source RGB候选相对原source预测轴median14.7640度/P9045.1578/P9983.1069，1600点大于60度。既有canonical60度的独立轴理想loss下界.00208803，只是轴几何条件、非GT误差或完整photo冲突最优值；目前无证据应放开cap或增大lambda。confidence源q norm中位1.029181/P991.804685/max3.592599，全99827中位.973277/P991.674602/max3.686656；不支持之前泛化为大raw norm导致学习失效，本轮保留原raw q Adam。

三组diag source/事件SHA保留inspection/early_training与rgb_axis_lifecycle。图和轴分布只读，不增加训练teacher、不筛原评分/点/帧。相对原GS normal <7度仍等待55k全部15七MAE，当前不commit/push/新分支。

## B10同一40k检查点中间诊断与实际目检（2026-10-10T11:30:39.834769+09:00）

真实40k保存审核child exit0，99827点finite/opacity-scale/NN32冻结，canonical位移max.03547572m/qmax60.00006485度，native pmaxdelta.61573005，RGB global E5.50377035（非GT）；实际加载新RGB资产/源图CPU复算/marker PASS，canonical轴loss.003838724。检查点父exec10940还在等待50k，不提前称父会话退出。

独立40k预览exec86565、15帧GS/Depth/Z exec27161、CPU复算exec21859均真实exit0并关闭。RGB PSNR18.29660847/SSIM.90320820/LPIPS.10866528（foreground10.95702402/bboxSSIM.78457834/bboxLPIPS.22543707）。GS原有向pixel-weighted15.71225125度/覆盖93.93837273%；Depth Normal25.68376882度/覆盖89.29189544%；ExpectedZ.07940795m/MedianZ.08330994m（93.93837%）。missing180=25.67076223度；共同GS+Depth370948/415433、MAE14.73601245/25.68376882度，此中间未生成Mesh，不能当三法线共同口径或正式全七MAE。独立CPU逐数组/帧/聚合和506来源SHA PASS。原GT评分全415433前景保持，无GT标签参与训练/源RGB候选。

实际首中尾对比B09同40k：黑背景正常、人物/平台出现；上衣/裤腿/手臂碎黑斑，脚周与地面黑灰伪形变，当前40k质量FAILED_EARLY。不把较暗的背光帧直接称缺失，GT中帧本身较暗。 首帧GS平台紫蓝、足周杂色和人物大块混色；Depth/Mesh列此时Mesh明确不可用，Depth高频噪声更明显，预测Z/GTZ对比仍有偏差。中间图不作为全量质量通过证据。

另source-only native alpha合成贡献诊断exec37263实际exit0/关闭，三TRAIN0001/0075/0149、0 optimizer/0 GT标签，source RGBconf只贡献自身前景alpha质量38.47498%/28.52117%/39.15403%，既有plane12.06668%/16.42836%/11.84679%。可靠RGB支持29621源opacity中位.012234、mean.253954，数量不等于像素覆盖；近半像素合成仍来自两个约束外的source点。这给出继续加lambda不足的证据，不把子组贡献当正式Normal覆盖或评分子集。

仅source XYZ/原NN全135TRAIN位置稳定/opacity>.05诊断：去掉source feature和轴cos筛选、保留motion-span≤.01，.2m band也只得到16110点（原15187外多923，全部source_static，无新增dynamic）；.05m band12647（原投影重合12051），仅52点重合RGBconf。说明简单去掉轴cos或source静态gate不会解决主要覆盖缺口，不据此改生产plane支持。现有资产/core/所有原点/帧/评分保持，不生成新训练prior/mask。等待本次55k完整20k/15七MAE与最终画质后决定下一迭代，不commit/push/新分支。
