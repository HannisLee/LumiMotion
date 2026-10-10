
## B10训练曲线与RGB约束生命周期只读证据（2026-10-10T11:18:58.872501+09:00）

两个诊断均实际exit0/关闭（curve exec27142、RGB lifecycle exec92303），固定事件字节快照SHA，0 optimizer、0 GT标签读取。实际查看两张图：warmup2000 B09/B10曲线重合，逐step PhotoL1最大差2.68e−7/均差2.39e−9，total最大3.13e−7；2001–2464 RGB L1均差−.00016006，新增total均差+.03294333，total含附加RGB loss不能直接比较成画质退化。全局RGB拟合强度几乎一致，此说明新几何phase前初始外观保持，不当最终MAE。

RGB轴约束首记录37100（2100步）loss .0992467552，39400（4400步）.0038797222，lambda2*ramp后当前 .0077594444；实际在优化但不是GT normal误差。29621 source RGB候选相对原source预测轴median14.7640度/P9045.1578/P9983.1069，1600点大于60度。既有canonical60度的独立轴理想loss下界.00208803，只是轴几何条件、非GT误差或完整photo冲突最优值；目前无证据应放开cap或增大lambda。confidence源q norm中位1.029181/P991.804685/max3.592599，全99827中位.973277/P991.674602/max3.686656；不支持之前泛化为大raw norm导致学习失效，本轮保留原raw q Adam。

三组diag source/事件SHA保留inspection/early_training与rgb_axis_lifecycle。图和轴分布只读，不增加训练teacher、不筛原评分/点/帧。相对原GS normal <7度仍等待55k全部15七MAE，当前不commit/push/新分支。
