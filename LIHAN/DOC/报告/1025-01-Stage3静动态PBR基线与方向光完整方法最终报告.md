# Stage3 静动态实验展示与复现说明

日期：2026-10-25。服务器 ushas，Conda lumimotion-ushas。逻辑 Stage3 对应原生代码 Stage2 的 PBR 外观训练；本次比较环境光与方向光两种照明模块。

| 数据 | 方法 | PSNR(dB) | SSIM | 前景PSNR(dB) |
|---|---|---:|---:|---:|
| 静态DL-V2 | 原生LumiMotion环境PBR | 21.473688 | 0.944001 | 15.035016 |
| 静态DL-V2 | 改进几何＋已知GT方向PBR | 29.424360 | 0.974929 | 23.173601 |
| 动态DL-V2 | 原生LumiMotion环境PBR | 18.528889 | 0.932108 | 12.087656 |
| 动态DL-V2 | 改进几何＋估计方向PBR | 19.934739 | 0.910839 | 13.670354 |

结论：静态已知方向条件下PSNR和SSIM均提升；动态估计方向条件下PSNR提升、SSIM下降，不能宣称全面超过原生。动态足部缺陷仍需独立解决。

同一数据内 Stage3 RGB/Alpha输入、512半球采样、400×400、20000新增循环/4999真实更新、原生PBR损失和混合追踪深度1一致；环境光与方向光属于方法变量，相应光照参数/正则按各入口语义处理。静态完整方法使用已知GT方向；动态使用父TRAIN估计方向及插值，不读GT灯向或GT强度。历史API参数名称gt_directional不表示动态数据读取了GT。动态原始RGBA净化与原生视图132张输入RGB/Alpha逐字节相同，证据在总控directional_input_comparison.json。15固定测试视角全图浮点sRGB逐帧PSNR/SSIM后均值；SSIM为项目11×11、sigma1.5；前景分数另列。NumPy/SciPy独立复算指标，保存float数组，配对GT和Mask逐元素一致。

改进父模型承接纯环境PBR的控制结果：

| 数据 | PSNR(dB) | SSIM | 前景PSNR(dB) |
|---|---:|---:|---:|
| 静态DL-V2 | 18.122988 | 0.928915 | 11.670847 |
| 动态DL-V2 | 17.718670 | 0.893789 | 11.357672 |

这两臂与方向PBR完整方法一起保留。父模型除几何以外还继承材质DC、SH、门控和MLP初始化，Gaussian数量也不同，因此不是只替换法线/几何的严格单因素消融。不能将完整方法收益全部归因于法线。

对齐的是新增外观预算，不是整条训练链的总计算成本；父检查点分别为iteration35000与16500，历史阶段和真实更新规则不同，不能只按iteration比较总计算成本。单场景/单种子实验不声明微小差异的统计显著性。

小参数诊断在读取本轮最终TEST分数前完成：动态方向PBR@20500的三个固定TRAIN比较射线偏移0.01/0.03/0.1/0.2，预先要求SSIM提升至少0.001且PSNR下降不超过0.05dB；所有候选失败，训练与最终推理均保留原生0.1。完整记录见训练计划1025-03和总控train_bias_diagnostic_4000。

静态父为原生S1@35000和历史GS最优F-T04@16500（2.663858°）；动态为原生S1@35000和完整路线预登记主检查点@16500（7.329092°，原生14.545605°）。这些法线来自父模型，不冒充当前Stage3的同checkpoint法线MAE。原生动态父见过135训练时刻、改进父117，且动态门控和颜色头设置不同，属于完整方法对照，不能证明法线单因素因果。

训练source移除了TEST；所有TEST指标均在对应模型完成固定预算后计算，TEST不参与优化或选参。早期冒烟只评分和查看TRAIN，但Scene初始化曾加载含TEST图像的评价视图；输入逐字节一致性审计也读取TEST输入，均未用于梯度、TEST评分或按图选参。后续TRAIN诊断改用无TEST视图。历史TEST已被多轮实验查看，不称新盲测。现有1024 EL44.685147dB/0.997835与DL29.316748dB/0.974770为不同照明/已知光设置，不能混入本表。

展示顺序：先定量与几何控制页，再静态RGB和诊断，再动态RGB、诊断和固定相机时序。局部框使用固定相同坐标，并保留完整图。视频目录包含两种：文件名“15测试视角”为按采集顺序的15个TEST视角，动态同时变化相机和时间；文件名“固定相机0149”为真正固定TRAIN相机的150帧模型时序，25fps、6秒，目标RGB/Alpha缓冲置零，未观测固定视角没有GT评分。两模型固定相机内外参逐元素相同，六类视频均完整程序解码150帧，程序解码不等于实际人工播放全部视频。PNG与PDF可直接插入PPT，HTML离线浏览；CSV可编辑。

## 来源与完整命令

- 静态原生：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-03-staticDL-Stage3原生基线；source=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-02-Stage3-PPT配对总控/static_train_source；parent=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp@35000；model=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-03-staticDL-Stage3原生基线/model_mlp@55000。完整训练/渲染命令及环境在该目录README.md、state.json；train.log和evaluation.log，evaluation/arrays与metrics.json保留。
- 静态改进：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-08-staticDL-Stage3已知方向光；source=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-02-Stage3-PPT配对总控/static_directional_train_source；parent=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp@16500；model=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-08-staticDL-Stage3已知方向光/model_mlp@36500。完整训练/渲染命令及环境在该目录README.md、state.json；train.log和evaluation.log，evaluation/arrays与metrics.json保留。
- 动态原生：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-05-dynamicDL-Stage3原生基线；source=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-02-Stage3-PPT配对总控/dynamic_train_source；parent=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-03-LH-dynamic-jumpingjacks_DL_V2-原生LumiMotion两阶段/model_mlp@35000；model=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-05-dynamicDL-Stage3原生基线/model_mlp@55000。完整训练/渲染命令及环境在该目录README.md、state.json；train.log和evaluation.log，evaluation/arrays与metrics.json保留。
- 动态改进：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-07-dynamicDL-Stage3估计方向光；source=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-02-Stage3-PPT配对总控/dynamic_directional_train_source；parent=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1021-06-LH-dynamic-jumpingjacks_DL_V2-完整路线复现/model_mlp@16500；model=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-07-dynamicDL-Stage3估计方向光/model_mlp@36500。完整训练/渲染命令及环境在该目录README.md、state.json；train.log和evaluation.log，evaluation/arrays与metrics.json保留。
- 静态环境承接：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-04-staticDL-Stage3改进几何；source=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-02-Stage3-PPT配对总控/static_train_source；parent=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp@16500；model=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-04-staticDL-Stage3改进几何/model_mlp@36500。完整训练/渲染命令及环境在该目录README.md、state.json；train.log和evaluation.log，evaluation/arrays与metrics.json保留。
- 动态环境承接：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-06-dynamicDL-Stage3改进几何；source=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-02-Stage3-PPT配对总控/dynamic_train_source；parent=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1021-06-LH-dynamic-jumpingjacks_DL_V2-完整路线复现/model_mlp@16500；model=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1025-06-dynamicDL-Stage3改进几何/model_mlp@36500。完整训练/渲染命令及环境在该目录README.md、state.json；train.log和evaluation.log，evaluation/arrays与metrics.json保留。

## 验收

工程审计和四类实际目检见实验目录 independent_audit.json / manual_review.json。工程完成和数值提升不能替代完整分解质量。最终科学结论见本次报告；没有通过的检查明确写FAILED，全部checkpoint/图片/统计/日志保留。

## 最终实际验收与交付结论

全部六组固定预算训练、15TEST评价、独立冻结/指标复算、视频全解码及中文展示整理完成，交付验收 **PASS**。四类目检与完整几何/材质恢复分开判断；父模型足部缺陷没有被Stage3位置更新修复。

| 实验 | 实际目检结论 | 适用范围 |
|---|---|---|
| 静态原生 | FAILED | 完整静态方向变化外观/材质分离质量；原生基线复现与指标记录有效 |
| 静态改进 | PASS | 已知GT方向静态DL的固定协议外观对比和PPT展示；不含完整三维/GT材质验收 |
| 动态原生 | FAILED | 完整动态物理外观/材质分离和足部质量；原生基线数据仍有效 |
| 动态改进 | FAILED | 完整动态几何恢复和PSNR/SSIM同时超过原生未通过；PSNR提升为有效局部收益 |
| 静态环境承接 | FAILED | 改进父模型接回环境PBR的外观质量，作为完整保留的负对照 |
| 动态环境承接 | FAILED | 改进父模型接回环境PBR的完整动态外观，作为负对照保留 |

PPT主结果建议使用静态已知方向条件：PSNR 21.473688→29.424360dB（+7.950671），SSIM 0.944001→0.974929（+0.030928）。动态估计方向仅部分收益：PSNR 18.528889→19.934739dB（+1.405850），SSIM 0.932108→0.910839（−0.021269）；作为部分成功/失败边界展示，不能宣称全面优于原生或足部恢复。静态GT方向与动态估计方向条件分开；完整父初始化不同，不能将全部增量归因于法线。材质分解本轮只有定性检查，没有GT材质MAE验收。

实际查看范围与逐实验结论保存在各实验manual_review.json，联系表在总控final_review。平台真实彩色棋盘格不单独作为伪影；按同视角GT的亮度、投影与结构判断。所有程序全解码均不冒充人工逐帧播放。

交付目录：/home/han.li/reproduce/LumiMotion-stage2-perlight/Exporter/1025-01-Stage3静动态实验展示。主展示：1025-01-Stage3静动态展示合集.pdf；原图与视频及CSV均在该目录。
