# 静态ELDL纯RGB方向光初始化Stage1计划

当前服务器ushas，使用lumimotion-ushas。S0、B00均使用来自albedo的Alpha，不符合本轮约束，排除复用；不读取历史GT指标作筛选。

原始输入仅使用Static ELDL-V3的image前三颜色通道及camera.json。原image的Alpha全不透明，明确丢弃。按图像边界8像素中位RGB、2/65535最大颜色距离与边界4连通性推断背景，内部同色孤岛保留。此为RGB推断轮廓，包含边缘误差，不是真值分割。

117 TRAIN（排除10的倍数及61—79块），18 DEV仅RGB筛选，15 TEST不参与训练。随机100k立方体从头初始化，静态共享几何，形变头置零冻结，关闭时变颜色。沿用原生35k预算，保存10k、20k、35k，法线正则8000后启用，dist权重1000，增密截至20k，reset每3000步，分辨率400×400。

先在output/smoke_test独立200步验证；正式训练输出output/1030-02-LH-static-ELDL-V3-RGB纯净Stage1。只运行Stage1，禁止Stage2。GPU1目前空闲，与已有GPU0进程隔离。

主选依据DEV RGB PSNR/SSIM/LPIPS，法线连续性、模型深度法线一致性、表面覆盖和异常尺度作内部诊断。禁止GT normal/depth/light/material/alpha参与。训练完成后冻结唯一几何，同一模型完成E0—E4，再冻结方法和预测后独立评测GT灯向。

冒烟与完整命令、输入和检查点SHA、渲染及日志在对应实验README和receipt记录。原生保存iteration k仅含k−1次optimizer更新。

执行更新：首轮smoke因TensorBoard bytes路径审计hook错误失败并保留；1030-02重试200步199次更新成功，几何头六项精确为零，最终25273点。正式35k已启动，GPU1训练PID591585，监督576080；后处理pipeline694633等待训练，10k提前DEV检查监督701005。后处理两个代理目检门禁，均在GT读取前。

RGB推断mask预览：output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/rgb_mask_preview.jpg，八视角目检人体与平台完整、外背景干净。

指标解释：原生normal_loss对整图mean(1-rend_normal·surf_normal)，背景零法线仍有常数1。本数据约77%背景产生约0.0154的加性常数；8000启用后total从约0.008升约0.025不能解释为法线退化。DEV内部法线诊断仅用RGB推断前景内缩和模型alpha>0.95的单位法线，不用总normal loss判几何。

GT读取前明确基线主指标作用域：DEV固定RGB推断mask>0.5、不腐蚀、不依预测alpha缩分母的前景PSNR；另报全图PSNR/SSIM/LPIPS。前景PSNR差0.1dB内才依内部GS/模型深度法线cos、覆盖、全图LPIPS破同值。10k原30.446为全图PSNR，原产物保留，补前景指标重渲染于early_dev_10000_foreground；不将背景优势混作前景重建质量。waiting pipeline因规则补齐有序替换为800945，训练PID591585不动。


## 2026-10-31完成与口径更新

正式35000步已完成，实际34999次更新，训练进程返回0，权威收据`model_mlp/rgb_only_stage1.json`为FINISHED，训练日志与10k/20k/35k保存点全部保留。该训练发生于用户允许渲染Alpha之前，纯RGB输入与RGB推断mask处理保持历史事实，不重训。

用户后续明确渲染自带Alpha／轮廓掩膜允许使用；本文最初仅凭Alpha排除S0/B00的历史口径已撤销，禁止项继续包括albedo颜色数值、GT光向／法线／深度／材质／阴影。S0/B00其余谱系完成无GT审计并重纳候选，B00真实逐帧DEV重建指标最优且内部几何目检可用，最终选为方向光初始化基线。B00的DEV18均为训练内，不能把较高指标等同留出泛化优势；此限制写入选择收据。无新Stage2训练。
