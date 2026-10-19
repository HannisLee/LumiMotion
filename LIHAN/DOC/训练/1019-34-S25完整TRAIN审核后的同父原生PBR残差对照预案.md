# S25完整TRAIN审核后的同父原生PBR残差对照预案

2026-10-19，ushas／lumimotion-ushas。S25仍按已登记12150固定预算运行；本文在任何新Stage3优化前登记，不代表已选父或启动训练。仅当完整TRAIN终点和实际目检支持时执行，不根据S25新TEST数值或图像决定父、预算或参数。

现有同父A39/A40完整135TRAIN显示，环境原生PBR＋一次继承主要DL基础的终点PSNR35.48815703/35.49497594，前景删失误差.02498913/.02495768。额外小残差增益很小，不称显著；维持原0/.1消融以交付同一新几何父的外观比较，不新扫分位、权重、采样或灯向。

父审核：S25固定12150真实正常退出，节点/源码/资产冻结，完整135TRAIN0/8100/12150结果齐全，独立重载通过；实际查看三个TRAIN的RGB/Alpha/Normal/Separation及物理分量。检查12150内部GS/Expected一致性是否低于S23同设置的6.531038976598669°，同时如实报告原Alpha分母覆盖、IoU、前景/主体/高亮误差与人体腿足鞋完整性及光度权衡，不添加最终RGB或覆盖数值门槛。仅依据这些TRAIN证据决定是否采用，写独立train_only父选择收据及证据SHA、记录当时尚未读取新S25 TEST数字/图片。若不支持，保留S25负结果，本文不启动任何新PBR训练，不按GT挑8100代替12150。

通过时：A41与A42同从S25@12150转换，几何父完全相同，不加载S25 Adam或旧A39/A40材质。A41额外lambda_residual=0，A42=.1；输出分别`output/1019-48-LH-static-jumpingjacks-ELDL-V3-A41追加预算父主要DL基础`、`output/1019-49-LH-static-jumpingjacks-ELDL-V3-A42追加预算父主要DL小残差`，GPU1/2各一训练，每卡显存允许最多3训练。

固定V12源码与相同135TRAIN、原16位颜色/原Alpha、黑背景、线性饱和删失、共同亮度与方向资产。继承lambda_mix=.5、normal_weight=.5、normal_protection=warmup与joint_polish_steps=4050父配置；这些几何/预算元数据在Stage3不触发几何loss或追加训练。Stage3自己的schedule_horizon与iterations仍各4050（冒烟20），必须实际检查不能误继承父12150步预算。

原生环境PBR16×32 exp、BRDF与SH间接项、256半球光线、每步2048像素、全BVH、native_half=.5Expected+.5Median、depth_ratio1、normal_existing_tmin起点偏置9.9785986mm均沿用已验证接口。基础=(1-.5)×原生环境PBR+.5×完整DL；DL=rho/pi×2π×V×max(n·L,0)，主要DL只加一次。总色=基础+lambda_residual×(.25sigmoid(raw_amp))×完整DL，初始amp=.01、上限.25，非负，无新光强或逐像素目标门控。

环境初始化沿用TRAIN均值/平均线性rho、范围[.05,4]；环境优化限制初值.5–2倍、anchor.01，残差幅度anchor.01。完整RGB/base辅助.25/res辅助.1/Alpha.1不改。albedo/roughness lr2e-4、environment1e-3、A42 residual_amplitude1e-3，全部按4050线性衰减至.1倍；四几何与opacity、完整MLP、共享SH、方向全部冻结，不增密/裁剪/reset。

两臂各固定4050新增真实更新，累计51199，保存0/1350/2700/4050。种子0和相机序列相同，跨GPU浮点数不保证训练后逐位相同；冻结状态必须同父逐位相同。冒烟先分别在`output/smoke_test/1019-64-ELDL-V3-追加预算父主要DL基础门禁`、`1019-65-ELDL-V3-追加预算父主要DL小残差门禁`运行20真实步、保存0/4/8/20；检查实际父12150和预算元数据继承、Stage3实际20步、四几何/完整32MLP/全部lighting/未授权PLY字段冻结、优化器范围及启用组实际变化、重载/无目标、native_half、实际PBR起点、独立环境/主要DL参考与两层加和。实际查看三TRAIN四类和全部物理分量后才正式启动4050，不能跳过新父门禁。

固定4050终点先完整135TRAIN四节点及实际目检，再同一最终模型隔离15TEST三法线/三深度/人体腿足鞋平台分区覆盖/固定分母距离比例、RGB与暗中亮表现、四类150帧及全部分量；A42至少完成三个固定TRAIN相机的时序四类。唯一最终数值验收仍是三法线同时严格低于ELV2原精度，不能拼接模型、节点或更改统计口径，不额外发明RGB/覆盖门槛。失败checkpoint、渲染、统计及日志全部保留，完整实际命令、source/model/iteration、指标、代表图视频、实际四类/分量目检和PASS/FAILED写入各README。未执行项目如实待评，本文不预先宣称完成。
