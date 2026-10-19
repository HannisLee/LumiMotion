# 原生PBR完整TRAIN反事实几何梯度结果

2026-10-19，ushas／lumimotion-ushas。


ushas／lumimotion-ushas，GPU3；正式A42@4050，model：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-49-LH-static-jumpingjacks-ELDL-V3-A42追加预算父主要DL小残差/model_mlp；source：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_train_v2；原生256射线/2048像素一次采样×完整135TRAIN，seed0。

完整实际命令：
```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控/snapshots/three_stage_v12_fixed_polish' diagnose_pbr_geometry_train_v1 --model '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-49-LH-static-jumpingjacks-ELDL-V3-A42追加预算父主要DL小残差/model_mlp' --train_diagnostic '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控/polished_parent_pbr_train_endpoint_v12/A42/results' --output '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-50-LH-static-jumpingjacks-ELDL-V3-A42原生PBR几何梯度零更新诊断/results'
```

日志diagnostic.log、queue.log；队列完整命令/真实PID/退出码queue_launch.json和job_state.json；全部结果results/receipt.json与cross_view_gradient_sums.npz。

实际训练optimizer仅albedo/roughness/environment/residual_amplitude，原几何lr0、原normal loss weight0。本诊断只在内存开放四几何求导，候选先验0.5来自父Stage2，不是原Stage3训练项。不构建/调用optimizer；checkpoint及内存几何不变。

首个TRAIN相同采样状态下requires_grad开关的render_linear/Alpha/mask差均0。几何先验用独立Expected depth_ratio0，PBR追踪保留native_half。

|参数|0.5normal/photo范数比逐帧均值|逐帧余弦均值|冲突帧比例|photo点相干均值|
|---|---:|---:|---:|---:|
|xyz|0.479015300|-0.028624667|0.800000000|0.469734521|
|rotation|0.450912838|-0.043877443|0.800000000|0.487816123|
|scaling|0.309817224|-0.008064665|0.525925926|0.542371142|
|opacity|0.212821928|-0.072742881|0.659259259|0.463939519|

photo+0.1Alpha目标均值0.03127282959305578，候选raw normal均值0.026462911883438075；环境/幅度anchor对几何恒常，未计loss值。

完整135视角不等于每帧全部像素光度梯度；photo每视角一次2048采样，Alpha/normal全图内缩。点相干包含少视角支持点，高值不证明多视角或GT准确。BVH与MLP输入停止梯度、未含Adam/曲率，局部导数不证明有限训练收益。

该梯度实验没有新增四类/视频或GT渲染，RGB/Alpha/Normal/Separation目检不适用；正式模型可视化由A42完整流程另行交付。最终诊断执行PASS，科学几何精度未由此证明；optimizer_updates=0，gt_read=false，test_read=false，失败资产保留。
