# 1008-19-训练RGB正向残差L1减权可选Preset

新增默认关闭 photometric_positive_residual_weight=1、photometric_positive_residual_threshold=.03。默认权重1直接执行旧L1浮点顺序；preset lambertian_geometry_area_robust_photo 设.25，仅对训练RGB中 mean(target-pred)>.03 的像素将L1乘.25，判定detach。分母仍为原有效像素数，不重归一化。DSSIM/材质先验/灯光先验/几何损失和normal支持二值mask保持；不新增BRDF、法线或状态资产。显式CLI覆盖preset，旧checkpoint缺字段取原默认。

108 CPU测试PASS，包括旧loss/gradient精确、单项符号/阈值/分母和参数校验。当前核心GPU旧4/16几何前向/loss逐位与反向等价、旧103loss在真实C4-49/L-52前向逐位与反向等价PASS；新C4真实46初始化源参数/材质/灯光/常规render逐位、新Adam、7训练RGB仅L1变化、全部geometry项和normal mask逐位、GeometryRun/AlbedoRun、冻结源PASS。全部为0优化诊断，不能作为训练源。

真实40/20+20/独立40、原参数/Adam/13视角界限与完整实际Trainer恢复PASS，新C4正式已经启动。代码 arguments/stage2.py、scripts/loss_stage2.py、tests/test_stage2_robust_photo.py；no_gt_normal_training_review.json保存当前108和旧103实际L正式启动SHA，活动L进程已加载的旧代码不受本次修改影响。GT normal/depth/material不训练/初始化/伪目标/资格，仅独立内部11离线评测。


实际启动验收：原XYZ/rotation/scale恢复最大差1.2154924e-6/2.8684735e-6/2.0936131e-5满足预声明2e-5/5e-5/1e-4；13视角render最大.000183105469<.001。严格1e-6 FAILED保留，独立重复覆盖PASS，不能称训练轨迹逐位一致；实际Trainer参数/Adam/图/相机/RNG精确加载PASS，初始化资产不可用仍恢复。正式CV4_z26初始参数/材质/灯光与真实46逐位、独立新Adam、源冻结/config/current108实际启动SHA均PASS。README完整实际CLI/source/model/iteration/日志及待渲染产物已验收；formal47/48/49质量和自身四类/HDR/Mesh仍PENDING，不能当达到5°。只有只读初始审计等待器在formal前增加保存mtime3秒避免读半写checkpoint，旧等待日志保留，训练/核心/界限不变。


实际49k完整结果：C4 GS6.527267096/DN9.011653893/Mesh8.339013335°，三覆盖.998150667460/.956240965904/.983543066066；布料11.483382170°比控制11.477634078°略差。GS原有效像素均值微降.007576602°，但缺失计罚6.848348305°比控制6.843560905°差.004787400°，不推广默认。自己的49四类常规+v2三帧、HDR3/linear/video和Mesh0058完整验收COMPLETED、执行PASS、法线目标FAILED_NORMAL_TARGET；全部产物保留。
