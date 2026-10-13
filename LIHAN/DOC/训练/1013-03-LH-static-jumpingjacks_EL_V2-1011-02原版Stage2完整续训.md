# 1011-02 原版 Stage2 完整续训

日期：2026-10-13（Asia/Tokyo）；服务器 ushas；环境 lumimotion-ushas；GPU 2（A40）。

目标：从 1011-02 已完成的 Stage1 iteration=35000 接上完整原版 Stage2，到 iteration=55000，共 20000 步。原版 Stage2 入口不恢复材质优化器状态且重新初始化环境光，因此从冻结的 Stage1 35000 启动完整阶段，避免把 37100 冒烟模型作为完整状态续训起点。

独立输出：/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-06-LH-static-jumpingjacks_EL_V2-1011-02原版Stage2续训；model=model_mlp；source=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-02-LH-static-jumpingjacks_EL_V2-原生LumiMotion两阶段/data_view。原始失败目录、冒烟结果均保留。源码使用 /home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1013-05-LH-static-jumpingjacks_EL_V2-原版Stage2兼容修复/original-source，与 original 分支算法一致；使用已修复的 nvdiffrast 与已验收原生光栅缓存。

训练：--load_iter 35000 --iterations 55000 --save_iterations 37000 40000 45000 50000 55000 --test_iterations 55000 --diffuse_sample_num 512 --depth_ratio 0 --resolution 2。原版 2000 步 diffuse 预热，后续 18000 步完整 PBR；不修改训练损失或更新频率。

同时修复 roughness GT 导出端的透明路径轮盘重加权，重导出合法GT，保持原严格范围检查、前景评分阈值、收缩半径和 MAE 定义。先以 37100 冒烟模型验收，再评测完整 55000 模型，并用保存浮点图独立 CPU 复算。

完成标准：训练退出 0，20000 条 loss 有限，检查点完整可重载；原版 PBR NVS 与材质、RGB/Alpha/Normal/分离渲染通过；全 15 测试帧 roughness MAE 有效且独立复算通过。图像质量按实际目检记录，不以运行成功替代材质质量结论。最终命令、指标、图片/视频和 PASS/FAILED 记录到实验 README。


## 完成与验收（2026-10-14 Asia/Tokyo）

最终结论：PASS（完整训练与评测链路）。训练exit=0；20000条loss连续35001–55000且全部有限，Gaussian点数101096，95071点roughness更新，最终参数范围[0.28353898,0.98488318]。35000/37000/40000/45000/50000/55000检查点保留。原版源码一致性检查PASS。

原版insights、materials、NVS均exit=0。四类视频及两类材质视频各完整解码150帧，代表帧已目检；RGB/Alpha/Normal完整，Separation以绿色为主。Albedo仍有阴影残留，roughness地面仍有纹理和阴影状误差。完整目检结论见实验README。

15帧NVS：PSNR=39.83271128，SSIM=0.99627268，LPIPS=0.00352455。严格roughness MAE=0.176275439363，coverage=100%，495065/495065像素有效；原GT独立重建15帧、40个输入哈希与MAE复算PASS。

相同新GT下37100参考MAE=0.170825119557，最终MAE增加0.005450319806（约3.19%）。RGB改善，但roughness精度未改善；PASS指训练完成和指标计算正确，无人为降低阈值或修改GT以降低分数。

完整命令、source/model/iteration、日志、视频及代表图见 [实验README](/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-06-LH-static-jumpingjacks_EL_V2-1011-02原版Stage2续训/README.md)。旧GT、失败评测和启动路径错误日志均保留。
