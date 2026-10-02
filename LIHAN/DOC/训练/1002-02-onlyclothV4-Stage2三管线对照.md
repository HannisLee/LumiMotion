# onlyclothV4 Stage2 三管线对照

日期：2026-10-02；服务器 venus；Conda lumimotion；GPU RTX3090Ti 24GB；代码分支 `1001-stage2-perlight`。

## 来源与计划

只读 Stage1 来源：`/home/lihan/reproduce/LumiMotion-perlight/output/1001-01-onlyclothV4-FOVxy修复重训/model_mlp`，iter35000，44,136个Gaussian；来自原 main 的FOV修复后SH训练。源数据：`/home/lihan/data/LH-data/transfer-static/only_clothV4`，105训练/15测试；GT元数据：`/home/lihan/data/LH-data/static/only_clothV4/lights.json`。

三组均从同一个SH检查点加载，resolution=2、depth_ratio=0、目标累计iter55000。original保留原始IR算法，diffuse_sample_num=512；GT/learned固定几何，20,000步每步优化albedo；learned额外优化105个训练方向。两种Lambertian共用光强6.032453522706779，为仅训练帧RGB/SH初始材质/几何法线标定的标量。AREA近似误差、无阴影/镜面/间接光和光照/相机/运动耦合均需在结论中说明。

正式输出：

- `output/1002-02-onlyclothV4-stage2_original/`
- `output/1002-03-onlyclothV4-stage2_GT/`
- `output/1002-04-onlyclothV4-stage2_learned/`

冒烟输出：`output/smoke_test/1002-02-onlyclothV4-stage2_GT/`、`1002-03-onlyclothV4-stage2_learned/`。各200步，包含15帧评估、120帧RGB/alpha/normal/albedo、72帧固定相机转灯、120帧HDR和6个视频。另保留GT/learned恢复测试目录04/05和首次严格浮点检查失败日志。

## 验收

按顺序执行CPU数值检查、相机/检查点预检查、GPU冻结/梯度/背景与恢复等价检查、两组冒烟、正式三组训练和同协议评估。显存允许时最多三个训练并行。完整实际命令和日志放在实验目录；不得覆盖已有checkpoint或渲染。当前CPU17项通过，120个相机中心最大误差1.43e-6、FOV误差0；原始Stage1/SH renderer/Gaussian/IR renderer四个源码文件与9cd834f字节一致。

实验README必须记录source/model/iteration、完整训练与渲染命令、日志、指标、代表图片视频和四类目检。PASS只涵盖管线功能完整性和定性可用性；材质真实性或新模型优于PBR需另凭证据判断。失败时全部产物保留。

最终：三组正式对照均完成iter55000、同协议15测试帧评估和四类可视化，最终功能验收PASS。原始IR有120帧HDR及5视频；GT/learned各有72帧转灯、120帧HDR及6视频。GT和learned正式结果的120帧alpha/normal逐像素相同，来源SH哈希与冻结指纹不变。两组200步冒烟及两组独立恢复实验也均完整渲染PASS。

| 管线 | PSNR | 前景PSNR | 最终结论 |
| --- | --- | --- | --- |
| 原始IR | 34.53980 | 24.28745 | PASS |
| Lambertian GT | 29.92783 | 19.78163 | PASS |
| Lambertian learned | 32.24665 | 22.09130 | PASS |

训练版本388db4b。渲染阶段另有HDR轴向、resume-only识别和原始HDR配置读取修复，各渲染provenance和代码差异记录执行版本。原始HDR最初与中间配置读取失败都未产生帧，logs/hdr.log、hdr_config_fixed.log和FAILED标记保留；最终hdr_config_fixed_mlp.log通过，RECOVERED与README记录最终状态。其他早期失败输出和测试日志也保留。

完整指标、代表图、恢复证据与解释见 [验收与限制报告](../报告/1002-30-Stage2三管线对照验收与限制.md)。PASS覆盖管线功能完整性，不表示真实albedo或真实光照已经恢复；learned方向平均误差60.048度。
