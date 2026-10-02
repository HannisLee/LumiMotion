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

按顺序执行CPU14项、相机/检查点预检查、GPU冻结/梯度/背景与恢复等价检查、两组冒烟、正式三组训练和同协议评估。显存允许时最多三个训练并行。完整实际命令和日志放在实验目录；不得覆盖已有checkpoint或渲染。

实验README必须记录source/model/iteration、完整训练与渲染命令、日志、指标、代表图片视频和四类目检。PASS只涵盖管线功能完整性和定性可用性；材质真实性或新模型优于PBR需另凭证据判断。失败时全部产物保留。

当前：两组200步冒烟训练及渲染已完成；正式对照等待GPU检查和目检通过后启动。后续结果继续更新此记录。
