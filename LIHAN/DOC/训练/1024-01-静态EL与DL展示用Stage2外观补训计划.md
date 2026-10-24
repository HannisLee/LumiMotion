# 静态EL与DL展示用Stage2外观补训计划

用户授权：补齐展示表中的法线MAE、PSNR、SSIM、Albedo MAE，缺少时可补训LumiMotion Stage2。服务器ushas，环境lumimotion-ushas。已有EL Stage2曾失败，F-T04未完成Stage2。

两个父分别为EL-V2原生Stage1@35000与DL-V2 F-T04@16500。原生PBR Stage2均新增20000循环，按原入口每4步更新且终点先保存的逻辑，真实优化器更新均4999次；不把循环数称优化器更新数。相机400×400，135训练/15测试，512采样，expected-depth，原损失与原默认优化参数，几何位置/旋转/尺度不进入优化器，opacity和材质颜色分支沿用原生设置。两组照明不同，不能解释为同数据单因素消融。

EL学习环境光。DL沿用已知方向；光强初始化为F-T04训练RGB拟合的6.5890622139，不取GT能量，Stage2继续学习全局光强倍率。GT法线和Albedo仅离线评测；RGB训练输入由原始RGBA转换。测试方向仅用于已知光条件的离线渲染。不会把Stage1法线与Stage2外观拼成同一checkpoint的一行。

先24循环工程冒烟，输出output/smoke_test/1024-01-staticEL-展示Stage2冒烟和1024-02-staticDL-展示Stage2冒烟。正式输出output/1024-03-staticEL-展示Stage2补训、output/1024-04-staticDL-展示Stage2补训。总控output/1024-02-静态ELDL-展示数值补全总控。启动脚本LIHAN/scripts/ppt_metrics_stage2_job.py保存实际argv、环境、PID、parent SHA和日志。独立CUDA缓存，使用已有sm86兼容nvdiffrast，不改共享安装。每卡一组。

评测统一15测试帧浮点渲染，PSNR/SSIM全图每帧均值，补充前景PSNR；Albedo线性RGB、固定GT腐蚀前景、无GT缩放拟合，记录有效像素与coverage。法线先保留历史三项，Stage2若另评则独立行。渲染与失败日志保留，不按TEST选择中间点或择优改写。最终表格/PDF/PNG/CSV及展示更新存根目录exports。

## EL输入纠正

冒烟离线RGB检查发现原EL图像Alpha全图有效，直接转换改变了历史基线的黑底/前景定义；该EL冒烟与首轮正式运行不作为比较结果，终止本次自有EL训练并完整保留现场。EL重试严格复用原EL基线data_view（历史采用GT Albedo文件的Alpha作轮廓，未读取其RGB作训练监督），保持原RGB和掩膜；DL仍使用原始RGBA。EL正式新目录追加重试1。此轮不掩盖两组历史掩膜来源差异。

## 实际启动与工程验收

EL重试与DL冒烟均完成24循环、5次优化器更新，训练/15帧RGB/15帧Albedo均exit0。正式EL重试1在GPU1、DL在GPU2，均从原父重新开始，未沿冒烟权重继续。当前正式状态以各state.json为准；冒烟指标不进入最终对比表。数值表初稿已生成至exports/1024-02-静态EL与DL-数值对比，外观栏待正式终点。

输入配对审计：全部150帧400×400 Alpha>0.5轮廓逐像素一致，相机矩阵最大绝对差3.7052e-6。两组掩膜文件来源不同但最终轮廓一致；RGB照明不同。审计位于总控paired_camera_mask_audit.json，核心源码SHA在scientific_code_manifest.json。
