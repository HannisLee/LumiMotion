# 静态JumpingJacks转换与原版两阶段执行

日期：2026-10-10。新增两套150帧静态数据，转换为transfer-static，各135/15，保留原始RGBA、相机与time=frame/150。转换器新增可选--geometry-kind，static模式要求全部骨骼世界矩阵差异≤1e-7，并在manifest记录静态属性；默认dynamic行为保持。time标签改为直接描述来源字段，消除固定150帧假设。训练算法代码使用原版main及既有FOV修复，无新算法变更。

输出17号env、18号dir，19号执行汇总；先独立1100/1120冒烟，再从头35k/55k正式训练，采用六步原版SH/IR及静态NVS；每步失败停止并保留产物。完整命令、输入哈希与实测验证保存在各组README与inspection。静态NVS使用专用黑背景GT视图，固定timestep0001；训练用straight RGBA，由原版loader合成一次。

## 原版相机平移兼容

输入审计发现原版reader的R采用相机局部Y/Z翻转，但T对三轴全部取负，横向平移与Blender定义不一致；新轨迹实际相机中心最大偏差0.101717451884米。新增显式--original-loader-translation参数，在转换矩阵的world_to_camera局部X平移中预先补偿符号。源矩阵保留在source_camera_to_world，原始数据与原版reader代码保持，默认转换行为未改变。新两套数据均显式启用此适配。未适配transforms副本在正式输出inspection/unadapted_transforms_{train,test}.json，输入审计将核对实际loader中心和源中心。

实际验证：两组完整1100/1120冒烟六步、20视频共3000帧解码及实际首中末目检PASS；17/18正式两组已启动，真实seed0默认10万点初始化PASS。直接使用转换器新增CLI参数独立生成150图像与两份transforms，逐文件SHA256/JSON与审计后数据完全一致，证据output/smoke_test/1010-19-static_jumpingjacks-转换复现/verification.json。初始化证据允许同内容复核并保留原文件，点云或既有证据不一致则报错；各组禁止覆盖STARTED与已训练模型。

## 最终执行结果（2026-10-10）

两套独立冒烟与正式六步均完成；正式Stage1 35,000次、Stage2累计55,000次（新增20,000次），2/2流程与产物验收PASS。正式执行起止：2026-10-10T14:51:49.023234+09:00 至 2026-10-10T16:59:26.681785+09:00，两组并行总墙钟7657.7秒（127.63分钟）。全部输出保留在当前项目，原版算法源码SHA256与启动前快照一致。

正式20个视频共3,000帧已完整解码；Stage1共1,200张、材质600张、HDR300张PNG及30套测试视角NVS输出完整，最终35k/55k高斯、形变与环境光checkpoint各属性有限。300个原版reader相机对源相机中心误差最大2.6645e-15米；独立转换复现、默认seed0十万点随机方块初始化核验均PASS。

完整15视角NVS：env PSNR39.241677 / SSIM0.995162 / LPIPS0.004567；dir PSNR20.307147 / SSIM0.921322 / LPIPS0.082773。指标按原版静态NVS与GT Alpha合成的黑背景整图计算。两组基色存在照明残留，dir明显更严重，平台法线斑块和边缘拖影也保留；流程PASS不代表材质恢复质量达标。GT法线/深度/Albedo/Roughness MAE不在本轮结果内。

最终训练报告：[env](../训练/1010-17-static_jumpingjacks_environment_light-原版LumiMotion两阶段训练报告.md)、[dir](../训练/1010-18-static_jumpingjacks_direction_light-原版LumiMotion两阶段训练报告.md)。完整命令、模型、日志、指标、代表图片/视频与实际目检结论已回填；验收证据见[执行汇总](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-19-static_jumpingjacks-原版两阶段执行汇总/README.md)及final_verification.json。
