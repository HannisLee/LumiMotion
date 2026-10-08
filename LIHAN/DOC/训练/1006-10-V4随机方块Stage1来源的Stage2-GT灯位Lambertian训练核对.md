# V4随机方块Stage1来源的GT Lambertian Stage2执行结果

用户已授权启动、全量MAE评测与四组报告；当前4/4 COMPLETED，流程PASS。

| 数据集 | 阶段 | Albedo MAE | bMAE | Normal MAE° | Depth MAE | SSIM↑ | 前景SSIM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV4 | SH Stage1 35k | 0.192875 | 0.129177 | 7.4027 | 0.026094 | 0.999192 | 0.995041 |
| CV4 | 原版IR Stage2 55k | 0.228857 | 0.131870 | 7.9667 | 0.030203 | 0.985574 | 0.907423 |
| CV4 | GT Lambertian 55k | 0.138418 | 0.088483 | 7.4027 | 0.026094 | 0.973556 | 0.830631 |
| CV4L | SH Stage1 35k | 0.177255 | 0.114688 | 7.5799 | 0.027284 | 0.999261 | 0.995261 |
| CV4L | 原版IR Stage2 55k | 0.198856 | 0.117014 | 8.0912 | 0.031638 | 0.985740 | 0.909815 |
| CV4L | GT Lambertian 55k | 0.137338 | 0.083037 | 7.5799 | 0.027284 | 0.976506 | 0.850182 |
| CV4D | SH Stage1 35k | 0.158172 | 0.118708 | 11.5613 | 0.102184 | 0.997447 | 0.989456 |
| CV4D | 原版IR Stage2 55k | 0.161104 | 0.121246 | 11.9397 | 0.108388 | 0.975469 | 0.877510 |
| CV4D | GT Lambertian 55k | 0.136366 | 0.101979 | 11.5613 | 0.102184 | 0.961797 | 0.808305 |
| CV4DL | SH Stage1 35k | 0.138899 | 0.099402 | 12.4948 | 0.087812 | 0.995352 | 0.982812 |
| CV4DL | 原版IR Stage2 55k | 0.122299 | 0.091330 | 12.8886 | 0.093827 | 0.975389 | 0.879943 |
| CV4DL | GT Lambertian 55k | 0.120561 | 0.085033 | 12.4948 | 0.087812 | 0.964946 | 0.828440 |

## 已执行统一设置


| 项目 | 设置 |
| --- | --- |
| 执行环境 | venus / lumimotion，当前项目1001-stage2-perlight分支 |
| 模式 / preset / 光照 | photometric_lambertian / lambertian_gt / gt_directional |
| 步数 | 35000→55000，新增20,000次Adam更新 |
| 数据 | 原实验data_view；105 train / 15 test；测试8/16/…/120；time 0–1 |
| 分辨率 / 背景 | resolution=2，640×360，黑背景，真实RGBA |
| 材质 | 每Gaussian三通道sigmoid albedo，跨时间共享；初始化为Stage1 SH DC颜色，LR=0.001 |
| 冻结 | 几何、尺度、旋转、opacity、SH、完整形变网络和法线；无增点/裁剪 |
| 动态 | 冻结网络仍按每帧真实time计算形变；Stage1颜色形变输出不参与Lambertian着色 |
| 光照方向 | GT light_pos_world减本组Stage1 canonical点云中心再归一化，方向表固定 |
| 光强 | 四组共用 **6.030241533058244**；参考1006-01的105训练RGB标定，显式固定，不各组拟合 |
| 损失 | 0.8前景L1 + 0.2遮罩图DSSIM + 0.001初始albedo先验；光照平滑0 |
| 监督 | GT alpha>0.5作为RGB mask；无Alpha/normal/depth/albedo真值训练监督 |
| 保存 | 初始化35000、40000、50000、55000完整状态 |
| 测试 | 离线render_stage2 --task all；test_iterations不触发此训练器在线测试 |
| 实际调度 | 最多2组并行，06/07后08/09；全程记录GPU显存 |
| 冒烟 | 本次执行独立200步35000→35200及完整渲染；通过后正式重新从原Stage1初始化 |


## 实际GT光照含义


沿用现有GT灯位方向近似实现。元数据标记type=point但source_type=AREA、energy/intensity=1800，现有管线仅使用其世界灯位形成场景中心方向，光强统一为6.030241533058244，参考1006-01的训练图标定并由四组共用。屏幕空间着色为线性albedo × 标定光强 × max(n·l,0) / π，再合成alpha并转sRGB。**它是GT灯位的方向近似，不是完整GT面积光模型**；没有面积光积分、逐点距离衰减、投射阴影、镜面或间接光。

## 共用固定光强标定结果


用户要求四组同场景共用一个光强，现统一设为 **6.030241533058244**；四组正式与冒烟命令均使用显式正数，禁止再次各组自动标定。

- 参考：1006-01 only_clothV4 的Stage1 iter35000（43,140个Gaussian）、105训练帧RGB、初始SH DC材质和冻结几何法线；分辨率640×360。
- 四份GT灯光元数据（含灯位/颜色/强度/面积）与105/15划分完全一致；共用数值按用户要求标定并固定，方向仍按各组现有GT方向实现计算。
- 有效像素累计：1,804,756；掩码为GT alpha>0.5、预测alpha>0.9、单位光强shading>0.1、GT各RGB通道<0.98；在线性RGB拟合标量，不使用测试帧或GT albedo。
- 执行：2026-10-06T16:43:04.408119+09:00 → 2026-10-06T16:43:17.851992+09:00（Asia/Tokyo），成功执行约13.44秒；模型来源哈希和参数冻结校验通过，优化步数0，未生成训练checkpoint。
- [完整标定记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/inspection/shared_light_calibration.json)（每组inspection中保留相同副本）；[成功日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/logs/shared_light_calibration_retry1.log)；[标定脚本](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/commands/calibrate_shared_light.py)。首次记录保存的路径处理失败日志保留于 logs/shared_light_calibration.log，修复后成功重跑。
- 数值属于现有Lambertian管线的RGB/SH代理材质尺度标定，不作为Blender AREA能量1800到物理辐照度的直接转换。四组checkpoint及stage2_config.json已保存固定值；显式正数不会触发自动标定或生成model_mlp/calibration.json。


## 最终报告与训练记录

- [CV4评测报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-06-CV4-固定共用光强GT-Lambertian-Stage2评测报告.md)。
- [CV4L评测报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-07-CV4L-固定共用光强GT-Lambertian-Stage2评测报告.md)。
- [CV4D评测报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-08-CV4D-固定共用光强GT-Lambertian-Stage2评测报告.md)。
- [CV4DL评测报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-09-CV4DL-固定共用光强GT-Lambertian-Stage2评测报告.md)。

[执行汇总](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/README.md)；[完整核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/final_verification.json)；[四组报告汇总](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-10-V4四组固定共用光强GT-Lambertian-Stage2评测汇总.md)。

本轮只优化材质，几何保持冻结，不修复Stage1几何缺陷。实际命令、checkpoint、日志、指标、代表图与PASS结论已逐组回填到1006-06～09训练记录及README；1006-01～04保留历史结果并添加后续报告入口。
