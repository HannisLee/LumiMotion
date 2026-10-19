# 冻结PBR的TRAIN误差与遮挡分解

2026-10-19，ushas／lumimotion-ushas。上一轮完成原生PBR同父对照，但完整135TRAIN已表明主体/高亮误差比Stage2退化；因此先做固定状态机制诊断，不根据TEST/GT换参数或延长训练。

新增scripts/diagnose_three_stage_pbr_error.py，仅加载A32@4050正式基础组，验证已保存native_half模式和全部冻结前向SHA；只读取已见TRAIN帧1/75/149。保存原生RGB/Alpha/Normal/Separation、材质、roughness、半球可见度、间接入射光及正欠拟合/非饱和过拟合图。核验漫反射+镜面在线性域等于PBR基础，保持Alpha，前后checkpoint SHA一致；无optimizer。

同一冻结状态另作去SH间接光与无遮挡两个机制探针。无遮挡同时令追踪alpha与局部返回颜色为0，仅用于判断可见度对斑驳的贡献，明确不是物理正确候选、最优超参或最终渲染；任何探针均不保存模型。必须保留原生追踪作为唯一正式结果，不以探针PSNR替代验收。

上向区域由原Alpha>.99、预测Alpha>.99和预测GS法线Z>.9固定定义，不读取GT平台分区；可见度<.5、.5～.8、≥.8仅作诊断分桶。前景整体误差仍用原Alpha>.5，饱和通道继续删失。报告可见度与正欠拟合相关性、正/负误差、材质及漫反射/镜面贡献，不将相关性称为因果。

输出output/1019-25-LH-static-jumpingjacks-ELDL-V3-PBR误差TRAIN分解，日志同目录diagnostic.log；完整实际命令、真实PID及退出码写入launch/job_state。失败产物保留，重试新目录。此项不是训练，未增加或修改任何训练预算。

## 首次诊断失败与修正

25目录因诊断误将截断显示diffuse/specular逆变换为HDR而断言失败，脚本原文、部分PNG、日志与failed_receipt完整保留。checkpoint SHA不变，未训练。修改为只旁观实际rendering_equation线性返回值，再按同一mask/Alpha组装漫反射和镜面图，执行后恢复原函数；不修改冻结渲染器源码。独立26目录重试，不能把首次诊断假设错误写成原生PBR的物理加和故障。

## 单一几何起点机制核验登记

26目录三帧诊断已完成。去间接与完全无遮挡都增大TRAIN误差；低可见度上向像素的正欠拟合较高，相关性只能提示追踪起点问题，不能据此移除遮挡。为区分起点自遮挡与真实全几何遮挡，仅增加一个法线起点探针：在PBR实际着色器内将position沿预测单位法线偏移现有native light_t_min=.009978598594665527，原沿射线偏移、全部材质/环境/几何与DL不变；不扫描偏移值、不读取GT、不训练。27目录独立执行，记录RGB、Alpha/Normal/深度差以及上向可见度均值/标准差/三邻域高频RMS。26源码另行归档，不改其结果。
