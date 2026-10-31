# 静态ELDL纯RGB输入与方向光初始化实现

新增static_rgb_light_prepare.py：仅访问原RGB及camera.json，丢弃原Alpha，通过边界颜色与连通性推断轮廓，保留16位颜色，生成随机点云及输入哈希。

新增static_rgb_light_stage1.py：沿用原生Trainer，关闭时变颜色及形变，隔离DEV/TEST，禁止加载外部实验模型或GT文件，计数真实更新，禁止覆盖模型。

采用独立新增入口，现有训练代码和用户工作区修改不覆盖。工程冒烟、基线筛选、方向估计及最终独立评测的结果将在本文件持续补充。

新增static_rgb_light_extract.py：DEV18 RGB PSNR/SSIM/LPIPS验证与内部法线/覆盖诊断；以固定Gaussian ID做world投影、模型深度遮挡、边界内缩、normal一致性与跨帧定向筛选。新增static_rgb_light_fit.py：E0—E4，单位光向、固定方向光强1、跨帧共享每表面rho/b，空间哈希留出及bootstrap稳定性，不读取GT。

工程冒烟发现审计hook对bytes路径不兼容，已用os.fsdecode修复；首轮失败资产保留。重试200步199更新成功，formal35k已启动。对应提取首轮发现GaussianModel必需no_binary_separation参数，补齐后新编号重跑。

准确的信息边界：prepare读取原RGBA容器并哈希完整文件以验证来源，但仅前三颜色通道参与任何分割/训练数值，原Alpha不参与监督。绝不打开albedo/normal/light等GT属性文件。

新增static_rgb_light_pipeline.py：持久化等待正式训练，完整DEV18 PSNR/SSIM/LPIPS与内部几何诊断后选择；选定基线代理目检和方法诊断代理目检两个独立门禁。全部通过后冻结所有预测、模型、观测、配置、代码/原生CUDA来源和环境，再单次独立评测；没有GT反馈分支。

新增static_rgb_light_report.py：从已有冻结评测产物汇总中文README、报告与GOAL，绝不修改模型、光向或方法配置。百分位/残差图、法线矩特征值/条件数与两次bootstrap属于无GT验收证据。

GT前展示审查修正：最终百分位图高亮严格使用实际共同fit Gaussian集与相同投票阈值，灰点保留全部可靠观测；旧10k诊断图完整保留。残差图是固定预测灯向后逐表面OLS重新标定rho/b的诊断，不是优化器内部Huber残差，也不是空间留出CV残差，图题/配置明确区分。此更正不改变任何预测、选参或留出分数。
