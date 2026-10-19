# 保留原生PBR与父Adam的Stage3末端继续入口

2026-10-19，ushas／lumimotion-ushas。实施训练计划1019-35；尚未启动新优化。

新增独立末端入口train_three_stage_refine，从A42@4050的自足Stage3加载完整几何、材质、环境、受限残差、32个MLP及照明；不走Stage2→Stage3材质转换。恢复父4组Adam moments/step/实际末端lr并保持该lr，再追加四几何空moments组。两组计算相同Expected一致性.5，四几何lr为0或既有Stage2末端向量。normal→opacity保护4冒烟/200正式步，photo/Alpha不冻结opacity。

独立geometry pipe.depth_ratio=0计算Expected先验，PBR继续depth_ratio=1及原半球/BRDF/BVH/法线起点。现有原生物理文件不改；MLP查询仍随当前canonicalXYZ前向变化，网络输入Jacobian及BVH可见度停止梯度，不能宣称精确全物理导数。

父RNG在构造和初始化核验后恢复，空相机栈由父4050完整135帧周期边界确定。新checkpoint兼容格式1，自足包含全部资产，累计数明确51199+新步数，修正旧Stage2父公式不能用于Stage3继续的问题。所有写入拒绝覆盖已有实验。监督入口仅增加显式新模块允许项。

新增独立审计：全部四节点初始PLY/材质/env/幅度、父Adam/RNG逐张量相等，后续32MLP/全部照明/未授权PLY字段冻结，8优化组、实际lr与step继承、累计计数；按实际逐步日志重算loss和opacity保护。另用独立Expected光栅重算先验，并实测受保护/未保护及photo+Alpha opacity梯度。原生目标置零、混合深度、起点和物理分量审计仍沿用独立既有入口。

静态编译通过；GPU冒烟及正式验收尚待执行，不预先声称门禁或精度PASS。代码SHA256：

- `scripts/train_three_stage_refine.py`：`29ac7875eca32f18765def2d3f5e951ba71d6c1d9de9ca92affb82e9f7430df0`
- `scripts/audit_three_stage_refine.py`：`3260ac6e6c3933c8fb5fce6cf4cba579df67144336d42ef7a91036bb622d4001`
- `scripts/run_three_stage_job.py`：`f3618f1343c7e08df19dd2d67c8ef75df8570997659f4bd8c364f36a8ebce5f4`
