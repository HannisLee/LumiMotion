# Stage2 HDR 轴向回归与来源写入保护

日期：2026-10-02。实现分支：`1001-stage2-perlight`。

## 修改缘由与结果

原始 `render_relight_with_hdr.py` 对世界方向使用矩阵 `[[0,-1,0],[0,0,1],[-1,0,0]]`，`EnvLight` 随后以 `atan2(env_x,-env_z)` 映射经度。组合后为世界 `atan2(-y,x)`。新增积分器最初使用 `atan2(y,x)`，会左右镜像非均匀环境照明。现通过 `world_direction_to_hdr_uv` 与原始轴向一致，世界 Z 向上，yaw 是经度偏移；常量白炉和非均匀亮区响应均通过。正式训练只使用方向光，训练参数及检查点不受此修复影响。

初次训练现在拒绝向 Stage1 来源目录及其子目录写入，并拒绝已有非空模型目录。恢复仍显式使用 `--resume_iteration`；已有 checkpoint 和渲染拒绝覆盖。GPU 检查实际验证了来源目录、来源子目录和已有实验三种拒绝路径，没有创建禁止目录。

统一渲染入口记录自身执行时的 Git 提交、未提交差异和模型路径到 `provenance.json` / `code_diff.patch`。训练启动时的版本另在实验根目录记录。这样可以追溯训练期间修复渲染工具的版本差异。

## 验证与保留

CPU 验证现为 17 项：包括原始 IR 总损失和所有梯度逐位一致、HDR 原始矩阵回归、非均匀经度亮区和白炉响应。IR回归覆盖原始Trainer强制开启的train_ray路径，使用非平凡部分遮罩及全部可选项。补测曾误将False视为原始可执行分支，该分支实际未定义Ll1；修正测试范围，保留unit_tests_15_final.log失败记录，最终通过记录另存unit_tests_15_verified.log。GPU 真实 checkpoint 验证通过，Python/NumPy/Torch/CUDA 随机状态与相机队列一致；跨进程浮点材质/优化器以 `rtol=1e-5, atol=1e-6` 验收，最大 albedo logit 差异 `4.77e-7`。

冒烟中旧轴向 HDR 输出及首次逐位比较失败日志全部保留。修复后的 HDR 必须写入新目录，实验 README 明确标注最终代表输出与早期失败尝试。全分支同步本文件不表示其他分支拥有这些代码。
