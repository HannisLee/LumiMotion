# 完整TRAIN可见Expected外环平面零优化复核

2026-10-19，ushas／lumimotion-ushas。实现计划1019-39，基于报告1019-121保留的canonical窄带不足反例，仅新增C中的可见支撑诊断与独立监督；不修改A46、训练代码、旧快照或旧诊断结果。

只读同一A45@4050和完整135TRAIN，以原生Expected几何副本收集固定XY外环中可见世界点、GS/Expected法线、像素索引。高度不参与初始外环选择；所有135帧各贡献总权重1，复用已经合成平面CPU验证的10轮50%截尾PCA拟合全局可见平面。另按原normal偏移宽度构造外环可见窄带，分别统计原始外环与窄带角度、点距、各分母与像素加权/等帧；保存只读参考与逐TRAIN冻结掩膜，不启用新训练或权重。

保留135帧原始外环NPZ、逐帧metrics、拟合history、三锚点3x3图及全部父checkpoint/PLY/32MLP/marker前后SHA。继承TRAIN读取守卫，无GT/新A46 TEST读取，0 optimizer update。不能把预测外环标签、拟合平面或窄带误差当正式GT精度；实际图像须检查人体/鞋/边缘混入，保留失效条件。RGB/PBR/BVH/Separation/视频未重做。

两文件AST通过，PCA复用原已验证函数并由监督核对其源SHA、计划与V15全部源码。真实数据收集/拟合/图像查看仍待完成；独立output56，命令/PID/日志/退出码留存，失败不覆盖重试。源码SHA256：

- `diagnose_visible_train_plane_support_v1.py`：`97f689d3b29c940e030eb98e27ac29a5a157f9517e16117dbda044a49c3c3448`
- `run_visible_train_plane_support_v1.py`：`8aaae9cf14bc02801e22d001a5916594f16e2e1221d1de54abc109e58ddafc52`
