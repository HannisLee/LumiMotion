# 原生Distortion完整TRAIN零更新诊断

2026-10-19，ushas／lumimotion-ushas。新增总控`diagnose_native_distortion_train_v1.py`及独立修正版v2；仅读取S23@8100、完整135TRAIN及现有冻结源码，0优化，不修改训练入口或历史模型。

检查发现当前三阶段loss没有启用native `rend_dist.mean()`；这与已登记的S23实现一致，不能把它称为S23实际训练项。诊断从冻结V10 `arguments/__init__.py`的AST读出原生默认lambda_dist=1000，仅作为反事实候选，不从其他数据集指标推系数。分别计算完整光度目标+.1Alpha、.5normal及候选distortion对XYZ/rotation/scale/opacity的真实梯度、范数、点积、跨帧均值；所有预期路径必须连通且有限，不把缺失导数静默当作零。

首次输出1019-45因bootstrap把cwd改为冻结快照、诊断误以cwd为项目根，在模型加载前被路径断言拦截；保留代码、日志及failure.json。v2明确用THREE_STAGE_REPO_ROOT判输出，兼容旧S23缺省normal_protection=warmup；独立重试1019-46完成135帧。未覆盖失败结果，未重启任何训练。

重试receipt状态PASS_ZERO_UPDATE_COMPLETE_TRAIN_DISTORTION_DIAGNOSTIC，checkpoint SHA与S23原值完全相同。候选加权distortion均值0.0000554337，实际光度均值0.06572316，.5normal均值0.01337794。候选梯度/当前几何总梯度逐帧范数比均值XYZ0.005714、rotation0.004670、scale0.001482、opacity0.000857；平均余弦约.117/.120/.115/.038。该量级暂不支持把缺少此项当作主要原因，不新增distortion正式训练、不放大系数。

完整命令、GPU2、PID、源码SHA：`output/1019-46-LH-static-jumpingjacks-ELDL-V3-S23原生Distortion零更新诊断重试/launch.json`，日志diagnostic.log，全部逐帧与汇总results/receipt.json。此项为无渲染视频的零更新梯度诊断，四类目检不适用；不得代替S23/S24或最终模型验收。
