# TRAIN原生PBR与直接项幅度只读诊断

2026-10-19，ushas／lumimotion-ushas。A35完整135TRAIN末点仍比共同S20退化，残差实际贡献约.15%；据此进行只读机制诊断，不读取TEST/GT，不启动新训练、不写模型参数。

使用冻结v5实际前向与A35@4050，来源和checkpoint及135TRAIN图像哈希核验；同固定原Alpha>.5、等帧RGB通道删失L2。基础为真实base_linear，单次DL为真实residual_linear除以已保存lambda_residual×.25sigmoid(residual_raw)，不从截断sRGB倒推。

分别固定基础缩放a=1仅拟合非负DL系数b，及同时拟合全局a/b；边界各[0,4]，L-BFGS-B最多100次、ftol1e-14/gtol1e-10，初值为保存的真实1/b。记录旧与拟合逐帧删失L2/Charbonnier、局部Hessian、边界和原实际残差系数范围[0,.025]。仅诊断表示能力；整体base缩放包括原生全部线性PBR，不等同单独调环境参数，形式比例与真实光能无关，不能由更宽诊断边界直接决定正式参数或强度。

外部脚本总控diagnose_pbr_direct_amplitude.py，冻结v5为导入根；输出output/1019-32-LH-static-jumpingjacks-ELDL-V3-TRAIN原生PBR直接项幅度诊断，完整实际命令、日志、PID/退出码、收据同目录。三TRAIN预览仅保存原A35，无拟合checkpoint；若有后续机制更改，须另登记计划并做工程门禁。S22按原8100预算继续，不由本诊断改参/停止。

已完整135TRAIN正常exit0，两个求解器均成功；固定基础仅调DL约.1593% L2改善，共同诊断a=.64219589、b=.49620560，L2 .01210605→.004561665。原checkpoint不变；详细解释、真实目检及不可直接部署限制见报告1019-84。
