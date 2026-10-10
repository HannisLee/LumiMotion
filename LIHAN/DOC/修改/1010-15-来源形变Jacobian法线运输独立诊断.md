# 1010-19 dir 来源形变Jacobian法线运输独立诊断

状态：只读诊断实际完成，exec82276退出0且会话关闭；0 optimizer、0 GT标签、0 TEST，不修改生产/训练/Preset。直接用逆转置Jacobian替换原旋转的候选不接受。

## 来源与方法

读取原Stage1 35000全部99827点和固定32个NN tensor，以B10 40k仅作相机/来源NN容器，未使用其学得XYZ/q/features/material。原MLP逐点计算位置变形，源native概率固定，J=I+∂d_xyz/∂sourceXYZ；输入XYZ求导，网络权重冻结。仅六TRAIN时间0001/0031/0061/0091/0121/0149。det>1e−4、condition<100且求解finite仅定义诊断可逆样本，所有点数组完整保留，不删训练点或评分像素。对比原来源旋转轴与J逆转置运输源轴的无向轴夹角；这不是GT有向法线MAE。

## 实际结果

| TRAIN帧 | 动态近平台点可逆条件合格/6836 | 动态近平台det≤0 | 两种轴平均差° | 其余动态det≤0/38747 | 有限差分相对最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0001 | 3727 | 2895 | 40.86929 | 17359 | 0.892% |
| 0031 | 3743 | 2905 | 40.88848 | 17795 | 2.167% |
| 0061 | 3674 | 2917 | 39.97558 | 18684 | 3.433% |
| 0091 | 3535 | 2961 | 36.48443 | 17929 | 1.698% |
| 0121 | 4123 | 2525 | 35.34572 | 17054 | 1.026% |
| 0149 | 3662 | 2984 | 39.94708 | 17291 | 3.124% |

动态近平台约37–44%、其余动态约44–48%出现det≤0；来源位置MLP没有受物理体积形变约束，不能据旋转与Jacobian不同就把后者认定为正确法线。有限差分16个固定样本、ε1e−5的相对最大差约0.9–3.4%，存在ReLU/高频编码数值误差，不称严格精确验证。近零角使用float32归一化向量再转float64点积，可能有约0.02°舍入误差；数十度动态差异不受该量级影响。

结论：执行与来源审计PASS；直接Jacobian法线替换候选未接受，未生成prior、新Preset或训练。该只读诊断在用户要求停止迭代前已经执行，现仅归档现有结果；完成B10后不继续迭代。

## 完整命令与产物

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-19-dir-来源形变Jacobian法线运输诊断/commands/audit_source_jacobian.py
```

日志logs/source_jacobian.log；inspection/source_jacobian.json含来源SHA和逐组统计，六source_J_*.npz保留全部点。此诊断没有RGB/Alpha/Albedo/Normal渲染或视频，不产生七MAE，不替代B10最终验收。
