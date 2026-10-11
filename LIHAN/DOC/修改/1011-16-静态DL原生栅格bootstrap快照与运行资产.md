# 静态DL原生栅格bootstrap快照与运行资产

日期：2026-10-11；服务器：ushas；环境：lumimotion-ushas。

## 修改原因

最初核心快照runner直接运行快照的训练模块，会让Python导入Conda环境中同名的`diff_surfel_rasterization` pip extension，绕过批次协议要求的仓库native CUDA rasterizer。已运行的旧SH smoke与T01正式输出全部保留并标记协议失败，不覆盖、不删除。旧T01进程已消失，日志止于读取training transforms；未保存退出码，退出原因未知。

## 修复与核验

- `goal_core_snapshot.py`现在复制最小训练代码、冒烟wrapper和完整小型`assets/`运行目录，并生成runner委托至`LIHAN/scripts/bootstrap_1011.py`。该bootstrap先从工作树导入`scene.pbr_rasterizer.native_module()`，随后切换到冻结快照运行目标模块。
- `train_static_dl.py`初始化护栏要求rasterizer provenance的method为`repository_lumimotion_cuda_rasterizer`，并记录CUDA源码和编译扩展SHA。绕过bootstrap时立即失败。
- 首个正确bootstrap smoke显示快照遗漏`assets/bsdf_256_256.bin`，训练在GaussianModel初始化前失败。资产SHA256为`aee514f7c7e561a357e529567222da99e84886c31c46a32fe767a5b066bbe196`；已纳入`T01-sh-native-assets-20261011`并重跑成功。`scene/`中native rasterizer `csrc`随源目录快照一并包含；native CUDA编译仍由仓库bootstrap在主工作树构建。
- 正确SH smoke结果：20步exit0；d_color梯度20/20非零且有限，三个几何head逐步输出严格零，151个时间采样严格零；独立checkpoint重载检查training/eval源码SHA及实际.so SHA一致，三个head权重/偏置全零。证据：`output/smoke_test/1011-06-LH-static-jumpingjacks_DL_V2-T01-SH-native-assets冒烟/`。
- B/C物理smoke也均经bootstrap运行，产生iteration20 checkpoint及物理梯度审计；C的首步BVH trace 100000 rays，mean visibility约0.1749。

失败快照和目录状态另存于`output/1011-09-静态DL-Goal训练总控/snapshots/`与对应smoke输出，不作为正式训练依据。
