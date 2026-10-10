# 1010-12 dir PBR原生光栅方向核验

PASS_NATIVE_FACING_AND_GRADIENT_DIAGNOSTIC，仅0步功能，不是新训练MAE或质量PASS。B05同55k、Source与camera-derived plane，不读GT标签；旧installed包hbb1 f7e2b68a normal.view_z sign与仓库ray dot实现不同。六实际相机0001/0038/0075/0076/0080/0150均在plane上方；旧kernel三帧0038/0075/0076全部朝下，独立namespace原生均dot≥.99999988。相同旧训练权重Native与legacy RGB对比保存，不作为重新训练后的质量提升。

合成已知单片normal梯度与finite difference max差.00035429；GGX/PBR/BVH六帧forward finite；source权重SHA/fingerprint不变、0optimizer、没有pip install或修改环境包/B05管线。代表inspection/native_comparison.png已实际查看：三坏角度平台定向修复，RGB人物和脚周伪影仍在。

首次build exec44427 exit1：原生嵌套GLM头文件目录为空；保留logs/build_native.log。只将仓库stable已有追踪的GLM头文件复制诊断快照后retry exec60130 exit0，日志build_native_retry.log。GPU exec29477 exit0。首次CPU指定两个不存在的test_stage1_loss/test_stage2_loss，exec58526 exit1且保留日志；实际discover exec11751全部187 PASS，不隐去失败调用；最后全部代码修订后的discover见smoke1010-13 logs/cpu_final.log（187 PASS）。

完整命令：

```bash
CUDA_VISIBLE_DEVICES=0 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-12-dir-PBR原生光栅方向核验/commands/build_native.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-12-dir-PBR原生光栅方向核验/commands/check_native.py
```

source/model：B05 model_mlp/55000，日志logs/build_native_retry.log及check_native.log；输出inspection/images/，全部保留。
