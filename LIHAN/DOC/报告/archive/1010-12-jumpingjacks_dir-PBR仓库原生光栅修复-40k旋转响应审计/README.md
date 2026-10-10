
## B06 40k加性旋转实际响应的只读审计（2026-10-10T07:23:22.796399+09:00）

假设核验：原生world四元数为qcanonical+dMLP再归一化，canonical信赖域角度不直接等于每时刻最终轴变化。0optimizer、未改变Gaussian/NN、没有GT标签输入；五TRAIN帧0001/0038/0075/0076/0149，source XYZ/NN/gate保持，比较source q与真实40k q加同一NN增量后的几何轴。projected组最终另有硬plane投影，本诊断不把该中间轴当最终normal。

45583未投影动态点canonical旋转中位8.364336°、P9935.336845°，只有15点≥59.9°（.032907%）；未投影静态39057点只有10点到界。动态点MLP增量/source q norm中位.133–.189，source world q近抵消（norm<.05×source q norm）在各组/五帧均0。

| TRAIN帧 | MLP增量/source q norm中位 | 实际world轴变化中位 | P99 world轴变化 | source world q近抵消点 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.138803 | 6.017214° | 31.412746° | 0 |
| 0038 | 0.133201 | 6.150679° | 32.343266° | 0 |
| 0075 | 0.184334 | 5.866048° | 29.638847° | 0 |
| 0076 | 0.188972 | 5.840733° | 29.400106° | 0 |
| 0149 | 0.137795 | 6.028020° | 31.427202° | 0 |


因此不支持“动态几何普遍卡在60°上限”或“整个NN增量压倒canonical使旋转无法生效”的假设；当前确有真实world轴变化，尾部也有变化较大点。不能据此加大信赖域、改全套四元数组合后宣称质量改善，也不从轴变化大小推定GT normal准确。若B06最终失败，优先基于已观测的未投影平台贡献/RGB和gate可导性/实际SUN预算选择后续候选；本轮不改。

首次exec96264因诊断相机fid留在CPU而NN为CUDA exit1，日志logs/audit_rotation_response.log及commands/audit_rotation_response_before_camera_device_fix.py保留；只在诊断增加ctx.load_camera/unload_camera，retry exec33080实际exit0且关闭，frozen/source和469核心SHA通过。命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_rotation_response.py
```

证据inspection/rotation_response.json与logs/audit_rotation_response_retry.log，完整分位数覆盖0/1/50/90/99/100百分位；source q norm保持与来源一致。本轮55k完整七MAE/四类/HDR仍PENDING，未commit/push/新建分支。
