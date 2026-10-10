
## B06 50k真实保存与独立检查点审计（2026-10-10T07:34:24.139755+09:00）

等待进程exec81677实际exit0并关闭，绑定train258688/startticks9357639，在真实50k marker后调用检查点审计，audit子进程也实际exit0。50k状态PASS_CHECKPOINT_ONLY：

```json
{
  "status": "PASS_CHECKPOINT_ONLY",
  "iteration": 50000,
  "gaussians": 99827,
  "all_finite": true,
  "opacity_scale_feature_frozen": true,
  "albedo_rgb_range": [
    0.030999988317489624,
    0.968999981880188
  ],
  "clamped_albedo_channel_fraction": 0.0,
  "channels_at_projected_bound_fraction": 0.33705644097622217,
  "max_displacement": 0.08000000566244125,
  "max_rotation_deg": 60.00005340576172,
  "actual_light_scale": 6.247264862060547,
  "deformation_state_tensors_unchanged": true,
  "deformation_tensor_count": 32,
  "deformation_input_exactly_source_xyz": true,
  "anchor_marker_bound": true,
  "source_files_unchanged": true,
  "final_mae": false
}
```

99827点/冻结opacity-scale-feature/32来源NNtensor原值和source SHA保持，source XYZ输入锚定正确，native source SHA/plane资产绑定通过，albedo可导DC硬clamp0。最大canonical位置0.08000000566m、旋转60.00005341°容差内；额外world plane投影不等于canonical信赖域。强度6.24726486由训练RGB拟合，不是GT强度。核心469文件与launch SHA仍一致。

完整训练仍真实存活：15477/20000，iteration50477；后处理真实PID258864仍等待train完成，不重启。本检查不是质量或七项MAE；最终55k全15帧七MAE、coverage/common/missing180及四类/HDR目检待完成，未commit/push/新分支。

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_50000_when_saved.py
# 该等待进程在50k实际保存后调用同环境命令：
/home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 50000
```

证据inspection/checkpoint_50000_audit.json、audit_50000_wait_state.json、model_mlp/pbr_directional/iteration_50000/lighting.json；日志logs/audit_50000_when_saved.log、checkpoint_50000_audit.log。全部来源与失败历史保留。
