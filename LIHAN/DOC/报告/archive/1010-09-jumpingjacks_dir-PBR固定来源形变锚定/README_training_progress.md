
## 正式启动记录

{"status": "TRAINING", "started_at": "2026-10-10T04:30:08.950000+09:00", "training_pid": 232645, "process_start_ticks": "8458195", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_anchor", "session_id": 16086}

实际配置/150帧RGBA来源/99827点源XYZ锚定SHA/材质初始化核验PASS，证据inspection/launch_configuration.json。实际train session16086；后处理session55026绑定原PID232645，40k预览session25518，均持续等待/执行。完整训练和MAE尚未完成，不提前提交/push或声称7°。

## B04 40k真实检查点与RGB预览

RGB全15帧：{"psnr": 18.517455841244086, "ssim": 0.8864702026049296, "lpips": 0.10901500582695008, "psnr_foreground": 11.48589137773118, "ssim_foreground_bbox": 0.7530547539393108, "lpips_foreground_bbox": 0.22960177361965178}

检查点：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 40000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "max_displacement": 0.02973010018467903, "max_rotation_deg": 60.00002670288086, "actual_light_scale": 5.28775691986084, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

实际查看0010/0080/0150对照GT/B03同40k：黑背景正常、人体平台存在；人物/衣服/头手斑驳稀疏、脚周平台伪结构和残影仍明显，与B03同阶段较接近，FAILED_SO_FAR。预览正常exit0。99827全保留、源MLP32tensor完全未改、anchor逐元素来源XYZ且marker绑定正确，实际XYZ/rotation更新有界、材质硬clamp比例0；功能PASS不等同最终质量/7°。继续完整20k、最终55k所有MAE和四类/HDR审核，不以40k代替最终结果、未commit/push。证据inspection/preview_40000_review.json、checkpoint_40000_audit.json和preview_40000_rgb_comparison.png。

## B04运行中135训练时刻来源平面覆盖诊断

只读真实35k、全部135训练time查询MLP，源SHA不变、0优化步、无GT输入、未生成训练先验。本轮训练参数和源码不改。

当前14556平面支持点的坐标运动box-span范数P90=.00699977m、P99=.22801237m、max=.68234813m；全部135时刻的最远平面距离P99=.07980215m、max=.23887041m。此前1/75/149三时刻诊断P99约.0193m未覆盖所有动作，不能称完整时间稳定；当前静态feature阈值可能误含运动点，应在后续修复候选中核验时间稳定性。

来源静态距plane≤.08m共有42785点；低Alpha≤.05点27400，opacity总和212.6719、未投影Alpha×scale0×scale1面积代理.0327114，而当前支持代理4.591814，约.71%。这只是面积代理不等同光栅贡献，不能仅依据大量低透明度点数认定它们主导残影。

来源静态距plane≤.2m但当前未支持32800点，opacity总和2643.1616、面积代理1.679724，可能有不可忽略的较远预测片。静态near.2且135time运动box-span≤.03m共45547点，面积代理6.007738；动态feature组near.2且同稳定条件0点。不能据此不加筛选地将所有近plane点投影：平台侧面/人体脚可能存在，仍需源方向与时间可信度及完整MAE验证。

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/audit_source_plane_coverage.py
```

实际exit0，日志logs/audit_source_plane_coverage.log，完整各band/Alpha/静动态/时间范围统计inspection/source_plane_coverage.json。不是最终MAE，不改评分、不删除Gaussian、不提前宣称7°。

## B04只读训练损失图

基于两轮实际TensorBoard，当前B04前缀和B03完整训练同坐标对照：inspection/training_losses_prefix_steps.png。RGB L1图采用尾部201训练步均值，其余原始scalar，每个点原值/iteration/wall_time/run/tag保留training_losses_prefix_steps.csv；生成信息training_losses_prefix_steps.json。图中x轴为Stage2训练步数，原生Adam每4步实际更新，不能称每步optimizer update。首次图training_losses_prefix.png的轴标签updates不准确，版本保留，正式引用更正为steps的版本。

RGB L1当前略低于B03同阶段，光强曲线接近；支持点canonical平面轴一致性降至约.001量级，而预测GS-depth一致性仍随训练相机波动。不能把内部source plane loss当全前景GT法线MAE，也不能由其下降直接声称7°。不读取GT、未改变训练；日志logs/plot_training_losses_steps.log正常exit0。

完整命令：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/plot_training_losses_steps.py
```

## B04 50k真实检查点

{"status": "PASS_CHECKPOINT_ONLY", "iteration": 50000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.34154086569765696, "max_displacement": 0.06370601058006287, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.186013221740723, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

全部99827 Gaussian/源32tensor保留不变、opacity/scale/feature冻结、sourceXYZ锚定与当前marker同SHA；真实几何更新有界、材质硬clamp比例0，34.15%通道位于内部投影边界，仍可导。光强6.18601322仅训练RGB拟合，不是GT能量。最终55k全15帧七项MAE/四类质量待完成，不能由有限性检查称质量或7°PASS。

命令：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/audit_checkpoint.py 50000
```

日志logs/audit_checkpoint_50000.log，实际exit0。

## B04完整20k训练结束，后处理开始

{"status": "TRAIN_COMPLETED", "started_at": "2026-10-10T04:30:08.950000+09:00", "training_pid": 232645, "process_start_ticks": "8458195", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_anchor", "session_id": 16086, "finished_at": "2026-10-10T05:14:31.955921+09:00", "iteration": 55000, "steps_completed": 20000, "log_completed_at": "2026-10-10T05:14:28+09:00", "loop_duration": "44:01", "training_exit_code": 0, "verified_training_session_closed": true}

启动04:30:08.950、55k保存及Training complete 05:14:28 JST；全部20000步、loop44:01，训练exec session16086实际exit0并关闭，55k PLY/PTH/灯光marker已保存。真实后处理childPID239457正在全量render，已核验/proc cmdline，不是仅依据状态文件。完整四类/HDR/动态Mesh/七项MAE和最终质量仍待完成，不提前提交或宣称7°。

## B04最终55k RGB与四类首中尾实际审阅

{"psnr": 19.475974080557172, "ssim": 0.8950952370961507, "lpips": 0.11134533832470576, "psnr_foreground": 12.349587008267934, "ssim_foreground_bbox": 0.7730405648549398, "lpips_foreground_bbox": 0.22556244830290476}

已实际查看RGB0010/0080/0150对照GT/B03同55k和RGB/Alpha/Normal/Albedo0001/0075/0150。黑背景正常，人物平台存在；人物/头手/衣服斑驳稀疏，脚周平台伪结构和残影仍明显，棋盘纹理部分恢复。Alpha主体平台基本完整但边缘/下方薄雾；Normal平台集中但脚周和人体异常斑驳；Albedo人体前后照明残留与脚周污染。当前四类质量FAILED，commit_allowed=false。GT背光本来较暗，不单凭背面暗判失败。仅首中尾实际人工看，不声称150逐张。

PSNR19.47597 vs B03 19.35354、LPIPS.11134534 vs .11383797，略改善不能当7°通过。完整15帧七项MAE/HDR/转灯/三法线与深度误差图待完成；保留本轮全权重与后处理日志。证据inspection/visual_review_early.json、rgb_final_55000_comparison.png、four_categories_final_55000.png。
