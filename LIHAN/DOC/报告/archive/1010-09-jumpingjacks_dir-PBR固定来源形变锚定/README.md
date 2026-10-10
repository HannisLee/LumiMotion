# 1010-09 jumpingjacks dir 原生PBR固定来源形变锚定

更新时间：2026-10-10T05:28:00.297574+09:00
状态：完整20000步训练及全量MAE完成；最终图像质量FAILED，7°目标FAILED，不能提交。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；保留SH间接光近似及每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度和二值静动态参数冻结。几何2000步warmup、1000步ramp，位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Anchor Preset继承B03全部参数，唯一新增：冻结MLP的XYZ输入固定为复制35k来源XYZ，实际渲染使用优化后真实XYZ/rotation加来源形变；训练、周期测试、离线全部评测一致读取哈希锚定资产，checkpoint marker绑定资产SHA。此修复避免canonical位移被MLP位置输入放大。继续保留来源PBR DC初始化及Adam更新后投影到[.031,.969]；原生PBR/GGX、SH损失、几何约束和其余学习率保持。初始化/投影只改PBR DC，原生SH DC/rest仍正常优化。记录pbr_albedo_initialization.json；源35k不改、离线直接读取保存PLY，评测不再初始化或投影。原生albedo_lr=.01，每4步更新，不缩短训练。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15600631 | linear RGB | 96.642930% |
| Roughness MAE | 0.21155915 | r | 96.913295% |
| GS Render Normal MAE | 18.91234950 | degree | 93.686587% |
| Depth Normal MAE | 27.51005704 | degree | 89.073328% |
| Mesh Normal MAE | 22.23676747 | degree | 94.081356% |
| Expected Z Depth MAE | 0.08637919 | m | 93.686587% |
| Median Z Depth MAE | 0.08998650 | m | 93.686587% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。共同有效像素：369183/415433，覆盖率88.867038%；GS缺失预测记180°后的惩罚均角29.08247777°。

| 法线来源 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 17.99532404 |
| depth_normal | 27.46234473 |
| mesh | 21.26058146 |
完整JSON另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.475974080557172, "ssim": 0.8950952370961507, "lpips": 0.11134533832470576, "psnr_foreground": 12.349587008267934, "ssim_foreground_bbox": 0.7730405648549398, "lpips_foreground_bbox": 0.22556244830290476}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。原生SH间接光转灯仍是来源照明近似。

## 四类实际首中尾目检

- RGB与背景：黑背景正常，人物平台存在；人物/头手/衣服斑驳、边缘稀疏、脚周平台伪结构。与B03同55k对照，略改善，质量FAILED。GT背光本来暗，不单凭暗判失败。
- Alpha：主体轮廓大体完整，头手边毛糙，平台边缘/下方薄雾残影，质量FAILED；覆盖率见上表。
- Albedo与静动态分离：棋盘纹理部分恢复，人体前后照明残留及脚周污染；静动态参数冻结，质量FAILED。
- Normal/Depth：0010/0080/0150实际查看GS/Depth/Mesh与GT/误差；平台脚周伪起伏、人体斑驳、Depth噪声及Mesh头手腿缺面，质量FAILED。

代表图：inspection/four_categories.png、rgb_comparison.png、relight.png；15帧误差图和浮点数组保留normal_depth、albedo、roughness；权重与日志保留，不删除失败结果。

## 审计及结论

163项CPU单元测试和真实权重单帧GPU梯度检查通过。checkpoint的有限性、xyz/rotation实际更新并在信赖域内、opacity/scale/feature冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；先完成前后景人工审核，必要时修复重训，再进入目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_anchor --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/write_results.log

## 实际训练时间与过程证据

{
  "status": "TRAIN_COMPLETED",
  "started_at": "2026-10-10T04:30:08.950000+09:00",
  "training_pid": 232645,
  "process_start_ticks": "8458195",
  "steps_planned": 20000,
  "source_iteration": 35000,
  "target_iteration": 55000,
  "loss_preset": "pbr_gt_directional_anchor",
  "session_id": 16086,
  "finished_at": "2026-10-10T05:14:31.955921+09:00",
  "iteration": 55000,
  "steps_completed": 20000,
  "log_completed_at": "2026-10-10T05:14:28+09:00",
  "loop_duration": "44:01",
  "training_exit_code": 0,
  "verified_training_session_closed": true
}

以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。

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

## 最终审核结论：FAILED

完整20000步及9步后处理均实际exit0。150×4类/72转灯/150 HDR2048及6视频全部解码、全部15测试帧七项MAE、独立复算和83来源SHA：PASS。150原始RGBA来源、99827点全保留、源MLP32tensor不变、anchor输入逐元素源XYZ/marker绑定：PASS。

实际RGB/Alpha/Normal/Albedo及HDR/转灯和0010/0080/0150法线/深度误差图已看。背景功能正常；人体/头手/衣服斑驳、边缘缺失、平台脚周伪起伏/残影、Albedo光照残留、Depth噪声/Mesh局部缺面仍在。最终图像质量FAILED，不能提交。仅首中尾实际人工看，不声称150逐张。

原GS Render Normal pixel weighted MAE18.91234950°（覆盖93.686587%）；Depth27.51005704°、Mesh22.23676747°；共同像素369183/415433（88.867038%）GS17.99532404°/Depth27.46234473°/Mesh21.26058146°；GS缺失180惩罚29.08247777°。没有删除Gaussian/帧/困难像素，没有改评分。7°目标FAILED。

与B03同55k：RGB PSNR19.47597 vs19.35354，Albedo .15600631 vs .15765579略改善；GS18.91235 vs18.89548未改善。Anchor确实消除更新XYZ输入冻结MLP的耦合，但不能据此称几何质量得到修复。根因审计继续关注来源平面支持的全时间可信度与未覆盖较远表面片；后续只用来源预测/原RGBA/GT方向构造修复，不以GT材质或几何作为先验。

commit_allowed=false；尚未commit/push、未建立目标2分支。全部失败权重、渲染、浮点数组、统计、日志保留。
