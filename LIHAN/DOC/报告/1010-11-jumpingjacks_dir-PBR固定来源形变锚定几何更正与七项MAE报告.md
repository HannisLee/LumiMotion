# 1010-11 B04原生PBR固定来源形变锚定：完整几何更正评测

更新时间：2026-10-10T06:04:10.503202+09:00
最终状态：FAILED（图像质量与原GS法线<7°）；运行时/全15离线重评/独立复算PASS。未commit/push/新建目标分支。

## 更正范围与真实训练

B04已经完整20000步35k→55k（2026-10-10 04:30:08.950→05:14:28 JST，loop44:01）；本次0训练步，不重跑Stage1或Stage2，不改权重/材质/光强。Stage1来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp。训练135帧/15测试帧、400×400黑背景，训练仅原RGB/自身RGBA、相机/time、来源预测与GT方向；GT材质/法线/深度/mesh/颜色/能量禁止训练，GT只离线评分。

旧GeometryRun漏读Anchor，旧GS18.91234950°、Depth27.51005704°、Mesh22.23676747°及depth/common/missing180标记SUPERSEDED。旧数组计算自洽不证明运行时正确，不能据此比较Anchor效果。旧权重/数组/Mesh/统计/日志全部保留，旧metrics_summary_legacy_unanchored.json保存；本次使用FrozenScene共享源XYZ查询真实渲染几何，再完整15time×135train重新TSDF Mesh及全部15几何评分。

RGB、albedo、roughness原已使用FrozenScene，不受该遗漏影响，本次原产物以symlink引用并再次独立复算/hash核验，不冒充重新训练或重新渲染。自包含报告归档仅引用同一55k来源，以哈希核对。

## 正式七项MAE：全15帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15600631 | linear RGB | 96.642930% |
| Roughness MAE | 0.21155915 | r | 96.913295% |
| GS Render Normal MAE | 18.77743450 | degree | 93.850031% |
| Depth Normal MAE | 27.21989900 | degree | 89.219922% |
| Mesh Normal MAE | 22.15133832 | degree | 94.229635% |
| Expected Z Depth MAE | 0.08569735 | m | 93.850031% |
| Median Z Depth MAE | 0.08932931 | m | 93.850031% |

共同有效像素369804/415433，覆盖89.016520%；GS缺失180惩罚28.69257202°。

| 法线 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 17.86533346 |
| depth_normal | 27.16722766 |
| mesh | 21.16834873 |


原评分口径保留：GS/Depth/Mesh为原32-bit世界着色Normal有向角度，GT alpha≥.5所有FG，没有按物体/困难像素过滤。Expected/Median Z单位米，无scale/shift/ICP校准。Albedo raw linear RGB、GT alpha≥.99内缩2，缺失预测0；Roughness归一化本征r，源EEVEE/Principled Alpha透射混合GT、同GT core且pred alpha≥.9，覆盖率单列。透明r为可见透射率×Alpha加权值，不声称唯一第一命中值。

Mesh每个测试time独立融合所有135 TRAIN相机的预测Expected Z，voxel.02/sdf_trunc.08/depth20/alpha.5；不跨time、不用test位姿/GT几何。原模型99827全保留、MLP32tensor不变、source_xyz Anchor与当前marker同SHA。无删点/删帧/改评分。完整原定义见1010-09训练文档及1010-08修改文档。

## RGB与目检

{"psnr": 19.475974080557172, "ssim": 0.8950952370961507, "lpips": 0.11134533832470576, "psnr_foreground": 12.349587008267934, "ssim_foreground_bbox": 0.7730405648549398, "lpips_foreground_bbox": 0.22556244830290476}

原B04 150×RGB/Alpha/Normal/Albedo、72方向光、150 HDR2048及6完整视频已生成并解码，原生FrozenScene正确，不因几何重评重复改写。原四类/HDR审核结果沿用FAILED：背景正常，人物平台存在但衣服/头手斑驳、脚周假几何/残影、Alpha边雾、Albedo光照残留。代表图仍在原B04 inspection及model_mlp/renders_stage2/ours_55000各分类目录；HDR与转灯只功能/目检，无GT精度结论。

本次实际查看更正后的normal_depth/comparison_0010.png、0080、0150：平台脚周伪起伏、身体法线斑驳、Expected/Median脚周明显Z误差；Depth法线较噪，Mesh局部手/腿/头缺面，0080更明显。仅3代表帧实际人工检查，不声称所有15/150逐张。真实源运行时修正略改善分数，但18.77743450°远高于7°，不能提交。B03同55k GS18.89548239°，B04更正18.77743450°小幅改善；训练RGB不变，不能称已解决图像质量。

## 可复核证据与限制

179 CPU实际exit0；真实GPU B04 55k/B05 source Plane逐像素 normal/alpha/Expected Z与原生PBR首中尾max error0、timeoverride一致、源hash不改、0优化步。重评3步执行实际exit0，独立CPU三法线/common/2depth/albedo与90来源SHA PASS；roughness独立复算15帧41SHA再次PASS；missing180单独CPU全15复算PASS。runtime_history.jsonl、inspection/independent_audit.json、missing180_independent.json、gpu_audit.json和logs可复核。完整权重/浮点数组保留，归档保存报告/metrics/代码与命令快照/logs/代表误差图，未删除失败产物。

B05训练持续，训练/renderer/MLP/参数不改，只有未执行离线GeometryRun修复，已保存旧launch manifest/snapshot和SHA修订记录。最终B05全部七项MAE仍待55k完成。

## 实际重评时间

{"step": "predicted_meshes", "exit_code": 0, "elapsed_seconds": 154.19239830970764}
{"step": "normal_depth", "exit_code": 0, "elapsed_seconds": 21.643505811691284}
{"step": "verify_all_mae", "exit_code": 0, "elapsed_seconds": 1.7644400596618652}

状态：{
  "status": "COMPLETED_REVIEW_PENDING",
  "step": "finished",
  "pid": 246942,
  "updated_at": "2026-10-10T06:00:12.118709+09:00",
  "frames": 15,
  "iteration": 55000
}


## 完整命令

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/logs/normal_depth.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/logs/verify_all_mae.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评/logs/verify_roughness.log

### 原B04完整训练命令

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_anchor --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

训练日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/logs/train.log

原全render/HDR/七项命令和人工审核见1010-09报告，几何分数以上方更正为准。
