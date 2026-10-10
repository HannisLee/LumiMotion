# 1010-02 jumpingjacks dir 原生PBR方向光完整基线

更新时间：2026-10-10T01:48:27.915814+09:00
状态：完整20000步训练、渲染、全部15测试帧MAE与独立复算完成；人工图像质量FAILED：人物斑驳、平台伪几何/阴影和材质残留；不能commit。7°目标FAILED。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

GT灯方向按帧去重、每帧白色单位光，纯数值固定辐照度π；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；保留SH间接光近似及每4步优化。优化albedo_dc、albedo_dc_stage1、albedo_rest、roughness、opacity与原生颜色形变优化组；位置、旋转、尺度和二值静动态参数冻结。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.19659954 | linear RGB | 96.275665% |
| Roughness MAE | 0.29154181 | r | 96.560916% |
| GS Render Normal MAE | 21.22875590 | degree | 93.078788% |
| Depth Normal MAE | 27.80211481 | degree | 88.449882% |
| Mesh Normal MAE | 22.79203567 | degree | 93.778058% |
| Expected Z Depth MAE | 0.08852290 | m | 93.078788% |
| Median Z Depth MAE | 0.09119155 | m | 93.078788% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。三法线共同像素结果另存metrics_summary.json。

## RGB及渲染

{"psnr": 18.384561590747985, "ssim": 0.8770897030830384, "lpips": 0.12493863701820374, "psnr_foreground": 11.308006909041387, "ssim_foreground_bbox": 0.7279389182726542, "lpips_foreground_bbox": 0.26818381746610004}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。原生SH间接光转灯仍是来源照明近似。

## 四类目检（最终首/中/尾审核）

- RGB与背景：55k首中尾背景为黑色，人物/平台存在；衣服/皮肤强斑驳，头手欠完整，脚周平台凸起与错误阴影，0080 GT也背光但预测平台缺棋盘格。质量FAILED。
- Alpha：主体/平台大致覆盖，头手边缘毛糙、局部漏检和平台残影；数值覆盖率见上表。
- Albedo与静动态分离：材质含强照明/暗背面结构、平台偏白并丢失棋盘格；静动态参数冻结，不能宣称修复分离。
- Normal/Depth：GS/Depth法线斑驳，0010平台脚周明显凸起/波纹，身体Z偏差；Mesh较平滑但仍有伪几何/缺面，三法线均未达到7°。冻结几何不能据材质训练声称已修复。

代表图：inspection/four_categories.png、rgb_comparison.png、relight.png；15帧误差图和浮点数组保留normal_depth、albedo、roughness；权重与日志保留，不删除失败结果。

## 审计及结论

137项CPU单元测试通过。checkpoint的有限性、原始几何冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

完整训练与评测执行成功，但图像质量FAILED，未commit/push；进入B01预测几何+训练RGB光强修复，完整重训和MAE后再审阅。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/logs/write_results.log

## 重光照与完整审计结论

转灯72帧有光照响应且Alpha固定，HDR150帧2048采样，首中尾有橙色照明与材质/伪几何残留；没有重光照GT。6视频完整解码，83来源文件哈希复核、三法线/Depth/Albedo独立数组复算及Roughness独立复算PASS。训练00:50:38→01:35:26、loop44:38，55k保存01:35:25；全部流程01:48:27完成。最终执行/产物PASS，图像质量FAILED，法线7°FAILED，失败权重/图/统计/日志保留。

## 输入来源声明纠正（2026-10-10 02:03）

旧转换mask_source实际为albedo-alpha，先前将它写成原始输入RGBA来源不准确。全部150帧原RGB有自身alpha，16-bit alpha与albedo alpha逐像素相同；旧转换RGBA亦与新的原RGB-only转换逐像素相同。数值指标保留，但旧转换的GT文件读取链不能作为严格来源验收；后续1010-05只从原RGB自身alpha生成输入，启动前核验全部来源哈希/像素，不读取GT材质/几何文件。固定35k初始化同样继承旧转换，数值一致性已审计披露，未重跑Stage1。B00人工质量FAILED结论不变。
