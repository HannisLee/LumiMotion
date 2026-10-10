# 1010-07 jumpingjacks dir 原生PBR材质初始化与投影修复

更新时间：2026-10-10T04:08:24.938182+09:00
状态：COMPLETED_QUALITY_FAILED：完整20000步训练、全15帧七项MAE和独立复算PASS；实际图像质量FAILED，法线7°FAILED，commit_allowed=false。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；保留SH间接光近似及每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度和二值静动态参数冻结。几何2000步warmup、1000步ramp，位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

新增refine仅将来源PBR DC初始化并在原生Adam更新后投影到[.031,.969]，修复硬clamp零梯度；原生PBR/GGX、SH损失、几何约束和其余学习率保持。初始化/投影只改PBR DC，原生SH DC/rest仍正常优化。记录pbr_albedo_initialization.json；源35k不改、离线直接读取保存PLY，评测不再初始化或投影。原生albedo_lr=.01，每4步更新，不缩短训练。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15765579 | linear RGB | 96.537645% |
| Roughness MAE | 0.20939310 | r | 96.811983% |
| GS Render Normal MAE | 18.89548239 | degree | 93.718121% |
| Depth Normal MAE | 27.50692138 | degree | 89.098844% |
| Mesh Normal MAE | 22.19361409 | degree | 94.091947% |
| Expected Z Depth MAE | 0.08631321 | m | 93.718121% |
| Median Z Depth MAE | 0.09000833 | m | 93.718121% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。三法线共同像素结果另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.35354474812191, "ssim": 0.8932221968968709, "lpips": 0.11383797228336334, "psnr_foreground": 12.234762806258436, "ssim_foreground_bbox": 0.7692404826482137, "lpips_foreground_bbox": 0.22925928036371868}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。原生SH间接光转灯仍是来源照明近似。

## 四类、几何与重光照实际目检：FAILED

实际查看四类0001/0075/0150、RGB0010/0080/0150、三法线/Depth0010/0080/0150、HDR0001/0076/0150及转灯0000/0036/0071。人工范围是首中尾，不声称150逐张人工；全量图片与6视频全解码由自动审计复核。

- RGB与前后景：黑背景正常、主体/平台存在；0080平台棋盘纹理比B02有所恢复，但人物衣服/皮肤斑驳、头手稀疏及脚周伪结构仍明显，正面平台多白。GT背光自身也暗，不把暗部单独判功能故障。
- Alpha：主体平台基本完整，头手边毛糙、平台边及下方薄雾/残影；不是全前景消失。
- Albedo与静动态：平台纹理恢复，人体正背面照明结构、头手斑驳与脚周污染仍残留；feature冻结。
- Normal/Depth/Mesh：台面整体较集中，脚周仍异常/伪起伏，人体法线斑驳；Depth Normal更噪、Mesh头手腿缺面、脚周Depth误差明显。
- HDR与转灯：光照方向变化有响应、场景/黑背景存在、无全屏全黑全白；HDR橙粉色，无HDR/转灯GT，不能声称其颜色准确性。斑驳与平台伪结构持续。

证据inspection/four_categories_final_55000.png、rgb_final_55000_comparison.png、relight_final_55000.png、visual_review.json及normal_depth/comparison_*.png。失败权重、浮点数组、渲染、统计、日志与首个失败审计脚本保留。

## 审计及结论

153项CPU单元测试和真实权重单帧GPU梯度检查通过。checkpoint的有限性、xyz/rotation实际更新并在信赖域内、opacity/scale/feature冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；人工质量已失败；保留本轮，继续来源/几何管线审计后修复完整重训。通过后才能提交/push及建立目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_refine --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/logs/write_results.log

## 时间、同像素与覆盖率

训练03:10:06.570至03:54:31 JST，完整20000步、loop44:07、实际exit0；后处理04:08:24.985结束exit0。三法线共同像素369317/415433（88.899293%），GS 17.99274139°、Depth 27.46065764°、Mesh 21.20051106°；GS缺失按180°惩罚 29.01587388°。全99827 Gaussian保留，无帧/难像素删选。

## 过程记录与审计限制

40k预览PSNR18.46988217、最终19.35354475；B02同55k18.91228960。Albedo .18603245→.15765579、Roughness .24401021→.20939310、GS19.40743705→18.89548239°，覆盖93.694049→93.718121%，改善不足以通过质量或7°。40k/50k真实DC均在[.031,.969]、硬clamp比例0，投影确实持续生效；50k32.9604%通道在内部投影边界，仍可导。MLP全32 state tensors与35k源相同，但XYZ变动可改变MLP输出；不宣称形变输出恒定。

来源平面14,556支持点的B02 canonical平面法线均差已降至.92838°，多数其MLP形变轴变化很小；当前.08m支持/信赖域不能修正更远来源片。来源局部32NN PCA通过可靠平面条件的点仅静态3.2077%/动态7.1628%，不能强制整个模型用该PCA方向。40k三训练帧一致性梯度分解显示RGB+SH整体主导（rotation一致性约其1.65～4.25%），不能认定一致性主导全部噪声。这些是来源预测诊断，不是GT正确性证据；0 optimizer steps、不读GT材质/法线/几何或测试RGB。首次梯度审计错误verify方法的exit1脚本/log保留，修正后完整重跑实际exit0。

完整过程记录README_training_progress.md、inspection/source_plane_deformation.json、source_local_geometry.json、source_plane_support_distribution.json、normal_depth_gradients.json及checkpoint_40000/50000_audit.json。训练开始、预览、所有诊断命令/日志与真实输入preflight保留；没有用诊断产物冒充训练权重或MAE。
