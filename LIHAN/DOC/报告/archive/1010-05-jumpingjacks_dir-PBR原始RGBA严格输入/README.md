# 1010-05 jumpingjacks dir 原生PBR原始RGBA严格输入

更新时间：2026-10-10T03:08:19.550962+09:00
状态：COMPLETED_QUALITY_FAILED：完整20000步训练、全15帧七项MAE与独立复算PASS；实际四类/前后景/重光照质量FAILED，法线7°FAILED，禁止commit。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；保留SH间接光近似及每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度和二值静动态参数冻结。几何2000步warmup、1000步ramp，位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.18603245 | linear RGB | 96.517409% |
| Roughness MAE | 0.24401021 | r | 96.793436% |
| GS Render Normal MAE | 19.40743705 | degree | 93.694049% |
| Depth Normal MAE | 27.60153971 | degree | 89.068995% |
| Mesh Normal MAE | 22.24380070 | degree | 94.072209% |
| Expected Z Depth MAE | 0.08647351 | m | 93.694049% |
| Median Z Depth MAE | 0.09010209 | m | 93.694049% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。三法线共同像素结果另存metrics_summary.json。

## RGB及渲染

{"psnr": 18.912289595136517, "ssim": 0.8901072422663371, "lpips": 0.11933969706296921, "psnr_foreground": 11.9928638250002, "ssim_foreground_bbox": 0.7621153076489766, "lpips_foreground_bbox": 0.24486602246761321}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。原生SH间接光转灯仍是来源照明近似。

## 四类与几何实际目检：FAILED

实际查看四类0001/0075/0150、RGB0010/0080/0150、三法线/Depth0010/0080/0150、HDR0001/0076/0150与转灯0000/0036/0071。仅首中尾人工审阅，不声称全150帧逐张人工审核；全量图片/视频完整性由自动审计复核。

- RGB：黑背景正常、人物和平台存在；人体皮肤/衣服仍斑驳、头手/边界缺失，平台棋盘格未恢复，脚周残色和伪结构。
- Alpha：主体轮廓基本完整，头手边毛糙、平台边残影；不是全前景消失。
- Albedo：明正面/暗背面照明结构严重残留，平台多白；材质分解未通过。静动态feature冻结。
- Normal/Depth/Mesh：台面较基线平滑，但脚周明显伪起伏、人体法线斑驳、Depth Normal更噪、Mesh头手腿缺面；脚周深度误差明显。
- HDR/转灯：随光变化、场景存在、背景正常；斑驳与伪结构持续，HDR橙粉色；没有对应GT，不判重光照颜色准确性。

证据inspection/four_categories_final_55000.png、rgb_final_55000_comparison.png、relight_final_55000.png、visual_review.json及normal_depth/comparison_*.png。全量权重、浮点数组、渲染、日志与失败证据保留。

## 审计及结论

147项CPU单元测试和真实权重单帧GPU梯度检查通过。checkpoint的有限性、xyz/rotation实际更新并在信赖域内、opacity/scale/feature冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；实际质量失败，保留本轮；开始独立B03完整修复训练，验收通过后才能提交/push及建立目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_geometry --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/logs/write_results.log

## 时间、覆盖率与修复诊断

训练02:10:11.814至02:54:30 JST，完整20000步，loop44:02、实际exit0；后处理03:08:19结束exit0。三法线共同像素369136/415433（88.855724%），GS 18.51964521°、Depth 27.54664188°、Mesh 21.26382698°；GS缺失按180°惩罚 29.53432483°。不删除帧、难像素或Gaussian，99827全保留。

50k预测albedo硬clamp通道71.7531%，来源35k为43.5029%；支持下一轮仅修复预测材质DC可导范围的方案，不使用GT材质/法线。40k预览/50k检查证据保留。B03 preset新增初始化和原生每4步更新后投影[.031,.969]，同数据/35k/20k、同几何与评测定义。
