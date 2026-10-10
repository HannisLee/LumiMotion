# 1010-12 jumpingjacks dir 原生PBR仓库原生光栅修复

更新时间：2026-10-10T07:58:10.829884+09:00
状态：完整20000步训练、全量渲染、全部15测试帧七项MAE与独立复算完成；实际前后景/四类/HDR/Normal Depth Mesh质量FAILED、7°FAILED，保留完整失败产物，未commit/push。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；保留SH间接光近似及每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度和二值静动态参数冻结。几何2000步warmup、1000步ramp，Canonical PLY位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Plane Native Preset继承B05全部模型参数，包括既有来源平面几何投影。只用未改的35k PLY/MLP、全部135训练相机与time筛选支持：feature<.5、Alpha>.05、canonical与所有训练时刻距source RANSAC/SVD平面≤.2m、全时间形变坐标box-span范数≤.01m、canonical及全部时刻法线轴与plane绝对cos≥.5。支持集合固定，全部99827点保留；实际渲染world XYZ投影到plane、actual quaternion的Z轴旋到source polarity×plane且保留切向旋转与raw norm。训练RGB、PBR/BVH/GGX、周期测试、离线GS Normal/Depth/Mesh/HDR/方向灯共享同一实际几何。不是输出后改法线图、不是改变评分或删除难像素。支持资产来源SHA、全训练相机SHA和精确重算及checkpoint marker绑定；GT材质/几何从未参与支持。

来源MLP输入仍固定源XYZ、feature/time/camera不变，源32tensor保持。Canonical PLY仍限制.08m/60°，但有效渲染位置还经过plane几何模型投影，其总世界位移不能宣称受canonical .08m限制。该额外平面模型假设可能误平坦化来源误分类或真实非平面细节，必须以原始全15帧七项MAE及目检验收。PBR DC初始化和更新后[.031,.969]内部投影保留；本轮继承B05的Plane模型，只改变仓库原生CUDA backend与Native Preset；材质/光强/loss/lr/间接光/几何参数不改。不是GT normal/plane/mesh约束。

## 本次实际原生CUDA修复

B05使用环境installed hbb1 f7e2b68a的diff_surfel_rasterization0.0.1，以camera-space normal.z符号定向。实测来源plane上方相机0038/0075/0076被错误反向。B06显式构建仓库LumiMotion原始CUDA，逐Gaussian按实际观察射线-dot(n_view,p_view)定向，forward/backward同规则；保留原始T/光栅/BRDF/GGX/BVH实现，不在最终normal图或评分中翻转法线。独立namespace，不pip install，不改变Stage1/env/旧Preset默认installed后端，不覆盖B05结果。

新增pbr_gt_directional_native和pbr_gt_directional_plane_native，本轮后者；pbr_rasterizer_backend=lumimotion。来源资产pbr_rasterizer.json记录CUDA/Python/GLM每文件SHA与综合SHA、真实构建.so SHA和torch版本；checkpoint marker绑定综合SHA，加载/评分核验源，禁止静默回退。原嵌套GLM为空，使用同仓库已追踪stable目录的GLM头文件。仅支持scale/rotation CUDA路径，拒绝compute_cov3D_python=True。训练、PBR RGB/alpha/normal、HDR/方向光、GeometryRun/逐time Mesh、albedo与roughness材质helper均共享显式后端；actual albedo/rough缓冲首中尾相等验证PASS。

187 CPU最终discover实际exit0；20步真实Trainer smoke保存35020并加载marker，三TRAIN帧GGX/BVH/material/XYZ/rotation梯度非零finite、0额外optimizer；GeometryRun alpha/worldnormal/ExpectedZ与实际PBR全buffer maxerror0、Mesh targettime override一致。独立原生0步已知plane+三坏视角GPU法线camera-facing正确，合成normal梯度有限差分maxerror.00035429。HDR16三帧只作smoke，正式仍150帧2048；smoke不作全量MAE/画质/7°通过。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.16231946 | linear RGB | 96.784312% |
| Roughness MAE | 0.21780360 | r | 97.047548% |
| GS Render Normal MAE | 17.29779705 | degree | 94.101335% |
| Depth Normal MAE | 26.04174483 | degree | 89.466894% |
| Mesh Normal MAE | 20.34490562 | degree | 94.485513% |
| Expected Z Depth MAE | 0.08001558 | m | 94.101335% |
| Median Z Depth MAE | 0.08431691 | m | 94.101335% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。共同有效像素：370823/415433，覆盖率89.261806%；GS缺失预测记180°后的惩罚均角26.89505457°。

| 法线来源 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 16.31552150 |
| depth_normal | 25.98854287 |
| mesh | 19.29566234 |
完整JSON另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.682202597909917, "ssim": 0.8977724393208821, "lpips": 0.11070925494035085, "psnr_foreground": 12.488568200962481, "ssim_foreground_bbox": 0.7782925049463908, "lpips_foreground_bbox": 0.221701983610789}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。原生SH间接光转灯仍是来源照明近似。

## 四类实际目检：FAILED

- 黑背景正常，人物/平台存在；头手/衣服仍斑驳稀疏，脚周平台伪凸起/灰黑污染，B05/B06整体接近。RGB PSNR略改善但不足验收；GT0080本身背光暗，不只凭暗失败。
- 主体和平台基本覆盖，腰手边缘/脚周细碎薄雾残影；数值覆盖率全部报告，不从alpha饱和推断正确几何。
- 正背面照明烘入材质、平台棋盘格和脚周污染；静动态feature冻结，来源误分未修复。Albedo与Roughness MAE较B05上升。
- GS平台整体向上，0075不再旧installed整个plane反向，实际原生核修复有效；身体/脚周混合轴斑驳及伪凸起仍有。Expected/Median Z脚周明显凸起，Depth Normal高频噪声，Mesh0080下腿等缺面、0010/0150身体和平台仍不平整。

HDR/转灯：HDR0001/0076/0150与转灯0000/0036/0071已实际看：黑背景/主体/光照响应正常但人物和脚周伪影仍明显；无重光照GT，不声明精度。

已实际view上列代表图，6视频逐帧完整解码PASS；代表帧实际目检不宣称人工逐帧观看全部视频。GT只用于独立离线评测，没有生成任何训练GT geometry/normal/material prior。

## 审计及结论

187项CPU单元测试与真实源权重PBR/BVH/GGX GPU梯度检查通过。GeometryRun已共享FrozenScene Anchor/Plane真实几何入口；本轮35020 native smoke首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致，材质helper一致；完整训练与离线使用同一native来源。完整launch源码/manifest和CUDA/GLM快照保存。checkpoint的有限性、canonical xyz/rotation实际更新并在信赖域内、opacity/scale/feature冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；已实际完成前后景/四类/HDR与Normal/Depth/Mesh审核且FAILED，需继续修复和新完整训练评测，质量验收后才提交并进入目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/logs/write_results.log

## 实际训练时间与过程证据

{
  "status": "TRAIN_COMPLETED",
  "training_pid": 258688,
  "process_start_ticks": "9357639",
  "session_id": 30746,
  "source_iteration": 35000,
  "target_iteration": 55000,
  "steps_planned": 20000,
  "loss_preset": "pbr_gt_directional_plane_native",
  "started_at": "2026-10-10T07:00:03.390000+09:00",
  "start_time_source": "proc_btime_plus_start_ticks",
  "observed_at": "2026-10-10T07:00:34.438075+09:00",
  "argv": [
    "/home/lihan/miniconda3/envs/lumimotion/bin/python",
    "-u",
    "-m",
    "scripts.train_stage2",
    "--source_path",
    "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/data_view",
    "--stage1_model_path",
    "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp",
    "--model_path",
    "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model",
    "--load_iter",
    "35000",
    "--iterations",
    "55000",
    "--save_iterations",
    "40000",
    "50000",
    "55000",
    "--test_iterations",
    "40000",
    "55000",
    "--render_mode",
    "original_ir",
    "--loss_preset",
    "pbr_gt_directional_plane_native",
    "--pbr_gt_lights_path",
    "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/gt_directions_only.json",
    "--pbr_light_intensity_scale",
    "3.141592653589793",
    "--depth_ratio",
    "0",
    "--resolution",
    "2",
    "--is_blender",
    "--eval",
    "--load2gpu_on_the_fly",
    "--gt_alpha_mask_as_scene_mask",
    "--train_light_folder",
    "images",
    "--test_light_folder",
    "images",
    "--deform-type",
    "mlp",
    "--sh_degree",
    "3",
    "--hyper_dim",
    "1"
  ],
  "finished_at": "2026-10-10T07:44:26.888854+09:00",
  "iteration": 55000,
  "steps_completed": 20000,
  "training_exit_code": 0,
  "verified_training_session_closed": true,
  "log_completed_at": "2026-10-10T07:44:25+09:00",
  "loop_duration": "44:02"
}

以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。

## 正式启动：真实进程

{"status": "TRAINING_VERIFIED_LIVE", "training_pid": 258688, "process_start_ticks": "9357639", "session_id": 30746, "source_iteration": 35000, "target_iteration": 55000, "steps_planned": 20000, "loss_preset": "pbr_gt_directional_plane_native", "started_at": "2026-10-10T07:00:03.390000+09:00", "start_time_source": "proc_btime_plus_start_ticks", "observed_at": "2026-10-10T07:00:34.438075+09:00", "argv": ["/home/lihan/miniconda3/envs/lumimotion/bin/python", "-u", "-m", "scripts.train_stage2", "--source_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/data_view", "--stage1_model_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp", "--model_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model", "--load_iter", "35000", "--iterations", "55000", "--save_iterations", "40000", "50000", "55000", "--test_iterations", "40000", "55000", "--render_mode", "original_ir", "--loss_preset", "pbr_gt_directional_plane_native", "--pbr_gt_lights_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/gt_directions_only.json", "--pbr_light_intensity_scale", "3.141592653589793", "--depth_ratio", "0", "--resolution", "2", "--is_blender", "--eval", "--load2gpu_on_the_fly", "--gt_alpha_mask_as_scene_mask", "--train_light_folder", "images", "--test_light_folder", "images", "--deform-type", "mlp", "--sh_degree", "3", "--hyper_dim", "1"]}

仅新Stage2，未重跑Stage1；完整MAE/最终质量尚无，不把source35000或20步smoke作本轮结果。

## B06启动配置/后处理核验

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T07:03:19.241733+09:00", "steps": 1470, "total_steps": 20000, "iteration": 36470, "source_cuda_sha256": "82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0", "process_handles": {"train": {"pid": 258688, "session_id": 30746, "start_ticks": "9357639"}, "postprocess": {"pid": 258864, "session_id": 47576, "start_ticks": "9362140"}}, "postprocess": "WAITING_LIVE_TRAIN", "final_mae": false}

正式配置仅loss_preset、backend和3本轮路径字段与B05不同，所有其它值一致；15187支持/所有来源plane数组逐元素与B05相同。源码manifest逐SHA核验PASS，实际仓库native CUDA综合SHA82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0绑定。首次inline审计误用默认vllm python缺smplx失败，未改训练；正确venus lumimotion执行归档commands/audit_launch.py实际exit0。真实后处理绑定同train PID/start ticks，完整9步将在train完成后执行，40k预览同PID绑定，不退出后盲重启。完整MAE/质量仍PENDING，未commit/push/新分支。

## B06实际进度与进程绑定更新

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T07:04:56.276364+09:00", "steps": 2261, "total_steps": 20000, "iteration": 37261, "process_handles": {"train": {"pid": 258688, "session_id": 30746, "start_ticks": "9357639"}, "postprocess": {"pid": 258864, "session_id": 47576, "start_ticks": "9362140"}, "preview_40000": {"pid": 258868, "session_id": 25842, "start_ticks": "9362256"}}, "postprocess": "WAITING_LIVE_TRAIN", "final_mae": false, "commit_allowed": false}

TRAIN/POST/40k-preview实际PID/start ticks/cwd全部复核，40k预览cmdline为相对脚本路径，按resolve核验，不因绝对路径子串不匹配盲重启。核心469源码/GLM/快照SHA仍匹配launch。仅修正文档生成器末尾遗留1010-10引用为本轮1010-12，保留修正前脚本；运行中训练/renderer/MLP/参数未改。

## B06 40k真实检查点、全15RGB预览与只读gate诊断（2026-10-10T07:15:26.674133+09:00）

完整训练仍运行：6962/20000，实际iteration41962；train258688/startticks9357639与post258864/startticks9362140/cwd重新核验，核心469来源/GLM源码SHA匹配launch，未改变本轮训练参数或源码。preview exec25842、checkpoint审计exec1420、gate诊断exec90456均实际exit0并关闭。

检查点：全部99827 Gaussian保留，opacity/scale/feature与source32NNtensor冻结不变，XYZ最大位移0.03539922m、canonical旋转最大60.00002670°，DC RGB范围[.031,.969]、硬clamp0，训练RGB拟合光强5.28000259。native CUDA marker来源与anchor/plane绑定通过；这些只是保存完整性，不是法线7°或质量PASS。

15 TEST帧40k RGB：PSNR18.51417174、SSIM0.88792369、LPIPS0.10788124、前景PSNR11.46161042、前景bbox SSIM0.75713942/LPIPS0.22566057。B05同40k PSNR18.39326180/LPIPS0.10931317，略改善，不能由此声称GS法线改善。

已实际view inspection/preview_40000_comparison.png中GT/B05/B06的0010/0080/0150：黑背景正常、人物和平台存在，但头手细节欠完整、衣服斑驳、脚周平台伪凸起/灰黑残影仍明显，两个预测总体接近；0080 GT本身暗，不以暗单独判失败。中间质量FAILED_SO_FAR，55k完整七MAE/四类/HDR仍PENDING，未commit/push/新建分支。

只读gate诊断：source NN隐藏层与32参数固定，只开放当前feature logit的输出乘法gate自动求导；当前位置/旋转形变与原FrozenScene路径逐bit相等。0001/0075/0149三TRAIN帧PBR/GGX/BVH的RGB+SH、normal-depth和自身输入Alpha梯度均finite且存在非零梯度，source SHA/状态fingerprint不变；0optimizer、feature/NN值不变、未读GT标签/测试RGB、未生成新训练prior。本轮仍冻结gate。来源预测筛选的6836动态plane候选不是GT物体标签，梯度方向不证明它们全部属于平台。

| TRAIN帧 | RGB+SH gate梯度L2 | 非零点 | 来源动态plane候选RGB+SH L2 | 候选非零点 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.0005630466 | 34817 | 0.00024462515 | 4672 |
| 0075 | 0.01247051 | 36220 | 0.010364828 | 5064 |
| 0149 | 0.002293421 | 34628 | 8.5120133e-05 | 4629 |


该机制可作为本轮最终失败后的新候选，仅证明可导性，未实施新训练/未声称能到7°。不能从0075的大梯度推定正确几何或把候选全部静态化；需要重新读取本GOAL、独立完整训练与全量MAE验证。

完整诊断命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_feature_gate_gradients.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 40000
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_preview_comparison.py
```

证据：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/inspection/feature_gate_gradients.json、checkpoint_40000_audit.json、preview_40000_review.json、preview_40000_comparison.png；logs/audit_feature_gate_gradients.log、checkpoint_40000_audit.log、preview_40000.log、preview_render_40000.log。预览完整命令另在commands/preview_40000_argv.json；所有失败轮保持原权重与统计，最终MAE尚无。

## B06原生40k联合几何贡献审计（2026-10-10T07:19:17.158800+09:00）

完整训练继续：8684/20000。0optimizer、无GT材质/normal/geometry、未改本轮源码/参数。只读实际40k六帧0001/0038/0075/0076/0149/0150的仓库native CUDA光栅，三原坏TRAIN视角0038/0075/0076的projected支持组都朝向camera所在的plane上方，所有signed dot最小≥.99999988。PBR的alpha/worldnormal/Expected Z与原始native光栅逐buffer maxerror0，当前NN和权重fingerprint不变、469核心/GLM SHA保持launch。

联合贡献核验：全部99827点每个pass均用真实原opacity/position/rotation，输入组one-hot颜色而非只渲单组，得到真实联合alpha合成权重；四组贡献总和与全alpha最大误差6.56e−7，重复pass几何buffer0。下表为全图alpha>.5内最大权重来源组的像素数量，不是GT物体分区、不是评测mask，不用于丢点/丢像素。

| 帧 | 投影plane主导像素 | 动态plane候选主导 | 其它动态主导 | 静态非投影主导 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 1990 | 4886 | 12057 | 1047 |
| 0038 | 4006 | 3861 | 8057 | 1150 |
| 0075 | 2610 | 2573 | 8774 | 2808 |
| 0076 | 2459 | 2688 | 8885 | 2742 |
| 0149 | 1893 | 5098 | 12314 | 954 |
| 0150 | 1922 | 4943 | 12090 | 1096 |


已实际view inspection/native_joint_geometry_comparison.png首中尾0001/0075/0150：投影组单独形成完整向上的平面，但联合图中平台被其它来源组大量遮盖；来源动态plane候选红组主要出现在平台，其它动态绿组也占据脚周/平台，人体主要绿组，衣服含白色静态非投影贡献。因此native符号修复真实生效，但它不会自动把未支持组的geometry变成平面。不能将source候选当真实平台标签或直接删除它们，不能以投影组单独的平整图片验收完整结果。

候选审计补充（尚未修改/启动）：gate只读梯度证明可通过原RGB/自身Alpha/normal-depth自洽调整时间形变幅度；若最终B06失败，可独立评估可导gate微调并绑定来源gate信赖域/先验，保持source NN与所有点、统一训练/评测入口及marker。当前Plane asset重算用source35k feature，不要求改变来源选择；需要明确当前可学习gate与固定source支持组语义。不能根据该分组图预设所有candidate必须静态、不得读取GT正常向量训练。其它候选actual SUN ray预算/Direct Preset仍未实施；需最终完整七MAE和读取GOAL后选择一个独立新实验。

第一次exec96086因诊断import误写get_rasterizer_types（真实API rasterizer_classes）exit1，未渲染/未改训练；原脚本commands/audit_native_joint_geometry_before_api_name_fix.py与logs/audit_native_joint_geometry.log保留。只修正诊断API后retry exec17809实际exit0并关闭，生成18张来源贡献/normal图和inspection/native_joint_geometry.json。实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_native_joint_geometry.py
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_joint_geometry_comparison.py
```

日志logs/audit_native_joint_geometry_retry.log；证据inspection/native_joint_geometry.json、native_joint_geometry_review.json、native_joint_geometry_comparison.png与native_joint_geometry_40000。只读机制PASS、最终质量/七MAE/7°仍PENDING，未commit/push/新建目标分支。

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

## B06 50k真实保存后审计的等待句柄

2026-10-10T07:24:13.431998+09:00：新增只读50k检查点审计等待进程262910/startticks9501104/exec81677；/proc实际存活/cwd/命令已核验，绑定train258688/startticks9357639及完整argv，在真实50k marker出现后执行commands/audit_checkpoint.py 50000。此刻WAITING_LIVE_TRAIN_50000，不预先声称审计完成；训练结束前若没有50k则拒绝自动重启。日志logs/audit_50000_when_saved.log及logs/checkpoint_50000_audit.log（待实际保存后生成），状态inspection/audit_50000_wait_state.json。训练与最终9步后处理未改，最终质量/七MAE仍PENDING。

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

## B06完整20000步结束、55k实际退出与最终RGB早期目检（2026-10-10T07:46:40.050962+09:00）

完整Stage2源35000→55000共20000步，07:00:03.390000 JST启动，log保存55k与Training complete均07:44:25，循环44:02；train exec30746实际exit0并关闭，真实PID258688已退出，不是观察超时。training_runtime.json已记录verified_training_session_closed/actual exit0，Stage1未重训。

55k独立检查exec88063实际exit0：全部99827 Gaussian保留、数值有限、opacity/scale/feature及32来源NNtensor原值、sourceSHA保持。最大canonical XYZ位移0.08000001311m、旋转60.00006485°容差内；DC [.031,.969]硬clamp0，原RGB拟合E6.61184454。source XYZ anchor、plane和native CUDA来源绑定通过，核心469文件/GLM hash仍匹配launch。该检查不是最终GT normal或质量验收。

55k最终15 TEST RGB：PSNR19.68220260、SSIM0.89777244、LPIPS0.11070925、前景PSNR12.48856820、前景bbox SSIM0.77829250/LPIPS0.22170198。B05同55k PSNR19.37506314→本轮19.68220260（+.30713946dB）；LPIPS.11125243→.11070925，稍改善，不等于normal7°或画质PASS。

已实际view inspection/rgb_final_55000_comparison.png中GT/B05/B06首中尾0010/0080/0150：黑背景/主体存在，衣服、头手稀疏斑驳，脚周平台伪凸起/灰黑污染仍明显；两轮总体接近，GT0080本身背光暗，不凭暗单独失败。当前FAILED_SO_FAR_EARLY_FINAL_RGB_ONLY，完整四类/HDR/方向光、七MAE仍PENDING，未commit/push/新建目标分支。

真实后处理PID258864/startticks9362140正在全量render；child265977/startticks9623907、cwd与scripts.render_stage2 argv实际核验存活，使用--task all --load_iter55000 --hdr_samples2048。后续albedo、15time×135TRAIN Mesh、normal_depth、roughness、独立七MAE复算、产物审核、报告仍须完成，不用RGB指标代替全部七MAE。没有重启训练或更改已运行参数/源。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 55000
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_rgb_comparison.py
```

证据training_runtime.json、inspection/checkpoint_55000_audit.json、final_55000_early_review.json、rgb_final_55000_comparison.png、model_mlp/renders_stage2/ours_55000/eval/metrics.json；日志logs/train.log、checkpoint_55000_audit.log、render.log、postprocess.log。完整训练/渲染命令在planned_commands.json和commands/train.sh/render.sh。

## B06最终四类实际目检（2026-10-10T07:47:59.665924+09:00，完整七MAE仍进行中）

150×4=600张insights的RGB/Alpha/normal/albedo已实际生成，已view首中尾0001/0075/0150的inspection/four_categories_final_55000.png。RGB黑背景正常/主体平台存在，但头手、衣服和脚周平台仍斑驳伪影；Alpha大部饱和，腰手边缘/脚周薄雾与细碎残影。Albedo正背面照明依旧烘入，平台棋盘与脚周污染；GS Normal整体平台向上、0075不再旧包整个平台朝下，但人体/脚周混合轴与斑驳仍有。

本轮40k原生来源支持组正确符号与55k全图正常plane相互支持实际kernel修复生效，但不能据此推定所有Gaussian或GT normal准确；最终Depth/Mesh及15帧七MAE/coverage/common/missing180待完成。转灯/HDR正渲染，不能提前声明已审核。阶段质量FAILED_SO_FAR，仍未commit/push/新分支。证据inspection/final_55000_four_category_review.json、four_categories_final_55000.png；拼图命令python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_four_categories.py（实际exit0）。

## B06完整渲染及HDR/转灯实际目检（2026-10-10T07:56:04.688664+09:00）

全量render子进程实际exit0（runtime_history.jsonl），结束07:54:59.887392 JST，用时632.9978秒；RENDER_COMPLETED=all。实际150×4四类、150 HDR2048样本、72转灯和6视频存在（完整视频解码另由最终产物审计执行，不提前标通过）；后处理进入材质/几何评测。实际view inspection/relight_final_55000.png：HDR0001/0076/0150和转灯0000/0036/0071，黑背景/主体平台存在、光照有响应，但人物正背面和脚周斑驳、伪起伏/材质污染仍明显，仍FAILED_SO_FAR；无重光照GT，不声明精度。方向光固定0001相机/time/geometry转灯，alpha逐bit保持，亮度全图均值范围.01817811653–.07329406589。

灯表重新核验150帧各SUN白色单位intensity1/light_rgb[1,1,1]，仅原GT方向；实际转灯强度是该白色单位表sum×保存的训练RGB拟合scale6.611844539642334。relight.json遗留generic字符串“frame GT RGB irradiance sum × saved scale”不是使用原GT color/intensity的证据，本轮保留原渲染metadata、另在报告澄清，不改已完成渲染/核心源码。最终结果后再读GOAL决定修正文案，当前输入边界未变。

命令python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_relight_comparison.py实际exit0；证据inspection/final_55000_relight_review.json、relight_final_55000.png、model_mlp/renders_stage2/ours_55000/directional/relight.json及hdr/hdr.json。全部15七MAE与Depth/Mesh/roughness审核仍PENDING，未commit/push/新建目标分支。

## B06最终完整验收与B05同55k比较（2026-10-10T08:01:26.105680+09:00）

| 指标 | B05同55k | B06原生CUDA | B06−B05 |
| --- | ---: | ---: | ---: |
| Albedo MAE | 0.15432221 | 0.16231946 | +0.00799725 |
| Roughness MAE | 0.21168068 | 0.21780360 | +0.00612293 |
| GS Render Normal MAE | 17.73081984 | 17.29779705 | -0.43302279 |
| Depth Normal MAE | 26.21282757 | 26.04174483 | -0.17108274 |
| Mesh Normal MAE | 20.41400002 | 20.34490562 | -0.06909441 |
| Expected Z Depth MAE | 0.08031497 | 0.08001558 | -0.00029939 |
| Median Z Depth MAE | 0.08444845 | 0.08431691 | -0.00013154 |

GS法线改善0.43302279°、覆盖率93.88493451%→94.10133523%；Depth/Mesh/Z略改善，Albedo和Roughness更差。RGB PSNR19.37506314→19.68220260dB，LPIPS0.11125243→0.11070925，略改善但人物/脚周伪影明显，最终质量FAILED、7°FAILED。全部99827点与15测试帧保留，评分/GT前景域不变。

共同370823/415433（89.26180636%），GS/Depth/Mesh共同MAE16.31552150°/25.98854287°/19.29566234°；missing180=26.89505457°。

完整训练exec30746实际exit0，20k/55k于07:44:25完成，loop44:02；后处理exec47576实际exit0并关闭、9步骤全0，于07:58:11结束。187 CPU/真实20步GPU/共享逐buffer入口已验证，533来源SHA、15time×135TRAIN Mesh、481 roughness来源与独立15帧复算、6视频全解码PASS；这些不等于质量PASS。469核心/GLM及launch快照SHA再次一致，source32NN/冻结opacity-scale-feature保持。Native源SHA82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0绑定、原生实际编译.so SHA记录在pbr_rasterizer.json，不回退installed。B05旧kernel的实际结果保持，不覆盖为新kernel评分。
