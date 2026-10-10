# 1010-10 jumpingjacks dir 原生PBR来源平面几何投影

更新时间：2026-10-10T06:47:48.996001+09:00
状态：完整20000步训练、全量渲染、全部15测试帧七项MAE及独立复算均完成；最终质量FAILED、7°FAILED，保留完整失败产物，未commit/push。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；保留SH间接光近似及每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度和二值静动态参数冻结。几何2000步warmup、1000步ramp，Canonical PLY位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Plane Preset继承B04全部参数，新增来源平面几何投影。只用未改的35k PLY/MLP、全部135训练相机与time筛选支持：feature<.5、Alpha>.05、canonical与所有训练时刻距source RANSAC/SVD平面≤.2m、全时间形变坐标box-span范数≤.01m、canonical及全部时刻法线轴与plane绝对cos≥.5。支持集合固定，全部99827点保留；实际渲染world XYZ投影到plane、actual quaternion的Z轴旋到source polarity×plane且保留切向旋转与raw norm。训练RGB、PBR/BVH/GGX、周期测试、离线GS Normal/Depth/Mesh/HDR/方向灯共享同一实际几何。不是输出后改法线图、不是改变评分或删除难像素。支持资产来源SHA、全训练相机SHA和精确重算及checkpoint marker绑定；GT材质/几何从未参与支持。

来源MLP输入仍固定源XYZ、feature/time/camera不变，源32tensor保持。Canonical PLY仍限制.08m/60°，但有效渲染位置还经过plane几何模型投影，其总世界位移不能宣称受canonical .08m限制。该额外平面模型假设可能误平坦化来源误分类或真实非平面细节，必须以原始全15帧七项MAE及目检验收。PBR DC初始化和更新后[.031,.969]内部投影保留；本轮仅改变Plane Preset及对应几何投影，材质/光强/loss/lr/其余参数不改。不是GT normal/plane/mesh约束。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15432221 | linear RGB | 96.660979% |
| Roughness MAE | 0.21168068 | r | 96.935993% |
| GS Render Normal MAE | 17.73081984 | degree | 93.884935% |
| Depth Normal MAE | 26.21282757 | degree | 89.256029% |
| Mesh Normal MAE | 20.41400002 | degree | 94.334345% |
| Expected Z Depth MAE | 0.08031497 | m | 93.884935% |
| Median Z Depth MAE | 0.08444845 | m | 93.884935% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。共同有效像素：369930/415433，覆盖率89.046850%；GS缺失预测记180°后的惩罚均角27.65368647°。

| 法线来源 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 16.77003773 |
| depth_normal | 26.15792626 |
| mesh | 19.37026547 |
完整JSON另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.375063141094465, "ssim": 0.8948765238126118, "lpips": 0.11125242908795675, "psnr_foreground": 12.246959743787187, "ssim_foreground_bbox": 0.7720984180768331, "lpips_foreground_bbox": 0.22454677522182465}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。原生SH间接光转灯仍是来源照明近似。

## 四类实际目检：FAILED

- 黑背景正常且主体平台存在；人物头手/衣服仍明显斑驳、稀疏，平台脚周伪起伏/灰块；与B04相近，PSNR略低。GT背光本来暗，不仅凭暗失败。
- 主体平台基本覆盖，边缘薄雾化与稀疏残影；完整数值覆盖率已报告。
- 正背面光照仍烘入衣服/身体，平台棋盘格和脚周污染；冻结feature未修复源静动态误分类。
- GS身体斑驳、平台脚周伪起伏/异常轴；Depth更噪，Mesh局部头手腿缺面，0080明显。15187 plane组实际轴正确，但旧installed kernel在0075等三个训练帧错误反向；不是评分后翻转。

HDR/转灯：HDR与转灯背景黑、人物平台可见且有照明响应；人体/脚周斑驳仍明显。未提供重光照GT，不作精度声明。

实际看0010/0080/0150测试图、0001/0075/0150四类图、HDR0001/0076/0150及转灯0000/0036/0071；六视频完整解码通过，代表图目检不宣称人工逐视频逐帧观看。代表证据：inspection/visual_review.json、rgb_final_55000_comparison.png、four_categories_final_55000.png、relight.png及normal_depth/comparison_0010/0080/0150.png。全部浮点图、Mesh、权重与日志保留。

## 审计及结论

179项CPU单元测试与真实源权重PBR/BVH/GGX GPU梯度检查通过。GeometryRun已共享FrozenScene Anchor/Plane真实几何入口；源模型Plane/B04实际55k首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致。修复只改尚未执行的离线入口，训练源码不改，旧launch代码/manifest与当前修订已保存。checkpoint的有限性、canonical xyz/rotation实际更新并在信赖域内、opacity/scale/feature冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；已完成前后景人工审核且FAILED；后续修复需新完整训练及全量评测后才允许提交和进入目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_plane --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/write_results.log

## 实际训练时间与过程证据

{
  "status": "TRAIN_COMPLETED",
  "training_pid": 245234,
  "process_start_ticks": "8927314",
  "steps_planned": 20000,
  "source_iteration": 35000,
  "target_iteration": 55000,
  "loss_preset": "pbr_gt_directional_plane",
  "session_id": 42059,
  "observed_at": "2026-10-10T05:48:32.269187+09:00",
  "started_at": "2026-10-10T05:48:20.140000+09:00",
  "start_time_source": "proc_btime_plus_start_ticks",
  "finished_at": "2026-10-10T06:33:51.337112+09:00",
  "iteration": 55000,
  "steps_completed": 20000,
  "log_completed_at": "2026-10-10T06:33:46+09:00",
  "loop_duration": "45:06",
  "training_exit_code": 0,
  "verified_training_session_closed": true
}

以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。

## 正式启动记录

{"status": "TRAINING", "training_pid": 245234, "process_start_ticks": "8927314", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_plane", "session_id": 42059, "observed_at": "2026-10-10T05:48:32.269187+09:00", "started_at": "2026-10-10T05:48:20.140000+09:00", "start_time_source": "proc_btime_plus_start_ticks"}

真实train session42059/PID245234/start ticks8927314；后处理session86603/PID245271绑定真实train；40k预览session2961同PID绑定，禁止退出后盲目重启。实际配置相比B04改变Plane Preset/projection、新增阈值.2/.01字段及本轮路径；其它既有参数相同，dir表字节相同。首次launch audit只列5个允许字段，漏列新阈值2项，断言失败；未修改训练，修正审计字段后PASS。150原RGBA来源、99827点全保留、source资产/cameraSHA/15187支持/代码manifest检查PASS，证据inspection/launch_configuration.json。最终MAE/质量待完成，未commit/push。

## 已核验实时进度

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T05:51:27.024252+09:00", "steps": 1395, "total_steps": 20000, "iteration": 36395, "training_pid": 245234, "process_start_ticks": "8927314", "postprocess": "WAITING_LIVE_TRAIN", "checkpoint_40000": false, "source_manifest_matches": true}

完整训练与七项MAE尚未完成，不提前提交/push，未标记目标完成。

## 2026-10-10 几何运行时遗漏更正（待完整重评）

后续源码审计确认旧scripts/eval_geometry_gt.py的GeometryRun仍直接用当前XYZ查询sourceMLP，未消费Anchor资产；逐time Mesh也使用这个入口。先前声明所有几何评测与训练一致不成立。旧GS18.91234950°/Depth27.51005704°/Mesh22.23676747°及2depth/common/missing180虽CPU复算自洽，但不是实际B04 Anchor模型结果，现标记SUPERSEDED。保留原权重/浮点图/原分数/日志，不用旧几何分数评价Anchor效果。RGB/Albedo/Roughness及原生四类渲染已用FrozenScene，受该遗漏影响的仅GeometryRun评测与Mesh；已有RGB四类质量FAILED不因此变PASS。新共享FrozenScene入口已改，179CPU PASS；真实GPU逐buffer验证和全15帧重新Mesh/几何评测待完成。B05训练运行中的source/renderer/参数不改，仅离线入口改并保存launch旧代码及hash记录，不重启训练。

## B05 40k真实检查点/完整15帧RGB预览

RGB：{"psnr": 18.393261800805856, "ssim": 0.8858578602472941, "lpips": 0.10931317210197448, "psnr_foreground": 11.360213718905596, "ssim_foreground_bbox": 0.7518338481585185, "lpips_foreground_bbox": 0.2303248941898346}

Canonical检查：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 40000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.3492876008828607, "max_displacement": 0.03536596521735191, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 5.298036098480225, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

全部99827 Gaussian保留，MLP32tensor原值、sourceXYZ anchor与marker绑定，额外15187支持plane资产/marker SHA及全time筛选统计核验PASS。canonical max位移.03536597m/rotation60.000038°，材质硬clamp0；这不是effective world总位移限制。真实光强5.29803610从RGB拟合，不取GT能量。

实际看0010/0080/0150对照B04同40k/GT：黑背景正常、主体存在；人物/头手/衣服斑驳稀疏、脚周平台伪结构/灰块/残影仍明显，FAILED_SO_FAR。PSNR18.39326 vsB04同40k18.51746略低，不能断言改善；GT背光本来暗，不凭该暗单独判失败。预览exec2961实际exit0，checkpoint CPU audit exec76226实际exit0。40k没有七MAE，不把此预览当最终55k成果；完整20000步/四类/HDR/15七MAE继续。证据inspection/preview_40000_review.json、preview_40000_rgb_comparison.png、checkpoint_40000_audit.json。

已核验实际/proc PID+start ticks：{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T06:05:17.355924+09:00", "steps": 7247, "total_steps": 20000, "iteration": 42247, "training_pid": 245234, "process_start_ticks": "8927314", "checkpoint_40000": true, "postprocess": "WAITING_LIVE_TRAIN", "offline_geometry_evaluator_fixed": true}

## B05运行中只读来源覆盖、法向运动和实际梯度审计

不修改运行中训练/renderer/MLP/参数，不读取GT材质/法线/几何，0optimizer步。所有脚本/log/JSON放本输出commands/logs/inspection；Source SHA及模型fingerprint保持，当前核心manifest仍匹配。

Source35k及既有full135统计：当前15187支持点未投影Alpha×scale0×scale1代理5.610998；同几何candidate条件的静态不稳定点1542（proxy.382483），动态candidate6836（proxy2.905280），全time距离≤.2m且axis abs cos≥.5。低Alpha静态near.2有29588点但proxy仅.047707，不据点数认定主导残影；proxy不是光栅实际贡献，不能据此判属于平台或删除/投影。

真实全部135time进一步查询sourceMLP：6836动态candidate沿plane法向运动span最小.01217797m、中位.12775928m、P99.31366219m，whole-coordinate boxspan中位.390175m。将稳定性从world改为normal span≤.01仍无动态candidate通过，不能声称它们只是沿plane移动而应全投影；推迟扩大支持，后续仍需RGB/来源几何证据。

首次诊断强求asset数组与native runtime重算逐bit一致失败（保留commands/*initial_failed.py及logs/audit_source_plane_normal_motion.log），未改训练或资产。GPU单独核验：asset sigmoid(2×fea)与native get_binary_feature先加减log(.5)路线有max1.19209e−7舍入、8152元素不同但static分类差0；实际span/maxdistance/mincos max差1.78814e−7/1.19209e−7/1.79163e−7。只读统计复查用1e−6有限容差实际PASS，训练资产严格重算校验没有放宽或修改。证据source_feature_roundoff.json、source_plane_normal_motion.json及*_retry.log。

B05实际40k三训练帧0001/0075/0149，真正PBR/BVH/GGX autograd分解；sourceMLP no_grad，Plane projection在其外求导，与Trainer相同，不误用FrozenScene no_grad投影当训练梯度。PBR+SH梯度等于分别求导之和；GS-side+depth-side等于完整ND梯度；全部finite、0optimizer、源SHA/fingerprint未改。动态旋转SH/PBR梯度L2比约2.9166%、1.2186%、2.4791%；ND/PBR约6.2180%、7.4410%、4.3123%。证据不支持SH为主要梯度干扰，也不能由量级推出GT法线准确；ND仍较弱但对自己的噪声深度一致不等于GT正确，不能随意加大后宣称7°。

额外原生预测depth distortion mean×现有Stage1 lambda_dist1000只读诊断：loss约2.64–3.10e−5，动态XYZ梯度/PBR约.0074%–.0193%，动态rotation约.0195%–.0223%；不是该项已用于本轮优化。只加默认1000不能预计明显修复当前geometry；若后续纳入或重标定必须新训练完整MAE验证，不把内部loss当GT normal改善。

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_unprojected_plane_groups.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_unprojected_plane_groups.log；实际exit0。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_source_plane_normal_motion.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_source_plane_normal_motion_retry.log；实际exit0。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_pbr_sh_geometry_gradients.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_pbr_sh_geometry_gradients.log；实际exit0。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_depth_distortion_gradients.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_depth_distortion_gradients.log；实际exit0。

## B05 50k真实检查点

{"status": "PASS_CHECKPOINT_ONLY", "iteration": 50000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.3420851406266174, "max_displacement": 0.07999999821186066, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.189603805541992, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

99827点全保留、source32NNtensor/opacity/scale/feature原值、marker绑定anchor+plane资产；canonical位移/旋转有界、PBR DC可导区间，actual world额外投影不宣称canonical边界。最终55k全部15帧七MAE/四类/HDR待完成，50k不代替最终结果；未commit/push/新建分支。

## B05完整20000步结束，55k与后处理启动

{"status": "TRAIN_COMPLETED", "training_pid": 245234, "process_start_ticks": "8927314", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_plane", "session_id": 42059, "observed_at": "2026-10-10T05:48:32.269187+09:00", "started_at": "2026-10-10T05:48:20.140000+09:00", "start_time_source": "proc_btime_plus_start_ticks", "finished_at": "2026-10-10T06:33:51.337112+09:00", "iteration": 55000, "steps_completed": 20000, "log_completed_at": "2026-10-10T06:33:46+09:00", "loop_duration": "45:06", "training_exit_code": 0, "verified_training_session_closed": true}

源35k→55k全部20k，log 55k保存06:33:45、Training complete06:33:46 JST，loop45:06；训练exec42059实际exit0并关闭。后处理真实/proc childPID252609命令scripts.render_stage2已核验，执行全量all渲染；完整15帧七MAE及四类/HDR最终质量仍待完成，不提前提交/7°。

55k真实canonical检查：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 55000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.34729749132666177, "max_displacement": 0.08000001311302185, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.492673397064209, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

## 来源预测的SUN像素预算只读GPU机制审计

实际B05 40k三TRAIN帧0001/0075/0149，不读测试RGB或GT材质/法线/几何，不优化或修改本轮训练。native_sampled（SH indirect256 enabled）选1024个photo像素，实际对应期望263168 rays（256 hemisphere+1SUN）；关闭间接光后原预算仍除256，direct_sampled只选1024像素/1024SUN rays。仅诊断通过独立args临时大cap40960000调用旧代码，使direct_dense选所有alpha>.9像素18927/15971/19186，SUN实际同数量，全部前/反向finite、source SHA/fingerprint不变，实际exit0。

动态点非零梯度覆盖 direct_sampled→direct_dense：14568→29613、3019→10757、14490→29873；native_sampled含indirect rays给其它Gaussians传播也有30077/30534/30436，不能声称direct_dense整体覆盖超过native_indirect。Forward+gradient时间sampled native约.19–.31s、direct_dense约.09–.13s（并行其它GPU任务环境，非正式benchmark）。只证明single SUN关闭SH间接光时可按实际光线数利用预算，并且全有效photo像素可跑，不能推断画质/7°通过。当前材料/光强已用indirect训练，offlinedirect loss与不同像素集L1不可作为重新训练后的质量比较。

若B05最终仍失败，候选是独立Direct Preset关闭间接光并按实际L个SUN预算训练photo像素，保留原生GGX/BVH/颜色、source和全部评分；改动需在读取完整B05结果后新训练/全七MAE验证。当前没有修改训练参数或新增训练。证据inspection/directional_ray_budget.json，完整命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_directional_ray_budget.py
```

日志logs/audit_directional_ray_budget.log；0optimizer、非最终MAE。

## B05 55k最终RGB与四类早期目检（全七MAE仍运行中）

RGB：{"psnr": 19.375063141094465, "ssim": 0.8948765238126118, "lpips": 0.11125242908795675, "psnr_foreground": 12.246959743787187, "ssim_foreground_bbox": 0.7720984180768331, "lpips_foreground_bbox": 0.22454677522182465}

已实际看全部测试集中首中尾0010/0080/0150及四类0001/0075/0150：背景黑且主体存在，人物头手/衣服斑驳稀疏、脚周平台伪几何/灰块仍明显；alpha基本覆盖但边缘残影，albedo光照烘入，normal身体斑驳。0075平台整体异常green/yellow正在用当前安装光栅器进行来源支持组只读核验，不先假定几何还是坐标问题。PSNR19.37506314 vsB04同55k19.47597408略低，未见足以通过的改善；GT背光本来暗，不凭暗单独失败。状态FAILED_SO_FAR，完整HDR/15帧七MAE仍待完成，未commit/push。证据inspection/final_55000_early_review.json及两张最终拼图。

## B05只读当前CUDA光栅来源与plane法线符号审计

真实installed diff_surfel_rasterization0.0.1的direct_url指向hbb1 f7e2b68a0f17d6d7bca3fa564a96c7678581ce5a，扩展SHA49de0cdbfae3b83f513c0c2a387eba317f9a159aaf69c115b1bff7c615a889e4；不是仓库LumiMotion内的computeTransMat实现。读取该commit本地Git源码可见forward/backward均以normal.view_z<0定向，仓库原生则以-dot(normal_view,position_view)>0定向。

55k 0步分组真实光栅验证：15187投影组world normal绝对plane夹角max.02798°，full alpha/worldnormal与nativePBR buffer maxerror0，证明轴与坐标一致但不证明朝向正确。0075组法线全部-plane而0001/0150全部+plane；其相机均在plane上方。全部150相机中0038/0075/0076的plane.view_z>0且camera-plane side>0，旧Z符号规则会错误朝下，三帧均TRAIN，不直接构成15测试GS MAE变化承诺；身体其它off-axis位置也可能有相同机制。分组独立渲染不是联合贡献，不能据此删除/扩大support或改变mask。

当前B05训练/模型/渲染/MAE仍使用同一旧installed包，保留其完整实际结果不覆盖；不把共享入口自洽误写为物理朝向正确。已启动output/smoke_test/1010-12-dir-PBR原生光栅方向核验独立namespace原仓库CUDA构建验证，不pip install、不改当前环境包/其它分支，不改本轮代码/参数。证据inspection/raster_kernel_provenance.json、plane_raster_normals.json、plane_sign_camera_diagnostic.json、plane_raster_group_comparison.png及commands/audit_plane_raster_normals.py；该实际exec26173 exit0，source权重fingerprint保持且无GT标签。

## 最终验收与B04同55k更正结果比较

| 指标 | B04已更正 | B05 | 差值B05−B04 |
| --- | ---: | ---: | ---: |
| Albedo MAE | 0.15600631 | 0.15432221 | -0.00168410 |
| Roughness MAE | 0.21155915 | 0.21168068 | +0.00012153 |
| GS Render Normal MAE | 18.77743450 | 17.73081984 | -1.04661466 |
| Depth Normal MAE | 27.21989900 | 26.21282757 | -1.00707143 |
| Mesh Normal MAE | 22.15133832 | 20.41400002 | -1.73733829 |
| Expected Z Depth MAE | 0.08569735 | 0.08031497 | -0.00538238 |
| Median Z Depth MAE | 0.08932931 | 0.08444845 | -0.00488086 |

GS改善1.04661466°，Depth/Mesh和两个Z误差均下降；roughness略升、RGB PSNR19.37506314 vs19.47597408略低，仍存在明显伪影，质量FAILED与7°FAILED。全前景评分定义不变，全部99827点/15测试帧保留。共同369930/415433（89.04684991%），GS/Depth/Mesh共同MAE16.77003773°/26.15792626°/19.37026547°，missing180为27.65368647°。

训练exec42059及后处理exec86603均实际exit0并关闭，9步骤均exit0；后处理结束06:47:49 JST。179CPU、逐buffer共享运行时、92来源SHA及15time×135train Mesh、roughness独立15帧/41来源、六视频完整解码PASS；不将数值执行PASS当质量PASS。核心manifest与当时源/快照逐SHA核验PASS。

旧installed CUDA实际kernel来源和法线Z符号缺陷已记录inspection/raster_kernel_provenance.json。B05训练/渲染/评分共享该旧kernel，分数是本轮实际模型结果，不覆盖为新kernel分数。后续独立原生构建诊断0optimizer、无GT标签且不改变环境installed包；不作为已完成的新训练或质量提升。
