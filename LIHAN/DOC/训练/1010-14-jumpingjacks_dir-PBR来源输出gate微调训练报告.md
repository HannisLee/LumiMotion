# 1010-14 jumpingjacks dir 原生PBR来源输出gate微调

更新时间：2026-10-10T09:52:24.482406+09:00
状态：完整20000步训练、全量渲染和15测试帧七MAE/独立复算完成；人工质量FAILED，原GS Normal <7°目标FAILED。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；关闭训练与重光照的Stage1 SH间接光近似，原生SH辅助RGB损失保留，每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度冻结；feature原值初始化并优化输出gate，来源MLP全部32tensor冻结。几何2000步warmup、1000步ramp，Canonical PLY位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Plane Native Direct Gate Preset继承B07全部材质/几何参数，包括既有来源平面几何投影。只用未改的35k PLY/MLP、全部135训练相机与time筛选支持：feature<.5、Alpha>.05、canonical与所有训练时刻距source RANSAC/SVD平面≤.2m、全时间形变坐标box-span范数≤.01m、canonical及全部时刻法线轴与plane绝对cos≥.5。支持集合固定，全部99827点保留；实际渲染world XYZ投影到plane、actual quaternion的Z轴旋到source polarity×plane且保留切向旋转与raw norm。训练RGB、PBR/BVH/GGX、周期测试、离线GS Normal/Depth/Mesh/HDR/方向灯共享同一实际几何。不是输出后改法线图、不是改变评分或删除难像素。支持资产来源SHA、全训练相机SHA和精确重算及checkpoint marker绑定；GT材质/几何从未参与支持。

来源MLP输入仍固定源XYZ、time/camera保持，feature gate按训练RGB/Alpha/预测ND优化，源32 MLP tensor保持。Canonical PLY仍限制.08m/60°，但有效渲染位置还经过plane几何模型投影，其总世界位移不能宣称受canonical .08m限制。该额外平面模型假设可能误平坦化来源误分类或真实非平面细节，必须以原始全15帧七项MAE及目检验收。PBR DC初始化和更新后[.031,.969]内部投影保留；本轮继承B07 Direct Native/Plane，只新增feature输出gate优化；其它材质/光强/loss/lr/几何参数保持。不是GT normal/plane/mesh约束。

## 本次来源输出gate与Preset

Preset pbr_gt_directional_plane_native_direct_gate继承B07，显式pbr_optimize_feature_gate=True；全99827点feature原值初始化，source MLP全部32 tensor关闭梯度和optimizer，sourceXYZ输入冻结。原native Deform乘法gate可导；geometry warmup2000/ramp1000/decay同步gate，lr.001、source raw-logit最大偏移4、anchor.001，每4步Adam后投影source±4。不强制source候选当静态，不筛/删Gaussian或难像素，旧Preset flagFalse保持。固定source plane15187与B07相同；额外world plane投影不等于canonical .08m。

gate metadata保存来源SHA/config、checkpoint marker绑定其SHA，FrozenScene/GeometryRun/材质helper统一核验source32tensor、gate finite/trust/source配置。全部渲染/离线评测用保存PLY feature，不退回source gate。MLP hidden冻结、只有原生输出乘法feature梯度；原GGX/BVH/native raster/Direct实际SUN预算及原SH辅助loss不改。Directions白色单位×训练RGB拟合scale，无GT光强/颜色。原生Adam状态未完整保存，不声明无损resume。

202 CPU回归实际exit0，20步真实Trainer及sourceNN/gate保存检查、五参数GPU梯度/实际SUN计数/共享geometry-material逐buffer与目标time一致PASS。GPU photo gate L2三帧.00085692690/.00269499072/.00030556315；保存68101 feature变化、maxlogit1.85966492e−5，source32tensor/opacity/scale保持。首次201项中2个mock缺新增字段错误日志保留，仅修fixture/增metadata测试后202通过，训练生产源码未改。全部四类150/4视频及HDR16/SUN仅功能smoke，实际质量FAILED_SMOKE_ONLY，无正式7°指标。

独立完整35k→55k验证gate是否改善，不使用20步smoke或B07训练后权重。全15七MAE/coverage/common/missing180和四类、Depth/Mesh/HDR/SUN按原口径验收。GT材质/法线/几何只能离线评测；模型、输入、失败产物保留。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.20137736 | linear RGB | 97.269167% |
| Roughness MAE | 0.20119953 | r | 97.494595% |
| GS Render Normal MAE | 16.84133987 | degree | 95.012674% |
| Depth Normal MAE | 25.82059126 | degree | 90.332737% |
| Mesh Normal MAE | 20.07707837 | degree | 95.004249% |
| Expected Z Depth MAE | 0.07851869 | m | 95.012674% |
| Median Z Depth MAE | 0.08287710 | m | 95.012674% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。共同有效像素：374301/415433，覆盖率90.099005%；GS缺失预测记180°后的惩罚均角24.97859493°。

| 法线来源 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 15.81731046 |
| depth_normal | 25.76318049 |
| mesh | 19.06113108 |
完整JSON另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.24938274959784, "ssim": 0.9107702334721883, "lpips": 0.1102573330203692, "psnr_foreground": 11.887298783047992, "ssim_foreground_bbox": 0.8042977293332417, "lpips_foreground_bbox": 0.21988034347693125}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。本轮关闭来源SH间接近似，HDR/SUN仍使用原生GGX/BVH可见性；无重光照GT准确率声明。

## 四类实际目检：FAILED

- 黑背景正常，人物与平台存在；头手/衣服/小腿仍稀疏斑驳破碎，脚周大块灰黑污染和伪凸起明显，B08与B07近乎相同。GT0080本身背光暗，失败不是单凭暗。
- 人物与平台主体大部饱和，身体/手指/平台边缘及脚周有薄雾和细碎残影；高Alpha不保证几何/颜色正确。
- 正背面色差和阴影继续烘入材质，平台棋盘格及脚周颜色污染、身体斑驳仍明显；没有GT材质先验。
- 实际单独view TEST0010/0080/0150的Normal/Depth/Mesh图：GS平台整体上向但人体/脚周混合轴斑驳、平台伪凸起；Depth Normal平台强波纹、人体噪声；Expected/Median Z脚周及人物明显错误（原Z米未拟合/裁剪）；Mesh脚周伪凸起、人物头手腿表面零碎，0080腿部不连续。没有缩小评分域。

HDR/SUN：实际view HDR0001/0076/0150与SUN0000/0036/0071：HDR暖色响应正常，但人物头手/躯干/腿斑驳与脚周平台伪凸起仍有；SUN0/71明显响应且高对比，36灯位于平台下方关闭间接光后黑属直接照明预期，未将该帧黑单独作为错误。所有重光照无GT精度声明。6视频自动全解码PASS，人工只查看列出代表图。

6视频逐帧解码通过；人工实际查看所列代表帧，不宣称人工逐帧观看所有视频。无重光照GT精度声明。

## 审计及结论

202项CPU单元测试与真实源权重PBR/BVH/GGX GPU梯度检查通过。GeometryRun已共享FrozenScene Anchor/Plane真实几何入口；本轮35020 native smoke首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致，材质helper一致；完整训练与离线使用同一native来源。完整launch源码/manifest和CUDA/GLM快照保存。checkpoint的有限性、canonical xyz/rotation实际更新并在信赖域内、opacity/scale冻结、feature gate信赖域微调、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；先完成前后景人工审核，必要时修复重训，再进入目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct_gate --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-14-jumpingjacks_dir-PBR来源输出gate微调/logs/write_results.log

## 实际训练时间与过程证据

{
  "status": "TRAIN_COMPLETED",
  "started_at": "2026-10-10T09:10:32.082100+09:00",
  "training_pid": 282299,
  "process_start_ticks": "10140426",
  "source_iteration": 35000,
  "target_iteration": 55000,
  "steps_planned": 20000,
  "preset": "pbr_gt_directional_plane_native_direct_gate",
  "gt_training_inputs": [
    "light_direction_only"
  ],
  "quality": "PENDING",
  "normal_goal": "PENDING",
  "train_session_id": 76498,
  "postprocess_session_id": 89846,
  "preview_session_id": 88932,
  "training_exit_code": 0,
  "verified_training_session_closed": true,
  "child_finished_at": "2026-10-10T09:39:51.188865+09:00",
  "iteration": 55000,
  "steps_completed": 20000,
  "finished_at": "2026-10-10T09:39:54.530013+09:00",
  "verified_finished_observed_at": "2026-10-10T09:40:57.309255+09:00",
  "training_complete_log_time": "10/10 09:39:50",
  "training_loop_elapsed": "28:59"
}

以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。

## B08完整20k实际启动（2026-10-10T09:11:05.296325+09:00）

真实PID282299/startticks10140426，exec76498，source35k→55k20000步、Preset pbr_gt_directional_plane_native_direct_gate；当前生产core473双snapshot SHA已锁定，不续B07/smoke权重。后处理PID282335/exec89846及40k预览PID282338/exec88932绑定同一真实train PID/ticks/argv，不自动重试。9步全量render/15七MAE/复算/文档已准备；真实train_runtime/process_handles保存。当前仍运行，无B08最终MAE或质量/7°结论，未commit/push/新分支。

## B08真实启动只读审计（2026-10-10T09:12:42.793456+09:00）

launch_audit实际exit0。473核心及双snapshot逐SHA一致；cfg仅8差异（3路径/Preset与4 gate字段），geometry warmup2000正式保持，source35k PLY/MLP SHA/32 NN optimizer0与feature组启用、方向表byte同B07白色单位、source plane全部数组bit同B07/15187支持、150帧自身RGBA Alpha provenance PASS；真实PID282299/ticks10140426/argv/cwd仍一致。该审计0优化、无GT标签或B08正式MAE。40k/50k/55k只读检查排入audit_checkpoints_when_saved.py，实际PID282503/exec3502绑定同train，未保存前退出则失败不重启；9步后处理exec89846等待本轮完成。

## B08训练期间B07全15原域误差分解（2026-10-10T09:14:41.800008+09:00）

只读取已完成B07的15张离线评测npz，实际exit0，原GS16.81871120°/GT415433/有效394675精确重组。按GT世界法线距固定source正plane轴有向夹角分箱，非GT物体语义：0–5°占有效75.53075315%，该箱MAE10.04343114°、贡献原MAE7.58587918°；5–30°占1.07607525%，MAE39.33107411°、贡献.42323195°；>30°占23.39317160%，MAE37.65885282°、贡献8.80960006°。这说明按B07分布，单把近plane方向组改善到0而保持其它组误差也不足7°，但不是B08预测或新评分，必须同时处理其它朝向；B08 gate对全部99827点开放，不仅平台候选。

≥30°误差占20.24881231%有效像素、贡献10.50594465°；完整各箱组成原域，没有筛帧/点/难像素，不生成任何GT训练监督/mask/prior，不修改live训练。命令commands/diagnose_b07_offline_error.py、日志logs/b07_offline_error_decomposition.log、证据inspection/b07_offline_error_decomposition.json及15来源npz SHA。GT只有本离线分析读取，0 optimizer。当前B08最终MAE/质量待完成。

## B08早期训练曲线实际快照（2026-10-10T09:16:37.128262+09:00）

固定事件字节快照到3663个Stage2步，100步均值photo L1/total loss及训练RGB拟合正E与B07曲线早期几乎重合。两轮Direct SUN/GGX/BVH/alpha>.9全可用photo一致，B08仅开放输出gate且total含新弱anchor；前2k仍warmup，早期曲线不能替代全15法线MAE或质量结论。实际view inspection/early_training/training_curves.png，原CSV、B07/B08事件字节snapshot、来源SHA与时刻保留metadata.json。作图exec59372实际exit0并关闭，命令commands/plot_early_training.py，日志logs/plot_early_training.log；0额外优化，无GT标签读取，不更改live训练。完整B08仍运行，40k预览与最终七MAE待完成。

## B08同40k检查点、全15 RGB与实际审核（2026-10-10T09:22:17.610367+09:00）

预览exec88932实际exit0并关闭，checkpoint等待审计exec3502已执行40k检查实际exit0（后续50k/55k仍绑定原训练等待）。99827点finite/全部保留，opacity/scale及32来源NNtensor保持；71553点raw feature变化，最大source logit偏移.92840385437（边界4），canonicalXYZ .03547000885m、q60.00005341°容差内；sourceXYZ/plane/native/gate marker与source35k SHA绑定，DC硬clamp0、边界41.16488191%，RGB拟合E5.49645996，非GT光强。

全部15 TEST RGB PSNR18.05530376dB（B07同40k18.05598159）、SSIM.89959086（B07.89944883）、LPIPS.11140581（B07.11159734），前景PSNR10.83801726。量化变化很小；实际view GT/B07/B080010/0080/0150图：GT/B07/B08实际首中尾拼图：黑背景正常，人物平台存在；人物头手/衣服/腿依旧破碎斑驳，脚周大块灰黑阴影污染和平台伪凸起明显。B08同40k外观几乎与B07相同，未通过质量。GT0080本身背光暗，失败不单凭暗。 当前FAILED_SO_FAR_RGB_ONLY，未测全15 Normal，不推断最终7°；正式训练持续进行，无commit/push/新分支。

命令commands/preview_40000.py282299、make_rgb_40000_comparison.py；日志logs/preview_40000_wait.log、preview_render_40000.log、checkpoint_40000_audit.log；证据preview_40000/eval/metrics.json、inspection/{rgb_40000_comparison.png,preview_40000_review.json,checkpoint_40000_audit.json}。

## B08同40k gate真实世界响应只读审计

commands/audit_gate_world_effect_40000.py实际exec29785 exit0并关闭；0 optimizer，未读GT标签/材质/法线/几何。固定同40k canonical XYZ/q/material/原32 MLP/sourceXYZ，仅比较saved gate与source35k gate的Deform+既有plane输出；不是B07另训权重，也不是GT normal准确率。4个source组完整分区全部99827点，没删点或改任何评分/训练参数、没生成标签/prior；source/checkpoint和473核心SHA保持。

概率偏移全点中位4.33647074e−8、P99.02914502、最大.25986132；source raw静动态阈值跨越91点（投影0、dynamic near-plane34、other dynamic14、static未投影43）。source dynamic near-plane6836点概率均值.97179770→.96945214、中位偏移2.45571136e−5、P99.08227499；组标签不是GT物体。TRAIN0001/0075/0149全点axis无向响应P99.13932712/.16727745/.12982376°、最大5.07943976/3.94833914/4.91278177°；近plane动态组P99.39499720/.41688162/.36503260°、XYZ P99.00792922/.00451554/.00697693m。多数source gate原已饱和，40k真实世界响应大多小；不能用raw feature变化数量宣称整体法线已修复，仍须本轮55k完整MAE/目检。完整JSON inspection/gate_world_effect_40000.json，日志logs/gate_world_effect_40000.log。

## B08训练期间原RGB拟合方向的原生四元数运输诊断（2026-10-10T09:30:29.890633+09:00）

只用原35k PLY/32 MLP、全部135 TRAIN相机/time/原RGB自身Alpha、白色单位GT方向及既有source-only plane；40000 FrozenScene仅相机/NN容器，learned40k XYZ/q/material/feature不用。重组source RGB对应/可见性Gram线性拟合，与B07统计同40948可靠点；仅RAM候选法线/四元数不写训练资产。已有硬plane支持11327可靠点的法线轴被投影固定，所以新旋转响应候选为其余29621点；是模型约束/源可观测性诊断，不改任何GT评分或筛Gaussian。候选四元数保留source norm/yaw、源轴最近无向符号及60°信赖域，全部点保留。

将RGB候选从canonical以source R_world R_canonical^T近似运输，逐135 TRAIN姿态与真实q_candidate+固定source NN偏置（并应用原plane）比较。全部候选的平均运输偏差中位0.00114575°、P999.84908840°、最大44.12652537°；最坏训练时刻偏差中位0.00442551°、P9927.51723807°、最大87.03892812°。这只是approx与exact原生模型差，不是GT normal准确率；未来RGB几何约束须考虑真实raw quaternion加NN偏置，线性Lambertian拟合仍有GGX/阴影/裁剪/对应误差，不能据此宣称7°或新训练改善。

首次exec86238实际exit1并关闭：第二轮相机fid卸载至CPU后未重新加载，造成CUDA/CPU Linear设备不匹配。原脚本/日志保留，独立retry脚本补齐逐camera load/unload；exec97008实际exit0并关闭。诊断0 optimizer、源35k PLY SHA8ebd88b6f847a20178458777d51aa191d5f67ba4bbc490fd1e99ef62f66d5529/40k fingerprint/473核心SHA保持。未读GT标签/光强/颜色，未保存任何法线teacher/prior/训练或评分mask，没有修改live B08。命令commands/audit_rgb_normal_native_transport{,_retry}.py，日志logs/rgb_normal_native_transport{,_retry}.log，统计inspection/rgb_normal_native_transport_diagnostic.json。B08正式最终七MAE与质量仍PENDING。

### 上述运输诊断的静动态分组补充

| 来源组（非GT物体） | RGB可靠非plane候选 | 平均运输偏差中位 | 平均偏差P99 | 最坏时刻偏差中位 | 最坏时刻偏差P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-static | 17389 | 0.00000734° | 0.65557366° | 0.00002910° | 2.39407082° |
| source-dynamic | 12232 | 1.61462750° | 12.19259415° | 5.54153050° | 32.80356603° |

总体中位很小由17389个近静态候选占比造成，不能概括动态12232点：动态平均偏差中位1.61462750°/P99 12.19259415°，最坏时刻中位5.54153050°/P99 32.80356603°、最大87.03892812°。这是后续方法选择的新证据：若使用原RGB生成的几何约束，需要直接拟合或检验原生q_raw+source NN偏置的各时刻法线，不能把简单R_world R_canonical^T运输当精确监督。当前只存诊断统计，尚未实现该prior/新Preset或改变B08；GT Normal仅离线评测。

## B08 50k保存与真实检查点审计（2026-10-10T09:37:01.357665+09:00）

原train PID282299已保存50k；audit_checkpoints_when_saved.py绑定原PID/ticks/argv，40k与50k子检查均实际exit0，父exec3502仍live等待55k，不重启本轮。50k全部99827点finite/保留，opacity/scale与32来源NNtensor原值、source35k SHA、sourceXYZ锚定和plane/native/gate marker核验。71123点raw feature变化，最大source logit偏移2.91029191（边界4）；canonicalXYZ .07999999821m、q60.00006485°容差内，额外world plane投影仍不等于canonical位移上限。DC [.031,.969]内部、硬clamp0、边界比例41.92152424%，RGB拟合E7.12114573，非GT能量。

本检查无GT标签/优化或正式MAE；只有权重有限性/来源/信赖域检查PASS，不判画质或7°。证据inspection/checkpoint_50000_audit.json、checkpoint_wait_state.json、model_mlp/pbr_directional/iteration_50000/lighting.json；命令commands/audit_checkpoint.py50000、audit_checkpoints_when_saved.py，日志logs/checkpoint_50000_audit.log、checkpoint_audits_wait.log。完整训练继续，最终55k全量render/15七MAE与目检PENDING，无commit/push/新目标分支。

## B08完整20k真实训练完成与55k审计（2026-10-10T09:40:57.309255+09:00）

固定35000→55000完整20000步，train exec76498实际exit0并关闭、原PID282299已退出；Training complete log 10/10 09:39:50，loop 28:59。检查点父exec3502实际exit0并关闭，40k/50k/55k三次子审计全部exit0。55k全部99827点finite/保留，opacity/scale及32源NNtensor原值、source35k SHA、sourceXYZ锚定、plane/native/gate marker绑定通过；70238点raw feature变化，最大source logit偏移3.157794（边界4），canonicalXYZ .08000001311m/q60.00005341°容差内。DC内部[.031,.969]硬clamp0，边界比例42.27981074%；RGB拟合E7.92472982，非GT光强/颜色。473核心与双snapshot逐SHA保持。

后处理真实PID282335/exec89846已开始all render，child287136实际argv/cwd/ticks核验；四类150×4、HDR150×2048、方向72/6视频及全部15七MAE/复算待完成。当前仅TRAIN_COMPLETED和权重审计，不判质量或7°，未commit/push/新目标分支。证据training_runtime.json、process_handles.json、inspection/checkpoint_55000_audit.json、checkpoint_wait_state.json、model_mlp/pbr_directional/iteration_55000/lighting.json及logs/train.log/checkpoint_55000_audit.log/postprocess.log/render.log。

## B08 55k raw gate与概率梯度只读比较（2026-10-10T09:46:01.299518+09:00）

commands/audit_probability_gate_gradients_55000.py实际exec81335 exit0并关闭，0 optimizer、原训练已结束；同55k源NN/sourceXYZ/native/plane/材质固定，TRAIN0001/0075/0149只读比较∂L/∂raw和∂L/∂概率，严格验证链式因子2g(1−g)。全点导数中位4.26760162e−5、dynamic source-near-plane6836点中位.00051473238，显示大量sigmoid饱和。没有读取GT标签/生成prior或修改任何权重、点数、评分。

| TRAIN帧 | 全点 raw梯度L2 | 概率梯度L2 | >1e−12 raw/概率梯度点数 | 同lr.001假设首步概率平均偏移 raw/direct |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.000849484 | 0.010315651 | 34923/49210 | 0.000004960/0.000232780 |
| 0075 | 0.001164197 | 0.015449737 | 34522/49417 | 0.000007103/0.000243716 |
| 0149 | 0.000726180 | 0.012277407 | 35059/49329 | 0.000004998/0.000231180 |

首步比较仅RAM中以lr.001/无动量估计并投影[0,1]，没有执行optimizer，真实训练采用ramp/decay、多步动量，不能据此预测MAE或7°改善。备选方案是使用概率变量优化但前向/保存仍为原native sigmoid PLY feature，需新协议/Preset与真实完整训练；尚未实施。最终B08全量七MAE与人工审核仍待完成。证据inspection/probability_gate_gradients_55000.json与logs/probability_gate_gradients_55000.log，473核心SHA与FrozenScene fingerprint保持。

## B08最终RGB与四类早期实际目检（2026-10-10T09:47:18.265752+09:00，七MAE/HDR仍进行中）

全15 TEST RGB PSNR19.24938275dB、SSIM0.91077023、LPIPS0.11025733、前景PSNR11.88729878；B07同55k19.23561417dB/.91049162/.11079910，改变量很小。

实际view GT/B07/B08首中尾0010/0080/0150与四类0001/0075/0150拼图：黑背景正常，人物与平台存在；头手/衣服/小腿仍稀疏斑驳破碎，脚周大块灰黑污染和伪凸起明显，B08与B07近乎相同。GT0080本身背光暗，失败不是单凭暗。 人物与平台主体大部饱和，身体/手指/平台边缘及脚周有薄雾和细碎残影；高Alpha不保证几何/颜色正确。 正背面色差和阴影继续烘入材质，平台棋盘格及脚周颜色污染、身体斑驳仍明显；没有GT材质先验。 平台整体上向，0075无旧kernel整面反转，但脚周/人体混合轴和伪起伏明显；最终Depth/Mesh未实际查看，七MAE待完成。 当前FAILED_SO_FAR_RGB_AND_FOUR_ONLY，无最终七MAE/7°结论。HDR仍生成；早期review只记录已实际看图，不提前填最终完整审核。未commit/push/新分支。

代表图inspection/rgb_final_55000_comparison.png、four_categories_final_55000.png及early_visual_review_55000.json；两作图命令实际exit0。RGB拼图仅修正继承B07的错误indirect标签为Direct，不修改图片/指标。

## B08完整验收与B07同55k比较（2026-10-10T09:53:11.304611+09:00）

| 指标 | B07直接SUN实际预算 | B08来源输出gate | B08−B07 |
| --- | ---: | ---: | ---: |
| Albedo MAE | 0.20195435 | 0.20137736 | -0.00057699 |
| Roughness MAE | 0.20060596 | 0.20119953 | +0.00059357 |
| GS Render Normal MAE | 16.81871120 | 16.84133987 | +0.02262867 |
| Depth Normal MAE | 25.83637410 | 25.82059126 | -0.01578285 |
| Mesh Normal MAE | 20.06540566 | 20.07707837 | +0.01167271 |
| Expected Z Depth MAE | 0.07846366 | 0.07851869 | +0.00005503 |
| Median Z Depth MAE | 0.08282073 | 0.08287710 | +0.00005637 |

GS覆盖率95.00328573%→95.01267352%；RGB PSNR19.23561417→19.24938275dB、SSIM0.91049162→0.91077023、LPIPS0.11079910→0.11025733。

共同374301/415433（90.09900513%），GS/Depth/Mesh共同MAE15.81731046°/25.76318049°/19.06113108°；missing180=24.97859493°。

完整20k训练与9步后处理实际exit0且会话关闭，202 CPU/20步Gate Preset、真实SUN光线数/材质/XYZ/q/feature梯度、共享几何与材质逐buffer均已验证。独立15帧七MAE复算与来源SHA、15time×135TRAIN预测Mesh、roughness来源复算、6视频全解码通过。473核心和双source快照逐SHA保持；native kernel SHA绑定，无GT训练材质/法线/几何/原能量/颜色。全部99827点/15测试帧及原评分定义保持，失败权重/图/数组/日志保留。

人工质量FAILED，原GS Normal <7°目标FAILED；本脚本不执行commit/push/新分支。
