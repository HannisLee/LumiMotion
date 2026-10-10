# 1010-13 jumpingjacks dir 原生PBR直接SUN与真实光线预算

更新时间：2026-10-10T08:55:28.290218+09:00
状态：完整20000步训练、全量渲染和15测试帧七MAE/独立复算完成；人工质量FAILED，原GS Normal <7°目标FAILED。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；关闭训练与重光照的Stage1 SH间接光近似，原生SH辅助RGB损失保留，每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度和二值静动态参数冻结。几何2000步warmup、1000步ramp，Canonical PLY位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Plane Native Direct Preset继承B06全部材质/几何参数，包括既有来源平面几何投影。只用未改的35k PLY/MLP、全部135训练相机与time筛选支持：feature<.5、Alpha>.05、canonical与所有训练时刻距source RANSAC/SVD平面≤.2m、全时间形变坐标box-span范数≤.01m、canonical及全部时刻法线轴与plane绝对cos≥.5。支持集合固定，全部99827点保留；实际渲染world XYZ投影到plane、actual quaternion的Z轴旋到source polarity×plane且保留切向旋转与raw norm。训练RGB、PBR/BVH/GGX、周期测试、离线GS Normal/Depth/Mesh/HDR/方向灯共享同一实际几何。不是输出后改法线图、不是改变评分或删除难像素。支持资产来源SHA、全训练相机SHA和精确重算及checkpoint marker绑定；GT材质/几何从未参与支持。

来源MLP输入仍固定源XYZ、feature/time/camera不变，源32tensor保持。Canonical PLY仍限制.08m/60°，但有效渲染位置还经过plane几何模型投影，其总世界位移不能宣称受canonical .08m限制。该额外平面模型假设可能误平坦化来源误分类或真实非平面细节，必须以原始全15帧七项MAE及目检验收。PBR DC初始化和更新后[.031,.969]内部投影保留；本轮继承B06原生CUDA及Plane模型，只改关闭SH间接近似和实际SUN像素预算；材质/光强/loss/lr/几何参数保持。不是GT normal/plane/mesh约束。

## 本次Direct SUN与真实光线预算

新增pbr_gt_directional_native_direct及pbr_gt_directional_plane_native_direct，本轮后者。wo_indirect/wo_indirect_relight=True，只关闭PBR来源SH间接近似，native SH辅助RGB损失不改。pbr_directional_actual_ray_budget=True，SUN每像素L条BVH光线，启用间接光的显式覆盖则额外diffuse_sample_num条。单灯直接预算262144允许全部alpha>.9可用photo像素；该规则、全99827点、所有训练/测试帧与原GT评分域不改。旧Preset默认仍历史diffuse+light分母，显式HDR保留原预算，拒绝非法模式/非正整数/不足一个完整像素预算。

194 CPU discovery实际exit0；20步真实Trainer35000→35020实际exit0，保存Direct flags和native marker。三TRAIN帧实际BVH分别18843/15904/19137条单SUN，等于全部可用photo像素；真实3SUN预算512选170像素/510条，with-indirect成本257选1020像素/262140条，旧direct默认仍1024像素。材质/roughness/XYZ/q梯度非零finite；独立诊断0额外optimizer且参数SHA保持。首次诊断复用projection图导致重复backward失败，源码/日志保存；修正仅诊断每variant重建图，实际retry exit0。

GeometryRun与PBR alpha/worldnormal/ExpectedZ全buffer maxerror0、Mesh目标time一致，albedo/roughness材质helper一致。150×四类与4视频、三帧HDR16/替换SUN仅smoke功能；实际目检质量仍FAILED，不作为正式MAE或7°。正式完整150 HDR仍2048样本。

继承B06显式仓库LumiMotion native kernel，逐Gaussian按-dot(n_view,p_view)定向，forward/backward一致。pbr_rasterizer_backend=lumimotion，来源资产保存CUDA/Python/GLM逐文件SHA、综合源SHA和实际.so SHA，checkpoint marker与加载核验源，不静默回退installed。训练、全部RGB/Alpha/Normal、HDR/SUN、GeometryRun与逐time Mesh、材质评测共享入口。没有GT训练材质/法线/几何/能量；方向表白色单位×训练RGB拟合正scale，重光照metadata明确light-table而非GT photometry。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.20195435 | linear RGB | 97.264792% |
| Roughness MAE | 0.20060596 | r | 97.490997% |
| GS Render Normal MAE | 16.81871120 | degree | 95.003286% |
| Depth Normal MAE | 25.83637410 | degree | 90.318054% |
| Mesh Normal MAE | 20.06540566 | degree | 95.002804% |
| Expected Z Depth MAE | 0.07846366 | m | 95.003286% |
| Median Z Depth MAE | 0.08282073 | m | 95.003286% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。共同有效像素：374246/415433，覆盖率90.085766%；GS缺失预测记180°后的惩罚均角24.97241395°。

| 法线来源 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 15.78935883 |
| depth_normal | 25.77777194 |
| mesh | 19.04545171 |
完整JSON另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.235614174536575, "ssim": 0.9104916214942932, "lpips": 0.11079909801483154, "psnr_foreground": 11.882659584201594, "ssim_foreground_bbox": 0.803943928082784, "lpips_foreground_bbox": 0.22023212313652038}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。本轮关闭来源SH间接近似，HDR/SUN仍使用原生GGX/BVH可见性；无重光照GT准确率声明。

## 四类实际目检：FAILED

- 实际查看GT/B06/B07测试0010/0080/0150：黑背景正常、人物与平台存在，但头手、衣服及小腿斑驳稀疏，脚周灰黑污染与伪凸起明显；Direct增加对比，未通过质量验收。GT0080本身背光暗，结论不单凭亮度。
- 实际查看四类0001/0075/0150：人物和平台大部饱和覆盖，腰手边缘、脚周与平台仍有薄雾和细碎残影；饱和Alpha不代表正确表面。
- 材质仍带正背面光照与色差，平台棋盘/脚周污染，身体破碎；固定source feature未纠正静动态误分，材质质量FAILED。
- 实际逐张查看0010/0080/0150 Normal/Depth/Mesh对比：平台GS法线整体上向，但脚周及人体存在混合轴和斑驳；Depth Normal平台呈强烈波纹，Expected/Median Z脚周均出现大块几何偏差；Mesh复现脚周隆起且0080/0150腿部细碎/缺口。与GT平整平台和连续人体有明显差异，质量FAILED。

HDR/SUN：实际查看HDR0001/0076/0150和SUN0000/0036/0071：存在光照响应，但身体破碎和平台脚周污染/伪凸起保持。SUN36位于平台下侧，Direct无间接光近黑可解释；其它方向仍有伪影。灯表为白色单位原GT方向乘RGB拟合scale7.94063759，未使用GT光强/颜色，无重光照GT精度声明。

6视频逐帧解码通过；人工实际查看所列代表帧，不宣称人工逐帧观看所有视频。无重光照GT精度声明。

## 审计及结论

194项CPU单元测试与真实源权重PBR/BVH/GGX GPU梯度检查通过。GeometryRun已共享FrozenScene Anchor/Plane真实几何入口；本轮35020 native smoke首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致，材质helper一致；完整训练与离线使用同一native来源。完整launch源码/manifest和CUDA/GLM快照保存。checkpoint的有限性、canonical xyz/rotation实际更新并在信赖域内、opacity/scale/feature冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；先完成前后景人工审核，必要时修复重训，再进入目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算/logs/write_results.log

## 实际训练时间与过程证据

{
  "status": "TRAIN_COMPLETED",
  "started_at": "2026-10-10T08:14:00.087990+09:00",
  "training_pid": 271624,
  "process_start_ticks": "9801108",
  "source_iteration": 35000,
  "target_iteration": 55000,
  "steps_planned": 20000,
  "preset": "pbr_gt_directional_plane_native_direct",
  "gt_training_inputs": [
    "light_direction_only"
  ],
  "quality": "PENDING",
  "normal_goal": "PENDING",
  "train_session_id": 44255,
  "postprocess_session_id": 86673,
  "finished_at": "2026-10-10T08:43:04.510646+09:00",
  "iteration": 55000,
  "steps_completed": 20000,
  "training_exit_code": 0,
  "verified_training_session_closed": true,
  "verified_finished_observed_at": "2026-10-10T08:43:45.200906+09:00",
  "training_complete_log_time": "10/10 08:43:01",
  "training_loop_elapsed": "28:42"
}

以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。

## B07完整训练实际启动（2026-10-10T08:14:00.087990+09:00）

实际PID271624/startticks9801108，cwd与scripts.train_stage2 argv核验。固定35000→55000共20000步，pbr_gt_directional_plane_native_direct，仅GT方向/原RGB与自身Alpha，不续B06权重。源码471逐SHA及双snapshot一致、9步后处理计划备齐；尚无B07最终MAE或质量结论。

## B07真实启动只读审计与B06误差分解（2026-10-10T08:18:12.598836+09:00）

训练实际PID271624/startticks9801108持续存活。launch_audit_retry实际exit0：当前配置仅7个差异（3路径、Preset、actual-budget开关、wo_indirect与wo_indirect_relight），471核心/双snapshot逐SHA匹配；plane全部数组与B06逐bit一致、支持15187/99827、150帧whiteunit方向表快照与B06逐byte相同、原35k PLY/MLP SHA保持、原RGB/自身Alpha150帧provenance PASS。初次audit误认stage1_source.json为files嵌套对象而KeyError，保留失败源码/日志；只修正读取现有平铺path→SHA schema，retry PASS，不改训练。

仅离线读取B06全部15评测npz并重组原始GS17.29779705°、GT415433/有效390928完全一致。按GT法线与来源正plane轴有向夹角分箱（不是物体语义/训练掩码）：0–5°占有效75.63771334%，该箱MAE10.47368676°、贡献原MAE7.92205717°；5–30°占1.05722793%，MAE39.04751487°、贡献.41282123°；>30°占23.30505873%，MAE38.45911204°、贡献8.96291865°。所有分箱组成完整原域，不丢点/帧/难像素、不提出替代分数、不生成训练GT标签/prior。高误差30–60°占14.86283919%贡献6.30380581°，≥60°占6.16251586%贡献4.83946485°，仍是B06结果，不是B07指标。提示后续需同时改善不同方向的法线，不能只凭平台外观验收；B07核心与loss未据此修改。

证据inspection/launch_audit.json、b06_offline_error_decomposition.json；命令commands/audit_launch.py、diagnose_b06_offline_error.py；日志logs/launch_audit_retry.log、b06_offline_error_decomposition.log。诊断0优化、GT严格离线。完整B07训练、七MAE、四类/HDR最终质量仍待完成，未commit/push/新目标分支。

## B07早期训练事件快照（2026-10-10T08:20:17.353326+09:00）

固定事件字节快照保存至3929步，0额外优化、不读GT标签。100步平均photo L1和total loss持续下降，B07开头高于B06，约3k后接近；正E由原训练RGB拟合至约5.2，初值π，并非GT能量。B06每次sample1024，B07all-eligible alpha>.9且物理模型不同，不能把训练loss曲线直接当七项MAE/质量比较。图inspection/early_training/training_curves.png，原CSV、两份事件字节snapshot和来源SHA/时刻见同目录metadata.json；作图命令commands/plot_early_training.py实际exit0，日志logs/plot_early_training.log。已实际view图。最终B07训练/15帧七MAE仍PENDING。

## B07同40k检查点与实际RGB早期审核（2026-10-10T08:22:54.566226+09:00）

预览exec67658实际exit0、检查exec67311实际exit0。全15 TEST RGB PSNR18.05598159dB（B06同40k 18.51417174），SSIM0.89944883（B06 0.88792369），LPIPS0.11159734（B06 0.10788124）；前景PSNR10.84037981。PSNR/LPIPS变差、SSIM提高，不等于法线MAE变化。实际view GT/B06/B07首中尾0010/0080/0150拼图：背景正常/主体平台存在，但头手/衣服斑驳破碎、脚周平台伪凸起和黑色阴影污染明显；B07前景/脚周更暗更高对比，GT0080自身背光暗，不能仅以暗失败。总体未通过质量，未推断最终Normal MAE。 当前FAILED_SO_FAR_RGB_ONLY，完整55k七MAE仍PENDING，未commit/push/新目标分支。

40k全部99827点finite、opacity/scale/feature及32源NNtensor原值、source35k SHA保持。最大canonical XYZ位移0.03549213707m、旋转60.00005341°容差内，sourceXYZ anchor/plane/native marker通过；albedo可导DC [.031,.969]硬clamp0，更新后位于投影边界比例41.128820%；RGB拟合E5.49740267，非GT能量。源码471与launch仍匹配。

证据preview_40000/eval/metrics.json、inspection/rgb_40000_comparison.png、preview_40000_review.json、checkpoint_40000_audit.json；命令commands/preview_40000.py 271624（绑定live train后调用记录的preview_40000_argv.json），audit_checkpoint.py40000；日志logs/preview_40000_wait.log、preview_render_40000.log、checkpoint_40000_audit.log。全训练PID271624持续进行；50k只读检查PID/状态记录在inspection/audit_50000_wait_state.json，9步后处理PID271790绑定原训练等待终态，不重启本轮。

## B07 40k来源未投影near-plane组的实际联合贡献（2026-10-10T08:26:50.574185+09:00）

已读取GOAL与40k当前FAILED_SO_FAR；只读诊断exec73154实际exit0，未改live训练。固定35k PLY/MLP和既有全135时刻plane数组、alpha>.05/全时间距plane≤.2/axis abs cos≥.5筛选，不读任何GT。四组完整分区99827点：原投影15187、source-static近plane但motion>.01未投影1542、source-dynamic近plane6836、其余76262。只是诊断颜色，全原opacity/xyz/q保留；alpha/worldnormal/ExpectedZ实际native buffer误差0，两pass0，组累计权重之和与alpha max≤6.56e−7。原source/checkpoint/471源码SHA不变，0 optimizer。

实际view inspection/unprojected_plane_groups_40000/comparison.png（TRAIN0001/0075/0149）：蓝=原投影、红=static未投影、绿=dynamic near-plane、白=其它；动态near-plane占平台大部，原投影被其它组遮挡，static未投影仅零散区域。static未投影motion span中位.05101065m，放宽阈值到.02/.04/.08也仅新增349/664/968点（全1542）；对应主导alpha>.5像素422/498/298，远小于dynamic near-plane5117/2903/5310与其它12855/10966/13154。仅放宽既有static motion阈值不能据此推断足以解决整体伪几何。dynamic近plane全XYZ span中位.39017498m、P90.89354634m（并非B05较小的法线轴运动span），不能把这6836点未经验证当真实静态平台强制投影；source候选不是GT语义标签，也不作为删点/评分mask。尚未实施下一轮gate/NN/plane参数修改，须先完成本轮55k七MAE与验收。

命令commands/audit_unprojected_plane_groups.py，日志logs/unprojected_plane_groups.log；完整来源stats/实际joint weights与buffer误差见inspection/unprojected_plane_groups.json，3帧groups/normal和实际已看拼图在inspection/unprojected_plane_groups_40000。无GT训练先验保存。

## B07训练期间固定35k预测/原RGB/方向可观测性只读诊断（2026-10-10T08:35:38.311823+09:00）

新增候选调查，非新训练或Preset：仅源35k PLY与原值32tensor MLP、既有source-only plane、全135 TRAIN相机/time/原RGB自身Alpha、白色单位GT方向，未读GT normal/albedo/roughness/depth/mesh/blend/原GT能量/颜色。40k FrozenScene仅作相机/固定源NN容器，不用训练后40k XYZ/q/material。逐原Gaussian中心投影到135训练RGB，源自身预测alpha>.9、原RGBA alpha≥.99、点与源预测camera Z差<.025m来测可见性（非GT depth）。按固定源R与camera-facing符号将单SUN unitdir送回canonical基底，以原训练linear RGB亮度做三维Lambertian线性Gram/rhs拟合；只存数值可观测性统计，不保存拟合法线/监督/prior/评分mask。全99827点保留，0 optimizer。

首次完整exec17823与增加投影/运输核验的独立exec77927实际exit0，原记录与脚本/日志均保留。全部135帧source投影→相机ray逆回world max2.92062759e−6m，source light-dot-normal运输恒等max1.07288361e−6；471核心source/训练权重SHA不变。固定原35k PLY SHA8ebd88b6f847a20178458777d51aa191d5f67ba4bbc490fd1e99ef62f66d5529。

全点≥12次观测80388，Gram λmin>.01λmax且≥12次77904；再加g长度>.01、相对线性残差<.3后40948（static28716/dynamic12232，projected11327，为重叠subset，不相加）。dynamic筛选点残差中位.20139334，与源canonical轴无向夹角中位15.82878590°；全筛选点.18728082/11.02228355°。这些只是对原RGB线性近似的拟合筛选，不能据此叫方向正确或GT normal提升：原native GGX、硬阴影、RGB裁剪、源对应/transport误差都未由线性模型精确解释，source标签也不是GT物体分割。候选只说明部分点有RGB-direction信号，未来若构造RGB几何约束必须另行设计/source资产绑定/完整训练/全15原评分与目检；本轮不修改。

证据inspection/rgb_direction_observability{,_projection_verified}.json；命令commands/audit_rgb_direction_observability{,_projection_verified}.py，日志logs/rgb_direction_observability{,_projection_verified}.log。完整B07训练及七MAE/验收仍未完成，未commit/push/新目标分支。最初即时归档文本脚本因多行引号SyntaxError、尚未执行任何写入；仅修正归档字符串后完成，不涉及诊断/训练参数。

## B07 50k真实检查点审计（2026-10-10T08:36:34.141250+09:00）

50k实际保存、绑定原训练的audit exec54247实际exit0并关闭，不是等待超时或重启。99827点finite/全部保留，opacity-scale-feature及32source NNtensor保持，source35k SHA、anchor/sourceXYZ输入、plane/native marker核验。最大canonical XYZ.07999999821m/rotation60.00005341°容差内，额外world plane投影不等于canonical .08m。DC [.031,.969]硬clamp0，projection边界channel比例41.87477670%；RGB拟合E7.12501812（非GT强度）。当前471核心/launch仍匹配；本检查无GT标签/优化/七MAE。

命令commands/audit_50000_when_saved.py绑定process_handles里的原train，实际调用audit_checkpoint.py50000，日志logs/audit_50000_when_saved.log、checkpoint_50000_audit.log；证据inspection/audit_50000_wait_state.json、checkpoint_50000_audit.json、model_mlp/pbr_directional/iteration_50000/lighting.json。55k同类只读检查已排入audit_55000_when_saved.py，9步后处理仍等待原训练终态，最终15帧七MAE/coverage/common/missing180及四类/HDR目检待完成，未commit/push/新目标分支。

## B07完整20k训练实际完成与55k审计（2026-10-10T08:43:45.200906+09:00）

固定35000→55000共20000步完整运行，train exec44255实际exit0并关闭、原PID271624已退出；Training complete log 10/10 08:43:01，loop 28:42。55k只读审计exec20440实际exit0，完整99827点finite，opacity-scale-feature冻结、32源NN原值及源35k PLY/MLP SHA保持；sourceXYZ anchor、plane/native marker绑定通过。最大canonicalXYZ 0.08000001311m/rotation 60.00005341°，DC [.031,.969]硬clamp0，projection边界比例42.158267%，RGB拟合E7.94063759，非GT强度。471核心source/launch仍匹配，未改已完成训练参数/源。

后处理真实PID271790已进入全量render，child276537（实际cwd/argv可核验），all150×4/HDR150×2048/方向72/6视频；后续完整15七MAE与独立复算/目检仍PENDING。没有将checkpoint数值PASS当画质或7°。证据training_runtime.json、inspection/checkpoint_55000_audit.json、audit_55000_wait_state.json、model_mlp/pbr_directional/iteration_55000/lighting.json；日志train.log、audit_55000_when_saved.log、checkpoint_55000_audit.log、postprocess.log、render.log。未commit/push/新目标分支。

## B07最终RGB与四类早期实际目检（2026-10-10T08:48:01.127092+09:00，七MAE/HDR仍进行中）

全15 TEST RGB PSNR19.23561417（B06同55k 19.68220260），SSIM0.91049162（B06 0.89777244），LPIPS0.11079910（B06 0.11070925）；前景PSNR11.88265958、bbox SSIM0.80394393/LPIPS0.22023212。SSIM提高、PSNR下降，不能替代七MAE。

实际view GT/B06/B07首中尾0010/0080/0150及四类0001/0075/0150拼图：黑背景正常/人物平台存在，但头手/衣服和小腿稀疏斑驳、脚周伪凸起/灰黑及阴影污染仍明显，Direct更高对比；GT0080自身背光暗，不仅凭暗失败。 主体平台大部饱和覆盖，腰手边缘与平台/脚周薄雾、细碎残影仍有；Alpha饱和不能推断真实表面正确。 正背面光照/色差仍烘入材质，平台棋盘和脚周污染、身体破碎；source feature保持，未修正来源静动态误分。 平台整体上向、0075无旧kernel整个反向，但人体/脚周混合轴斑驳、伪起伏仍明显；Depth/Mesh与最终15七MAE待完成。 当前FAILED_SO_FAR_RGB_AND_FOUR_ONLY，未提前评定7°，未commit/push/新目标分支。HDR尚在生成，后处理真实render child276537仍活跃，实际SUN72完成但完整relight审核待HDR150。

代表图inspection/rgb_final_55000_comparison.png、four_categories_final_55000.png，early review JSON同inspection；命令commands/make_final_rgb_comparison.py、make_final_four_categories.py实际exit0；真实model_mlp/renders_stage2/ours_55000/eval/metrics.json、insights四类各150已生成，6视频全解码最终另审计。

## B07完整验收与B06同55k比较（2026-10-10T09:00:56.974578+09:00）

| 指标 | B06原生+间接光 | B07直接SUN实际预算 | B07−B06 |
| --- | ---: | ---: | ---: |
| Albedo MAE | 0.16231946 | 0.20195435 | +0.03963489 |
| Roughness MAE | 0.21780360 | 0.20060596 | -0.01719764 |
| GS Render Normal MAE | 17.29779705 | 16.81871120 | -0.47908585 |
| Depth Normal MAE | 26.04174483 | 25.83637410 | -0.20537073 |
| Mesh Normal MAE | 20.34490562 | 20.06540566 | -0.27949996 |
| Expected Z Depth MAE | 0.08001558 | 0.07846366 | -0.00155192 |
| Median Z Depth MAE | 0.08431691 | 0.08282073 | -0.00149618 |

GS覆盖率94.10133523%→95.00328573%；RGB PSNR19.68220260→19.23561417dB、SSIM0.89777244→0.91049162、LPIPS0.11070925→0.11079910。

共同374246/415433（90.08576594%），GS/Depth/Mesh共同MAE15.78935883°/25.77777194°/19.04545171°；missing180=24.97241395°。

完整20k训练与9步后处理实际exit0且会话关闭，194 CPU/20步Direct Preset、真实SUN光线数/材质/XYZ/q梯度、共享几何与材质逐buffer均已验证。独立15帧七MAE复算与来源SHA、15time×135TRAIN预测Mesh、roughness来源复算、6视频全解码通过。471核心和双source快照逐SHA保持；native kernel SHA绑定，无GT训练材质/法线/几何/原能量/颜色。全部99827点/15测试帧及原评分定义保持，失败权重/图/数组/日志保留。

人工质量FAILED，原GS Normal <7°目标FAILED；本脚本不执行commit/push/新分支。
