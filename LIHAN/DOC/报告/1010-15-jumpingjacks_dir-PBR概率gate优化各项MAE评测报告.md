# 1010-15 jumpingjacks dir 原生PBR概率gate优化

更新时间：2026-10-10T10:45:17.268449+09:00
状态：完整20000步训练、全量渲染和15测试帧七MAE/独立复算完成；人工质量FAILED，原GS Normal <7°目标FAILED。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；关闭训练与重光照的Stage1 SH间接光近似，原生SH辅助RGB损失保留，每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度冻结；feature原值初始化并优化输出gate，来源MLP全部32tensor冻结。几何2000步warmup、1000步ramp，Canonical PLY位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Plane Native Direct Probability Gate Preset继承B08全部材质/几何参数，包括既有来源平面几何投影。只用未改的35k PLY/MLP、全部135训练相机与time筛选支持：feature<.5、Alpha>.05、canonical与所有训练时刻距source RANSAC/SVD平面≤.2m、全时间形变坐标box-span范数≤.01m、canonical及全部时刻法线轴与plane绝对cos≥.5。支持集合固定，全部99827点保留；实际渲染world XYZ投影到plane、actual quaternion的Z轴旋到source polarity×plane且保留切向旋转与raw norm。训练RGB、PBR/BVH/GGX、周期测试、离线GS Normal/Depth/Mesh/HDR/方向灯共享同一实际几何。不是输出后改法线图、不是改变评分或删除难像素。支持资产来源SHA、全训练相机SHA和精确重算及checkpoint marker绑定；GT材质/几何从未参与支持。

来源MLP输入仍固定源XYZ、time/camera保持，feature gate按训练RGB/Alpha/预测ND优化，源32 MLP tensor保持。Canonical PLY仍限制.08m/60°，但有效渲染位置还经过plane几何模型投影，其总世界位移不能宣称受canonical .08m限制。该额外平面模型假设可能误平坦化来源误分类或真实非平面细节，必须以原始全15帧七项MAE及目检验收。PBR DC初始化和更新后[.031,.969]内部投影保留；本轮继承B08 Direct Native/Plane，改为feature概率坐标优化；其它材质/光强/loss/lr/几何参数保持。不是GT normal/plane/mesh约束。

## 本次概率gate与Preset

继承B08 Direct/native CUDA/sourceXYZ/原平面15187/材质/XYZ与q/loss/白色单位SUN和全局RGB拟合E。只将输出gate的Adam变量从raw logit改为概率坐标，Preset pbr_gt_directional_plane_native_direct_probability_gate；lr.001、概率来源最大偏移1（同时投影[0,1]）、弱anchor.001，geometry warmup2000/ramp1000/decay和每4步更新保持。概率逐bit初始化为原35k native gate，不使用B08/20步训练后权重。

feature optimizer组只绑定概率叶子，原32NN全部关闭梯度/optimizer。前向逐bit使用真实PLY raw feature的native sigmoid值，梯度以概率坐标传递；更新后有限逆sigmoid写回原PLY（端点±80，未改概率保留source raw原值），再同步概率叶子到实际native量化值。所有FrozenScene/GeometryRun/材质/HDR/SUN直接加载保存PLY feature，没有另存teacher/概率prior。v2来源SHA/config/概率信赖域与checkpoint marker绑定，拒绝混用v1；旧Preset默认raw方式保持。

opacity/scale/来源NN32tensor原值、全部99827点保留，无强制静态/删点/筛难像素。canonical .08m/q60°只限制canonical参数，概率改变来源NN world形变须用实际RGB/Alpha/覆盖及完整MAE验证。GT材质/法线/粗糙度/深度/mesh/blend/灯强度颜色不参与训练、初始化、loss、prior，GT只离线评分。原生完整Adam状态未保存，不声明无损resume。

211 CPU、20步真实Trainer、五参数GPU梯度/真实SUN光线预算/保存加载/共享geometry-material和150×4/4视频PASS功能。smoke实际质量FAILED_SMOKE_ONLY、未测全15七MAE；69971点native概率变化、max偏移1.85966492e−5（短步ramp），source32NN/opacity/scale保持。三TRAIN photo概率梯度L2 .015394565/.035415199/.013251405，前向与实际PLY精确一致。功能PASS不能作为完整质量或7°达标证据。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.19121691 | linear RGB | 97.544001% |
| Roughness MAE | 0.19018926 | r | 97.749537% |
| GS Render Normal MAE | 15.94231363 | degree | 95.478453% |
| Depth Normal MAE | 25.18807492 | degree | 90.790091% |
| Mesh Normal MAE | 19.88648893 | degree | 95.363392% |
| Expected Z Depth MAE | 0.07691848 | m | 95.478453% |
| Median Z Depth MAE | 0.08110909 | m | 95.478453% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。共同有效像素：376095/415433，覆盖率90.530844%；GS缺失预测记180°后的惩罚均角23.36025968°。

| 法线来源 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 14.92166238 |
| depth_normal | 25.11806842 |
| mesh | 18.86326307 |
完整JSON另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.810345333543953, "ssim": 0.9202187975247701, "lpips": 0.10036607980728149, "psnr_foreground": 12.319447497693583, "ssim_foreground_bbox": 0.8192175626754761, "lpips_foreground_bbox": 0.2063336710135142}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。本轮关闭来源SH间接近似，HDR/SUN仍使用原生GGX/BVH可见性；无重光照GT准确率声明。

## 四类实际目检：FAILED

- 黑背景正常，人物/平台存在，概率gate相对B08头/衣服/部分腿略更完整。但人物手臂/衣服/双腿仍光度斑驳稀疏，脚周灰黑伪凸起明显；GT0080自身背光暗，不以暗单独判失败，预测肩颈轮廓和脚周仍与原RGB不同。
- 人物与平台Alpha主体完整且大部饱和；手臂、身侧、腿和平台边缘细碎残影/薄雾仍明显。黑RGB区域不能一概叫透明度缺失；覆盖和缺失180指标按原全前景记录。
- 衣服/正背面光照和阴影烘入材质，脚周灰黑污染；平台棋盘格可见但污染明显。来源MLP32tensor冻结，native gate概率优化未解决材质-法线歧义。
- 实际分别查看TEST0010/0080/0150 Normal/Depth/Mesh图。GS平台大部分上向但脚周伪凸起/弯曲与人体轴混合斑驳仍明显；Depth Normal人体/平台噪声更强，Mesh脚周凸起、背面缺面/腿脚差异仍明显，Expected/Median深度保留假凸起。七MAE/coverage/common/missing180完整记录，15.94度未达原7度。

HDR/SUN：实际看HDR0001/0076/0150与转灯0000/0036/0071拼图，并直接查看0036黑帧：HDR暖色/方向变化有响应，但人物斑驳和平台/脚周伪几何仍明显。SUN背向0036接近黑，闭合轨迹照到背向/下半球且无间接光，是当前直接光模型的预期，不以黑单独判失败；0000/0036平均亮度.0742719/5.4159e-7、PNG均差18.93937确认响应，拼图像素与各原图一致。无重光照GT精度结论。

6视频逐帧解码通过；人工实际查看所列代表帧，不宣称人工逐帧观看所有视频。无重光照GT精度声明。

## 审计及结论

211项CPU单元测试与真实源权重PBR/BVH/GGX GPU梯度检查通过。GeometryRun已共享FrozenScene Anchor/Plane真实几何入口；本轮35020 native smoke首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致，材质helper一致；完整训练与离线使用同一native来源。完整launch源码/manifest和CUDA/GLM快照保存。checkpoint的有限性、canonical xyz/rotation实际更新并在信赖域内、opacity/scale冻结、feature gate信赖域微调、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；先完成前后景人工审核，必要时修复重训，再进入目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct_probability_gate --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/logs/write_results.log

## 实际训练时间与过程证据

{
  "status": "TRAIN_COMPLETED",
  "started_at": "2026-10-10T10:03:46.073168+09:00",
  "training_pid": 291976,
  "process_start_ticks": "10459825",
  "source_iteration": 35000,
  "target_iteration": 55000,
  "steps_planned": 20000,
  "preset": "pbr_gt_directional_plane_native_direct_probability_gate",
  "gt_training_inputs": [
    "light_direction_only"
  ],
  "quality": "PENDING",
  "normal_goal": "PENDING",
  "train_session_id": 91145,
  "postprocess_session_id": 16931,
  "preview_session_id": 1259,
  "audit_session_id": 19913,
  "training_exit_code": 0,
  "verified_training_session_closed": true,
  "child_finished_at": "2026-10-10T10:32:59.638003+09:00",
  "iteration": 55000,
  "steps_completed": 20000,
  "finished_at": "2026-10-10T10:33:02.959598+09:00",
  "verified_training_session_at": "2026-10-10T10:33:46.559551+09:00"
}

以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。

## B09完整20k实际启动与只读审计（2026-10-10T10:05:44.168512+09:00）

真实PID291976/startticks10459825/exec91145，10:03:46.073168 JST独立35000→55000共20000步，当前实际tqdm约1285步/20000，正式warmup2000，Preset pbr_gt_directional_plane_native_direct_probability_gate。当前474核心及双snapshot SHA已锁定；未续smoke/B08权重。原NN32/opacity/scale冻结，所有99827点保留、原RGB/自身Alpha/源35k/相机time/白色单位GT方向唯一训练输入。

audit_launch实际exit0，cfg相较B08只有6差异（3路径、Preset、2新概率配置）；source35k PLY/MLP SHA保持，plane全部数组与B08逐bit一致/15187支持，150自身RGBA Alpha provenance PASS，方向表逐byte同B08白色单位，MLP optimizer0。0诊断优化、未读GT标签。真实train/postprocess/preview/audit四PID/ticks/argv/cwd核验；postprocess292132/exec16931、40k预览292130/exec1259、40/50/55k审计292131/exec19913绑定原训练，不自动重启。九步render/15七MAE/独立复算与文档已准备，尚无B09最终MAE/画质/7°结论，未commit/push/新目标分支。

smoke归档989文件SHA3，新增概率feature叶子仅优化坐标，所有渲染和保存仍真实native PLY sigmoid。训练文档、README、source manifest/train_argv/planned_commands和日志保存；commands/write_results.py在启动前纠正继承报告日期/继承B08表述，仅文档命令变化，生产核心与冒烟474 SHA保持。

## B09早期曲线实际快照（2026-10-10T10:16:35.143803+09:00）

exec53315实际exit0并关闭，固定B08/B09事件字节snapshot至3327步，100步均值实际view inspection/early_training/training_curves.png。前2000 warmup photo/total平均delta −2.16e−9/−2.41e−9、最大绝对delta2.68e−7/3.28e−7；geometry阶段1327步photo均值delta −.00135368、total −.00165965，比raw gate下降。相同Direct SUN/native/all-eligible alpha>.9；total包含不同坐标弱anchor，不能替代GT MAE/质量/7°，光强仍仅RGB拟合。0额外优化/无GT标签，生产474核心不改。CSV/原事件字节及metadata来源SHA保留；命令commands/plot_early_training.py，日志logs/plot_early_training.log。

## B09同40k检查点/全15 RGB与实际响应（2026-10-10T10:16:35.143803+09:00）

预览exec1259与gate-world等待诊断exec14352均实际exit0并关闭，检查点父exec19913已完成40k子审计exit0、仍等待50k/55k。全部99827点finite/保留，opacity/scale与32NNtensor原值，max概率delta .60847592；raw max79.74610138是有限端点/逆sigmoid编码，不沿用旧raw±4或声称同量世界变化。CanonicalXYZ .03549813m/q60.00006485°容差内，DC硬clamp0/边界40.58320895%、RGB拟合E5.50689125非GT强度，source/plane/native/v2 marker绑定。

全15 RGB PSNR18.35268777（B08同40k18.05530376）、SSIM.90386886（.89959086）、LPIPS.10786089（.11140581），前景PSNR10.98395868。实际view GT/B08/B09 TEST0010/0080/0150：黑背景正常，人物/平台存在；B09头/衣服/腿略更完整，平台脚周黑灰污染稍有变化，但人物斑驳/手臂与腿稀疏、平台脚周伪凸起仍明显，未通过质量。GT0080自身背光暗，不单以暗失败。没有正式40k Normal/七MAE，不推断最终7°。

同一40k canonical/material/sourceNN/XYZ/plane，仅比较saved概率与source35k native概率，0 optimizer/noGT。全点概率delta中位.00163957/P99.35333336（B08对应4.34e−8/.02914502），source raw阈值跨越227；动态near-plane6836点概率均值.97179770→.89238799、delta中位.05443400。TRAIN1/75/149全点axis响应P99 3.53469853/4.60136210/3.66612515°（B08约.139/.167/.130），动态near-plane axis响应P99 2.37774202/2.08524452/2.21833535°、XYZ P99 .06728367/.02283315/.07268968m。概率方案的实际响应比raw更大，但不等于GT法线改善/7°；source组不是GT物体，全99827点完整分组、不改评分。生产474核心/SHA保持，当前完整train291976持续，最终七MAE/四类/HDR/质量待完成，无commit/push/新分支。

命令commands/{preview_40000,make_rgb_40000_comparison,audit_gate_effect_when_40000_saved,audit_gate_world_effect_40000}.py，检查audit_checkpoint.py40000；证据preview_40000/eval/metrics.json、inspection/{preview_40000_review,rgb_40000_comparison,checkpoint_40000_audit,gate_world_effect_40000,gate_effect_wait_state}.json/png与对应logs。

## B09训练期间全135来源姿态旋转响应原型（2026-10-10T10:21:54.575849+09:00）

commands/audit_source_local_rotation_response.py实际exec33765 exit0并关闭，0 optimizer/无GT标签；40k容器仅source32 NN/原相机time，全部诊断使用原35k XYZ/q/feature与既有source-only plane，未用learned40k geometry/material/feature。全99827点、135 TRAIN姿态，RAM中人工10°canonical法线轴扰动，保留norm/yaw；不存candidate quaternion/normal/teacher/prior/评分mask，不改变B09生产或训练。

| source组（非GT语义） | 点数 | 平均时刻原加法响应中位 | 原加法最坏时刻P99 | source-local原型平均响应中位 |
| --- | ---: | ---: | ---: | ---: |
| source_projected | 15187 | 0.00000170° | 0.00001971° | 0.00000172° |
| source_dynamic_near_plane | 6836 | 9.25706546° | 13.59882556° | 10.00000078° |
| source_other_dynamic | 38747 | 9.09878417° | 15.11675295° | 10.00000070° |
| source_static_unprojected | 39057 | 9.99999483° | 10.16521196° | 10.00000074° |

投影15187点法线被原plane固定，不应响应，仍全部保留。其它动态38747点10°canonical扰动在原加法平均响应中位9.09878417°，near-plane动态9.25706546°：没有证明原加法存在大范围严重旋转响应抑制，不能据此盲目扩大角度或宣称仅更改旋转运输就可达7°。原加法时间响应有差异和极少大响应（全点最大88.05979930°）；source-local四元数原型未投影点响应约10°，和R_source_world R_source_canonical^T运输最大偏差4.22423871e−5°，自身初始化返回原raw deformation逐bit一致。它只是全来源的数学候选，不是已实现/已训练的Preset，也不是GT Normal准确率；若今后采用RGB方向约束，可能给固定source运动的局部修正提供一致表达，须另行接入全部入口/功能/完整训练验收。当前B09管线474 SHA保持且继续训练。

## 训练期间追加证据（完整性快照）


## B09同40k原生PBR阴影与深度起点只读诊断（2026-10-10T10:26:03.958797+09:00）

只读commands/audit_shadow_visibility.py，3 TRAIN帧原RGB/自身Alpha，同40k checkpoint所有参数冻结，0 optimizer、无GT标签/prior。完整diagnostic JSON与最终日志存在、生产474 SHA全部一致。原执行工具session ID在上下文压缩后未恢复，原进程exit code不作0声明；本次独立完整性复核实际exit0。实际view inspection/shadow_visibility/comparison.png：全部variant人物/平台存在，但人物斑驳、稀疏肢体与脚周伪几何未修复。

| TRAIN frame | native偏移.1 | 无遮挡 | 偏移.02 | 偏移.05 | 深度half混合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0001 | 0.06200587 | 0.05853464 | 0.11891680 | 0.07548539 | 0.06196247 |
| 0075 | 0.11945500 | 0.10437177 | 0.17422567 | 0.13582467 | 0.12160864 |
| 0149 | 0.06132580 | 0.05798720 | 0.11716985 | 0.07596797 | 0.06138846 |

表为相同native alpha>.9像素上的原RGB photo L1，非正式15 TEST七MAE；所有variant GS normal/alpha逐bit保持。原偏移沿光方向而非法线。减小bias明显增大photo L1；关闭遮挡局部改善但不修复斑驳/伪几何且抹去真实阴影，half混合差异小。目前没有依据将上述variant接入生产/Preset。own bright仅原RGB+自身Alpha诊断子集，不改变训练域/原评分/全99827点/帧。不是新训练、GT法线或7度结论；B09原训练继续，完整MAE和质量待完成。

## B09 50k检查点实际审计（2026-10-10T10:27:37.224705+09:00）

50k子审计实际exit0，完成时间2026-10-10T10:25:45.199833+09:00；父exec19913仍等待55k，train PID291976/startticks10459825仍live，未重启。全部99827点finite/保留，opacity/scale与源32NNtensor原值；sourceXYZ/marker/sourceSHA保持。max概率delta1.00000000达到允许[0,1]全域，raw最大88.77782440为有限逆编码，不沿用raw±4。canonical XYZ0.08000000m、q60.00005341°信赖域容差内；DC hardclamp0、投影边界40.20655734%，RGB拟合E7.04603338非GT光强。此审计只验证实际checkpoint/provenance，不是最终MAE或质量/7度结论。未commit/push/新分支。

## B09原RGB损失几何梯度关系只读检查（2026-10-10T10:28:13.018917+09:00）

commands/audit_photo_gradient_competition.py，exec83830实际exit0并关闭，同40k冻结checkpoint TRAIN0001/0075/0149，原RGB/自身Alpha，0optimizer/无GT标签/无prior；474生产SHA、sourceNN与模型SHA保持。所有99827点保留，source plane支持仅诊断分组，不是GT物体或新增评分mask。

| TRAIN | SH/PBR XYZ梯度L2比 | XYZ aggregate cosine | SH/PBR q梯度L2比 | q aggregate cosine | SH/PBR gate梯度L2比 | gate aggregate cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0001 | 0.10236965 | -0.20877363 | 0.07872238 | -0.08671552 | 0.07137501 | -0.05572955 |
| 0075 | 0.09448496 | -0.13924229 | 0.06909085 | -0.12017505 | 0.12816550 | 0.01781042 |
| 0149 | 0.02023082 | 0.01829717 | 0.01062575 | 0.06026068 | 0.03685447 | 0.03010815 |

前两帧SH与PBR部分几何梯度方向相反，第三帧略同向；但SH仅约1%–13% PBR梯度范数，不能据此断言SH是主要失败原因或移除SH就能修复法线。预测normal-depth/ownAlpha也有局部反向，这不等于正则有害。仅局部当前梯度，不是20k Adam轨迹/GT准确率或正式七MAE；不改变当前B09 loss或Preset、不存逐点梯度/normal teacher。完整训练和最终质量/MAE继续等待。

## B09完整20k实际终止与55k审计（2026-10-10T10:33:46.559551+09:00）

原训练exec91145实际exit0并关闭，source35000→55000独立20000步，启动2026-10-10T10:03:46.073168+09:00、child完成2026-10-10T10:32:59.638003+09:00，train PID291976已终止；不是观察timeout或重启。三检查点父exec19913实际exit0并关闭，40k/50k/55k全部子审计exit0。55k所有99827点finite/保留，opacity/scale和来源32NNtensor原值、sourceXYZ/marker/sourceSHA保持；max概率delta1.00000000，rawmax88.77782440是有限逆编码，不按raw±4审计。canonical XYZ0.08000001m/q60.00006485°在容差内，DC硬clamp0/投影边界41.10544575%，RGB拟合E7.74885416非GT。当前474核心及双snapshot保持；GT方向白色单位，未读GT材质/法线/几何训练。

后处理原PID292132/exec16931仍live，其render子PID296753已于2026-10-10T10:33:02.962058+09:00启动；将完成全部150四类/150 HDR2048/72SUN/6videos与15七MAE、独立复算。尚无最终MAE、正常前后景/质量或7度结论，未commit/push/新分支；不会用数值/运行PASS代替人工质量。

## B09 55k全15 RGB与实际首中尾预览（2026-10-10T10:34:31.179322+09:00）

make_final_rgb_comparison实际exit0。当前render任务仍live且尚未全部HDR/七MAE完成；all15 RGB PSNR19.81034533dB（B08 19.24938275）、SSIM0.92021880（.91077023）、LPIPS0.10036608（.11025733）、前景PSNR12.31944750。实际view GT/B08/B09 TEST0010/0080/0150拼图：黑背景正常，人物/平台存在，B09衣服/腿略更完整，但人物斑驳、手臂/腿稀疏和脚周灰黑伪凸起仍明显，FAILED_EARLY_RGB_ONLY。GT0080自身背光暗，不以暗单独判失败。尚无最终Normal/七MAE或7度结论，四类/HDR完整人工审核待齐备；不以PSNR改善或运行PASS代替质量。未commit/push/新分支。

## B09 55k四类实际目检（2026-10-10T10:35:21.393426+09:00）

make_final_four_categories实际exit0，RGB/Alpha/GS Normal/Albedo各150PNG齐备；实际view TRAIN0001/0075/0150四列拼图，FAILED_EARLY_FOUR_CATEGORIES_ONLY。黑背景正常、人物/平台存在；Alpha主体完整且大部饱和，手臂/身侧/平台边缘仍薄雾与细碎残影，因此黑RGB区域不能一概叫透明度缺失。人物材质保留前后光照色差/阴影、脚周灰黑污染，平台棋盘格可见但污染明显。GS平台整体上向但脚周伪凸起、人体轴混合斑驳。Depth/Mesh全15及HDR最终产物尚待齐备，不宣称已审核；150视频全解码审计也待后处理完成。当前原render child仍live，不重启/不改生产474核心，不作7度或正常前后景通过结论，未commit/push/新分支。

## B09完整验收与B08同55k比较（2026-10-10T10:46:04.857837+09:00）

| 指标 | B08 raw gate | B09概率gate | B09−B08 |
| --- | ---: | ---: | ---: |
| Albedo MAE | 0.20137736 | 0.19121691 | -0.01016045 |
| Roughness MAE | 0.20119953 | 0.19018926 | -0.01101026 |
| GS Render Normal MAE | 16.84133987 | 15.94231363 | -0.89902624 |
| Depth Normal MAE | 25.82059126 | 25.18807492 | -0.63251633 |
| Mesh Normal MAE | 20.07707837 | 19.88648893 | -0.19058944 |
| Expected Z Depth MAE | 0.07851869 | 0.07691848 | -0.00160022 |
| Median Z Depth MAE | 0.08287710 | 0.08110909 | -0.00176801 |

GS覆盖率95.01267352%→95.47845260%；RGB PSNR19.24938275→19.81034533dB、SSIM0.91077023→0.92021880、LPIPS0.11025733→0.10036608。

共同376095/415433（90.53084372%），GS/Depth/Mesh共同MAE14.92166238°/25.11806842°/18.86326307°；missing180=23.36025968°。

完整20k训练与9步后处理实际exit0且会话关闭，211 CPU/20步Gate Preset、真实SUN光线数/材质/XYZ/q/feature梯度、共享几何与材质逐buffer均已验证。独立15帧七MAE复算与来源SHA、15time×135TRAIN预测Mesh、roughness来源复算、6视频全解码通过。474核心和双source快照逐SHA保持；native kernel SHA绑定，无GT训练材质/法线/几何/原能量/颜色。全部99827点/15测试帧及原评分定义保持，失败权重/图/数组/日志保留。

人工质量FAILED，原GS Normal <7°目标FAILED；本脚本不执行commit/push/新分支。
