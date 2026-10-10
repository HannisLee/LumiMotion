# 1010-16 jumpingjacks dir 原生PBR训练RGB来源局部法线约束

更新时间：2026-10-10T11:54:26.210739+09:00
状态：完整20000步训练、全量渲染和15测试帧七MAE/独立复算完成；人工质量FAILED，原GS Normal <7°目标FAILED。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；关闭训练与重光照的Stage1 SH间接光近似，原生SH辅助RGB损失保留，每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度冻结；feature原值初始化并优化输出gate，来源MLP全部32tensor冻结。几何2000步warmup、1000步ramp，Canonical PLY位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Source RGB Normal Preset继承B09全部材质/几何参数，包括既有来源平面几何投影。只用未改的35k PLY/MLP、全部135训练相机与time筛选支持：feature<.5、Alpha>.05、canonical与所有训练时刻距source RANSAC/SVD平面≤.2m、全时间形变坐标box-span范数≤.01m、canonical及全部时刻法线轴与plane绝对cos≥.5。支持集合固定，全部99827点保留；实际渲染world XYZ投影到plane、actual quaternion的Z轴旋到source polarity×plane且保留切向旋转与raw norm。训练RGB、PBR/BVH/GGX、周期测试、离线GS Normal/Depth/Mesh/HDR/方向灯共享同一实际几何。不是输出后改法线图、不是改变评分或删除难像素。支持资产来源SHA、全训练相机SHA和精确重算及checkpoint marker绑定；GT材质/几何从未参与支持。

来源MLP输入仍固定源XYZ、time/camera保持，feature gate按训练RGB/Alpha/预测ND优化，源32 MLP tensor保持。Canonical PLY仍限制.08m/60°，但有效渲染位置还经过plane几何模型投影，其总世界位移不能宣称受canonical .08m限制。该额外平面模型假设可能误平坦化来源误分类或真实非平面细节，必须以原始全15帧七项MAE及目检验收。PBR DC初始化和更新后[.031,.969]内部投影保留；本轮继承B09 Direct Native/Plane，新增source-local真实旋转与lambda2训练RGB派生轴约束；其它材质/光强/原loss/lr保持。不是GT normal/plane/mesh约束。

## 本次训练RGB来源局部法线与Preset

继承B09概率gate、仓库原生Direct SUN/GGX/BVH与SH辅助、材质、真实XYZ/q、信赖域、source-only plane15187及原photo/Alpha/ND损失和全部lr。独立Preset pbr_gt_directional_source_rgb_normal，新增pbr_source_local_rotation=True与lambda_pbr_rgb_normal_prior=2；正式warmup2000/ramp1000。source q/raw norm从原35k保存，固定NN32 tensor按当前gate查询来源world旋转，再按R_source_world(p) R_source_canonical^T R_current_canonical运输，随后执行既有plane。初值原d_rotation逐bit保持。实际Gaussian covariance、normal、depth、PBR/BVH共享真实q，绝非评测法线图覆盖。周期预览/FrozenScene/GeometryRun/Mesh/material/HDR/SUN都使用相同入口和顺序。

全部99827点数组保留；附加canonical轴loss=sum(confidence*(1-dot))/sum(confidence)，仅新增loss按29621个confidence点归一化，29621非plane点有候选支持（来源static17389/dynamic12232）。来自原135TRAIN RGB线性luminance与自身Alpha、来源预测可见性、源35k qXYZ/固定NN/相机time/白色单位GT方向的独立LS候选；9个全点数组、source6 SHA/135逐图SHA、TRAINkeys/TEST0、finite与CPU全点Gram/解/残差/confidence复算绑定模型。该候选是训练RGB派生，不是GT法线，也不保证GGX/阴影准确；其confidence只约束新增loss，不改变原photo、完整点/帧/难像素或原评分。既有plane点没有重复该轴约束，也没有删除。

模型保存pbr_rgb_normal{,_input}.{npz,json}，source quaternion/target/confidence和原独立input本地复制、各SHA与checkpoint metadataSHA绑定，所有加载严格复算并检查源32NN冻结。新增0 optimizer参数，仍用原canonical rotation Adam raw q lr.001/norm和60度信赖域；probability gate/native sigmoid/有限逆sigmoid原实现不变。canonical .08m不保证gate/plane引起的实际world位移同界。未保存完整Adam状态，不声明无损resume。

226 CPU、独立20步35000→35020实际Trainer/保存加载/GPU五参数梯度/共享buffer/HDR16/SUN/四类150×4和4视频全解码功能PASS，实际画质FAILED_SMOKE_ONLY；RGB约束loss .102662175894、q梯度L2 .00525612570345非零。476核心与双snapshot保持。此完整20k从原35k初始化，不续20步或B09学得checkpoint。GT材质/normal/roughness/depth/mesh/blend/原灯强度颜色全部不参与训练/初始化/约束；仅独立离线评测读取GT标签，原有完整七MAE/覆盖/common/missing180定义保持。

## 各项MAE：全部15测试帧

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.19107403 | linear RGB | 97.566972% |
| Roughness MAE | 0.18083939 | r | 97.768637% |
| GS Render Normal MAE | 15.54006647 | degree | 95.486878% |
| Depth Normal MAE | 25.16398643 | degree | 90.801164% |
| Mesh Normal MAE | 19.75904152 | degree | 95.383130% |
| Expected Z Depth MAE | 0.07666492 | m | 95.486878% |
| Median Z Depth MAE | 0.08065818 | m | 95.486878% |

Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。共同有效像素：376203/415433，覆盖率90.556841%；GS缺失预测记180°后的惩罚均角22.96234465°。

| 法线来源 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 14.51173016 |
| depth_normal | 25.10148879 |
| mesh | 18.71977060 |
完整JSON另存metrics_summary.json。

## RGB及渲染

{"psnr": 19.754226714113255, "ssim": 0.9193544467290242, "lpips": 0.10111725976069769, "psnr_foreground": 12.278676023812173, "ssim_foreground_bbox": 0.8186597545941671, "lpips_foreground_bbox": 0.2067933867375056}

渲染路径：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp/renders_stage2/ours_55000

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。本轮关闭来源SH间接近似，HDR/SUN仍使用原生GGX/BVH可见性；无重光照GT准确率声明。

## 四类实际目检：FAILED

- 实际GT/B09/B10同55k TEST10/80/150对比：黑背景正常，人物和椭圆平台可辨认；人物上衣/手臂/裤腿有明显黑色碎斑，脚周及平台有大块黑灰伪几何/阴影混杂，最新外观与B09近似。背光中帧GT本身较暗，预测仍更暗，不把黑色RGB直接解释为Alpha缺失。RGB质量FAILED。
- insights1/75/150均保留人物和完整平台Alpha，手臂/腿/脚轮廓与孤立飞点、平台边缘有模糊碎片。中帧RGB近黑但Alpha人物完整；未删平台/难点。
- 人物首中尾均可见，反照率保留照明方向、阴影和碎斑，手臂及裤腿脏乱；平台有颜色斑块和脚周污染。没有干净材质分解或静动态分离的证据。
- 实际查看normal_depth/comparison_0010/0080/0150完整三行GT/GS/Depth/Z/Mesh对照：GS平台主体紫蓝，但脚周和人物表面有杂色、伪起伏；Depth Normal高频波纹更严重。Expected/Median Z在脚周平台有明显假起伏，与GT平面相比偏差集中在人物下方；Mesh也保留平台凸起，中帧身体和腿部有孔洞/缺片，末帧脚部不完整，三种法线人物误差明显。未删难像素或使用共同像素替代目标；整体几何质量FAILED。

HDR/SUN：实际查看HDR1/76/150和SUN0/36/71，并单独查看SUN0036原PNG：HDR有橙色人物及平台色彩响应，但人物/腿部明显碎斑、反照率照明污染和平台脚周假起伏仍保留。SUN0/71人物与平台受光，0036几乎全黑（仅小点），对应背/下半球直接照明且关闭间接的情况，不把该单帧黑色当独立渲染错误。重光照功能有响应，未提供重光照GT精度或质量达标证明。

6视频逐帧解码通过；人工实际查看所列代表帧，不宣称人工逐帧观看所有视频。无重光照GT精度声明。

## 审计及结论

226项CPU单元测试与真实源权重PBR/BVH/GGX GPU梯度检查通过。GeometryRun已共享FrozenScene Anchor/Plane真实几何入口；本轮35020 native smoke首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致，材质helper一致；完整训练与离线使用同一native来源。完整launch源码/manifest和CUDA/GLM快照保存。checkpoint的有限性、canonical xyz/rotation实际更新并在信赖域内、opacity/scale冻结、feature gate信赖域微调、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：FAILED。法线<7°目标：FAILED

未依据本报告自动提交或push；先完成前后景人工审核，必要时修复重训，再进入目标分支。

## 完整命令

### train

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_source_rgb_normal --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_rgb_normal_prior_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-17-dir-训练RGB法线约束原型/candidate_assets'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/train.log

### render

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/render.log

### albedo

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/albedo' --save_float
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/albedo.log

### predicted_meshes

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/predicted_meshes'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/predicted_meshes.log

### normal_depth

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_geometry_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --load_iter 55000 --gt_root /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light --gt_depth_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_depth --split test --resolution 2 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/normal_depth' --pred_mesh_manifest '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/predicted_meshes/manifest.json' --save_arrays
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/normal_depth.log

### roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_roughness_gt --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/model_mlp' --iteration 55000 --gt_roughness_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_source_alpha' --split test --output_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/roughness.log

### verify_roughness

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/roughness'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/verify_roughness.log

### verify_all_mae

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/commands/verify_all_mae.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/verify_all_mae.log

### audit_artifacts

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/commands/audit_artifacts.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/audit_artifacts.log

### write_results

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/commands/write_results.py'
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束/logs/write_results.log

## 实际训练时间与过程证据

```json
{
  "status": "TRAIN_COMPLETED",
  "started_at": "2026-10-10T11:11:25.974063+09:00",
  "training_pid": 306459,
  "process_start_ticks": "10865815",
  "source_iteration": 35000,
  "target_iteration": 55000,
  "steps_planned": 20000,
  "preset": "pbr_gt_directional_source_rgb_normal",
  "gt_training_inputs": [
    "light_direction_only"
  ],
  "quality": "FAILED",
  "normal_goal": "FAILED",
  "training_exit_code": 0,
  "verified_training_session_closed": true,
  "child_finished_at": "2026-10-10T11:41:41.045005+09:00",
  "iteration": 55000,
  "steps_completed": 20000,
  "finished_at": "2026-10-10T11:41:41.517407+09:00",
  "unified_session_id": 4419,
  "session_exit_verified_at": "2026-10-10T11:42:38.690439+09:00",
  "iterations_stopped_by_user": true,
  "new_training_started": false,
  "final_gs_normal_mae_deg": 15.540066466250822
}
```

以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。

## B10完整训练已实际启动（2026-10-10T11:12:19.216698+09:00）

原35k→55k独立20k，TRAIN PID306459/ticks10865815，exec4419实际仍运行；11:11:25.974063 JST启动，warmup2000/ramp1000、lambda2训练RGB轴约束，当前无正式MAE/质量结论。检查点独立共享加载helper先在smoke35020真实exit0核验，无GT标签、0 optimizer、29621支持/全99827点。检查点审计等待exec10940/后处理等待exec65390，精确进程信息见process_handles.json；保存40k/50k/55k后检查，训练完成后9步原全量渲染/七MAE自动顺序执行，人工质量单独验收。源476核心和双snapshot冻结。

训练准备helper首attempt在执行前因字符串引号嵌套语法错误exit1，未创建正式目录/模型、未启动训练；随后shell尝试不存在retry文件exit2，原日志保留。修正helper独立retry实际exit0，正式目录只创建一次，没有训练重启或改生产代码。全部准备脚本、失败JSON/log保留smoke1010-18。

### B10启动只读核对与当前进度（2026-10-10T11:13:35.904698+09:00）

audit_launch实际exit0，配置相比B09只有运行路径/Preset和新增3字段（source_local=True、RGB资产路径、lambda2）共7项差异，原warmup2000/概率gate/lr/源plane15187/全部99827点/ownAlpha150/冻结NN32/direction表byte一致。输入/资产SHA正常，source RGB约束支持29621；0诊断optimizer、0 GT标签读取。真实train PID306459仍运行，已完成1395/20000步，源码476 SHA锁定；exec4419/10940/65390均在运行中，不能提前填exit0。正式七MAE/前后景质量/7度均PENDING。

## B10训练曲线与RGB约束生命周期只读证据（2026-10-10T11:18:58.872501+09:00）

两个诊断均实际exit0/关闭（curve exec27142、RGB lifecycle exec92303），固定事件字节快照SHA，0 optimizer、0 GT标签读取。实际查看两张图：warmup2000 B09/B10曲线重合，逐step PhotoL1最大差2.68e−7/均差2.39e−9，total最大3.13e−7；2001–2464 RGB L1均差−.00016006，新增total均差+.03294333，total含附加RGB loss不能直接比较成画质退化。全局RGB拟合强度几乎一致，此说明新几何phase前初始外观保持，不当最终MAE。

RGB轴约束首记录37100（2100步）loss .0992467552，39400（4400步）.0038797222，lambda2*ramp后当前 .0077594444；实际在优化但不是GT normal误差。29621 source RGB候选相对原source预测轴median14.7640度/P9045.1578/P9983.1069，1600点大于60度。既有canonical60度的独立轴理想loss下界.00208803，只是轴几何条件、非GT误差或完整photo冲突最优值；目前无证据应放开cap或增大lambda。confidence源q norm中位1.029181/P991.804685/max3.592599，全99827中位.973277/P991.674602/max3.686656；不支持之前泛化为大raw norm导致学习失效，本轮保留原raw q Adam。

三组diag source/事件SHA保留inspection/early_training与rgb_axis_lifecycle。图和轴分布只读，不增加训练teacher、不筛原评分/点/帧。相对原GS normal <7度仍等待55k全部15七MAE，当前不commit/push/新分支。

## B10同一40k检查点中间诊断与实际目检（2026-10-10T11:30:39.834769+09:00）

真实40k保存审核child exit0，99827点finite/opacity-scale/NN32冻结，canonical位移max.03547572m/qmax60.00006485度，native pmaxdelta.61573005，RGB global E5.50377035（非GT）；实际加载新RGB资产/源图CPU复算/marker PASS，canonical轴loss.003838724。检查点父exec10940还在等待50k，不提前称父会话退出。

独立40k预览exec86565、15帧GS/Depth/Z exec27161、CPU复算exec21859均真实exit0并关闭。RGB PSNR18.29660847/SSIM.90320820/LPIPS.10866528（foreground10.95702402/bboxSSIM.78457834/bboxLPIPS.22543707）。GS原有向pixel-weighted15.71225125度/覆盖93.93837273%；Depth Normal25.68376882度/覆盖89.29189544%；ExpectedZ.07940795m/MedianZ.08330994m（93.93837%）。missing180=25.67076223度；共同GS+Depth370948/415433、MAE14.73601245/25.68376882度，此中间未生成Mesh，不能当三法线共同口径或正式全七MAE。独立CPU逐数组/帧/聚合和506来源SHA PASS。原GT评分全415433前景保持，无GT标签参与训练/源RGB候选。

实际首中尾对比B09同40k：黑背景正常、人物/平台出现；上衣/裤腿/手臂碎黑斑，脚周与地面黑灰伪形变，当前40k质量FAILED_EARLY。不把较暗的背光帧直接称缺失，GT中帧本身较暗。 首帧GS平台紫蓝、足周杂色和人物大块混色；Depth/Mesh列此时Mesh明确不可用，Depth高频噪声更明显，预测Z/GTZ对比仍有偏差。中间图不作为全量质量通过证据。

另source-only native alpha合成贡献诊断exec37263实际exit0/关闭，三TRAIN0001/0075/0149、0 optimizer/0 GT标签，source RGBconf只贡献自身前景alpha质量38.47498%/28.52117%/39.15403%，既有plane12.06668%/16.42836%/11.84679%。可靠RGB支持29621源opacity中位.012234、mean.253954，数量不等于像素覆盖；近半像素合成仍来自两个约束外的source点。这给出继续加lambda不足的证据，不把子组贡献当正式Normal覆盖或评分子集。

仅source XYZ/原NN全135TRAIN位置稳定/opacity>.05诊断：去掉source feature和轴cos筛选、保留motion-span≤.01，.2m band也只得到16110点（原15187外多923，全部source_static，无新增dynamic）；.05m band12647（原投影重合12051），仅52点重合RGBconf。说明简单去掉轴cos或source静态gate不会解决主要覆盖缺口，不据此改生产plane支持。现有资产/core/所有原点/帧/评分保持，不生成新训练prior/mask。等待本次55k完整20k/15七MAE与最终画质后决定下一迭代，不commit/push/新分支。

## B10完整验收与B09同55k比较（2026-10-10T11:54:42.629402+09:00）

| 指标 | B09概率gate | B10训练RGB轴约束 | B10−B09 |
| --- | ---: | ---: | ---: |
| Albedo MAE | 0.19121691 | 0.19107403 | -0.00014289 |
| Roughness MAE | 0.19018926 | 0.18083939 | -0.00934987 |
| GS Render Normal MAE | 15.94231363 | 15.54006647 | -0.40224716 |
| Depth Normal MAE | 25.18807492 | 25.16398643 | -0.02408849 |
| Mesh Normal MAE | 19.88648893 | 19.75904152 | -0.12744741 |
| Expected Z Depth MAE | 0.07691848 | 0.07666492 | -0.00025355 |
| Median Z Depth MAE | 0.08110909 | 0.08065818 | -0.00045091 |

GS覆盖率95.47845260%→95.48687755%；RGB PSNR19.81034533→19.75422671dB、SSIM0.92021880→0.91935445、LPIPS0.10036608→0.10111726。

共同376203/415433（90.55684069%），GS/Depth/Mesh共同MAE14.51173016°/25.10148879°/18.71977060°；missing180=22.96234465°。

完整20k训练与9步后处理实际exit0且会话关闭，226 CPU/20步RGB轴Preset、真实SUN光线数/材质/XYZ/q/feature梯度、共享几何与材质逐buffer均已验证。独立15帧七MAE复算与来源SHA、15time×135TRAIN预测Mesh、roughness来源复算、6视频全解码通过。476核心和双source快照逐SHA保持；native kernel SHA绑定，无GT训练材质/法线/几何/原能量/颜色。全部99827点/15测试帧及原评分定义保持，失败权重/图/数组/日志保留。

人工质量FAILED，原GS Normal <7°目标FAILED；本脚本不执行commit/push/新分支。

## 最终停止状态

按用户要求，当前B10完整训练、全部九步后处理、独立七MAE与实际画质检查均已结束。训练/检查点审计/后处理三个会话实际退出0并关闭；人工质量FAILED、原GS Normal<7°FAILED。停止后续迭代，未启动新训练/候选实验，未commit/push/创建新分支。完整总结：[1010-17总结报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1010-17-jumpingjacks_dir-原生PBR方向光迭代总结报告.md)。上方过程记录中的等待/运行中状态为对应历史时刻，当前状态见training_runtime/postprocess_state及final_completion.json。
