# F-T03｜从F-T02@16000恢复训练状态续训2k计划

## 目的与冻结配置

沿F-T02正式16000终点继续2000次优化，验证从已保存状态恢复Gaussian与光强Adam、增密统计、随机数和相机余栈后，跳过后续opacity reset的延续表现。不得从PLY重新初始化Adam，也不得从smoke产物启动正式训练。F-T03将以F-T02@16000的完整状态作为唯一训练起点，固定原训练配置、`detach_opacity`、ambient=0与原135帧相机/方向样本；本实验不添加新损失、物理变量或几何监督。

- 起点：`output/1012-10-LH-static-jumpingjacks_DL_V2-F-T02-跳过重置足部恢复2k/model_mlp`，iteration 16000；检查其PLY、deform、lighting、optimizer/iteration_16000.pth、训练marker、continuation marker和finished marker SHA。
- source：沿用F-T02真实训练视图 `output/1011-09-静态DL-Goal训练总控/data_view`，核对恰为原135帧；仅原RGB/Alpha、相机与允许的GT光照方向。不得用测试图像训练，也不得使用GT法线、深度、材质、阴影、鞋部语义或审计标签监督。
- 训练步：真实clock 16001–18000，2000次optimizer更新。恢复F-T02保存的Gaussian和光强Adam、增密统计、RNG与相机余栈；BVH仍每步重建。完整历史逐位重放不作为声明。
- 正式checkpoint：16500、17000、17500、18000（相对保存步500、1000、1500、2000）。reset策略为入口固定的`skip_all`，逐条检查reset事件和实际更新顺序。
- 真实烟测：从F-T02@16000恢复110步，global 16001–16110，`--smoke --steps 110 --save_steps 100 110`，保存16100与16110。烟测只诊断真实余栈恢复、初始审计和真实时间推进，不以前移clock替代，不用于正式指标或正式训练初始化。

## 入口、快照与输出

- 预留独立输出：`output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k`；不得覆盖其中现存文件，不预建`model_mlp`。
- 拟用入口：`scripts.train_static_dl_foot_resume`；稳定实现、source SHA清单、状态恢复审计、smoke与Fresh检查通过后才正式启动。
- 代码快照必须不可变，bootstrap `argv[2]`指向快照，涵盖续训entry、恢复模块、F-T02训练entry/scene及其依赖；记录逐文件SHA，不把主repo当快照。
- 完整启动命令、argv、GPU UUID/PID/真实退出码、活跃进程runtime/native/tracer来源与SHA、日志、全部checkpoint以及失败产物均落入独立输出目录。GPU调度按实时资源核验，并遵守每GPU最多2任务、4卡总计最多8任务，不打断其他任务。

## 评测与验收

使用冻结原15帧与原掩膜评测RGB／Alpha／Normal／Separation及Goal三类法线与局部鞋部诊断。对比F-T02@16000与冻结基线；分别报告全局数值门槛、Expected/Mesh鞋尖长度和高度、多视角腿足完整性。保留渲染命令、source/model/iteration、渲染目录、日志、定量结果、代表图/视频、四类目检及最终`PASS`/`FAILED`。在完成目检前不得将训练exit 0或fresh通过视为最终通过；失败结果及模型、渲染、统计和日志必须保留。

当前状态：resume入口已稳定；py_compile与CPU静态接口审查通过。真实恢复烟测已从F-T02@16000运行110步至16110并保存16100/16110；93项状态恢复检查、GPU/runtime capture和source provenance均PASS。fresh parity已通过；root批准后正式2k从正式F-T02@16000重新恢复并已启动，绝不从smoke续训。正式完整命令与argv已存档于输出目录`commands/train.json`。

## 真实状态恢复烟测记录（2026-10-12）

- smoke目录：`output/smoke_test/1012-17-F-T03-F02状态恢复真实续训110步smoke`；model=`model_mlp`，真实clock 16000→16110，optimizer updates=110，saved iterations 16100/16110，point_count=54760。
- immutable snapshot：`output/1012-08-LH-static-jumpingjacks_DL_V2-G2-F脚部reset单因素配对总控/snapshots/foot-resume-stable`，manifest共101文件，所需13项科学代码均列入SHA，且同时包括`train_static_dl_foot_recovery.py`和`train_static_dl_foot_resume.py`。
- 训练PID 2564275，物理GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`，真实退出码0；supervisor与活跃runtime采集均PASS。raster/tracer SHA与配对训练一致，见smoke目录`actual_runtime_capture.json`。
- `state_restore_audit.json`的93项均为true；Adam组和LR、增密统计、RNG、相机余栈匹配。恢复后余栈25项，期望下一相机`image_0054`（index48），restored xyz LR=`6.861567188727149e-05`。这只是审计起点结果；下一相机与真实16101更新的逐项parity还需fresh工具终审。
- 首次supervisor预检因output目录和model_path不在同一任务目录而拒绝，未创建训练PID、无CUDA更新；原日志和拒绝状态保留于smoke目录`attempts/preflight-output-mismatch`。修正为整个smoke任务都在`output/smoke_test`后第二次执行成功。
- 当前烟测不是正式结果；fresh parity通过后，root已批准正式续训并于2026-10-12启动；实际GPU/PID/退出码及评测结果待运行结束后补录。


## 真实16110 Fresh parity与正式训练启动

- 独立fresh：`output/smoke_test/1012-17-F-T03-F02状态恢复真实续训110步smoke/fresh_parity_16110`，真实checkpoint@16110，status=PASS。11项真实前向输出最大绝对差为0；实际首个训练camera为`image_0054`、global iteration 16001，与保存的camera_rng/余栈和metadata一致；Adam组LR一致。训练新科学源13项、起点资产7项及相关177条来源通过审查，fresh checkpoint哈希一致；审计详情见`parity.json`和`logs/fresh_parity_16110.log`。
- root已批准正式续训从正式F-T02@16000重新初始化完整训练状态，2000次更新至18000；GPU1，启动、source、reset和runtime都以正式输出目录状态和日志为准。命令为`output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/commands/train.sh`，实际child argv在`commands/train.json`。烟测模型不参与正式训练。

## 正式训练执行记录（2026-10-12）

正式训练从真实F-T02@16000 optimizer/checkpoint与camera余栈恢复，不使用烟测产物；global 16001–18000完成2000次optimizer更新，保存16500、17000、17500、18000。训练使用固定`detach_opacity`、ambient=0、`reset_policy=skip_all`；训练终点point_count=55957，Opacity reset performed/skipped均为0，clock_shift_diagnostic=false，initial_sources_unchanged=true。监督边界仍为原135张训练图的RGB/Alpha、相机及允许的GT光照方向，不使用测试图或其他GT监督。

- 完整child argv、命令JSON：`output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/commands/train.json`；shell入口`commands/train.sh`；bootstrap `argv[2]`指向不可变`foot-resume-stable`快照。命令SHA256 `c86c0a811240a240ba0f96af435d206ba59b5f66dcf1a430755bcb14e88819c1`。
- PID 2567197，GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`，2026-10-12T02:10:42Z真实退出码0，GPU验证通过。监督器状态`TRAINING_COMPLETED_PENDING_EVAL`；supervisor与实际活跃runtime捕获PASS。
- raster SHA256 `f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3`；surfel tracer SHA256 `5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e`。日志在输出目录`logs/train.log`，runtime/source审计见`actual_runtime_capture.json`与`model_mlp/foot_resume_training.json`。
- 训练完整性通过不表示定量或视觉效果通过。本阶段Goal已暂停，因此尚未启动正式fresh/原15/150步评测；render命令、MAE、代表图/视频、RGB/Alpha/Normal/Separation目检及最终`PASS`/`FAILED`均保持`PENDING`。完整训练checkpoint和日志保留，等待恢复评测指令。


## 正式18000收尾与最终验收（2026-10-12）

当前状态：**FAILED**（法线数值PASS，足部部分恢复，联合验收FAILED）。source=`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`，model=`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/model_mlp`，iteration18000，render输出=`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000`，日志=`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/logs`及根`logs/`。

- 原15帧GS／Depth／Mesh MAE：2.777640687397°／5.562334364716°／4.242261114472°；相对冻结基线：+0.021530°／+0.138259°／-0.045092°，三项预注册上限全部满足。
- 鞋object4局部MAE：15.977355896626°／23.809666767439°／24.092253131401°；Expected2厘米比例76.313662%，Mesh75.650768%，近台面代理3.556993%。
- 代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/four_classes/four_classes_10_80_150.png`；足部图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/spatial_diagnostic/figures/footwear_zoom_0010.png`、`0080.png`、`0150.png`。视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/four_classes/rgb.mp4`、`alpha.mp4`、`normal.mp4`、`separation.mp4`。
- fresh正式18000 PASS、11前向差0；六评测子命令exit0，600PNG／4×150视频完整解码PASS、checkpoint不变。训练93项状态恢复及活跃runtime证据保留。
- 人工检查范围：四类10／80／150、鞋10／20／40／80／90／140／150、全景诊断10／150。未逐帧人工检查所有视频。
- 四类结论：RGB：双腿与鞋主体可见且连续；暗侧较黑、投影模糊。相较T02鞋主体和末端继续改善，鞋尖仍偏低，不能仅凭黄色鞋面判定完整恢复。；Alpha：人物与台面合并轮廓连续，鞋object4 Alpha覆盖100%；台面遮住鞋部缺损对Alpha的影响，无法用该覆盖率证明鞋面完整。；Normal：腿部主体连续，鞋轮廓接近GT且较T02改善。0010／0020／0090／0140／0150仍见鞋尖偏低或末端小幅短缩；Expected及Mesh均留有残差，未达到完整稳定恢复验收。；Separation：人物与平台近同橄榄色，静态门控不能解读为运动分离或鞋部独立保护。
- 最终终审：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/FINAL_REVIEW_COMPLETED.json`；报告：`/home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1012-05-足部恢复三轮训练归档与验收报告.md`。指定子Agent因usage limit未能参与本次补评，root直接完成原脚本及终审，执行限制如实记录。
- 失败结果保留：全部checkpoint／optimizer、渲染、统计、日志未覆盖或删除；不从smoke继续训练，不启动下一轮。

完整实际渲染／评测命令（沿原15帧协议）：

### predicted_meshes

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.extract_geometry_mesh_dynamic --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 18000 --frames 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 --split test --resolution 2 --voxel_size .02 --sdf_trunc .08 --depth_trunc 20 --alpha_threshold .5 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/predicted_meshes'
```

### normal_depth

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.eval_geometry_gt --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 18000 --frames 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 --gt_root /mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2 --gt_depth_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/gt_depth' --split test --resolution 2 --alpha_threshold .5 --interior_radius 2 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/normal_depth' --pred_mesh_manifest '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/predicted_meshes/manifest.json' --save_arrays
```

### audit

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal_eval_audit.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000' 18000 --protocol '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/protocol.json'
```

### goal3_numeric

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal3_foot_resume_target.py --evaluation '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000' --output '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/goal3_numeric'
```

### spatial_diagnostic

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal2_error_spatial_diagnostic.py --evaluation '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000' --gt_depth_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/gt_depth' --output '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/spatial_diagnostic' --figures
```

### four_classes

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight LIHAN.scripts.goal3_foot_resume_visuals --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 18000 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-11-LH-static-jumpingjacks_DL_V2-F-T03-F02完整状态恢复续训2k/evaluation_18000/four_classes'
```
