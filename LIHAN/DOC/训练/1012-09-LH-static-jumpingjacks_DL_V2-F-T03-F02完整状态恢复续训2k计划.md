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

当前状态：resume入口已稳定；py_compile与CPU静态接口审查通过。真实恢复烟测已从F-T02@16000运行110步至16110并保存16100/16110；93项状态恢复检查、GPU/runtime capture和source provenance均PASS。fresh parity仍待执行/完成，故正式2k未启动，不能从烟测继续。正式命令以parity通过后存档的`commands/train.json`为准。

## 真实状态恢复烟测记录（2026-10-12）

- smoke目录：`output/smoke_test/1012-17-F-T03-F02状态恢复真实续训110步smoke`；model=`model_mlp`，真实clock 16000→16110，optimizer updates=110，saved iterations 16100/16110，point_count=54760。
- immutable snapshot：`output/1012-08-LH-static-jumpingjacks_DL_V2-G2-F脚部reset单因素配对总控/snapshots/foot-resume-stable`，manifest共101文件，所需13项科学代码均列入SHA，且同时包括`train_static_dl_foot_recovery.py`和`train_static_dl_foot_resume.py`。
- 训练PID 2564275，物理GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`，真实退出码0；supervisor与活跃runtime采集均PASS。raster/tracer SHA与配对训练一致，见smoke目录`actual_runtime_capture.json`。
- `state_restore_audit.json`的93项均为true；Adam组和LR、增密统计、RNG、相机余栈匹配。恢复后余栈25项，期望下一相机`image_0054`（index48），restored xyz LR=`6.861567188727149e-05`。这只是审计起点结果；下一相机与真实16101更新的逐项parity还需fresh工具终审。
- 首次supervisor预检因output目录和model_path不在同一任务目录而拒绝，未创建训练PID、无CUDA更新；原日志和拒绝状态保留于smoke目录`attempts/preflight-output-mismatch`。修正为整个smoke任务都在`output/smoke_test`后第二次执行成功。
- 当前烟测不是正式结果；fresh parity通过前不启动正式续训。
