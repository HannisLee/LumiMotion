# Stage2 独立外观评测入口与浮点 PSNR 口径

本修改仅新增 `LIHAN/scripts/goal4_appearance_eval_common.py`、`render.py`、`parity.py`、`rgb.py`、`visuals.py`、`run.py`、`target.py`（各文件均以 `goal4_appearance_eval_` 为前缀），不修改旧 helper 或原六固定计量源码。服务为 ushas，环境 `lumimotion-ushas`。

## 接入边界

新门禁只接受真实 `scripts.train_static_dl_appearance`，固定 G2-T06@16000 原六初始化资产与137个原TRAIN输入 SHA。对实际启动 argv、READY snapshot/manifest/entry SHA、marker/配置/finished、累计 appearance 更新、保存点元数据、精确 state、原样 deform、真实首步梯度、11 maps 起点同义审计、全部冻结叶张量、存活期 native/tracer、真实退出和 GPU 证据分别核验。新执行器真实证据为 `runtime_module_audit.json`、`runtime_shared_library_audit.json`、`launch_provenance.json`、`child_environment_audit.json`、`runtime_capture_exit_code.txt` 与 `state.json`；不要求本轮没有生成的旧supervise格式。采集时间、PID、start_ticks、argv、物理GPU UUID、原冻结二进制SHA、子环境及真实退出逐项交叉检查。旧足部预算及旧foot阈值不套用新入口。完整恢复需原 state、Adam累计步数、同损失与来源；不称为原Stage1 Adam无缝恢复。

训练保存的 `appearance_forward/initial.npz` 与 `iteration_N.npz` 来自真实第一张 TRAIN 相机。独立 fresh 用原真实图像读取器/相机读取器，比较保存前向、T06初始或resume精确state前向、新 PLY 前向与重建state前向的11项 maps；逐项记录实际最大差，工程前向tol为2e−6，不用该tol放宽法线科学阈值。

真实 `save_ply` 保存 `_albedo_dc_stage1` 对应三字段，`load_ply` 优先读取；只有历史文件缺字段才从当前dc派生副本。本轮新 PLY 有三字段时必须严格核对全部冻结hash，不能排除该副本；缺字段才显式记录fallback，部分字段缺失直接拒绝。然后独立从真实T06 PLY重建、核对全部起始冻结hash，只apply精确state反照率和light。fresh另外比较实际新PLY与重建state全部11maps，避免副本问题掩盖错误几何或错误前向。

## 固定计量

runner沿用原 `extract_geometry_mesh_dynamic`、`eval_geometry_gt`、`goal_eval_audit` 和原离线分区工具；保持10,20,…,150完整15测试、原掩膜、原实际有效像素加权MAE、Depth支持腐蚀和每时刻135训练相机TSDF。不改变原六源码SHA。GT仅进入离线评测和分区，训练来源严格不含TEST图片或GT几何。

原三精确基线为2.6683778383745302 / 5.404577799172339 / 4.20848728697561°。每项同时保存原始Δ、严格不退化`current≤baseline`、原GPU重复测量工程一致性`|Δ|≤1e−5°`；tol内正Δ不得写为改善。冻结几何阶段预期指标在工程一致性范围内相同。

主PSNR为float sRGB `clamp[0,1]` 后逐帧全图三通道MSE→PSNR，再对15个PSNR求均值；原T06起点和新候选都必须由同入口重新以float渲染，禁止用新float对旧8bit缓存的微小差值验收。原GT RGB先Alpha黑底合成和uint8截断，再独立RGB/Alpha PIL BICUBIC缩放至400×400、float32除255。未使用GT Alpha替换预测背景。

未钳位raw float PSNR、GT自身Alpha>0.5前景PSNR、GT满码饱和通道比例、预测超1通道比例、饱和下预测量和合并MSE后PSNR均明确列为辅助。主全图收益预注册`Δ>0.05dB`，前景守卫`Δ≥−0.05dB`。raw用于理解loss效果，不更换主目标。

四类实现复用旧完整150帧RGB/Alpha/Normal/Separation和物理/几何maps一致性检查，仅新physical renderer从精确state与独立EVAL纯方向加载。训练方向严格135个，完整评测表150个，135共同方向逐项一致。脚部、黑边与整体视觉独立验收；冻结几何阶段不能以PSNR提升宣称补齐鞋尖。

## 当前验证状态

七文件CPU语法通过；已知MSE=0.25对应PSNR=6.020599913279624dB、raw/clipped区分、超1比例以及原六计量源码SHA验证通过。原缓存CPU基线见报告1013-01。**新GPU fresh、15float基线、新模型三MAE和四类渲染尚未执行，GPU接入NOT RUN；CPU检查不等于准入或科学PASS。** 后续真实smoke/fresh结果须追加，不覆盖失败现场。

完整执行命令由runner `plan.json`/README逐条保存，子步骤真实退出及日志在 `executed.json`/`logs/`。结果独立output子目录、禁止覆盖。数值FAILED与流水线失败均保留checkpoint、数组、渲染、统计、日志；联合成功须数值、足部和黑边目检同时成立。

## 最终收尾范围

收到用户“收尾当前目标，落档文档”后，七helper保留为实现草案，**仅CPU语法和计量检查通过，未执行任何真实GPU fresh、float RGB或新模型三MAE验收**。持久GPU runner与float baseline spec仅准备，未启动；不能把代码齐备或CPU PASS写为GPU准入通过。

GT读取器已进一步按原代码逐字保留float64计算次序（RGBA/255→RGB×Alpha→×255→uint8），避免代数重排在截断处产生极少1LSB差。旧缓存最终值与首版差异见报告1013-01，两版产物均保留。

实际native编译门禁两probe：首次真实exit1为ninja PATH缺失，retry1真实exit1为私有重编译二进制SHA仍不匹配原f951。共享原生源码不变、共享.so未被本Agent写入；所有私有cache失败结果和日志保留。原native来源约束未放宽，未用不同二进制运行模型。详细PID、exit、命令、版本与源码/二进制SHA见总控02 `native_rebuild_probe*` 和 `evaluation_closeout.json`。此收尾不产生新训练、PSNR提升或脚部恢复结论。
