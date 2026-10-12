# F-T04｜当前最佳基线增密统计配对2k

**优化思路：** 从G2-T06@16000延续相同法线/物理损失，保持原增密向量和范数作为控制，不新增监督。

## 输入与控制变量

源`output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp`@16000；原135TRAIN data_view。起点无optimizer资产，明确新建零动量Adam、清零增密统计，seed1012及独立不放回相机序列相同，30k XYZ LR日程从16000继续，不挪用其他轨迹状态。

固定route=`original`；normal_opacity_route=detach_opacity、ambient0、reset_policy=skip_all；原censored RGB/Alpha/expected normal/distortion损失与权重、阈值0.0002及原拓扑算法保持。两组唯一实验变量为增密统计路线，输出路径属于运行元数据。

## 步数、输出及执行门槛

正式新增2000次optimizer更新，global16001—18000；保存16500/17000/17500/18000的PLY/deform/lighting与optimizer、随机状态、相机余栈、增密统计。输出`output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k`，日志同目录`logs/`。16000边界不补结构或reset，终点18000不执行无后续优化的结构/重置。

正式前需新独立snapshot来源冻结、GPU1真实110步smoke至16110，跨16100增密；首次真实梯度、原拓扑算子、屏幕统计/分母/parameter.grad核验及存活runtime、fresh重载均PASS。正式从原G2-T06@16000重新初始化，不接续smoke。候选T05还需零更新诊断证实固定阈值下存在新增选择；若没有则登记未启动/方案否决，不改变阈值。

当前未启动训练；完整argv/命令、GPU/PID/真实退出码、snapshot SHA、日志与checkpoint待实际准备后记录。每卡最多2任务，重新检查显存，不中断其他任务。失败目录和日志不覆盖。

## 联合验收

原15测试10—150、原掩膜和MAE数学不变；完整六评测、四类150视角、正面/侧面/关键鞋部人工检查。三类主MAE须来自同一checkpoint，比较当前最佳G2-T06及旧冻结G2-T04；新更严格上限GS≤2.7683778383745303°、Depth≤5.604577799172339°、Mesh≤4.40848728697561°，旧上限同时报告不改写。必须腿足完整且数值保持才整体PASS，不以coverage或总点数代替几何。

训练结果：GS／Depth／Mesh未评测；足部恢复未评测；可视化未生成，最终状态PENDING。本计划不假定候选有效，完整命令/输出/指标/四类目检最终写入实验README。


## 本次收尾归档状态（2026-10-12）

按用户“继续，收尾实验，归档文档”要求，本计划以 **NOT RUN（未执行）** 归档。本轮未启动GPU诊断、smoke、正式训练或新模型评测；无本轮PID、训练日志、checkpoint、MAE及四类可视化。以上均为预注册方案，不能写作已执行结果；足部是否恢复与法线是否退化均无本轮证据。现存起点模型脚部仍为部分恢复、完整性FAILED。

命令草案、未冻结snapshot状态和资源核查见`output/1012-15-LH-static-jumpingjacks_DL_V2-F-T04-T05增密配对总控/README.md`。未来若另行继续，须先完成输入SHA／native来源／snapshot门禁审查、真实诊断及smoke/fresh验证；不能直接将本草案视为已通过训练验收。


## 重新激活阶段真实正式结果与独立终审

此前NOT RUN收尾保持为历史；本次用户重新激活后已真实完成16001—18000共2000更新，训练来源及独立fresh PASS。
## Sol独立18000正式评测与最终验收（2026-10-12）

最终 **FAILED**：本轮新严格数值FAILED（GS超上限），历史门槛PASS；腿足主体部分恢复，鞋尖/脚掌边缘仍短缩、偏低，几何FAILED。完整六步骤、fresh、来源及文件有效性PASS不能替代联合验收。

source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`；model：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp`；iteration18000；route `original`；reset skip_all；真实2000更新，无前移clock。
原15TEST三MAE（度）：GS **2.771108769218906**／Depth **5.579782566593626**／Mesh **4.182807652901499**。
相对同一G2-T06基线变化：`{'gs_render': 0.10273093084437557, 'depth_normal': 0.17520476742128732, 'mesh': -0.025679634074111668}`。新上限：`{'gs_render': '2.7683778383745302', 'depth_normal': '5.604577799172339', 'mesh': '4.40848728697561'}`；历史上限：`{'gs_render': '2.856110384679971', 'depth_normal': '5.624075573848699', 'mesh': '4.487353582304789'}`。
GS／Depth／Mesh覆盖率：`{'gs_render': 0.9980156202986353, 'depth_normal': 0.9660477247933015, 'mesh': 0.9930648236855562}`；原eligible541731，缺失惩罚另见goal_mae_summary.json，不修改原conditional-valid数学。
原6源码SHA不变；协议 `7cf1e7f6e1ce7f223319899febdfb1d443440add68cddee191fe5bf47e1646be`，互斥8对象+unknown复算恢复三主MAE与有效像素数量。
新Adam、零统计、seed首相机image_0096、真实训练内17/18/40与增密梯度算子审计、训练存活native/tracer及194来源SHA均独立fresh核查。11前向max_abs全0，真实exit0，4资产前后不变。

### 完整实际fresh与六步骤执行命令

```bash
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_CUDA_ARCH_LIST=8.6 'LD_LIBRARY_PATH=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu' /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight LIHAN.scripts.goal3_densify_parity --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view' --load_iter 18000 --frame 96 --output_dir '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/fresh_parity_18000'
```

```bash
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_CUDA_ARCH_LIST=8.6 'LD_LIBRARY_PATH=/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu' /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal3_densify_run.py --model '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp' --iteration 18000 --output '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000' --gpu 1 --execute
```

真实PID：fresh 2661665，评测总控2661960；两实际exit0。日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/logs/fresh18000.log`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/logs/evaluation18000.log`，各六子命令与日志完整列于 `/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/plan.json`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/executed.json`及下面渲染清单。
渲染输出：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/four_classes`，RGB/Alpha/Normal/Separation各150 PNG与150帧MP4，完整CPU解码PASS。

### 四类人工目检

- rgb：侧向10、斜俯视80和斜正面150均有连续双腿与双鞋主体；鞋尖仍钝、短缩，颜色存在不等于完整三维鞋形。真实a0 RGB含直射和阴影，未新增训练光照。
- alpha：轮廓和平台连续；平台与脚共用Alpha，可在鞋射线上保持1，因此Alpha完整不能排除平台替代或鞋尖偏低。
- normal：双腿主体连续、没有大段消失；鞋尖/鞋底边缘仍偏平，鞋部GT外轮廓内存在局部低高度与边缘大角误差，完整足形FAILED。
- separation：原static feature flag分离完整输出；单一静态标签显示符合现入口，但不表示动作/语义足部区域，也不能证明鞋尖完整。

人工范围：四类PNG及真实视频抽帧10/80/150；鞋部10/20/40/80/90/140/150。自动完整视频解码不称人工逐帧验收。代表图：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/four_classes/four_classes_10_80_150.png`，真实视频抽帧：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/manual_review/actual_video_10_80_150.png`；关键鞋证据：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/spatial_diagnostic/figures/footwear_zoom_0020.png`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/spatial_diagnostic/figures/footwear_zoom_0140.png`。四视频：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/four_classes/rgb.mp4`, `/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/four_classes/alpha.mp4`, `/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/four_classes/normal.mp4`, `/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/four_classes/separation.mp4`.
同GT object_4鞋6185原eligible像素局部MAE：{'gs_render': 17.309347445789648, 'depth_normal': 24.61855227809896, 'mesh': 26.119686577771745}；Expected/Median/Mesh同射线2cm内率：{'expected': 0.7510105092966856, 'median': 0.8080840743734843, 'mesh_depth': 0.7346806790622473}；近平台替代代理：{'expected': 0.041875505254648344, 'median': 0.08003233629749394, 'mesh_depth': 0.18787388843977365}。
GT语义仅离线；Alpha包含平台，不能拿全白鞋区域替代三维足形证据。终审：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/FINAL_REVIEW_COMPLETED.json`、`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/evaluation_18000/goal3_final/goal3_acceptance.json`。checkpoint、失败渲染、统计及日志原样保留。

完整训练与六渲染命令、source/model/iteration、日志和所有代表图视频见实验根README；本组最终FAILED，不修改原门槛或历史归档。
