
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
