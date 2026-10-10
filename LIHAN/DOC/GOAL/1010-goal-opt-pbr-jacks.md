# 1010-goal-opt-pbr-jacks：Stage2 原生PBR方向光优化

日期：2026-10-10（Asia/Tokyo）。状态：进行中，未达到目标。输入来源纠正：旧转换实际albedo-alpha；B01已中止，B02使用原RGB自身alpha重新完整训练。

## 原始目标与验收

1. 在1001-stage2-perlight完成方向光原生PBR完整Stage2训练、四类渲染与前后景检查；若不支持则审计并修复。通过后commit、push一个版本，记录本地和远端SHA。
2. 从通过目标1的版本新建用户指定分支1010-goal-opt-pbr-jacks。仅训练Stage2，在jumpingjacks_direction_light上迭代；每次完整训练文档、完整MAE和修改思路/结果都落档；每次新改动前读取本文件。
3. 主目标为全15测试帧、全部前景的GS Render Normal角度MAE < 7°；同步报告覆盖率、Depth/Mesh Normal、Albedo、Roughness与Expected/Median Z、RGB指标。不得缩小帧集、换评价定义、丢弃困难像素来达标。覆盖率不得通过删减Gaussian显著降低。

## 数据边界

训练只用原始RGB/输入RGBA轮廓、相机/time、已有Stage1预测PLY/MLP、GT世界方向。GT albedo、normal、roughness、depth、mesh、blend不得参与训练、初始化、损失或生成训练几何先验。输入RGBA轮廓为原始图像自带的alpha，不从材质/法线GT导出。

为严格遵守“只能使用GT light direction”，灯表从lights.json只提取方向并去重；每帧为白色单位光，原SUN intensity/color不传入训练。基线PBR总辐照度固定π（纯数值初值，使albedo/pi项在正对光时为albedo），不是GT强度。后续如优化强度，仅使用训练RGB估计，不能使用测试RGB或GT强度。GT数据只在独立离线评测命令读取。

## 来源与评测固定口径

Stage1来源：output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，iteration35000；数据data_view，135训练帧/15测试帧，resolution2。禁止重跑Stage1。

原1009-04冻结几何Lambertian参考：GS21.52504346°/覆盖92.937489%，Depth27.60512758°，Mesh22.88577862°；其强度标定不直接作为本轮监督或参数。MAE使用已有通用协议，GT几何/深度来自dir自己的blend。Mesh使用所有135训练相机位姿在各测试目标时间融合预测深度；不使用测试相机几何或GT几何。

## 迭代记录

### B00：原生PBR方向光完整基线（目标1，训练进行中）

- 输出：output/1010-02-jumpingjacks_dir-PBR方向光基线。
- 分支：1001-stage2-perlight，来源HEAD a9e4d1061fec9986d82362f78b42c0c269e6146e加方向光实现。
- 简化思路：先保留原生PBR材质/SH/opacity优化范围、间接光和2000步diffuse warmup，检查现有SUN delta/GGX/BVH渲染真实可用。方向去重、强度固定π；不用GT材质或几何。
- 完整Stage2：35000→55000，20000步；原生每4步更新，固定几何故不能预先声称法线改善。
- 渲染：15帧RGB测试及全150帧四类可视化；转灯/HDR另检查；前景与黑背景均检查。
- 完整MAE：Albedo、Roughness、GS/Depth/Mesh Normal、Expected/Median Z、共同像素角度与覆盖率；待评测，无结果。
- gate：完整训练及前后景渲染通过后才提交/push，再建目标分支；质量目标7°未达到。

## 当前证据

2026-10-10开始核对：GPU空闲、无训练进程，当前代码尚未运行验证。所有运行以进程和实际checkpoint为准，state文件不是完成证据。未提交、未push、未建目标分支。

### B00执行进度（2026-10-10）

- 完整训练实际PID195148，开始00:50:38 JST；exec session3592。完整离线后处理PID195873，session1953，核对同PID的/proc启动时间+命令行，训练正常结束和55k真实checkpoint后才执行，不重启训练。
- 光照回归及原有回归35项通过（logs/unit_tests.log），完整unit discovery另执行（logs/all_unit_tests.log）。这不是完整训练或图像质量PASS。
- Roughness GT旧导出因透明头发材质显式失败，失败日志/空输出保留；新增--alpha_composite显式开关，透明部分按原Alpha混合透明与r自发光，定义透射层alpha加权有效r。原不透明默认行为不变。15帧dir roughness EXR完成，单独输出output/1010-02-jumpingjacks_dir-PBR方向光基线-GT评测专用/roughness_alpha_composite，不输入训练。
- 当前无MAE结果，未commit/push，目标分支待目标1通过后创建。

### B00补充审计

- 全部137项CPU unit tests通过，日志logs/all_unit_tests.log。
- Cycles16样本的透明roughness导出在低Alpha头发边缘出现r/alpha>1，不能拿来评测；原文件与日志保留，未裁剪。复核dir原始scene_metadata为BLENDER_EEVEE，因此新增导出--engine source，保持源引擎与透明混合语义。最终GT为独立GT评测目录roughness_source_alpha；全15帧load_target严格范围检查通过，未裁剪任何GT值。
- 只读Stage1预测审计：99827个Gaussian、54244静态；只用静态高opacity预测位置进行种子1010的RANSAC+SVD，主平面方向约[-.002546,-.001926,.999995]，offset .008620。高opacity静态点约59.6%在.03距离内。未使用任何GT几何/normal；诊断文件inspection/source_plane_audit.json。
- 下一步候选思路（未实施）：当前原生Stage2排除了xyz/rotation，不能直接学习更好法线。待基线验收、提交和建目标分支后，可用模型自身位置估计的静态主平面和预测法线/深度自洽项开放几何优化；人体仍使用RGB+GT方向，不以GT normal/albedo监督。必须完整评测，不能仅凭平面拟合声称7°已达标。

### B00 40k预览实测（完整训练仍进行）

15测试帧按自身相机/time/方向渲染，RGB PSNR17.76884325、SSIM.87406203、LPIPS.12072294、前景PSNR10.66144015。首/中/尾人工图见output/1010-02-jumpingjacks_dir-PBR方向光基线/inspection/preview_40000_rgb.png：黑背景正常，无全屏异常；人物颜色斑驳、手/头/腿边缘欠完整，平台脚周伪几何/阴影误差明显，背光0080人物较暗（对应GT亦背光，但平台预测缺少棋盘格）。此预览没有质量PASS，不能据此commit。待55k最终全量评测后，若仍存在这些问题，先在目标1分支审计修复和完整重训，前后景验收后才commit/push、建立目标分支。

目标1的产物自动审计、三法线/Albedo/Depth独立CPU复算、Roughness复算以及训练/评测归档脚本已准备，完整训练结束后顺序执行。自动产物PASS不代表人工图像质量通过。

### B01候选修改（准备中，尚未开始训练）

已重新读取GOAL后审计：原生Stage2过滤掉xyz/rotation，无法直接修复法线/平台几何；40k预览仍斑驳，不能凭产物完整性提交。准备新增独立pbr_gt_directional_geometry preset，旧preset保持原生材质范围。

新分支在原生Gaussian PLY的真实xyz/rotation上微调，复用原PBR/GGX/BVH、原MLP；几何约束仅来自训练RGB/输入Alpha、模型预测GS-Depth一致性，以及Stage1静态高opacity预测点的固定种子RANSAC主平面。没有外部GT法线/几何先验文件，不读取GT albedo/normal/roughness/depth/mesh/blend。平面模型必须由来源点重新计算、记录拟合证据。默认不启用此修改，B00进程及其保存配置不变；待B00完整MAE/审阅后决定是否作为目标1修复重训。

将几何更新延后至2000步材质warmup后，有界位移/旋转、轮廓保持、源点/旋转弱锚定，冻结opacity以避免降低覆盖率换分数。位置/旋转保存回原生PLY，所有离线渲染/三法线/Mesh自动使用真实修改几何。未训练、无新MAE；不宣称达到7°。

### B01接入验证（尚非训练实验）

新增原生几何preset和真实PLY xyz/rotation更新；位置最大位移.08、旋转最短弧最大60°，2000步几何warmup、1000步ramp，固定源点/旋转弱锚定、输入Alpha轮廓保持，默认冻结opacity。主平面只拟合来源静态高opacity位置，支持集合还包含低opacity的近面表面片；不使用外部GT几何/法线。

CPU新增5项测试（倾斜来源平面、无静态支持拒绝、最短弧/位移信赖域、新preset opt-in、四元数法线梯度），与已有测试共142项执行，倾斜平面精确性测试仍失败。前一记录中“通过”判断过早，已更正：即便把outlier移远仍存在微小偏斜；审计发现分量中位数通常不在倾斜平面上，直接以该中心SVD会有偏差。修复为以inlier均值中心SVD、沿拟合法线投影的中位数定offset；正在重测，失败日志保留，未放宽拟合门槛。

实际35k来源权重、训练frame0001单帧GPU前向/反向与参数投影通过：xyz/rotation/albedo/roughness梯度均有限且非零，源PLY/PTH哈希未改。不运行训练，不产生候选checkpoint，没有新MAE。审计output/smoke_test/1010-03-dir-PBR几何GPU梯度/audit.json。主平面支持14553点，方向约[-.002546,-.001926,.999995]；当前基线B00仍使用原先加载的材质优化代码，不启用新几何。

### B01数值修复与强度处理计划

倾斜平面SVD修复后，实际日志unit_tests_geometry_extension_svd_fixed.log明确142项OK；更新后的真实单帧GPU检查正在复核，旧结果文件保留。旧记录中提前判定142项通过已更正，依据实际日志而非预期。

为遵守只用GT方向而避免固定π导致亮度上限，准备仅由训练RGB优化一个全局正光强标量；初值仍π，白色/方向固定、不给训练提供GT颜色/能量。不进行测试RGB标定。新几何preset可启用，旧方向光preset默认固定不变。保存初始scale和实际拟合scale到同iteration lighting marker，离线渲染必须恢复该真实值；不能把该标量叫作已恢复的GT强度。无GT材质/法线约束，光强与albedo仍有可辨识性限制，报告必须如实说明。

### B01光强接入数值检查

143项CPU测试实际OK，日志unit_tests_geometry_intensity.log；实际35k权重单帧GPU检查PASS，log光强梯度-0.13456645607948303有限且非零，xyz/rotation/albedo/roughness梯度亦通过。更新来源拟合主平面[-0.0029613651534769734, -0.002400413027768733, 0.9999927341404656]，支持14556点。此为接入检查，无新训练或MAE。旧失败/修复前审计保留。B00仍在完整训练，未提交。

### B00完整训练结束

实际训练开始00:50:38、最后55k保存01:35:25、正常Training complete 01:35:26 JST；训练loop44:38、完整20000步，真实PLY/PTH与同iteration lighting marker已保存。15测试帧RGB最终PSNR18.38456159、SSIM.87708970、LPIPS.12493864、前景PSNR11.30800691。全150帧可视化、转灯/HDR、完整MAE与独立复算仍在后处理；不能标记质量PASS。最终RGB首中尾图inspection/rgb_final_55000.png，待审阅；未commit/push、未建目标分支。

### B00最终四类目检（首/中/尾；全部数值评测待后处理完成）

实际查看55k RGB与GT0010/0080/0150，以及四类RGB/Alpha/Normal/Albedo0001/0075/0150图。RGB：黑背景正常、人体/平台存在，但衣服/皮肤强斑驳、手头部欠完整、脚周平台伪几何和阴影误差；0080背光GT也暗，预测平台仍缺棋盘格。Alpha：主体与平台大致覆盖，头手边界毛糙/漏覆盖、平台边缘残影。Normal：人体/平台斑驳，脚周强异常，几何冻结不能修复。Albedo：明显残留照明/暗背面结构、平台偏白，未恢复棋盘格。当前人工质量gate=FAILED；不提交此版本。完整MAE和全部视频审计继续完成，保存失败结果后进入已准备的B01修复完整训练。

B00 Albedo完整15帧先出结果：raw全内部前景MAE.19659954015133715、覆盖96.27566479615834%。其余MAE待完成，不预填。

### B00全量结果（2026-10-10T01:48:27.915814+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.19659954 | linear RGB | 96.275665% |
| Roughness MAE | 0.29154181 | r | 96.560916% |
| GS Render Normal MAE | 21.22875590 | degree | 93.078788% |
| Depth Normal MAE | 27.80211481 | degree | 88.449882% |
| Mesh Normal MAE | 22.79203567 | degree | 93.778058% |
| Expected Z Depth MAE | 0.08852290 | m | 93.078788% |
| Median Z Depth MAE | 0.09119155 | m | 93.078788% |

RGB：{"psnr": 18.384561590747985, "ssim": 0.8770897030830384, "lpips": 0.12493863701820374, "psnr_foreground": 11.308006909041387, "ssim_foreground_bbox": 0.7279389182726542, "lpips_foreground_bbox": 0.26818381746610004}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-02文档。

### B00最终审核与下一步

完整20k训练、四类150帧、转灯72/HDR150、6视频完整解码、全部15帧七项MAE、83来源哈希与独立CPU复算完成；执行/产物PASS。人工RGB/Alpha/Normal/Albedo、0010三法线/Depth图和HDR/转灯首中尾已审阅，图像质量FAILED，法线21.22875590°/覆盖93.07878767%未达7°；不commit/push。最终训练/MAE文档和archive1010-02已回填失败结论并同步三个checkout，失败权重/图/日志保留。

决定在目标1分支运行B01修复：同35k源和独立20k Stage2、pbr_gt_directional_geometry，预测几何+输入RGBA/训练RGB，方向-only白光，光强由训练RGB拟合。输出1010-04-jumpingjacks_dir-PBR预测几何与RGB光强；完整评测和报告命令已准备，无GT材质/法线/几何训练输入。此为目标1修复，质量通过后才提交/push，再建立用户目标分支。

### B01完整修复训练启动

状态：B00完整训练和全量MAE已完成，人工图像质量FAILED；本B01作为目标1修复已启动，尚无最终MAE。真实训练PID206152，开始2026-10-10T01:54:52.575485+09:00，来源1009-02/35k，独立20000步Stage2至55k；后处理核对同一PID结束和真实55k后执行，不重启训练。源码快照见commands/code_snapshot、SHA256见code_manifest.json。

实际开训preset=pbr_gt_directional_geometry；143项CPU与真实35k单帧GPU梯度检查通过仅证明接入，完整图像质量/七项MAE待实际评测。所有训练输入同GOAL边界，无GT材质/法线/几何或测试RGB。未commit/push，未创建目标分支。

B01运行核对：实际配置为pbr_gt_directional_geometry，opacity/scale/feature冻结；37700实际训练光强4.64452362、各几何损失有限。只读40k预览等待实际保存完成；完整MAE仍待55k。postprocess PID206970等待同训练PID，无新增训练。源码与评测脚本快照已落档，文档索引更新为B00质量FAILED/B01正在训练。

### 输入来源审计纠正：B01中止、B02重新完整训练

# 1010-04 jumpingjacks dir PBR预测几何与RGB光强：中止记录

状态：FAILED_INPUT_PROVENANCE，2026-10-10 02:03:55 JST。训练实际PID206152；完整20k计划在4177步（累计39177）被主动TERM，exit143。没有新的40k/50k/55k checkpoint，只有复制的来源35k，不能把它当训练结果评测。

## 原因与纠正

审计旧dataset_manifest.json与转换代码发现mask_source=albedo-alpha。此前“原始RGBA自带alpha、无GT材质输入”的声明不准确。原RGB实际有自身alpha，全150帧16-bit alpha与albedo alpha逐像素完全相同（差异0）；旧输入文件数值无额外轮廮信息，但来源链不满足严格不读取GT文件的流程。为消除文件来源歧义，立即停止，不覆盖/删除失败产物，新建1010-05重新完整训练。已有35k来源保持不变；其旧alpha与原RGB相同的继承限制明确披露。

## MAE和质量

本轮新权重各项Albedo/Roughness/GS/Depth/Mesh Normal/Expected/Median Z MAE均N/A：中止前未保存任何新checkpoint，没有可评分的新训练产物。不能用35k来源或B00 MAE冒充本轮结果；B02会重新完整20k并评测全部15帧七项MAE。40k预览及最终后处理均真实退出FAILED，未自动重启。

四类目检/前后景：N/A（未保存候选新权重、无离线渲染）。法线7°、图像质量、完整Stage2：均未通过。未commit/push，未建目标分支。

## 保留证据

源model_mlp/point_cloud/iteration_35000、deform/iteration_35000、方向灯表、训练TensorBoard、完整train.log、postprocess.log、preview_wait.log、POSTPROCESS_FAILED、源码快照/哈希、所有计划命令均保留。实际单元/单帧梯度检查通过仅证明之前代码接入，不能弥补输入来源声明错误。输入审计inspection/input_provenance_failure.json。

完整原训练命令train_argv.json与commands/train.sh；原渲染/MAE命令planned_commands.json（中止后没有执行）。日志均位于本输出logs。

### B02简化思路与计划

# 1010-05 dir原生PBR预测几何与RGB光强完整Stage2计划

状态：B02准备开训，B01因输入文件来源不符主动中止；同35k来源、同几何/材质参数重新完整训练20000步至55k。新source_path=本实验data_view，135/15固定划分；原RGB自身RGBA、相机/time。训练启动前逐图核验原图SHA/转换SHA及像素，不读取GT材质/几何。全部150帧与旧转换RGBA数值完全相同，来源链已纠正；无新MAE。

来源同1009-02 Stage1/35k，独立20k Stage2至55k；135训练/15测试、400×400、黑背景。只用RGB/输入RGBA、相机/time、Stage1预测、GT方向，原GT颜色/强度丢弃。pbr_gt_directional_geometry开启真实PLY xyz/rotation，2000步warmup、1000步ramp、位移.08/旋转60°、预测GS-depth一致性、来源静态RANSAC平面、轮廓与弱锚定；opacity/scale/feature冻结。白色正光强初值π，仅训练RGB拟合，不读GT能量或测试RGB。GT材质/normal/depth/mesh/blend仅离线评测。

完整MAE：Albedo、Roughness、三类Normal、Expected/Median Z、覆盖率/共同像素和RGB。全150四类可视化、转灯72、HDR150/2048采样；实际四类目检/前后景结论待回填。所有失败结果保留。不能凭GPU单帧或CPU143项通过代替此完整训练/质量验收。


旧输入RGBA与新输入150帧逐像素相同，只纠正来源链，不调整评分或几何参数。147项CPU测试实际OK。完整MAE待真实55k产物；未commit/push，未建目标分支。

## B02实际启动

训练PID208981，实际开始2026-10-10T02:10:11.814472+09:00；新source_path=/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/data_view。原生启动前150帧原始RGBA来源/哈希/像素审核PASS，模型保存rgb_alpha_provenance.json。完整20000步及全量MAE/前后景验收待完成，未commit/push。

B00机器可读metrics_summary/visual_review/artifact_checks现明确标记strict_input_source_gate=FAILED_LEGACY_ALBEDO_ALPHA_SOURCE，修正仅GT方向输入的旧声明；实际albedo RGB/normal/roughness/depth/mesh没有训练，旧albedo Alpha与原RGB自身Alpha全150帧完全一致。MAE值与人工质量失败结论保留，完整archive SHA更新并同步。独立离线错误分布中位数11.1313°、P90 54.1042°、>=30°像素占24.3972%，仅诊断，不生成训练监督或改变评分集。

### B02运行中的来源材质初始化审计（候选修复，未修改训练）

只读35k预测DC，不使用GT材质：native get_albedo将SH2RGB硬clamp至[.03,.97]，全部43.5029%通道、高opacity54.5934%、静态高opacity58.8220%通道落在区间外；全部28.9431% Gaussian三个通道同时clamp。CPU反向实际验证43.5029%通道梯度为零，可能导致平台棋盘纹理/人物斑驳难以恢复。原模型SHA未改，B02继续原完整训练。

候选后续修复：仅将来源PBR DC初值投影到[.031,.969]内部，保留原材质公式和SH来源；CPU初始外观max差约.001、零梯度消失。尚未落实或训练，不能预报MAE改善。须在B02完整结果后结合质量审阅决定、每次实际新修改前读GOAL、独立完整训练及各项MAE。证据inspection/source_albedo_clamp_diagnostic.json与source_albedo_clamp_gradient.json。

## B02 40k只读预览（完整训练继续）

15帧RGB实测：{"psnr": 18.152288464182984, "ssim": 0.8832480907440186, "lpips": 0.1147700935602188, "psnr_foreground": 11.229736152077928, "ssim_foreground_bbox": 0.745617703596751, "lpips_foreground_bbox": 0.24591667254765828}。已实际查看inspection/preview_40000_rgb_comparison.png，0010/0080/0150对照GT及B00 40k/55k：黑背景正常，人物/平台存在；平台起伏有所减轻，衣服/皮肤仍明显斑驳，头手/边缘漏覆盖，背光0080 GT也暗但预测平台棋盘格仍未恢复。预览人工质量FAILED_SO_FAR，不能commit；完整55k七项MAE待完成。

真实40k checkpoint：99827 Gaussian，opacity/scale/feature冻结，最大位移.03003785、旋转最大60.00003°，均在界限内；真实训练RGB拟合光强4.75412703，离线预览已恢复该值；未用GT能量。证据inspection/checkpoint_40000_audit.json与preview_40000_review.json。

### B03候选材质初始化接入（未完整训练）

重新读取GOAL后新增独立pbr_gt_directional_refine与pbr_albedo_interior_init默认false。只投影35k预测PBR DC至[.031,.969]，保留SH来源，原生函数RGB2SH/SH2RGB、不改renderer/GGX/MAE定义。NativeTrainer在optimizer setup前初始化一次并记录source SHA/比例/外观差；真实PLY保存，离线渲染不重新初始化。151项CPU实际OK，真实35k/frame0001 GPU单帧前后向PASS，32964个原本clamp通道恢复非零光度梯度，原SH与source文件未变、0 optimizer steps。首次独立脚本模块路径错误保留，修正后实际exit0。无新checkpoint或MAE，不提前宣称7°或质量通过。B02完整训练仍继续；待最终结果后才决定新完整B03。文档修改1010-05与smoke1010-06。

## 真实NativeTrainer构造补充

check_constructor.py实际正常exit0：从只读35k构造新refine Trainer、产生pbr_albedo_initialization.json、源SHA保持不变、RGBA150帧审核PASS；材质投影后clamp通道比例0、最大初始外观差.0010000467；优化组xyz/albedo_dc/albedo_rest/roughness/rotation/albedo_dc_stage1，opacity/scale冻结，几何/正光强优化已接入。iterations_executed=0，无新训练checkpoint；constructor_model_mlp仅保存复制来源35k和初始化/配置记录，不能作为训练结果评测。证据constructor_audit.json与logs/check_constructor.log。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-06-dir-PBR材质初始化审计/commands/check_constructor.py
```

## 更新后材质投影补充

审计实际native albedo_lr=.01，单步RGB变化可约.00282，大于初始化margin .001；只初始化可能再次越过硬clamp边界。Refine增加pbr_albedo_project_after_step=true，原生每4步Adam更新后把真实PBR DC投影回[.031,.969]，维持梯度；不修改shader/GGX/MAE公式。初始化只执行一次，更新后投影持续执行，离线渲染直接读保存PLY，不执行任一操作。

新增连续8次原生LR Adam越界后恢复梯度、投影保留SH/拒绝共享存储/NaN/margin等测试；全量153项实际OK，logs/unit_tests_project_after_step_guard.log。新增真实GPU越界后投影+原PBR单帧反向检查PASS，原先clamp通道中33951个得到非零光度梯度，源文件/SH未变、0 optimizer steps；此检查故意将一个材质通道扰动越界以测试投影，不是训练。证据audit_with_projection.json、check_gpu_with_projection_guard.log。初始化单独版本和投影安全护栏前版本的日志/源码快照保留，当前code_manifest/code_snapshot是最新护栏版本。

“SH不变”的检查范围是初始化/投影与预训练源文件；原生SH损失仍正常优化_stage1 SH DC与SH rest，不能把训练中SH参数说成冻结。B02在开训时已加载旧NativeTrainer与geometry preset，未启用新增初始化/投影，运行继续到55k；当前source renderer未改。

### B03完整运行准备（尚未启动）

输出1010-07-jumpingjacks_dir-PBR材质初始化与投影修复，refine preset；同原始RGBA、135/15、35k来源、独立20k至55k，仅初始化与更新后投影PBR DC维持梯度，其余参数/原生PBR/MAE保持。153 CPU与真实GPU已通过，源码/全部训练及9步完整MAE/渲染/独立复算/报告命令准备。等待B02完整最终结果审閱后决定开训；无B03权重/MAE，不提前提交或标记目标完成。计划见训练1010-07文档。

## B02 50k checkpoint实测（完整训练继续）

{"status": "PASS_CHECKPOINT_ONLY", "iteration": 50000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "max_displacement": 0.06531433016061783, "max_rotation_deg": 60.000064849853516, "clamped_albedo_channel_fraction": 0.7175313234329224, "actual_light_scale": 4.703611850738525, "final_mae": false}

99,827 Gaussian全部保留，opacity/scale/feature未改，位移/旋转在信赖域内，权重全部有限。材质硬clamp通道71.7531%，高于35k约43.5029%；仍是预测参数诊断，不是GT MAE，不能替代最终图像质量/7°验收。Refine更新后投影候选有对应证据，B03仍等待本次55k完整后处理及人工审核，不提前开训。

## B02完整训练完成（后处理进行中）

实际启动02:10:11.814、55k保存及Training complete 02:54:30 JST，完整20000步、loop44:02，训练exec实际exit0。最终15帧RGB PSNR18.91228960、SSIM.89010724、LPIPS.11933970、前景PSNR11.99286383。已查看55k RGB0010/0080/0150对照GT/B00/40k：平台较平整、黑背景正常，人物衣服/皮肤仍斑驳，头手/边界缺失，平台棋盘纹理未恢复；初步最终RGB质量FAILED，不能提交，四类全部和转灯/HDR/七项MAE待后处理完成。图inspection/rgb_final_55000_comparison.png。

training_runtime.json补回真实开始/结束/步数/exit0；B03后处理准备脚本改为合并原runtime记录，避免丢失开始时间。其余训练/渲染/评分代码不改。

## B02最终四类首中尾实际目检

已实际查看inspection/four_categories_final_55000.png（0001/0075/0150）。黑背景正常，RGB人物斑驳、头手稀疏/缺失、平台棋盘纹理未恢复；Alpha轮廓基本完整但边毛糙、平台边残影；Normal平台比基线平滑但脚周/头手仍异常；Albedo正背面照明严重残留、平台多白。四类最终质量FAILED，不能commit；全部七项MAE、HDR/转灯/视频检查待后处理，不预填结果。证据visual_review_early.json。

### B02全量结果（2026-10-10T03:08:19.550962+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.18603245 | linear RGB | 96.517409% |
| Roughness MAE | 0.24401021 | r | 96.793436% |
| GS Render Normal MAE | 19.40743705 | degree | 93.694049% |
| Depth Normal MAE | 27.60153971 | degree | 89.068995% |
| Mesh Normal MAE | 22.24380070 | degree | 94.072209% |
| Expected Z Depth MAE | 0.08647351 | m | 93.694049% |
| Median Z Depth MAE | 0.09010209 | m | 93.694049% |

RGB：{"psnr": 18.912289595136517, "ssim": 0.8901072422663371, "lpips": 0.11933969706296921, "psnr_foreground": 11.9928638250002, "ssim_foreground_bbox": 0.7621153076489766, "lpips_foreground_bbox": 0.24486602246761321}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-05文档。

## B02完整最终审核与B03决策

全量后处理03:08:19 JST正常exit0，七项15帧MAE/独立复算PASS，150×4/72转灯/150HDR2048与6视频全解码PASS。GS19.40743705°覆盖93.694049%、Depth27.60153971°、Mesh22.24380070°，GS共同像素18.51964521°（88.855724%），缺失180惩罚29.53432483°；Albedo.18603245、Roughness.24401021、ExpectedZ.08647351m、MedianZ.09010209m。较B00 GS21.22875590°改善但远未7°。

已实际四类/三法线深度/HDR转灯首中尾审核：背景正常，人体斑驳、头手边缺失、平台纹理与照明残留、脚周伪起伏、Mesh缺面，最终质量FAILED，commit_allowed=false。原始RGBA来源PASS；未改评测全集/删除Gaussian。完整文档训练/报告1010-05及archive同步。决定启用已通过153 CPU和GPU检查的B03 refine初始化+更新后投影，同35k独立完整20k；不使用GT材质/法线。未commit/push、未建目标2分支。

## B03实际启动

{"status": "TRAINING", "started_at": "2026-10-10T03:10:06.570000+09:00", "training_pid": 219507, "process_start_ticks": "7977957", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_refine", "session_id": 26125}

B02完整最终FAILED已归档；fresh源码manifest匹配，150帧白色单位方向灯表核验PASS，启动前无候选model、无其它训练。等待真实模型配置/初始化记录和150RGBA preflight接入，再完整20k与全部MAE。没有GT材质监督、没有新MAE/7°声明。

B03实际启动配置PASS：refine、初始化true、更新后投影true，150帧原始RGBA来源PASS；99827 Gaussian真实材质初始化后硬clamp通道0、初始外观最大差.0010000467，source SHA记录吻合。不是最终MAE或训练质量结论。证据inspection/launch_configuration.json及model_mlp/pbr_albedo_initialization.json。

## B03运行中只读来源平面形变诊断

原生“binary feature”实际是sigmoid(2×feature)软门，静态支持仍有少量MLP形变；实际只读35k三训练时刻0001/0075/0149检查，支持14556点中feature P90=.01290935、P99=.37615985。形变轴偏差P90≤.05595°、P99≤1.99525°；位置形变P90约.0005m、P99约.0193m、少数max.20857m。大多数支持的形变近零，不能归因全部平台误差于MLP。

同一固定支持上，35k canonical平面法线均角17.9940°，B02真实55k为.92838°（中位.24708°、P902.71599°），平面位置绝对均差.0184996→.0099137m。说明已支持的平面参数确实修正，但画面脚周/噪声仍可能由支持之外的来源片影响；不能盲目增加相同平面权重或把该预测一致性当GT法线MAE。source SHA不变、0 optimizer steps、GT输入false，B03训练未改。证据inspection/source_plane_deformation.json、commands/audit_source_plane_deformation.py与日志。最终结果待完整训练。

## B03 40k只读预览与真实检查点

15帧RGB：{"psnr": 18.469882165991756, "ssim": 0.8856826225916544, "lpips": 0.11016974796851477, "psnr_foreground": 11.435341690063554, "ssim_foreground_bbox": 0.7512827237447103, "lpips_foreground_bbox": 0.23224637508392335}

实看0010/0080/0150：黑背景正常，0080平台棋盘纹理开始出现；人体/脚周仍明显斑驳、头手边稀疏，正面平台多白及影周伪结构。FAILED_SO_FAR，完整55k与七项MAE待完成，不能commit。比B02同40k PSNR18.15228846、LPIPS.11477009略改善，不将不同迭代效果或早期预览当最终结论。

真实40k检查：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 40000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "max_displacement": 0.03078901208937168, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 5.285261631011963, "final_mae": false}

99827全保留，材质DC仍[.031,.969]且clamp比例0，xyz/rotation信赖域内，opacity/scale/feature冻结；原生材质投影实际持续生效。光强为训练RGB拟合5.28526163，离线预览恢复同值。预览实际exit0，证据preview_40000_review.json、checkpoint_40000_audit.json及preview_40000_rgb_comparison.png。

## B03运行中来源局部几何与平面支持分布诊断

只读35k点云：按来源soft feature分组、32最近邻PCA，在opacity>.2、最小/次小协方差特征值<.2、邻域半径≤3倍来源最大scale条件下，静态1740/54244（3.2077%）、动态3265/45583（7.1628%）通过可靠平面筛选。来源轴与其PCA轴平均15.1489°/16.4298°，B02轴与同一来源局部轴15.6124°/20.4865°；这是来源自洽诊断，不是GT MAE，邻域含低opacity点、体积和边界会影响PCA，不能直接强制整个人体用这些PCA方向。未导出训练目标、0 optimizer steps、源SHA保持，B03未改。证据source_local_geometry.json与可复现commands/audit_source_local_geometry.py。

来源静态opacity>.05点19416，其中距来源RANSAC平面≤.08m15385、≤.12m16881、≤.2m17768；距平面P90=.14797m，当前support14556点还应用轴方向筛选。高opacity>.5点12603，≤.08m9786、≤.2m11589。当前几何信赖域.08m及support band.08不能校正较远的来源片；不能盲目只增已支持点的平面权重。此为下一步候选诊断，需结合B03完整真实MAE/质量后决定是否扩大源预测支持及有界位移；不读取GT几何定义支持，不删除Gaussian，不提前声称7°。证据source_plane_support_distribution.json。

## B03实际40k三训练帧梯度分解诊断

对0001/0075/0149真实40k权重，原生RGB+SH目标与lambda=.05的GS-Depth一致性梯度分别计算；所有梯度有限、full一致性梯度=GS侧+Depth侧（max差<1e-8）、源权重fingerprint/SHA不变，实际exit0、0 optimizer steps，未用GT材质/法线/深度或测试RGB。

rotation梯度L2：RGB+SH .10920/.69956/.12194，一致性full .00464/.01153/.00505（约1.65～4.25%）；GS侧 .00421/.01054/.00438，大于Depth侧 .00217/.00496/.00221。前两帧与RGB梯度cosine为-.10677/-.23217，最后帧+.09898。xyz同样RGB目标量级大于一致性。说明GS侧确实可受预测深度影响，但当前RGB目标整体主导，不能断言一致性项主导剩余噪声，也不能凭梯度冲突证明其GT准确性差。若后续尝试位置侧一致性，仍需完整训练/MAE验证。

首次独立脚本在最后校验误用verify_unchanged方法导致exit1，已保留失败script/log；改用真实FrozenScene.verify_frozen后重跑三帧正常exit0。当前训练未改、不重启。证据inspection/normal_depth_gradients.json、commands/audit_normal_depth_gradients.py、logs/audit_normal_depth_gradients_retry.log。

## B03 50k真实检查点

{"status": "PASS_CHECKPOINT_ONLY", "iteration": 50000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.3296035474704572, "max_displacement": 0.06572393327951431, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.202476978302002, "final_mae": false, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32}

材质DC真实保持[.031,.969]，native硬clamp比例仍0（约32.96%通道在投影边界，但位于native硬clamp内、仍可导）；同Gaussian全集、opacity/scale/feature冻结，XYZ与旋转有限且在信赖域内。MLP全部state tensor与35k源逐项完全相同；不能据此说不同XYZ下输出不变。光强6.20247698仅训练RGB拟合；不是GT能量。最终55k七项MAE/四类图像质量尚待完成。

## B03完整20k训练结束，后处理开始

启动03:10:06.570、55k保存及Training complete 03:54:31 JST；全部20000步，loop44:07，train exec实际exit0，真实55k PLY/PTH/lighting marker已保存。后处理绑定原PID结束后正常开始render，实际childPID226468。training_runtime原开始记录已保留，补充真实log完成时间/loop/exit0；完整四类、重光照、七项MAE及人工前后景质量仍待完成，不提前commit/push或声称7°。

## B03最终55k RGB实际评测/审閱（全量后处理继续）

{"psnr": 19.35354474812191, "ssim": 0.8932221968968709, "lpips": 0.11383797228336334, "psnr_foreground": 12.234762806258436, "ssim_foreground_bbox": 0.7692404826482137, "lpips_foreground_bbox": 0.22925928036371868}

已实际查看0010/0080/0150对照GT/B02同55k：背景正常，背面平台棋盘纹理有恢复；人物斑驳、头手边界稀疏及脚周伪结构仍明显，正面平台多白。PSNR19.35354 vs B02 18.91229略改善；最终RGB FAILED_SO_FAR，不能提交，完整四类与七项MAE尚待完成。不把GT背光本身较暗单独判失败。证据inspection/visual_review_early.json与rgb_final_55000_comparison.png。

## B03最终四类实际首中尾审閱

已查看RGB/Alpha/Normal/Albedo 0001/0075/0150，图inspection/four_categories_final_55000.png：Alpha主体平台基本完整，头手毛糙、平台边及下方薄雾/残影；Normal平台总体集中但脚周大块异常、人体斑驳；Albedo平台棋盘较B02恢复，但正背面光照残留、头手与脚周污染明显。四类质量FAILED，commit_allowed=false；材质投影确实持续生效仍不能修复来源几何与覆盖残影，完整MAE/HDR视频审计继续，不预填法线结果。仅首中尾实际人工审核，不声称150逐张。

### B03全量结果（2026-10-10T04:08:24.938182+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15765579 | linear RGB | 96.537645% |
| Roughness MAE | 0.20939310 | r | 96.811983% |
| GS Render Normal MAE | 18.89548239 | degree | 93.718121% |
| Depth Normal MAE | 27.50692138 | degree | 89.098844% |
| Mesh Normal MAE | 22.19361409 | degree | 94.091947% |
| Expected Z Depth MAE | 0.08631321 | m | 93.718121% |
| Median Z Depth MAE | 0.09000833 | m | 93.718121% |

RGB：{"psnr": 19.35354474812191, "ssim": 0.8932221968968709, "lpips": 0.11383797228336334, "psnr_foreground": 12.234762806258436, "ssim_foreground_bbox": 0.7692404826482137, "lpips_foreground_bbox": 0.22925928036371868}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-07文档。

## B03完整最终审核：FAILED

全部20k训练及后处理exit0；七项全15帧MAE/独立复算/83来源SHA/全量渲染和6视频解码PASS。Albedo.15765579、Roughness.20939310、GS18.89548239°覆盖93.718121%、Depth27.50692138°、Mesh22.19361409°、ExpectedZ.08631321m、MedianZ.09000833m；GS共同像素17.99274139°覆盖88.899293%、缺失180惩罚29.01587388°。高于7°；全部Gaussian/15帧保留。

实际四类/三法线深度/HDR转灯首中尾质量FAILED：主体/背景功能正常，人物和头手斑驳/缺失、脚周平台伪起伏、Albedo照明残留、Mesh缺面。材质投影改善RGB/Albedo/roughness，但不能解决来源几何。commit_allowed=false，未提交或建目标2分支。完整训练/报告1010-07和archive同步；开始下一处几何/形变管线审计后再定修复，不能仅换指标或删Gaussian。

## B03后定位：移动XYZ的冻结MLP输入耦合



B03动态高Alpha点canonical位置更新中位.00344391m、P90 .00766179m、P99 .01308086m，最大.02846621m；仅改变冻结MLP的输入，额外形变位移中位.00715～.00877m、P90 .02663～.03126m、P99 .08884～.10446m、最大.35186～.51564m；额外法线轴变化P99约8.50～10.68°、少数最大53.27～71.62°。B02也出现同类放大，完整静态/动态分布见audit.json。

NativeTrainer当前no_grad查询实际更新后pc.xyz；几何梯度不包括MLP输入位置的Jacobian，且canonical .08m约束不约束这部分额外形变。权重固定不能等同来源形变固定。此证据支持后续独立Anchor preset：仅将MLP输入锚定至真实复制35k XYZ，原生渲染仍使用优化后的真实XYZ/rotation与来源MLP输出；训练、TensorBoard测试、离线渲染/全部评测必须一致并校验资产。没有借GT定义人体或平面先验。

诊断不证明Anchor会提高GT精度，不宣称质量/7°通过。先实现接入与验证，再独立完整20k Stage2和全15帧七项MAE；当前尚未修改管线或开始Anchor训练。


实际审计output/smoke_test/1010-08-dir-PBR移动XYZ形变审计，完整命令/哈希/分布/exit0见README/audit.json/log。Source和候选文件SHA不变，0 optimizer steps。下一步实现独立pbr_gt_directional_anchor，保存真实35k XYZ锚定资产并在训练、周期测试、所有离线渲染/MAE一致读取；先保持B03其余参数，隔离修复效果，不同时变更评分或删除Gaussian。未实现、无新MAE、未提交/建目标分支。

## B04 Anchor实现与正式训练准备

新增pbr_gt_directional_anchor，仅将冻结来源MLP输入XYZ锚定至复制35k预测XYZ，实际渲染仍优化后的XYZ/rotation；训练/周期报告/FrozenScene全评测一致，资产SHA和marker绑定。保留B03其余参数、全99827Gaussian及评分协议。163CPU测试PASS（首次2旧mock错误日志保留并修复兼容），真实三训练帧GPU/周期报告形变一致、实际位置仍移动，PBR反向有限非零，0 optimizer steps，源SHA不变。首次GPU检查BVH缺参失败保留，独立retry exit0。功能PASS不代表最终质量/MAE。

正式output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定已准备完整20000步35k→55k、9步后处理及完整文档；尚未启动，无新MAE/7°。修改文档1010-06、训练文档1010-09；未提交/push或建目标分支。

## B04实际启动与配置核验

## 正式启动记录

{"status": "TRAINING", "started_at": "2026-10-10T04:30:08.950000+09:00", "training_pid": 232645, "process_start_ticks": "8458195", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_anchor", "session_id": 16086}

实际配置/150帧RGBA来源/99827点源XYZ锚定SHA/材质初始化核验PASS，证据inspection/launch_configuration.json。实际train session16086；后处理session55026绑定原PID232645，40k预览session25518，均持续等待/执行。完整训练和MAE尚未完成，不提前提交/push或声称7°。

## B04配置隔离与实时等待核验

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T04:36:08.891390+09:00", "steps": 2743, "total_steps": 20000, "iteration": 37743, "training_pid": 232645, "process_start_ticks": "8458195", "checkpoint_40000": false, "postprocess": "WAITING_LIVE_TRAIN"}

实际B03/B04 stage2_config逐字段对照：只改变loss_preset、pbr_deformation_anchor及各自source/model/light快照路径；Stage1来源相同、灯表字节完全相同、其它hyperparameters完全相同，启动源码manifest仍匹配。证据inspection/configuration_isolation.json。已核验/proc真实PID+启动ticks+cmdline，不根据状态文件误判停止或重启；完整训练/MAE仍在进行。

## B04 40k真实检查点与RGB预览

RGB全15帧：{"psnr": 18.517455841244086, "ssim": 0.8864702026049296, "lpips": 0.10901500582695008, "psnr_foreground": 11.48589137773118, "ssim_foreground_bbox": 0.7530547539393108, "lpips_foreground_bbox": 0.22960177361965178}

检查点：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 40000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "max_displacement": 0.02973010018467903, "max_rotation_deg": 60.00002670288086, "actual_light_scale": 5.28775691986084, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

实际查看0010/0080/0150对照GT/B03同40k：黑背景正常、人体平台存在；人物/衣服/头手斑驳稀疏、脚周平台伪结构和残影仍明显，与B03同阶段较接近，FAILED_SO_FAR。预览正常exit0。99827全保留、源MLP32tensor完全未改、anchor逐元素来源XYZ且marker绑定正确，实际XYZ/rotation更新有界、材质硬clamp比例0；功能PASS不等同最终质量/7°。继续完整20k、最终55k所有MAE和四类/HDR审核，不以40k代替最终结果、未commit/push。证据inspection/preview_40000_review.json、checkpoint_40000_audit.json和preview_40000_rgb_comparison.png。

## B04运行中135训练时刻来源平面覆盖诊断

只读真实35k、全部135训练time查询MLP，源SHA不变、0优化步、无GT输入、未生成训练先验。本轮训练参数和源码不改。

当前14556平面支持点的坐标运动box-span范数P90=.00699977m、P99=.22801237m、max=.68234813m；全部135时刻的最远平面距离P99=.07980215m、max=.23887041m。此前1/75/149三时刻诊断P99约.0193m未覆盖所有动作，不能称完整时间稳定；当前静态feature阈值可能误含运动点，应在后续修复候选中核验时间稳定性。

来源静态距plane≤.08m共有42785点；低Alpha≤.05点27400，opacity总和212.6719、未投影Alpha×scale0×scale1面积代理.0327114，而当前支持代理4.591814，约.71%。这只是面积代理不等同光栅贡献，不能仅依据大量低透明度点数认定它们主导残影。

来源静态距plane≤.2m但当前未支持32800点，opacity总和2643.1616、面积代理1.679724，可能有不可忽略的较远预测片。静态near.2且135time运动box-span≤.03m共45547点，面积代理6.007738；动态feature组near.2且同稳定条件0点。不能据此不加筛选地将所有近plane点投影：平台侧面/人体脚可能存在，仍需源方向与时间可信度及完整MAE验证。

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/audit_source_plane_coverage.py
```

实际exit0，日志logs/audit_source_plane_coverage.log，完整各band/Alpha/静动态/时间范围统计inspection/source_plane_coverage.json。不是最终MAE，不改评分、不删除Gaussian、不提前宣称7°。

## B04只读训练损失图

基于两轮实际TensorBoard，当前B04前缀和B03完整训练同坐标对照：inspection/training_losses_prefix_steps.png。RGB L1图采用尾部201训练步均值，其余原始scalar，每个点原值/iteration/wall_time/run/tag保留training_losses_prefix_steps.csv；生成信息training_losses_prefix_steps.json。图中x轴为Stage2训练步数，原生Adam每4步实际更新，不能称每步optimizer update。首次图training_losses_prefix.png的轴标签updates不准确，版本保留，正式引用更正为steps的版本。

RGB L1当前略低于B03同阶段，光强曲线接近；支持点canonical平面轴一致性降至约.001量级，而预测GS-depth一致性仍随训练相机波动。不能把内部source plane loss当全前景GT法线MAE，也不能由其下降直接声称7°。不读取GT、未改变训练；日志logs/plot_training_losses_steps.log正常exit0。

完整命令：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/plot_training_losses_steps.py
```

## B04 50k真实检查点

{"status": "PASS_CHECKPOINT_ONLY", "iteration": 50000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.34154086569765696, "max_displacement": 0.06370601058006287, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.186013221740723, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

全部99827 Gaussian/源32tensor保留不变、opacity/scale/feature冻结、sourceXYZ锚定与当前marker同SHA；真实几何更新有界、材质硬clamp比例0，34.15%通道位于内部投影边界，仍可导。光强6.18601322仅训练RGB拟合，不是GT能量。最终55k全15帧七项MAE/四类质量待完成，不能由有限性检查称质量或7°PASS。

命令：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/audit_checkpoint.py 50000
```

日志logs/audit_checkpoint_50000.log，实际exit0。

## B04完整20k训练结束，后处理开始

{"status": "TRAIN_COMPLETED", "started_at": "2026-10-10T04:30:08.950000+09:00", "training_pid": 232645, "process_start_ticks": "8458195", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_anchor", "session_id": 16086, "finished_at": "2026-10-10T05:14:31.955921+09:00", "iteration": 55000, "steps_completed": 20000, "log_completed_at": "2026-10-10T05:14:28+09:00", "loop_duration": "44:01", "training_exit_code": 0, "verified_training_session_closed": true}

启动04:30:08.950、55k保存及Training complete 05:14:28 JST；全部20000步、loop44:01，训练exec session16086实际exit0并关闭，55k PLY/PTH/灯光marker已保存。真实后处理childPID239457正在全量render，已核验/proc cmdline，不是仅依据状态文件。完整四类/HDR/动态Mesh/七项MAE和最终质量仍待完成，不提前提交或宣称7°。

## B04最终55k RGB与四类首中尾实际审阅

{"psnr": 19.475974080557172, "ssim": 0.8950952370961507, "lpips": 0.11134533832470576, "psnr_foreground": 12.349587008267934, "ssim_foreground_bbox": 0.7730405648549398, "lpips_foreground_bbox": 0.22556244830290476}

已实际查看RGB0010/0080/0150对照GT/B03同55k和RGB/Alpha/Normal/Albedo0001/0075/0150。黑背景正常，人物平台存在；人物/头手/衣服斑驳稀疏，脚周平台伪结构和残影仍明显，棋盘纹理部分恢复。Alpha主体平台基本完整但边缘/下方薄雾；Normal平台集中但脚周和人体异常斑驳；Albedo人体前后照明残留与脚周污染。当前四类质量FAILED，commit_allowed=false。GT背光本来较暗，不单凭背面暗判失败。仅首中尾实际人工看，不声称150逐张。

PSNR19.47597 vs B03 19.35354、LPIPS.11134534 vs .11383797，略改善不能当7°通过。完整15帧七项MAE/HDR/转灯/三法线与深度误差图待完成；保留本轮全权重与后处理日志。证据inspection/visual_review_early.json、rgb_final_55000_comparison.png、four_categories_final_55000.png。

### B04全量结果（2026-10-10T05:28:00.297574+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15600631 | linear RGB | 96.642930% |
| Roughness MAE | 0.21155915 | r | 96.913295% |
| GS Render Normal MAE | 18.91234950 | degree | 93.686587% |
| Depth Normal MAE | 27.51005704 | degree | 89.073328% |
| Mesh Normal MAE | 22.23676747 | degree | 94.081356% |
| Expected Z Depth MAE | 0.08637919 | m | 93.686587% |
| Median Z Depth MAE | 0.08998650 | m | 93.686587% |

RGB：{"psnr": 19.475974080557172, "ssim": 0.8950952370961507, "lpips": 0.11134533832470576, "psnr_foreground": 12.349587008267934, "ssim_foreground_bbox": 0.7730405648549398, "lpips_foreground_bbox": 0.22556244830290476}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-09文档。

## B04完整最终审核

## 最终审核结论：FAILED

完整20000步及9步后处理均实际exit0。150×4类/72转灯/150 HDR2048及6视频全部解码、全部15测试帧七项MAE、独立复算和83来源SHA：PASS。150原始RGBA来源、99827点全保留、源MLP32tensor不变、anchor输入逐元素源XYZ/marker绑定：PASS。

实际RGB/Alpha/Normal/Albedo及HDR/转灯和0010/0080/0150法线/深度误差图已看。背景功能正常；人体/头手/衣服斑驳、边缘缺失、平台脚周伪起伏/残影、Albedo光照残留、Depth噪声/Mesh局部缺面仍在。最终图像质量FAILED，不能提交。仅首中尾实际人工看，不声称150逐张。

原GS Render Normal pixel weighted MAE18.91234950°（覆盖93.686587%）；Depth27.51005704°、Mesh22.23676747°；共同像素369183/415433（88.867038%）GS17.99532404°/Depth27.46234473°/Mesh21.26058146°；GS缺失180惩罚29.08247777°。没有删除Gaussian/帧/困难像素，没有改评分。7°目标FAILED。

与B03同55k：RGB PSNR19.47597 vs19.35354，Albedo .15600631 vs .15765579略改善；GS18.91235 vs18.89548未改善。Anchor确实消除更新XYZ输入冻结MLP的耦合，但不能据此称几何质量得到修复。根因审计继续关注来源平面支持的全时间可信度与未覆盖较远表面片；后续只用来源预测/原RGBA/GT方向构造修复，不以GT材质或几何作为先验。

commit_allowed=false；尚未commit/push、未建立目标2分支。全部失败权重、渲染、浮点数组、统计、日志保留。

## B05候选：来源时间稳定平面几何投影

已在修改前阅读最新B04失败及135time覆盖诊断。单纯canonical source plane loss未保证actual world平面，故新增独立pbr_gt_directional_plane Preset，继承B04全部参数，仅增加source-only实际world XYZ/quaternion平面投影。135训练time稳定性≤.01m、全部time/canonical距plane≤.2m及axis abs cos≥.5、staticfeature/opacity筛选；真实支持15187/99827点，全部点保留。plane只从35k预测RANSAC/SVD拟合，不取GT几何/法线。投影真实几何同时作用RGB/PBR/BVH/法线/深度/mesh及重光照，不改评分、不滤难像素。Canonical边界.08m/60°不能宣称等于额外投影后的effective world边界，假设可能误平坦细节，需全15帧验收。

173项CPU实际exit0；真实GPU source full135重算、train/FrozenScene/周期报告一致、原生PBR/BVH/GGX有限非零梯度、源SHA不变，0优化步PASS。证据smoke_test/1010-10-dir-PBR来源平面投影验证；不能当最终质量或MAE。

B05全20k计划与命令/代码快照已准备，output/1010-10-jumpingjacks_dir-PBR来源平面几何投影；Stage1未重跑。完整训练/七MAE/四类/HDR/转灯审核PENDING。未commit/push/新建目标分支。

## 正式启动记录

{"status": "TRAINING", "training_pid": 245234, "process_start_ticks": "8927314", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_plane", "session_id": 42059, "observed_at": "2026-10-10T05:48:32.269187+09:00", "started_at": "2026-10-10T05:48:20.140000+09:00", "start_time_source": "proc_btime_plus_start_ticks"}

真实train session42059/PID245234/start ticks8927314；后处理session86603/PID245271绑定真实train；40k预览session2961同PID绑定，禁止退出后盲目重启。实际配置相比B04改变Plane Preset/projection、新增阈值.2/.01字段及本轮路径；其它既有参数相同，dir表字节相同。首次launch audit只列5个允许字段，漏列新阈值2项，断言失败；未修改训练，修正审计字段后PASS。150原RGBA来源、99827点全保留、source资产/cameraSHA/15187支持/代码manifest检查PASS，证据inspection/launch_configuration.json。最终MAE/质量待完成，未commit/push。

## 已核验实时进度

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T05:51:27.024252+09:00", "steps": 1395, "total_steps": 20000, "iteration": 36395, "training_pid": 245234, "process_start_ticks": "8927314", "postprocess": "WAITING_LIVE_TRAIN", "checkpoint_40000": false, "source_manifest_matches": true}

完整训练与七项MAE尚未完成，不提前提交/push，未标记目标完成。

## 2026-10-10 几何运行时遗漏更正（待完整重评）

后续源码审计确认旧scripts/eval_geometry_gt.py的GeometryRun仍直接用当前XYZ查询sourceMLP，未消费Anchor资产；逐time Mesh也使用这个入口。先前声明所有几何评测与训练一致不成立。旧GS18.91234950°/Depth27.51005704°/Mesh22.23676747°及2depth/common/missing180虽CPU复算自洽，但不是实际B04 Anchor模型结果，现标记SUPERSEDED。保留原权重/浮点图/原分数/日志，不用旧几何分数评价Anchor效果。RGB/Albedo/Roughness及原生四类渲染已用FrozenScene，受该遗漏影响的仅GeometryRun评测与Mesh；已有RGB四类质量FAILED不因此变PASS。新共享FrozenScene入口已改，179CPU PASS；真实GPU逐buffer验证和全15帧重新Mesh/几何评测待完成。B05训练运行中的source/renderer/参数不改，仅离线入口改并保存launch旧代码及hash记录，不重启训练。

## GeometryRun实际入口审计修复与B04重评

已修复scripts/eval_geometry_gt.py为开启Anchor/Plane时使用共享FrozenScene的真实pc/deformation，默认旧入口保持；六项新CPU回归，全部179 CPU实际exit0。真实B04 55k及B05源35k GPU逐像素alpha/worldnormal/Expected Z与原生PBR缓冲首中尾max error均0，targettime override一致，源hash/fingerprint不变；0步/noGT标签，不作为最终MAE。此前旧GeometryRun未共享，不能用旧数组复算证明Anchor实际几何正确，旧B04 geometry MAE标记SUPERSEDED，不能断言Anchor无改善，实际结果待更正。

B04完整15time×135train Mesh和15帧几何重评已实际启动exec session28509，output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评，复用已正确的RGB/albedo/roughness同55k产物并再次独立复算/hash，不重训Stage1/Stage2。B05真实训练PID245234继续；只改未执行离线入口并记录launch旧快照/new SHA，未改运行中train/renderer/MLP/参数。详细1010-08修改文档。最终7°、质量/提交仍待完成。

## B04完整几何更正最终结果

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15600631 | linear RGB | 96.642930% |
| Roughness MAE | 0.21155915 | r | 96.913295% |
| GS Render Normal MAE | 18.77743450 | degree | 93.850031% |
| Depth Normal MAE | 27.21989900 | degree | 89.219922% |
| Mesh Normal MAE | 22.15133832 | degree | 94.229635% |
| Expected Z Depth MAE | 0.08569735 | m | 93.850031% |
| Median Z Depth MAE | 0.08932931 | m | 93.850031% |

共同有效像素369804/415433，覆盖89.016520%；GS缺失180惩罚28.69257202°。

| 法线 | 共同像素MAE（度） |
| --- | ---: |
| gs_render | 17.86533346 |
| depth_normal | 27.16722766 |
| mesh | 21.16834873 |

旧B04几何分数SUPERSEDED且保留，更正全15time×135train Mesh/全15 geometry/独立复算及90来源SHA/rough41来源/missing180实际PASS；GPU原生buffer逐像素一致PASS。实际看0010/0080/0150更正图：平台脚周伪起伏、身体法线斑驳、Depth噪声、Mesh局部手/腿/头缺面，0080更明显；质量/7°FAILED。B04 actual18.77743 vsB03 18.89548小幅改善，不能再以旧18.91235声称Anchor未改善。RGB/albedo/rough原运行时正确不改，本次0训练步；B05继续最终20k，未commit/push/新建分支。

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

### B05全量结果（2026-10-10T06:47:48.996001+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.15432221 | linear RGB | 96.660979% |
| Roughness MAE | 0.21168068 | r | 96.935993% |
| GS Render Normal MAE | 17.73081984 | degree | 93.884935% |
| Depth Normal MAE | 26.21282757 | degree | 89.256029% |
| Mesh Normal MAE | 20.41400002 | degree | 94.334345% |
| Expected Z Depth MAE | 0.08031497 | m | 93.884935% |
| Median Z Depth MAE | 0.08444845 | m | 93.884935% |

RGB：{"psnr": 19.375063141094465, "ssim": 0.8948765238126118, "lpips": 0.11125242908795675, "psnr_foreground": 12.246959743787187, "ssim_foreground_bbox": 0.7720984180768331, "lpips_foreground_bbox": 0.22454677522182465}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-10文档。

## B05最终完整验收（2026-10-10T06:50:04.859858+09:00）

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

## B06新修改：显式仓库原生CUDA，Native Preset

已读取GOAL及B05全量结果：GS17.73081984°，较B04更正值改善1.04661466°，但RGB/四类/HDR质量FAILED和7°FAILED，全部177归档文件SHA三处同步核验通过。新修改只修复运行时实际CUDA来源/法线朝向，不同时关闭SH间接光或改ray预算。

## 实际证据与修改

环境installed diff_surfel_rasterization0.0.1的Git来源hbb1 f7e2b68a，以normal.view_z<0确定方向；本仓库LumiMotion原始CUDA以-dot(n_view,p_view)>0确定方向，forward/backward一致。世界plane与相机已证明0038/0075/0076在plane上方但旧包朝下，均是TRAIN帧；不是某个GT normal错误或评分mask问题。独立原生六相机0步GPU核验全部朝向正确，合成normal梯度finite difference最大误差.00035429；不根据旧权重替换渲染宣称新训练质量。

scene/pbr_rasterizer.py独立按CUDA/GLM来源hash构建namespace，原LumiMotion kernel不改、不pip install；GLM优先native嵌套目录，缺失时用仓库stable目录的已追踪同库头文件，缺失则拒绝构建。不会静默回退installed同名包。pbr_rasterizer.json保存每文件SHA/综合SHA、.so SHA、torch/CUDA版本；checkpoint绑定综合SHA，加载和各评测核验，不允许不同native源码被当同checkpoint。旧Preset与Stage1/env默认仍installed；新pbr_gt_directional_native和pbr_gt_directional_plane_native显式pbr_rasterizer_backend=lumimotion。仅支持scale/rotation CUDA路径，compute_cov3D_python=True拒绝。

训练render_ir、周期测试、FrozenScene、全四类/方向光/HDR、GeometryRun/逐time Mesh、原始材质albedo/roughness helper全部同后端；原有评分/GT域/coverage/common/missing180不改，不把最终混合normal后翻转。eval source包含kernel与GLM每文件SHA。render provenance保存本次真实构建记录。

## 验证与限制

187 CPU最终discover PASS，真实20步35000→35020 Trainer保存加载marker；三TRAIN帧PBR/GGX/BVH对材质与真实geometry梯度非零finite；source权重保持。GeometryRun三个test frame alpha/normal/Z逐buffer0、Mesh timeoverride一致，453来源SHA通过。材质helper albedo0、roughness黑0/白5.96046e−8。四类150与HDR16/方向光三帧已渲染且首中尾实际看：背景/主体存在、0075平台定向修复，人物/脚周伪影仍有，只FUNCTIONAL PASS；不代替完整20000步/15七MAE/7°验收。

初次独立build缺GLM头文件失败、首次CPU指定两个不存在测试名失败，日志均保留；补本地GLM快照retry/实际discover通过。详见两个smoke README及命令/log/inspection。B05所有权重/指标仍对应旧实际kernel，不覆盖、不改分数定义。

## 完整训练计划

B06 output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复：固定35k→55k全20000步，仅Stage2，B05所有参数一致只换Native backend/preset，从原始来源独立训练。完整all与15测试帧七MAE及独立复算/每time135train Mesh、四类/HDR目检全部必须完成。原RGB/自身RGBA/相机/time/source预测/GT direction-only；禁止GT材质/normal/geometry/GT强度与颜色用于训练。最终质量/7°PENDING，未commit/push/新分支。

完整命令/source/model/iteration/log/产物及验收计划见LIHAN/DOC/训练/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复训练报告.md和输出README/planned_commands.json。修改仅当前1001-stage2-perlight代码，LIHAN文档/index三处同步。

## 正式启动：真实进程

{"status": "TRAINING_VERIFIED_LIVE", "training_pid": 258688, "process_start_ticks": "9357639", "session_id": 30746, "source_iteration": 35000, "target_iteration": 55000, "steps_planned": 20000, "loss_preset": "pbr_gt_directional_plane_native", "started_at": "2026-10-10T07:00:03.390000+09:00", "start_time_source": "proc_btime_plus_start_ticks", "observed_at": "2026-10-10T07:00:34.438075+09:00", "argv": ["/home/lihan/miniconda3/envs/lumimotion/bin/python", "-u", "-m", "scripts.train_stage2", "--source_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/data_view", "--stage1_model_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp", "--model_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model", "--load_iter", "35000", "--iterations", "55000", "--save_iterations", "40000", "50000", "55000", "--test_iterations", "40000", "55000", "--render_mode", "original_ir", "--loss_preset", "pbr_gt_directional_plane_native", "--pbr_gt_lights_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/gt_directions_only.json", "--pbr_light_intensity_scale", "3.141592653589793", "--depth_ratio", "0", "--resolution", "2", "--is_blender", "--eval", "--load2gpu_on_the_fly", "--gt_alpha_mask_as_scene_mask", "--train_light_folder", "images", "--test_light_folder", "images", "--deform-type", "mlp", "--sh_degree", "3", "--hyper_dim", "1"]}

仅新Stage2，未重跑Stage1；完整MAE/最终质量尚无，不把source35000或20步smoke作本轮结果。

## B06启动配置/后处理核验

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T07:03:19.241733+09:00", "steps": 1470, "total_steps": 20000, "iteration": 36470, "source_cuda_sha256": "82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0", "process_handles": {"train": {"pid": 258688, "session_id": 30746, "start_ticks": "9357639"}, "postprocess": {"pid": 258864, "session_id": 47576, "start_ticks": "9362140"}}, "postprocess": "WAITING_LIVE_TRAIN", "final_mae": false}

正式配置仅loss_preset、backend和3本轮路径字段与B05不同，所有其它值一致；15187支持/所有来源plane数组逐元素与B05相同。源码manifest逐SHA核验PASS，实际仓库native CUDA综合SHA82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0绑定。首次inline审计误用默认vllm python缺smplx失败，未改训练；正确venus lumimotion执行归档commands/audit_launch.py实际exit0。真实后处理绑定同train PID/start ticks，完整9步将在train完成后执行，40k预览同PID绑定，不退出后盲重启。完整MAE/质量仍PENDING，未commit/push/新分支。

## B06实际进度与进程绑定更新

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T07:04:56.276364+09:00", "steps": 2261, "total_steps": 20000, "iteration": 37261, "process_handles": {"train": {"pid": 258688, "session_id": 30746, "start_ticks": "9357639"}, "postprocess": {"pid": 258864, "session_id": 47576, "start_ticks": "9362140"}, "preview_40000": {"pid": 258868, "session_id": 25842, "start_ticks": "9362256"}}, "postprocess": "WAITING_LIVE_TRAIN", "final_mae": false, "commit_allowed": false}

TRAIN/POST/40k-preview实际PID/start ticks/cwd全部复核，40k预览cmdline为相对脚本路径，按resolve核验，不因绝对路径子串不匹配盲重启。核心469源码/GLM/快照SHA仍匹配launch。仅修正文档生成器末尾遗留1010-10引用为本轮1010-12，保留修正前脚本；运行中训练/renderer/MLP/参数未改。

## B06 40k真实检查点、全15RGB预览与只读gate诊断（2026-10-10T07:15:26.674133+09:00）

完整训练仍运行：6962/20000，实际iteration41962；train258688/startticks9357639与post258864/startticks9362140/cwd重新核验，核心469来源/GLM源码SHA匹配launch，未改变本轮训练参数或源码。preview exec25842、checkpoint审计exec1420、gate诊断exec90456均实际exit0并关闭。

检查点：全部99827 Gaussian保留，opacity/scale/feature与source32NNtensor冻结不变，XYZ最大位移0.03539922m、canonical旋转最大60.00002670°，DC RGB范围[.031,.969]、硬clamp0，训练RGB拟合光强5.28000259。native CUDA marker来源与anchor/plane绑定通过；这些只是保存完整性，不是法线7°或质量PASS。

15 TEST帧40k RGB：PSNR18.51417174、SSIM0.88792369、LPIPS0.10788124、前景PSNR11.46161042、前景bbox SSIM0.75713942/LPIPS0.22566057。B05同40k PSNR18.39326180/LPIPS0.10931317，略改善，不能由此声称GS法线改善。

已实际view inspection/preview_40000_comparison.png中GT/B05/B06的0010/0080/0150：黑背景正常、人物和平台存在，但头手细节欠完整、衣服斑驳、脚周平台伪凸起/灰黑残影仍明显，两个预测总体接近；0080 GT本身暗，不以暗单独判失败。中间质量FAILED_SO_FAR，55k完整七MAE/四类/HDR仍PENDING，未commit/push/新建分支。

只读gate诊断：source NN隐藏层与32参数固定，只开放当前feature logit的输出乘法gate自动求导；当前位置/旋转形变与原FrozenScene路径逐bit相等。0001/0075/0149三TRAIN帧PBR/GGX/BVH的RGB+SH、normal-depth和自身输入Alpha梯度均finite且存在非零梯度，source SHA/状态fingerprint不变；0optimizer、feature/NN值不变、未读GT标签/测试RGB、未生成新训练prior。本轮仍冻结gate。来源预测筛选的6836动态plane候选不是GT物体标签，梯度方向不证明它们全部属于平台。

| TRAIN帧 | RGB+SH gate梯度L2 | 非零点 | 来源动态plane候选RGB+SH L2 | 候选非零点 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.0005630466 | 34817 | 0.00024462515 | 4672 |
| 0075 | 0.01247051 | 36220 | 0.010364828 | 5064 |
| 0149 | 0.002293421 | 34628 | 8.5120133e-05 | 4629 |


该机制可作为本轮最终失败后的新候选，仅证明可导性，未实施新训练/未声称能到7°。不能从0075的大梯度推定正确几何或把候选全部静态化；需要重新读取本GOAL、独立完整训练与全量MAE验证。

完整诊断命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_feature_gate_gradients.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 40000
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_preview_comparison.py
```

证据：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/inspection/feature_gate_gradients.json、checkpoint_40000_audit.json、preview_40000_review.json、preview_40000_comparison.png；logs/audit_feature_gate_gradients.log、checkpoint_40000_audit.log、preview_40000.log、preview_render_40000.log。预览完整命令另在commands/preview_40000_argv.json；所有失败轮保持原权重与统计，最终MAE尚无。

## B06原生40k联合几何贡献审计（2026-10-10T07:19:17.158800+09:00）

完整训练继续：8684/20000。0optimizer、无GT材质/normal/geometry、未改本轮源码/参数。只读实际40k六帧0001/0038/0075/0076/0149/0150的仓库native CUDA光栅，三原坏TRAIN视角0038/0075/0076的projected支持组都朝向camera所在的plane上方，所有signed dot最小≥.99999988。PBR的alpha/worldnormal/Expected Z与原始native光栅逐buffer maxerror0，当前NN和权重fingerprint不变、469核心/GLM SHA保持launch。

联合贡献核验：全部99827点每个pass均用真实原opacity/position/rotation，输入组one-hot颜色而非只渲单组，得到真实联合alpha合成权重；四组贡献总和与全alpha最大误差6.56e−7，重复pass几何buffer0。下表为全图alpha>.5内最大权重来源组的像素数量，不是GT物体分区、不是评测mask，不用于丢点/丢像素。

| 帧 | 投影plane主导像素 | 动态plane候选主导 | 其它动态主导 | 静态非投影主导 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 1990 | 4886 | 12057 | 1047 |
| 0038 | 4006 | 3861 | 8057 | 1150 |
| 0075 | 2610 | 2573 | 8774 | 2808 |
| 0076 | 2459 | 2688 | 8885 | 2742 |
| 0149 | 1893 | 5098 | 12314 | 954 |
| 0150 | 1922 | 4943 | 12090 | 1096 |


已实际view inspection/native_joint_geometry_comparison.png首中尾0001/0075/0150：投影组单独形成完整向上的平面，但联合图中平台被其它来源组大量遮盖；来源动态plane候选红组主要出现在平台，其它动态绿组也占据脚周/平台，人体主要绿组，衣服含白色静态非投影贡献。因此native符号修复真实生效，但它不会自动把未支持组的geometry变成平面。不能将source候选当真实平台标签或直接删除它们，不能以投影组单独的平整图片验收完整结果。

候选审计补充（尚未修改/启动）：gate只读梯度证明可通过原RGB/自身Alpha/normal-depth自洽调整时间形变幅度；若最终B06失败，可独立评估可导gate微调并绑定来源gate信赖域/先验，保持source NN与所有点、统一训练/评测入口及marker。当前Plane asset重算用source35k feature，不要求改变来源选择；需要明确当前可学习gate与固定source支持组语义。不能根据该分组图预设所有candidate必须静态、不得读取GT正常向量训练。其它候选actual SUN ray预算/Direct Preset仍未实施；需最终完整七MAE和读取GOAL后选择一个独立新实验。

第一次exec96086因诊断import误写get_rasterizer_types（真实API rasterizer_classes）exit1，未渲染/未改训练；原脚本commands/audit_native_joint_geometry_before_api_name_fix.py与logs/audit_native_joint_geometry.log保留。只修正诊断API后retry exec17809实际exit0并关闭，生成18张来源贡献/normal图和inspection/native_joint_geometry.json。实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_native_joint_geometry.py
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_joint_geometry_comparison.py
```

日志logs/audit_native_joint_geometry_retry.log；证据inspection/native_joint_geometry.json、native_joint_geometry_review.json、native_joint_geometry_comparison.png与native_joint_geometry_40000。只读机制PASS、最终质量/七MAE/7°仍PENDING，未commit/push/新建目标分支。

## B06 40k加性旋转实际响应的只读审计（2026-10-10T07:23:22.796399+09:00）

假设核验：原生world四元数为qcanonical+dMLP再归一化，canonical信赖域角度不直接等于每时刻最终轴变化。0optimizer、未改变Gaussian/NN、没有GT标签输入；五TRAIN帧0001/0038/0075/0076/0149，source XYZ/NN/gate保持，比较source q与真实40k q加同一NN增量后的几何轴。projected组最终另有硬plane投影，本诊断不把该中间轴当最终normal。

45583未投影动态点canonical旋转中位8.364336°、P9935.336845°，只有15点≥59.9°（.032907%）；未投影静态39057点只有10点到界。动态点MLP增量/source q norm中位.133–.189，source world q近抵消（norm<.05×source q norm）在各组/五帧均0。

| TRAIN帧 | MLP增量/source q norm中位 | 实际world轴变化中位 | P99 world轴变化 | source world q近抵消点 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.138803 | 6.017214° | 31.412746° | 0 |
| 0038 | 0.133201 | 6.150679° | 32.343266° | 0 |
| 0075 | 0.184334 | 5.866048° | 29.638847° | 0 |
| 0076 | 0.188972 | 5.840733° | 29.400106° | 0 |
| 0149 | 0.137795 | 6.028020° | 31.427202° | 0 |


因此不支持“动态几何普遍卡在60°上限”或“整个NN增量压倒canonical使旋转无法生效”的假设；当前确有真实world轴变化，尾部也有变化较大点。不能据此加大信赖域、改全套四元数组合后宣称质量改善，也不从轴变化大小推定GT normal准确。若B06最终失败，优先基于已观测的未投影平台贡献/RGB和gate可导性/实际SUN预算选择后续候选；本轮不改。

首次exec96264因诊断相机fid留在CPU而NN为CUDA exit1，日志logs/audit_rotation_response.log及commands/audit_rotation_response_before_camera_device_fix.py保留；只在诊断增加ctx.load_camera/unload_camera，retry exec33080实际exit0且关闭，frozen/source和469核心SHA通过。命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_rotation_response.py
```

证据inspection/rotation_response.json与logs/audit_rotation_response_retry.log，完整分位数覆盖0/1/50/90/99/100百分位；source q norm保持与来源一致。本轮55k完整七MAE/四类/HDR仍PENDING，未commit/push/新建分支。

## B06 50k真实保存后审计的等待句柄

2026-10-10T07:24:13.431998+09:00：新增只读50k检查点审计等待进程262910/startticks9501104/exec81677；/proc实际存活/cwd/命令已核验，绑定train258688/startticks9357639及完整argv，在真实50k marker出现后执行commands/audit_checkpoint.py 50000。此刻WAITING_LIVE_TRAIN_50000，不预先声称审计完成；训练结束前若没有50k则拒绝自动重启。日志logs/audit_50000_when_saved.log及logs/checkpoint_50000_audit.log（待实际保存后生成），状态inspection/audit_50000_wait_state.json。训练与最终9步后处理未改，最终质量/七MAE仍PENDING。

## B06 50k真实保存与独立检查点审计（2026-10-10T07:34:24.139755+09:00）

等待进程exec81677实际exit0并关闭，绑定train258688/startticks9357639，在真实50k marker后调用检查点审计，audit子进程也实际exit0。50k状态PASS_CHECKPOINT_ONLY：

```json
{
  "status": "PASS_CHECKPOINT_ONLY",
  "iteration": 50000,
  "gaussians": 99827,
  "all_finite": true,
  "opacity_scale_feature_frozen": true,
  "albedo_rgb_range": [
    0.030999988317489624,
    0.968999981880188
  ],
  "clamped_albedo_channel_fraction": 0.0,
  "channels_at_projected_bound_fraction": 0.33705644097622217,
  "max_displacement": 0.08000000566244125,
  "max_rotation_deg": 60.00005340576172,
  "actual_light_scale": 6.247264862060547,
  "deformation_state_tensors_unchanged": true,
  "deformation_tensor_count": 32,
  "deformation_input_exactly_source_xyz": true,
  "anchor_marker_bound": true,
  "source_files_unchanged": true,
  "final_mae": false
}
```

99827点/冻结opacity-scale-feature/32来源NNtensor原值和source SHA保持，source XYZ输入锚定正确，native source SHA/plane资产绑定通过，albedo可导DC硬clamp0。最大canonical位置0.08000000566m、旋转60.00005341°容差内；额外world plane投影不等于canonical信赖域。强度6.24726486由训练RGB拟合，不是GT强度。核心469文件与launch SHA仍一致。

完整训练仍真实存活：15477/20000，iteration50477；后处理真实PID258864仍等待train完成，不重启。本检查不是质量或七项MAE；最终55k全15帧七MAE、coverage/common/missing180及四类/HDR目检待完成，未commit/push/新分支。

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_50000_when_saved.py
# 该等待进程在50k实际保存后调用同环境命令：
/home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 50000
```

证据inspection/checkpoint_50000_audit.json、audit_50000_wait_state.json、model_mlp/pbr_directional/iteration_50000/lighting.json；日志logs/audit_50000_when_saved.log、checkpoint_50000_audit.log。全部来源与失败历史保留。

## B06完整20000步结束、55k实际退出与最终RGB早期目检（2026-10-10T07:46:40.050962+09:00）

完整Stage2源35000→55000共20000步，07:00:03.390000 JST启动，log保存55k与Training complete均07:44:25，循环44:02；train exec30746实际exit0并关闭，真实PID258688已退出，不是观察超时。training_runtime.json已记录verified_training_session_closed/actual exit0，Stage1未重训。

55k独立检查exec88063实际exit0：全部99827 Gaussian保留、数值有限、opacity/scale/feature及32来源NNtensor原值、sourceSHA保持。最大canonical XYZ位移0.08000001311m、旋转60.00006485°容差内；DC [.031,.969]硬clamp0，原RGB拟合E6.61184454。source XYZ anchor、plane和native CUDA来源绑定通过，核心469文件/GLM hash仍匹配launch。该检查不是最终GT normal或质量验收。

55k最终15 TEST RGB：PSNR19.68220260、SSIM0.89777244、LPIPS0.11070925、前景PSNR12.48856820、前景bbox SSIM0.77829250/LPIPS0.22170198。B05同55k PSNR19.37506314→本轮19.68220260（+.30713946dB）；LPIPS.11125243→.11070925，稍改善，不等于normal7°或画质PASS。

已实际view inspection/rgb_final_55000_comparison.png中GT/B05/B06首中尾0010/0080/0150：黑背景/主体存在，衣服、头手稀疏斑驳，脚周平台伪凸起/灰黑污染仍明显；两轮总体接近，GT0080本身背光暗，不凭暗单独失败。当前FAILED_SO_FAR_EARLY_FINAL_RGB_ONLY，完整四类/HDR/方向光、七MAE仍PENDING，未commit/push/新建目标分支。

真实后处理PID258864/startticks9362140正在全量render；child265977/startticks9623907、cwd与scripts.render_stage2 argv实际核验存活，使用--task all --load_iter55000 --hdr_samples2048。后续albedo、15time×135TRAIN Mesh、normal_depth、roughness、独立七MAE复算、产物审核、报告仍须完成，不用RGB指标代替全部七MAE。没有重启训练或更改已运行参数/源。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 55000
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_rgb_comparison.py
```

证据training_runtime.json、inspection/checkpoint_55000_audit.json、final_55000_early_review.json、rgb_final_55000_comparison.png、model_mlp/renders_stage2/ours_55000/eval/metrics.json；日志logs/train.log、checkpoint_55000_audit.log、render.log、postprocess.log。完整训练/渲染命令在planned_commands.json和commands/train.sh/render.sh。

## B06最终四类实际目检（2026-10-10T07:47:59.665924+09:00，完整七MAE仍进行中）

150×4=600张insights的RGB/Alpha/normal/albedo已实际生成，已view首中尾0001/0075/0150的inspection/four_categories_final_55000.png。RGB黑背景正常/主体平台存在，但头手、衣服和脚周平台仍斑驳伪影；Alpha大部饱和，腰手边缘/脚周薄雾与细碎残影。Albedo正背面照明依旧烘入，平台棋盘与脚周污染；GS Normal整体平台向上、0075不再旧包整个平台朝下，但人体/脚周混合轴与斑驳仍有。

本轮40k原生来源支持组正确符号与55k全图正常plane相互支持实际kernel修复生效，但不能据此推定所有Gaussian或GT normal准确；最终Depth/Mesh及15帧七MAE/coverage/common/missing180待完成。转灯/HDR正渲染，不能提前声明已审核。阶段质量FAILED_SO_FAR，仍未commit/push/新分支。证据inspection/final_55000_four_category_review.json、four_categories_final_55000.png；拼图命令python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_four_categories.py（实际exit0）。

## B06完整渲染及HDR/转灯实际目检（2026-10-10T07:56:04.688664+09:00）

全量render子进程实际exit0（runtime_history.jsonl），结束07:54:59.887392 JST，用时632.9978秒；RENDER_COMPLETED=all。实际150×4四类、150 HDR2048样本、72转灯和6视频存在（完整视频解码另由最终产物审计执行，不提前标通过）；后处理进入材质/几何评测。实际view inspection/relight_final_55000.png：HDR0001/0076/0150和转灯0000/0036/0071，黑背景/主体平台存在、光照有响应，但人物正背面和脚周斑驳、伪起伏/材质污染仍明显，仍FAILED_SO_FAR；无重光照GT，不声明精度。方向光固定0001相机/time/geometry转灯，alpha逐bit保持，亮度全图均值范围.01817811653–.07329406589。

灯表重新核验150帧各SUN白色单位intensity1/light_rgb[1,1,1]，仅原GT方向；实际转灯强度是该白色单位表sum×保存的训练RGB拟合scale6.611844539642334。relight.json遗留generic字符串“frame GT RGB irradiance sum × saved scale”不是使用原GT color/intensity的证据，本轮保留原渲染metadata、另在报告澄清，不改已完成渲染/核心源码。最终结果后再读GOAL决定修正文案，当前输入边界未变。

命令python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_relight_comparison.py实际exit0；证据inspection/final_55000_relight_review.json、relight_final_55000.png、model_mlp/renders_stage2/ours_55000/directional/relight.json及hdr/hdr.json。全部15七MAE与Depth/Mesh/roughness审核仍PENDING，未commit/push/新建目标分支。

### B06全量结果（2026-10-10T07:58:10.829884+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.16231946 | linear RGB | 96.784312% |
| Roughness MAE | 0.21780360 | r | 97.047548% |
| GS Render Normal MAE | 17.29779705 | degree | 94.101335% |
| Depth Normal MAE | 26.04174483 | degree | 89.466894% |
| Mesh Normal MAE | 20.34490562 | degree | 94.485513% |
| Expected Z Depth MAE | 0.08001558 | m | 94.101335% |
| Median Z Depth MAE | 0.08431691 | m | 94.101335% |

RGB：{"psnr": 19.682202597909917, "ssim": 0.8977724393208821, "lpips": 0.11070925494035085, "psnr_foreground": 12.488568200962481, "ssim_foreground_bbox": 0.7782925049463908, "lpips_foreground_bbox": 0.221701983610789}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-12文档。

## B06最终完整验收与B05同55k比较（2026-10-10T08:01:26.105680+09:00）

| 指标 | B05同55k | B06原生CUDA | B06−B05 |
| --- | ---: | ---: | ---: |
| Albedo MAE | 0.15432221 | 0.16231946 | +0.00799725 |
| Roughness MAE | 0.21168068 | 0.21780360 | +0.00612293 |
| GS Render Normal MAE | 17.73081984 | 17.29779705 | -0.43302279 |
| Depth Normal MAE | 26.21282757 | 26.04174483 | -0.17108274 |
| Mesh Normal MAE | 20.41400002 | 20.34490562 | -0.06909441 |
| Expected Z Depth MAE | 0.08031497 | 0.08001558 | -0.00029939 |
| Median Z Depth MAE | 0.08444845 | 0.08431691 | -0.00013154 |

GS法线改善0.43302279°、覆盖率93.88493451%→94.10133523%；Depth/Mesh/Z略改善，Albedo和Roughness更差。RGB PSNR19.37506314→19.68220260dB，LPIPS0.11125243→0.11070925，略改善但人物/脚周伪影明显，最终质量FAILED、7°FAILED。全部99827点与15测试帧保留，评分/GT前景域不变。

共同370823/415433（89.26180636%），GS/Depth/Mesh共同MAE16.31552150°/25.98854287°/19.29566234°；missing180=26.89505457°。

完整训练exec30746实际exit0，20k/55k于07:44:25完成，loop44:02；后处理exec47576实际exit0并关闭、9步骤全0，于07:58:11结束。187 CPU/真实20步GPU/共享逐buffer入口已验证，533来源SHA、15time×135TRAIN Mesh、481 roughness来源与独立15帧复算、6视频全解码PASS；这些不等于质量PASS。469核心/GLM及launch快照SHA再次一致，source32NN/冻结opacity-scale-feature保持。Native源SHA82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0绑定、原生实际编译.so SHA记录在pbr_rasterizer.json，不回退installed。B05旧kernel的实际结果保持，不覆盖为新kernel评分。

## B07下一轮选择：原生SUN直接光与真实光线预算（2026-10-10T08:04:20.443641+09:00，尚未实现/开训）

已重新读取本GOAL及B06全15帧七MAE/四类/HDR最终结果。B06修复实际native方向核：GS17.29779705°，较B05下降.43302279°/覆盖率提高，但Albedo.16231946和Roughness.21780360反而升高，RGB19.68220dB仅略改善，最终质量与7°FAILED。train和9步后处理实际exit0，1064归档文件SHA三处核验，源/快照保留；未commit/push/新建目标分支。

下一轮优先采用独立Direct Native Preset：关闭来源Stage1 SH间接光近似（训练/重光照），保留原生BRDF/GGX与BVH SUN可见性，以及B06全部材质/几何/source anchor/plane等设置；PBR物理训练像素预算按实际SUN数量计算，每像素L条SUN，若启用间接光则额外diffuse_sample_num条。默认其它Preset沿用历史预算，新显式开关绑定保存配置并验证适用original_ir/gt_directional。此候选并不删点或改变GT评分域，也不是仅重渲已有with-indirect模型后声称改善。

选择依据：B05只读真实SUN预算诊断证明关闭SH后旧代码仍按256分母，预算只选1024个SUN photo像素，实际可用all-valid18927/15971/19186单SUN射线；B06真native已验证GGX/BVH/物理材质与几何可导。当前SH间接近似带来源照明，材质烘光/脚周结构问题仍明显；用仅训练RGB/自身Alpha的新完整训练检验直接SUN物理模型更符合该输入的可能性。不是依据GT材质设置albedo/roughness或对GT法线拟合。原生SH辅助RGB分支先保留，避免同时改多个损失/源NN/gate或支持组。GT方向-only白色单位输入，全局强度仍只用训练RGB拟合，不读原GT强度/颜色。

gate梯度、未投影平台贡献/时间运动仍为后续候选；本轮不同时放开gate/NN，不强制候选全部静态。旋转审计不支持普遍卡60°，不据此扩大trust。下一步需新增参数/Preset/预算helper的CPU与实际native GPU smoke，保证默认回归、SUN L预算/所有photo像素/真实前反向/source指纹；通过功能smoke后从固定源35k独立完整Stage2 20k，仍全15七MAE/coverage/common/missing180及四类/HDR目检。当前未实现/未启动B07，无B07指标或通过结论。

另将在新代码迭代修正重光照元数据generic“frame GT RGB irradiance”文案，明确本轮白色单位direction table×训练RGB拟合scale；B06历史渲染metadata不覆盖，已另附勘误。

## B07实现与CPU验证（2026-10-10T08:09:56.647402+09:00）

新增两个Direct Native Preset及pbr_directional_actual_ray_budget=True，训练/重光照wo_indirect=True；native SH辅助RGB不改。SUN每像素L、with-indirect L+diffuse_samples，默认旧预算/显式HDR保持。CPU discovery194项实际exit0/OK，新增7预算/Preset边界测试。真实20步Trainer与GPU trace/梯度/共享buffer下一步，当前未启动完整B07或产生MAE；未commit/push/新目标分支。源码471文件快照在output/smoke_test/1010-14-dir-PBR原生SUN预算与DirectPreset。

## B07功能冒烟实际完成

## 实际冒烟结论（2026-10-10T08:13:31.388800+09:00）

194 CPU/20步真实Trainer/三帧all-eligible SUN及L3/with-indirect真实光线预算、材质与XYZ/q梯度、GeometryRun与材质helper逐buffer一致均PASS。参数SHA不变、0额外诊断优化，源码471 SHA核验。首次诊断重复backward图失败、保存后修正仅诊断图重建，retry实际exit0。35020 checkpoint flags/native来源绑定通过。

四类150×4和4视频逐帧解码PASS，已实际看首中尾RGB/Alpha/normal/albedo及SUN/HDR16。实际质量FAILED_SMOKE_ONLY：人物破碎/背面暗、albedo烘光、脚周伪几何仍有；不是完整MAE/7°，不commit/push。功能允许下一步独立完整20k训练，不沿用20步权重。

命令：check_pbr_gradients.py（失败和retry日志）、check_shared_buffers.py依次调用check_geometry_buffers.py/check_material_buffers.py；render_stage2 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-14-dir-PBR原生SUN预算与DirectPreset/model_mlp --load_iter35020 --task insights --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-14-dir-PBR原生SUN预算与DirectPreset/insights；日志logs/{pbr_gradients_retry,geometry_buffers,material_buffers,shared_buffers,insights}.log。代表图inspection/four_categories.png、relight.png；各检查JSON同inspection。

## B07完整训练准备（2026-10-10T08:13:43.091603+09:00）

# 1010-13 jumpingjacks dir 原生PBR直接SUN与真实光线预算完整Stage2计划

状态PREPARED，完整B07尚未启动。已重新读取GOAL及B06最终FAILED：GS17.29779705°、覆盖94.10133523%，RGB PSNR19.68220260dB，材质烘光和脚周伪几何仍明显。本轮只改变Direct光照与实际光线预算，不同时放开gate/源NN或调整几何trust/loss/lr。

固定Stage1 model_mlp35000→55000共20000步，135train/15test，resolution2（400×400）黑背景。原RGB/自身RGBA Alpha、相机/time、来源35k预测、GT方向；禁止GT albedo/normal/roughness/depth/mesh/blend/强度/颜色进入训练/初始化/loss/prior，GT只离线评测。来源35k继承旧albedo Alpha转换但全部150帧已证明与自身RGBA及新转换逐像素一致，披露且不重跑Stage1。

Preset pbr_gt_directional_plane_native_direct，wo_indirect与wo_indirect_relight=True，pbr_directional_actual_ray_budget=True。关闭PBR Stage1 SH间接近似，保留native SH辅助RGB；SUN每像素L条BVH射线、with-indirect则L+diffuse_samples。单SUN预算262144覆盖全部alpha>.9可用photo像素，原选择规则和GT域保持。旧Preset及显式HDR预算不变。沿用B06原生CUDA/GGX/BVH、15187/99827 source plane支持、DC内部投影、固定sourceXYZ锚定、32来源NN/opacity-scale-feature冻结、geometry warmup2000/ramp1000、canonical .08m/60°、原loss/lr及RGB拟合全局正E。plane额外world投影不等于canonical位移边界。全部点保留，不筛掉难像素/帧。

194 CPU PASS、真实20步Trainer/GPU trace与所有材质/几何梯度PASS；三TRAIN直接SUN18843/15904/19137实际光线/all-eligible像素，3SUN170像素/510条，with-indirect1020像素/262140条；旧direct仍1024。诊断0额外优化/SHA保持，失败诊断及重试保留。共享GeometryRun/材料helper逐buffer PASS。150×4与三HDR16/SUN smoke功能PASS、实际质量FAILED；不是完整MAE或7°。

本轮来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp35k独立开始，不续B06的55k。计划保存40k/50k/55k，正式全量all render：四类各150，HDR150×2048，转灯72，6视频；全部15帧七MAE/coverage/common/missing180；每测试time全部135训练位姿预测Depth Mesh融合，固定原评分定义，无对齐/缩放/筛物体。原生无完整Adam state，不声称无损resume。

训练/各评测/日志完整命令如下。代码471文件双快照及native源SHA记录；旧失败run与本run失败权重/图/数组/日志全部保留。


完整命令见对应训练报告与train_argv/planned_commands，仍未开训或产生B07七MAE。

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

### B07全量结果（2026-10-10T08:55:28.290218+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.20195435 | linear RGB | 97.264792% |
| Roughness MAE | 0.20060596 | r | 97.490997% |
| GS Render Normal MAE | 16.81871120 | degree | 95.003286% |
| Depth Normal MAE | 25.83637410 | degree | 90.318054% |
| Mesh Normal MAE | 20.06540566 | degree | 95.002804% |
| Expected Z Depth MAE | 0.07846366 | m | 95.003286% |
| Median Z Depth MAE | 0.08282073 | m | 95.003286% |

RGB：{"psnr": 19.235614174536575, "ssim": 0.9104916214942932, "lpips": 0.11079909801483154, "psnr_foreground": 11.882659584201594, "ssim_foreground_bbox": 0.803943928082784, "lpips_foreground_bbox": 0.22023212313652038}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-13文档。

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

## B08候选：原生Direct PBR来源输出gate微调（2026-10-10T09:02:36.005467+09:00，实施前计划）

已重新读取本GOAL和B07完整验收。B07全20k/9步后处理exit0，15帧GS16.81871120°/覆盖95.00328573%，共同15.78935883°、missing18024.97241395°；全部七MAE与实际RGB/Alpha/Albedo/Normal/Depth/Mesh/HDR/SUN审核完成，1057归档文件三处SHA核验。质量FAILED、7°FAILED，无commit/push/新分支。

选择单独开放每Gaussian原生feature输出乘法gate，来源35k MLP全部32tensor和sourceXYZ输入冻结，原生NN结构及输出公式不改。依据B06实际只读gate梯度非零以及B07source joint-group图中dynamic near-plane点对平台有明显贡献；source候选不是GT物体标签，不把这些点强制当静态、不删点/遮评分。全99827feature保留原值初始化，全部点RGB/自身Alpha/预测ND可驱动gate，geometry warmup2000/ramp1000同调度；拟定独立gate lr.001、source raw-logit最大变化4、弱源anchor.001，optimizer只新增feature组，source MLP所有组关闭/requires_gradFalse，训练Deform查询仅在gate候选开启时允许乘法梯度。

新Preset pbr_gt_directional_plane_native_direct_gate继承B07，默认其它Preset/Stage1保持。原DC/roughness/XYZ/q/plane/opacity/scale/Direct SUN预算、原评分不改。来源35k SHA和feature/NN绑定、保存PLY gate及metadata/lighting marker校验，FrozenScene/GeometryRun/材料helper共享加载。先CPU和真实20步Trainer/梯度/实际buffer/四类与重光照验证，再从35k独立完整20k及全15七MAE验收；功能通过不等于质量或7°。仅原RGB/自身Alpha/相机time/来源预测与GT单位方向，不读GT材质/法线/几何/能量/颜色进入训练/初始化/loss/prior。当前尚未实现、未开训，无B08指标。

## B08来源输出gate实现与功能冒烟

## 实际冒烟验收（2026-10-10T09:08:06.181380+09:00）

202 CPU discovery实际exit0；真实20步Trainer实际exit0并关闭，35k→35020保存全部99827点。新增gate参数组真实训练68101点raw feature发生变化，max delta1.85966492e−5，低步ramp仅验证更新；opacity/scale不变，source32 MLP tensor完全原值，初始feature原值，保存source metadata/lighting marker并共享加载核验。无额外诊断优化、无GT材质/法线/几何/能量/颜色训练输入。

真实native三TRAIN0001/75/149 SUN18843/15904/19137条，all-eligible alpha>.9；albedo/roughness/XYZ/q/feature梯度非零finite。gate photo L2 .00085692690/.00269499072/.00030556315。3SUN/with-indirect/历史预算variant同checkpoint零优化全部finite，参数SHA保持，不宣称不同variant已独立训练。GeometryRun alpha/worldnormal/Expected Z全buffer一致、目标timeoverride一致；albedo/roughness helper一致。

150×四类及4视频全解码PASS，实际查看0001/0075/0150 RGB/Alpha/Normal/Albedo及三HDR16/SUN。功能PASS、实际质量FAILED_SMOKE_ONLY：人物破碎/背面暗、材质烘光、平台脚周伪起伏仍明显；不作为正式全15七MAE/7°，未commit/push/新分支。正式候选须从35k独立完整20k。

首次CPU201项有2个旧GeometryRun mock缺少新增FrozenScene字段的错误，失败日志和原source snapshot保留；真实20步Trainer生产源码未改，之后仅修mock/增加metadata用例，202 retry全PASS。code_manifest_train_launch与现manifest唯一差异是tests/test_geometry_eval_source_runtime.py，生产核心完全相同。

完整训练命令：
```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-15-dir-PBR来源输出gate/model' --load_iter 35000 --iterations 35020 --save_iterations 35020 --test_iterations 35020 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct_gate --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_geometry_warmup 0
```

诊断命令commands/{check_gate_checkpoint,check_pbr_gradients,check_shared_buffers,check_geometry_buffers,check_material_buffers,make_comparison,finish_smoke}.py；对应日志logs/{gate_checkpoint,pbr_gradients,shared_buffers,geometry_buffers,material_buffers}.log；渲染命令python -m scripts.render_stage2 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-15-dir-PBR来源输出gate/model_mlp --load_iter35020 --task insights --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-15-dir-PBR来源输出gate/insights，日志logs/insights.log。图inspection/four_categories.png、relight.png；JSON和视频完整路径见inspection/visual_review.json。

## B08完整训练准备（2026-10-10T09:10:03.381895+09:00）

# 1010-14 jumpingjacks dir 原生PBR来源输出gate完整Stage2训练计划

状态PREPARED，完整B08未开训，无B08正式MAE/质量结论。已读GOAL及B07最终GS16.81871120°/覆盖95.00328573%、质量FAILED，全部失败产物保存。只新增原生feature输出gate，source35k MLP32tensor/XYZ输入冻结；本轮source/model35k→55000完整20000步，仅Stage2，135TRAIN/15TEST、400×400、黑背景、相机time原值。

输入仅原RGB/自身RGBA Alpha、源35k预测和GT方向白色单位，不用GT材质/法线/粗糙度/深度/mesh/blend/强度/颜色初始化/loss/prior；GT仅离线评分。源35k来自旧转换，但150帧原/新/旧Alpha逐bit一致已披露，不重训Stage1。

Preset pbr_gt_directional_plane_native_direct_gate：继承B07 Direct/native kernel/sourceXYZ/15187 source plane/DC内部投影/原geometry/loss/lr与全局RGB拟合E。只新增feature optimizer组，source32 MLP tensor冻结、opacity/scale冻结，全部99827点保留。原feature初始化，geometry warmup2000/ramp1000/decay同步，gate_lr.001、max_source_logit_delta4、weakanchor.001，每4步Adam投影source±4。gate metadata/lighting marker绑定来源SHA、加载实际gate而非source，FrozenScene/GeometryRun/材料helper共享。canonicalXYZ .08m/q60°保持，额外plane world投影不受canonical边界。原生没有完整Adam状态，不声明无损resume。

202 CPU retry/真实20步/GPU五参数梯度/源32tensor与gate保存/实际SUN预算/共享buffer PASS；初次201项2个旧mock缺字段错误只修fixture后202通过，原生产训练core保持。全部150四类及4视频解码PASS，实际smoke质量FAILED，只验证功能，无正式MAE。完整run从35k独立训练，不沿用smoke或B07权重。

保存40k/50k/55k；全量四类150×4、HDR150×2048、方向72、6视频。全部15帧七MAE，原GS有向world normal GTalpha≥.5全前景pixel-weighted，coverage/common/missing180保持；Depth Z米不拟合，Mesh每个测试time所有135训练姿态模型预测深度独立融合，无GT/testpose重建。质量未通过前不commit/push/新目标分支。


完整命令见训练报告及train_argv/planned_commands，尚未开训。

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

### B08全量结果（2026-10-10T09:52:24.482406+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.20137736 | linear RGB | 97.269167% |
| Roughness MAE | 0.20119953 | r | 97.494595% |
| GS Render Normal MAE | 16.84133987 | degree | 95.012674% |
| Depth Normal MAE | 25.82059126 | degree | 90.332737% |
| Mesh Normal MAE | 20.07707837 | degree | 95.004249% |
| Expected Z Depth MAE | 0.07851869 | m | 95.012674% |
| Median Z Depth MAE | 0.08287710 | m | 95.012674% |

RGB：{"psnr": 19.24938274959784, "ssim": 0.9107702334721883, "lpips": 0.1102573330203692, "psnr_foreground": 11.887298783047992, "ssim_foreground_bbox": 0.8042977293332417, "lpips_foreground_bbox": 0.21988034347693125}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-14文档。

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

## B09概率gate优化坐标方案与CPU验证（2026-10-10T09:56:41.896727+09:00）

# 1010-16 dir 原生PBR概率gate Preset功能冒烟

状态：211 CPU回归实际exit0；20步Trainer、GPU概率梯度、保存加载与共享buffer/四类/HDR/SUN待完成。

已先读取GOAL，B08完整20k/七MAE/人工质量FAILED，GS16.84133987°/覆盖95.01267352%、原7°目标FAILED，1055文件归档SHA3保持。55k只读梯度诊断显示sigmoid因子全点中位4.2676e−5，原raw Adam概率改变很小。本轮仅改gate优化坐标，不根据GT标签生成prior。

Preset pbr_gt_directional_plane_native_direct_probability_gate；source35000→35020（20步），源35k PLY/32 MLP、sourceXYZ、RGB/自身Alpha/相机time/GT白色单位方向唯一训练输入。禁GT材质/normal/roughness/depth/mesh/blend/原灯强度颜色。原NN32tensor、opacity/scale冻结，99827点全部保留；source平面15187继续由原35k全部135time推导。Direct SUN/GGX/BVH、原SH RGB辅助loss、DC投影与XYZ/q信赖域/loss保持。

概率叶子逐bit初始化为原native sigmoid gate，Adam feature组仅绑定概率叶子；前向逐bit使用真实PLY logit的native sigmoid值并对概率传递单位梯度。每4步更新后投影概率到来源±1及[0,1]、有限逆sigmoid写回PLY；端点使用±80有限raw，未改变的概率保留source raw逐bit值，然后同步概率叶子为实际native量化值。权重保存与所有FrozenScene/GeometryRun/材质/HDR/SUN均只读取实际PLY原native feature，另无训练概率资产；metadata v2与checkpoint SHA绑定，严格sourceNN/概率信赖域/协议验证。v1 raw gate和旧Preset保留。未保存完整Adam状态，不声明无损resume。

smoke warmup0、1000步ramp，仅验证20步功能；正式完整20k须恢复warmup2000，不从smoke/B08训练后权重继续。lr.001/max_probability_delta1/anchor.001，其余B08相同。CPU新增9项：来源初值/饱和单位梯度/有限端点/量化/Adam保存/边界/来源协议/legacy/auto，共211实际PASS。真实短训练与图像尚待检查；无正式MAE/7°结论，未commit/push/新目标分支。


## B09概率gate实现与真实功能冒烟

## 实际冒烟完成（2026-10-10T10:01:14.420317+09:00）

211 CPU、20步Trainer/GPU五参数梯度/保存/共享geometry-material/insights六会话均实际exit0且关闭；474核心和双snapshot逐SHA一致，未在训练后改生产源码。35000→35020保留99827点finite，opacity/scale和32来源NN tensor原值。raw feature变化70033点、实际native概率变化69971点，max概率delta 1.859664917e-05；raw最大偏移74.53421021是逆sigmoid/有限端点编码，不等于世界几何的大改变，不沿用raw±4的旧信赖域。

TRAIN0001/0075/0149 photo概率梯度L2 0.015394565,0.035415199,0.013251405，albedo/roughness/XYZ/q/概率全部finite非零；前向逐bit仍实际PLY native sigmoid，源码NN全冻结。单SUN实际18843/15904/19137光线，3SUN/间接/旧预算诊断都通过且0 optimizer，不称已独立训练。GeometryRun alpha/worldnormal/ExpectedZ最大误差0与目标time一致；albedo helper误差0，roughness≤5.96e−8。

概率全域[0,1]允许源gate修正，未强制任何source组静态、不删点/帧/难像素；canonicalXYZ .08m/q60°仍只限制canonical参数，gate改变原NN输出的world形变另需完整RGB/Alpha/覆盖与MAE验收。输入没有GT材质/法线/几何/灯强度颜色，GT方向白色单位，RGB拟合强度非GT。原生完整Adam未保存，不声明无损resume。

四类150×4、4视频全部解码150帧，实际看0001/0075/0150四类与HDR16/SUN代表图：

- 黑背景正常，人物/平台存在；20步初始材质人物斑驳/破碎、背面黑，脚周灰黑污染仍明显。
- 人物与平台主体覆盖完整/大部饱和，手臂/身体轮廓和平台边缘有薄雾及细碎残影。
- 正背面的光照色差/身体阴影烘入材质、平台脚周污染，未通过材质质量。
- Normal平台整体上向但脚周伪凸起、人体混合轴斑驳；GeometryRun仅buffer/time一致审计，不是GT精度，短训不作正式Depth/Mesh/7°评价。
- 实际view三帧独立替换SUN和HDR16，照明/暖色响应存在，身体斑驳/缺失和平台伪凸起仍明显；仅功能，最终全量HDR2048须重训后验收。

最终功能PASS、画质FAILED_SMOKE_ONLY、正式全15七MAE及7°未评；下一完整B09只从原35k独立训练，warmup恢复2000，所有评测口径/覆盖/common/missing180原值，不以20步smoke或B08训练后checkpoint继续。未commit/push/新目标分支。

## B09完整20k准备（2026-10-10T10:02:58.635227+09:00）

# 1010-15 jumpingjacks dir 原生PBR概率gate完整Stage2训练计划

状态PREPARED，未启动完整B09，无正式MAE/7°或质量结论。已先读GOAL，B08完整20k/七MAE/实际质量FAILED：GS16.84133987°/覆盖95.01267352%，失败1055文件归档保留。sigmoid饱和诊断及211 CPU/真实20步/GPU/共享buffer/4视频功能PASS，20步实际质量FAILED_SMOKE_ONLY。

只从原Stage1 source35000→55000独立20000步，仅Stage2，135TRAIN/15TEST、400²/黑背景、原相机time。输入仅原RGB/自身RGBA Alpha、source35k PLY/MLP、GT方向白色单位；不用GT材质/法线/粗糙度/深度/mesh/blend/灯强度颜色训练或prior。GT仅离线评测。旧source转换Alpha来源已披露、150帧原/新/旧Alpha bit一致，不重训Stage1。

Preset pbr_gt_directional_plane_native_direct_probability_gate：继承B08 Direct/native CUDA/sourceXYZ、固定plane15187、DC内部投影、XYZ/q/loss、全局RGB拟合E，只有gate优化坐标不同。211 CPU/GPU短训核验，无后训练源码改动；474核心双snapshot。概率叶子native源概率逐bit初始化，feature Adam组仅该叶子，源NN32tensor/opacity/scale冻结；前向精确使用PLY native sigmoid，更新后概率来源±1∩[0,1]投影，有限逆sigmoid存PLY并同步native量化。lr.001/anchor.001、正式warmup2000/ramp1000/decay/每4步保持，v2 metadata/marker SHA绑定真实source。旧Preset raw/v1兼容。没有概率teacher/GT prior；canonical .08m/q60°不保证gate引起的world形变有相同上限，须实际全域验收。未存完整Adam状态，不声明无损resume。

保存40k/50k/55k；150×4四类/HDR150×2048/SUN72/6视频。全部15 TEST七MAE+coverage/common/missing180原评分：GTalpha≥.5全前景pixel-weighted有向GS；Expected/Median camera Z米不拟合，Mesh每个目标time全部135 TRAIN预测深度融合，不用GT/测试位姿重建。全部点/帧/难像素保留，源opacity/scale不优化。未通过质量前不commit/push/新目标分支。


全部命令见训练文档和planned_commands/train_argv，尚未启动。

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

## 训练RGB候选方向约束独立原型准备

# 1010-17 dir 训练RGB法线约束独立原型

状态：PREPARED_SOURCE_RGB_PROTOTYPE；未改生产、未开始新完整训练，B09正式MAE仍待当前原后处理进程完成。2026-10-10T10:38:14.411234+09:00

已先读取GOAL，B09完整20k实际exit0但RGB/四类实际质量FAILED_EARLY；不能仅继续放大gate更新认定达到7度。已有135 TRAIN RGB线性可观测性只读诊断显示约40948点可拟合，source-local四元数运输与来源矩阵一致，但原additive大部分10度响应约9.1度，不能称原加法大范围失效。

本独立原型验证一个有观测来源的候选：只从原35k预测PLY/冻结来源NN、全部135 TRAIN RGB与自身Alpha/相机time/白色单位GT方向，估计来源canonical轴的RGB光照方向约束。GT albedo/normal/roughness/depth/mesh/blend/原光颜色能量不读。RGB估计方向是原训练图像派生，明确不称GT法线；全部99827点保留，可靠性权重不改变原photo loss/测试评分/帧。135个TRAIN键不含15 TEST；逐图像SHA/RGBA alpha和来源SHA绑定。

原型仅使用35k source XYZ/q/feature/scale/opacity与既有source-only plane，40k只作为固定来源NN/相机容器，学得40k XYZ/q/feature/material不用于构建。投影/来源方向运输逐点核验，source预测Z可见性.025m/ownAlpha≥.99/source rasterAlpha>.9。线性luminance最小二乘只作弱Lambertian方向候选，>=12观测、gram最小特征值>.01最大、系数norm>.01、relative residual<.3；原plane支持不重复施加该候选。原SHA/plane支持保持，所有完整点数组保存，未可靠点权重0而非删除点。此不是已通过GGX遮挡拟合或GT准确率。

必须先审计候选来源、有限性/维度/不含测试帧、confidence与原只读诊断逐点复算；后续若接入source-local渲染和训练，需独立Preset、全部入口/保存加载、CPU/GPU冒烟与完整20k/15帧七MAE/四类/HDR人工验收。B09生产474 SHA与当前后处理不改，不提前提交或宣称7度。无完整Stage2训练，无正式MAE结论。

## 训练RGB候选约束原型实际完成（未接入生产）


## 独立候选生成与CPU复算（2026-10-10T10:42:33.274030+09:00）

首attempt exec23962真实exit1并关闭，source PLY根路径错误，初次脚本/log和failure JSON保留；独立retry exec56679实际exit0并关闭，source BASE修正，native_probability精确源gate，不续任何学得40k参数。CPU verifier真实exit0，独立float64 gram/solve/残差/condition/轴与source SHA复算，confidence逐点差异0；CPU/GPU轴max绝对差2.4733824e-06、source轴复算误差5.9383892e-07，原输出angle字段未除float轴norm，.02952度包含float单位长度舍入，依maxabs判断计算一致，不当GT角误差。6来源文件与135 TRAIN RGBA SHA全核验，TEST0、GT标签0、optimizer0。

| source组（非GT语义） | 全部点数 | 可靠线性拟合点 | 实际候选约束点 |
| --- | ---: | ---: | ---: |
| all | 99827 | 40948 | 29621 |
| source_static | 54244 | 28716 | 17389 |
| source_dynamic | 45583 | 12232 | 12232 |
| projected_source | 15187 | 11327 | 0 |

实际非投影候选29621（静态17389/动态12232）；source-only plane支持已固定法线，不重复施加，可靠投影点11327仍完整保留。全99827点canonical轴/confidence/count/gram/rhs/残差数组保存candidate_assets/training_rgb_normal_constraint.npz，source_manifest/TRAIN_RGBA_manifest严格绑定。候选方向由原训练RGB生成，绝非GT normal；confidence仅此附加约束的支持，不改变原photo loss/点/TEST评分。方法仅线性Lambertian近似，未证明GGX/阴影/强度鲁棒或原7度准确率。当前B09正式15七MAE仍待完成，生产474SHA不变；未实现新生产Preset/完整Stage2训练，未用候选初始化现有B09或评测，不commit/push/新分支。后续若采用必须全部入口来源矩阵一致、保存加载检查和独立完整20k/15七MAE/质量验收。

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-17-dir-训练RGB法线约束原型/commands/build_training_rgb_constraint_retry.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-17-dir-训练RGB法线约束原型/commands/verify_training_rgb_constraint.py
```

### B09全量结果（2026-10-10T10:45:17.268449+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.19121691 | linear RGB | 97.544001% |
| Roughness MAE | 0.19018926 | r | 97.749537% |
| GS Render Normal MAE | 15.94231363 | degree | 95.478453% |
| Depth Normal MAE | 25.18807492 | degree | 90.790091% |
| Mesh Normal MAE | 19.88648893 | degree | 95.363392% |
| Expected Z Depth MAE | 0.07691848 | m | 95.478453% |
| Median Z Depth MAE | 0.08110909 | m | 95.478453% |

RGB：{"psnr": 19.810345333543953, "ssim": 0.9202187975247701, "lpips": 0.10036607980728149, "psnr_foreground": 12.319447497693583, "ssim_foreground_bbox": 0.8192175626754761, "lpips_foreground_bbox": 0.2063336710135142}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-15文档。

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

## B10训练RGB来源局部法线Preset实现与CPU（准备冒烟）

# 1010-18 dir 原生PBR训练RGB来源局部法线 Preset冒烟

状态：226 CPU实际exit0；真实20步Trainer/GPU/保存加载/共享buffer与实际四类/HDR/SUN待完成。2026-10-10T10:58:17.196226+09:00

已先读GOAL，B09完整20k/15七MAE/实际质量FAILED，原GS Normal15.94231363度/覆盖95.47845260%，1085文件SHA三checkout保留，未commit/push/新分支。独立训练RGB原型135 TRAIN、全99827点，有29621非投影来源点约束（静态17389/动态12232），其source/原RGB SHA与独立CPU全点LS/confidence复算PASS，首路径失败保留。约束由原RGB/自身Alpha/源35k/相机time/单位灯方向派生，绝不来自GT normal/material/depth/roughness/mesh/blend/灯颜色能量；GT只最终离线评分。

本Preset pbr_gt_directional_source_rgb_normal继承B09概率gate/native Direct/原GGX/BVH/SH辅助/material/geometry/lr/warmup/ramp信赖域与plane15187，新增source-local quaternion运输和lambda2 RGB轴约束。source q35k独立常量、原NN32tensor固定；对当前p gate查询原NN的world qsource，按R_world_source(p) R_canonical_source^T R_canonical_current运输；source初值raw deformation逐bit原值。只修改真实Gaussian渲染world quaternion，XYZ/depth来自原位置与真实raster，plane继续在运输后应用。原RGB拟合canonical轴约束平均1-dot，原photo/SH/Alpha/预测几何损失及GT评分不变；可靠性仅附加项权重，不删点/帧/难评分像素，existing plane支持不重复施加。B09概率gate前向/保存仍原native PLY sigmoid，无强制静态。

create/load均核验local35k来源/NN无梯度、135 TRAIN无TEST、全部观测图SHA/camera/白色单位方向、input协议/SHA与完整LS数组复算；模型复制输入asset+metadata并保存source quaternion/轴/confidence本地npz，checkpoint绑定其metadataSHA。FrozenScene/周期预览/GeometryRun/Mesh/材质/HDR/SUN共享运输入口，模型原数值/资产任何变化报错。canonical .08m/q60度保持，不称保证world位置最大.08（可学gate改变原NN motion）。原生Adam仍无完整状态，不声明无损resume。

独立原35k→35020，20步，仅冒烟warmup0/ramp1000；正式完整必须从原35k另训20k、恢复warmup2000，不续本smoke或B09。当前476核心两snapshot锁定；新增15 CPU实际PASS（初次CPU矩阵引用repo CUDA工具造成device mismatch失败保留并修正测试独立CPU矩阵），全部226 PASS。不提前声称图像/7度通过。

## 完整训练命令

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-18-dir-PBR训练RGB来源局部法线/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-18-dir-PBR训练RGB来源局部法线/model' --load_iter 35000 --iterations 35020 --save_iterations 35020 --test_iterations 35020 --render_mode original_ir --loss_preset pbr_gt_directional_source_rgb_normal --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-18-dir-PBR训练RGB来源局部法线/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_geometry_warmup 0 --pbr_rgb_normal_prior_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-17-dir-训练RGB法线约束原型/candidate_assets'
```

日志logs/train.log；来源source35k/model_mlp/point_cloud与deform SHA保持。后续命令commands/check_gate_checkpoint.py、check_pbr_gradients.py、check_shared_buffers.py和insights渲染，输出inspection与insights；真实四类首中尾/HDR16/SUN目检和4视频全帧解码待完成，最终PASS/FAILED待填。正式15七MAE尚未测。

## B10真实Preset功能冒烟与实际人工目检

## B10实际冒烟完成（2026-10-10T11:08:01.965361+09:00）

226 CPU、真实20步Trainer及保存加载/GPU/共享buffer/insights均实际exit0且会话关闭；476核心和双snapshot SHA保持。35000→35020，全部99827点finite，opacity/scale及来源NN32tensor冻结，概率gate改变69981点、maxdelta1.85966491699e-5；raw变化70033/max74.534210205只是有限逆sigmoid编码。独立源RGB候选29621点，lambda2，canonical轴loss0.102662175894、其q梯度L2 0.00525612570345非零，三TRAIN photo q梯度L2 .022683673/.602086246/.024002507，概率 .015508413/.038564317/.013200786；albedo/roughness/XYZ/q/gate均finite且非零。不能从全局梯度L2推断完整训练质量。

真实DirectSUN/HDR16梯度预算、三SUN/间接/legacy只读0 optimizer检查PASS；共享Geometry alpha/GSnormal/ExpectedZ误差0，material albedo0/roughness≤5.96e−8，plane真实世界位置约2.7e−8m/轴cos≥.99999964。source-local运输应用在真实Gaussian q，所有入口同顺序先运输再plane。4类150×4及4视频各150帧全部解码；Depth/Mesh及全15七MAE尚未在smoke生成或评分。

- RGB黑色背景正常，首尾人物和平台保留；中帧人物接近黑色但Alpha主体完整，不能解释为缺失。首尾人物/裤腿有明显黑斑，平台足周黑灰杂点，20步画质不通过。
- 首中尾人物和完整椭圆平台均有Alpha；手臂、身体及脚周有少量模糊边缘和孤立点。没有删去难点或平台。
- 人物首尾和中帧均可见，但明暗及纹理斑块混入反照率，手臂/裤腿碎斑，平台颜色受阴影污染；不认定已经完成材质分解。
- 实际渲染GS法线可见人物和平台，首尾平台主要紫蓝，中帧人物黄绿且表面杂色，足周法线噪点明显。此拼图只检查GS法线，未额外生成Depth/Mesh，不称其已验收。
- 同首中尾的方向光与HDR16均通过共享入口有实际亮度/色彩响应；HDR有橙色轮廓与高光、平台蓝色响应。方向光中帧人物近黑，不能仅据此宣称代码异常，也不能据重光照有响应认定几何正确。

最终功能PASS，画质FAILED_SMOKE_ONLY，原GS Normal<7度NOT_EVALUATED。GT方向白色单位；新轴来自原135TRAIN RGB/自身Alpha和原35k预测，不是GT normal；GT材质/法线/粗糙度/深度/mesh/blend/灯能量颜色全部未参与训练/初始化/约束。全部点/帧/原photo/评分保留。正式B10需独立原35k→55k完整20k，恢复warmup2000/ramp1000；不用本smoke或B09学得参数继续训练。不commit/push/新目标分支。

## B10独立完整20k准备（2026-10-10T11:10:37.954539+09:00）

# 1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束 完整Stage2训练计划

状态PREPARED，完整训练未启动，正式七MAE及7度和质量结论待执行。B09完整20k原GS MAE15.94231363度/覆盖95.47845260%，实际质量FAILED，全部失败产物保留。已读GOAL；B10实际226 CPU/20步/GPU/保存加载/共享buffer/4视频功能PASS，画质FAILED_SMOKE_ONLY。

## 本次训练RGB来源局部法线与Preset

继承B09概率gate、仓库原生Direct SUN/GGX/BVH与SH辅助、材质、真实XYZ/q、信赖域、source-only plane15187及原photo/Alpha/ND损失和全部lr。独立Preset pbr_gt_directional_source_rgb_normal，新增pbr_source_local_rotation=True与lambda_pbr_rgb_normal_prior=2；正式warmup2000/ramp1000。source q/raw norm从原35k保存，固定NN32 tensor按当前gate查询来源world旋转，再按R_source_world(p) R_source_canonical^T R_current_canonical运输，随后执行既有plane。初值原d_rotation逐bit保持。实际Gaussian covariance、normal、depth、PBR/BVH共享真实q，绝非评测法线图覆盖。周期预览/FrozenScene/GeometryRun/Mesh/material/HDR/SUN都使用相同入口和顺序。

全部99827点数组保留；附加canonical轴loss=sum(confidence*(1-dot))/sum(confidence)，仅新增loss按29621个confidence点归一化，29621非plane点有候选支持（来源static17389/dynamic12232）。来自原135TRAIN RGB线性luminance与自身Alpha、来源预测可见性、源35k qXYZ/固定NN/相机time/白色单位GT方向的独立LS候选；9个全点数组、source6 SHA/135逐图SHA、TRAINkeys/TEST0、finite与CPU全点Gram/解/残差/confidence复算绑定模型。该候选是训练RGB派生，不是GT法线，也不保证GGX/阴影准确；其confidence只约束新增loss，不改变原photo、完整点/帧/难像素或原评分。既有plane点没有重复该轴约束，也没有删除。

模型保存pbr_rgb_normal{,_input}.{npz,json}，source quaternion/target/confidence和原独立input本地复制、各SHA与checkpoint metadataSHA绑定，所有加载严格复算并检查源32NN冻结。新增0 optimizer参数，仍用原canonical rotation Adam raw q lr.001/norm和60度信赖域；probability gate/native sigmoid/有限逆sigmoid原实现不变。canonical .08m不保证gate/plane引起的实际world位移同界。未保存完整Adam状态，不声明无损resume。

226 CPU、独立20步35000→35020实际Trainer/保存加载/GPU五参数梯度/共享buffer/HDR16/SUN/四类150×4和4视频全解码功能PASS，实际画质FAILED_SMOKE_ONLY；RGB约束loss .102662175894、q梯度L2 .00525612570345非零。476核心与双snapshot保持。此完整20k从原35k初始化，不续20步或B09学得checkpoint。GT材质/normal/roughness/depth/mesh/blend/原灯强度颜色全部不参与训练/初始化/约束；仅独立离线评测读取GT标签，原有完整七MAE/覆盖/common/missing180定义保持。

只从原Stage1 source35000→55000独立20000步，仅Stage2，135TRAIN/15TEST、400²/黑背景/原相机time。原RGB/自身Alpha/预测35k/GT direction白色单位，新增轴约束由135TRAIN RGB估计，没有GT normal/albedo/roughness/depth/mesh/blend/原光强颜色。sourceplane15187、全部99827点/原photo/评分保持。正式warmup2000/ramp1000，保存40k/50k/55k。训练初始化没有使用候选替换source q或学得B09权重。


全部9步原评分命令见train_argv/planned_commands/训练文档，完整训练尚未启动。

## B10完整训练已实际启动（2026-10-10T11:12:19.216698+09:00）

原35k→55k独立20k，TRAIN PID306459/ticks10865815，exec4419实际仍运行；11:11:25.974063 JST启动，warmup2000/ramp1000、lambda2训练RGB轴约束，当前无正式MAE/质量结论。检查点独立共享加载helper先在smoke35020真实exit0核验，无GT标签、0 optimizer、29621支持/全99827点。检查点审计等待exec10940/后处理等待exec65390，精确进程信息见process_handles.json；保存40k/50k/55k后检查，训练完成后9步原全量渲染/七MAE自动顺序执行，人工质量单独验收。源476核心和双snapshot冻结。

训练准备helper首attempt在执行前因字符串引号嵌套语法错误exit1，未创建正式目录/模型、未启动训练；随后shell尝试不存在retry文件exit2，原日志保留。修正helper独立retry实际exit0，正式目录只创建一次，没有训练重启或改生产代码。全部准备脚本、失败JSON/log保留smoke1010-18。

### B10启动只读核对与当前进度（2026-10-10T11:13:35.904698+09:00）

audit_launch实际exit0，配置相比B09只有运行路径/Preset和新增3字段（source_local=True、RGB资产路径、lambda2）共7项差异，原warmup2000/概率gate/lr/源plane15187/全部99827点/ownAlpha150/冻结NN32/direction表byte一致。输入/资产SHA正常，source RGB约束支持29621；0诊断optimizer、0 GT标签读取。真实train PID306459仍运行，已完成1395/20000步，源码476 SHA锁定；exec4419/10940/65390均在运行中，不能提前填exit0。正式七MAE/前后景质量/7度均PENDING。

### B10新增RGB轴损失归一化说明（2026-10-10T11:15:03.711278+09:00）

核对实际实现：新增loss为sum(confidence*(1-dot))/sum(confidence)，29621支持点归一化，并非除99827；全99827数组/原photo/全部评测点帧保留。准备报告的该句已依实际代码更正，不修改源码、已有模型或训练，不把新增候选支持定义为正式评分范围。

## B10训练曲线与RGB约束生命周期只读证据（2026-10-10T11:18:58.872501+09:00）

两个诊断均实际exit0/关闭（curve exec27142、RGB lifecycle exec92303），固定事件字节快照SHA，0 optimizer、0 GT标签读取。实际查看两张图：warmup2000 B09/B10曲线重合，逐step PhotoL1最大差2.68e−7/均差2.39e−9，total最大3.13e−7；2001–2464 RGB L1均差−.00016006，新增total均差+.03294333，total含附加RGB loss不能直接比较成画质退化。全局RGB拟合强度几乎一致，此说明新几何phase前初始外观保持，不当最终MAE。

RGB轴约束首记录37100（2100步）loss .0992467552，39400（4400步）.0038797222，lambda2*ramp后当前 .0077594444；实际在优化但不是GT normal误差。29621 source RGB候选相对原source预测轴median14.7640度/P9045.1578/P9983.1069，1600点大于60度。既有canonical60度的独立轴理想loss下界.00208803，只是轴几何条件、非GT误差或完整photo冲突最优值；目前无证据应放开cap或增大lambda。confidence源q norm中位1.029181/P991.804685/max3.592599，全99827中位.973277/P991.674602/max3.686656；不支持之前泛化为大raw norm导致学习失效，本轮保留原raw q Adam。

三组diag source/事件SHA保留inspection/early_training与rgb_axis_lifecycle。图和轴分布只读，不增加训练teacher、不筛原评分/点/帧。相对原GS normal <7度仍等待55k全部15七MAE，当前不commit/push/新分支。

## B10同一40k检查点中间诊断与实际目检（2026-10-10T11:30:39.834769+09:00）

真实40k保存审核child exit0，99827点finite/opacity-scale/NN32冻结，canonical位移max.03547572m/qmax60.00006485度，native pmaxdelta.61573005，RGB global E5.50377035（非GT）；实际加载新RGB资产/源图CPU复算/marker PASS，canonical轴loss.003838724。检查点父exec10940还在等待50k，不提前称父会话退出。

独立40k预览exec86565、15帧GS/Depth/Z exec27161、CPU复算exec21859均真实exit0并关闭。RGB PSNR18.29660847/SSIM.90320820/LPIPS.10866528（foreground10.95702402/bboxSSIM.78457834/bboxLPIPS.22543707）。GS原有向pixel-weighted15.71225125度/覆盖93.93837273%；Depth Normal25.68376882度/覆盖89.29189544%；ExpectedZ.07940795m/MedianZ.08330994m（93.93837%）。missing180=25.67076223度；共同GS+Depth370948/415433、MAE14.73601245/25.68376882度，此中间未生成Mesh，不能当三法线共同口径或正式全七MAE。独立CPU逐数组/帧/聚合和506来源SHA PASS。原GT评分全415433前景保持，无GT标签参与训练/源RGB候选。

实际首中尾对比B09同40k：黑背景正常、人物/平台出现；上衣/裤腿/手臂碎黑斑，脚周与地面黑灰伪形变，当前40k质量FAILED_EARLY。不把较暗的背光帧直接称缺失，GT中帧本身较暗。 首帧GS平台紫蓝、足周杂色和人物大块混色；该图仅含GS/Depth/Z，未包含Mesh；中间Mesh MAE不可用，Depth高频噪声更明显，预测Z/GTZ对比仍有偏差。中间图不作为全量质量通过证据。

另source-only native alpha合成贡献诊断exec37263实际exit0/关闭，三TRAIN0001/0075/0149、0 optimizer/0 GT标签，source RGBconf只贡献自身前景alpha质量38.47498%/28.52117%/39.15403%，既有plane12.06668%/16.42836%/11.84679%。可靠RGB支持29621源opacity中位.012234、mean.253954，数量不等于像素覆盖；近半像素合成仍来自两个约束外的source点。这给出继续加lambda不足的证据，不把子组贡献当正式Normal覆盖或评分子集。

仅source XYZ/原NN全135TRAIN位置稳定/opacity>.05诊断：去掉source feature和轴cos筛选、保留motion-span≤.01，.2m band也只得到16110点（原15187外多923，全部source_static，无新增dynamic）；.05m band12647（原投影重合12051），仅52点重合RGBconf。说明简单去掉轴cos或source静态gate不会解决主要覆盖缺口，不据此改生产plane支持。现有资产/core/所有原点/帧/评分保持，不生成新训练prior/mask。等待本次55k完整20k/15七MAE与最终画质后决定下一迭代，不commit/push/新分支。

## 用户要求停止迭代与已有Jacobian诊断归档

用户明确要求完成当前B10后停止迭代并生成完整总结。B10训练已经完成35000→55000全部20k、exit0/会话关闭；40k/50k/55k审计均exit0，后处理继续至完整七MAE及实际目检。不启动B11或其他新训练/原型。原持续目标已暂停；7°与质量/提交分支条件尚未达成。

此前已完成只读Jacobian诊断exec82276真实exit0/关闭：六TRAIN、原35k全部99827点、0 GT/optimizer，动态近平台37–44%及其余动态44–48%det≤0，可逆动态轴与native来源轴仍差约35–52°。源位置MLP不满足物理可逆体积形变，直接用J逆转置替换法线候选未接受，生产/训练未改；完整来源SHA/六全点数组及数值局限归档1010-19诊断。见修改1010-15。

### B10全量结果（2026-10-10T11:54:26.210739+09:00，人工审核待完成）

| 指标 | MAE | 单位 | 覆盖率 |
| --- | ---: | --- | ---: |
| Albedo MAE | 0.19107403 | linear RGB | 97.566972% |
| Roughness MAE | 0.18083939 | r | 97.768637% |
| GS Render Normal MAE | 15.54006647 | degree | 95.486878% |
| Depth Normal MAE | 25.16398643 | degree | 90.801164% |
| Mesh Normal MAE | 19.75904152 | degree | 95.383130% |
| Expected Z Depth MAE | 0.07666492 | m | 95.486878% |
| Median Z Depth MAE | 0.08065818 | m | 95.486878% |

RGB：{"psnr": 19.754226714113255, "ssim": 0.9193544467290242, "lpips": 0.10111725976069769, "psnr_foreground": 12.278676023812173, "ssim_foreground_bbox": 0.8186597545941671, "lpips_foreground_bbox": 0.2067933867375056}

完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-16文档。

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

## 用户停止迭代后的完整总结（2026-10-10T11:54:52.820177+09:00）

B10全部20k和九步后处理实际exit0/会话关闭；原GS MAE15.54006647°/覆盖95.48687755%，人工质量FAILED、<7°FAILED。停止后续迭代，未commit/push/新分支，持续目标暂停。完整总结：报告/1010-17-jumpingjacks_dir-原生PBR方向光迭代总结报告.md；现有全部失败产物保留。
