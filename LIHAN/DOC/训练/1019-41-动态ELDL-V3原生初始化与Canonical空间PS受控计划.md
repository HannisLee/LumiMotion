# 动态ELDL-V3原生初始化与Canonical空间PS受控计划

日期：2026-10-19 UTC。状态：执行前预登记；完整目标尚未完成。

## 目标与禁止事项

完整需求来自附件 `3196f07e-7f67-496f-a823-7d34efce3038/pasted-text-1.txt`。新数据从头完成原生动态Stage1；Stage2实际运行SH环境辐照度加可见度Lambertian与纯可见度Lambertian，并分别关闭、开启独立Canonical空间PS。另做一组主线极小灯向学习率对照。本轮不做Stage3/PBR，不复用静态B00或任何旧模型初始化。GT/TEST指标不得进入训练、选择和停止决策。

## 服务器、模型与输入

已实测服务器ushas；唯一Python为 `/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。GPU2空闲；GPU0/1/3分别运行santo的任务，不干预。每卡最多3训练，启动前再次查询资源。

主Agent服务端实际模型ID当前接口不可读，不能声明已核验为6.1 sol/high。Training Monitor请求模型ID为可选的`gpt-6-luna`、medium；该Agent同样无法独立读取服务端采样元数据。Ostral与Soul 6.1不在可选ID中，已向用户报告并询问两个角色是否允许使用`gpt-6.1-sol`/medium；未静默替换。

源：`data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3`。150帧，800×800，30fps，原始16位RGBA、world空间有符号法线。元数据记载动态相机、骨骼和混合光照；不得把它视为单一方向光数据。

固定划分：TEST为10、20、…、150，共15；DEV为61至79间非10倍数，共18；TRAIN为其余117帧。Stage1与后续任何估光/伪法线拟合只读取这117帧，DEV只用于冻结配置后的比较，不参与Stage1优化。整个序列使用`(frame-1)/149`，各子集不能重新归一化。

image的Alpha实测全满码；原始物体Alpha来自albedo PNG的Alpha通道。仅打包该Alpha与原image RGB；albedo RGB不进入统计、点云或训练。16位精度和缩小前满码通道标记保留。Stage1遵循原生码值损失；Stage2统一逆sRGB至线性后进行着色和Alpha合成。TRAIN亮度均值与分位使用原Alpha、固定每帧8192样本，等帧权重，seed1019；q_base=P50、q_bright=P90。

## 原生Stage1固定预算

入口：新隔离包装器调用未经修改的`scripts/train_stage1.py::Trainer`。该文件与`original`分支无差异。包装器不修改损失、前向、几何、静动态分离、外观头、增密、透明度重置或学习率。只禁止加载外部资产、禁用TEST回调、核验输入并记录状态。

100k原生随机点云，seed0，范围[-1.3,1.3]^3；无历史权重、GT几何、GT法线或GT灯向。其随机初始化单独生成，不把旧点云复制为初始化。deformation随机初始化，全部原生参数参与优化。

训练35000步，resolution2（400×400），SH最高3阶，depth_ratio1；densify_until_iter20000；normal开始阈值8000；lambda_dist1000、normal权重0.02；lambda_separation0.001、d_xyz_loss_weight0.001、d_color_reg_loss_weight0.01；binarization_warm_up1000；其余沿原生默认值并保存完整实参。保存10000/20000/30000/35000与原生warmup资产。原生终步保存但不optimizer.step，因此35000标记对应34999次优化器更新，所有对照需声明。

先20步独立冒烟，输出到`output/smoke_test/1019-*-LH-dynamic-jumpingjacks-ELDL-V3-原生Stage1输入冒烟`，保存与重载检查。通过后单独新目录启动35000步；禁止把smoke续训为正式Stage1。正式目录预定`output/1019-59-LH-dynamic-jumpingjacks-ELDL-V3-N00原生Stage1`。累计/新增步数与权重加载状态由marker记录，最终Stage1重载和动态法线检查仍必须另做。

## Stage2首轮固定设计（尚未满足启动门槛）

下面为本轮预登记设计；在动态法线/完整场景BVH/PS独立梯度冒烟通过、TRAIN可辨识性核验完成前不启动正式四组。如接口证据迫使修订，先写新预登记，不能根据TEST/GT结果改计划。

四组均从N00@35000独立重载；每组新增8100次真实优化器更新，保存1350/4050/8100，最终点固定8100；不增密、不分裂，保持track ID稳定。前1350冻结全部几何与deformation完成外观适配；1351开始开放canonical XYZ、rotation、scale；4051开始小步开放deformation。所有组共享相同采样seed1019、完整前景RGB/Alpha损失和原有几何一致性。A0/A1只有PS权重不同，B0/B1同理。

环境分支为world法线共享2阶辐照度SH（每通道9系数），softplus非负，禁止逐Gaussian/逐视角环境颜色。A先在固定几何拟合环境基础，再冻结SH执行方向分量残差阶段，完整前向仍为EL+lambda_DL*DL；lambda_DL共享固定0.5。B无SH/环境常数/隐式环境扣除。DL为rho/pi*D*V_light*relu(n·L)，D为全局受限标量，范围[0.1,20]；rho在[0.03,0.97]，用共享尺度规范约束其TRAIN可见表面中位数为0.5，杜绝逐帧曝光。

初始材质lr0.001、SH lr0.001（仅基础阶段）、D lr0.001；canonical XYZ lr1.6e-6、rotation1e-4、scale2e-4、opacity5e-4；deformation为原生当前学习率的0.05倍；所有组共用法线一致性0.02、distortion1000与Alpha0.1。光度完整前景损失保持，亮区软权重只提高训练优先级；饱和通道采用下界删失。透明度3000/6000重置，后200步normal路径不更新opacity，RGB/Alpha仍可更新。

灯向只用新TRAIN、当前时刻完整动态几何与预测阴影估计。若TRAIN证据需要时变方向，则每TRAIN时刻一个单位方向，按全序列time线性插值并归一化，时间平滑权重0.01；不得复制旧灯向或假定世界固定。四组先冻结同一份估计Lt和全局D初始化。BVH遮挡使用完整当前G_t，不等同相机可见性。初始缓存/几何更新需明确重建规则，不能借用静态单帧缓存。

PS在固定当前deformation/Lt/visibility下，使用真实跨时刻图像观测估计局部canonical伪法线，目标detach。A扣除受限且detach的EL预测并声明误差；B直接使用观测。至少6条可靠观测；有效方向矩阵最小/最大奇异值比>=0.05；相对深度误差<1%、前景3×3内缩、预测Alpha>0.95；饱和、混合边界、近零直射光不得除以V放大。PS权重0.02，几何解冻后启用。记录有效覆盖、方向条件数、运输与真实切平面一致性、PS对实际XYZ/rotation/scale的梯度；明显非刚性处使用完整局部Jacobian逆转置运输并删失病态区域。

主线由TRAIN/DEV完整光度与稳定性、覆盖率和预登记几何代理选择，四组终点全部保留。随后仅主线追加独立一组从同一N00起点的8100步微调对照，灯向lr1e-5、角偏离惩罚0.1、时间平滑0.01。方向梯度不穿透BVH时明确固定visibility近似，候选需重算真实遮挡。TEST/GT不能反调选择。

## 评测与交付门槛

EL_V2真实旧Stage1为`output/1011-04-LH-dynamic-jumpingjacks_EL_V2-原生LumiMotion两阶段/stage1_frozen`，35000；已有metrics和15帧NPZ可追溯。旧GS/ExpectedDepth/Mesh条件MAE分别5.661060810285603、10.552236907360648、7.550293044074406度，不能当作新固定分母结果。没有核验到旧综合定义，预登记三项等权均值；统一固定GT前景分母，缺失预测按180度计，另外报告原条件MAE与覆盖率。新旧均使用15个相同frame ID、400×400、原world方向/朝向与相同Alpha阈值。统一评测脚本和重算结果未完成前不宣称超过旧基线。

旧time为frame/150，新time为(frame-1)/149；每个模型按实际训练时间映射渲染相同物理帧，禁止直接复用错误fid。原始c2w逐帧相同已初查；骨骼、网格拓扑、区域与Alpha对应还需完整核验。旧Stage1用135帧，新Stage1用117帧，存在预算/数据划分差异。必须保留新Stage1同数据对照并报告跨数据归因边界。

完整报告：三法线、Expected/Median/Mesh深度、人体/腿足鞋局部、完整性/动态抖动、RGB暗中亮区；EL/DL分量、正侧背/鞋尖与连续动作视频。所有实验README记录完整训练和渲染命令、source/model/iteration、日志/输出、指标、代表产物、四类目检、最终PASS/FAILED。未运行项目标注待完成，不能冒充通过；失败资产全部保留。

正式输出与审计在`output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控/`。当前尚无本轮Stage1终点、Stage2或PS收益证据，完整Goal保持ACTIVE。

## 首次执行收据

原生20步工程冒烟位于`output/smoke_test/1019-244-LH-dynamic-jumpingjacks-ELDL-V3-原生Stage1输入冒烟`，实际19次optimizer更新；三个TRAIN帧1/81/149重载与动态梯度通过。原生warmup权重尚未形成可用重建，目检质量FAILED，不能把工程PASS当科学通过。XYZ、rotation、scale、opacity、形变warp与rotation头梯度有限非零；形变scale头为零，保留该事实。完整遮挡和PS仍待验证。

正式N00已由`dynamic_v3_job.py`在GPU2启动，启动时资源为1/49140MiB、利用率0。固定训练35000步，不从smoke权重加载。实验`output/1019-59-LH-dynamic-jumpingjacks-ELDL-V3-N00原生Stage1`；监督器初始PID2301541，训练PID2301614，实际活状态须查看`job.json`结合`ps`与`train.log`，不能仅凭PID文件判定运行。完整命令为总控`planned_commands.json`及实验`command.json`；日志`train.log`、`supervisor.log`；结束退出码写`job.json`。不自动重启失败实验。

旧EL_V2完整15帧历史预测由同一CPU评测函数复算，条件GS/Depth/Mesh分别5.661060810285603、10.552236907360650、7.550293044074406；固定分母分别5.947001855919448、16.394499793395735、8.864881619570324；综合10.402127756295170度。报告见总控`eval_only/ELV2_historical_fixed_denominator_v1/audit.json`。该值仅证明历史NPZ、固定checkpoint与指标计算一致；仍需新运行时统一重渲染确认，不能混称该工作已完成。

逐帧原始camera c2w与共同物体world矩阵最大差均为0；旧150帧均缺少可比较bones字段，不能证明蒙皮动作和网格相同，评测区域等价尚未证明。

Planning/Evaluation指定ID不可用已报告，等候用户偏好期间已明确告知采用`gpt-6.1-sol`/medium进行只读独立审查，原角色模型匹配不成立。独立审查指出静态GT构建固定复用frame1 mesh，禁止直接用于本动态序列，需逐时刻GT；此修复只作用隔离评测。
