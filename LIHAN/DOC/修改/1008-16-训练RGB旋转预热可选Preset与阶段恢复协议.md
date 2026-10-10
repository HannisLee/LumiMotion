# 1008-16-训练RGB旋转预热可选Preset与阶段恢复协议

当前最佳是真实训练模型CV4 49k GS6.534843698°，CV4L 52k GS6.144370784°，两者仍未达到5°。CV4区域减权只得到0.003387503°的GS改善而DN/Mesh略回退；预测深度锚定单项减权GS6.541937721°回退，不采用。所有失败和各模型自己的输出完整保留。原15视角未进入选参或训练；内部11视角曾被Stage1看到，不能作为完全独立测试。

本次只读诊断从真实L52模型取7个optimization RGBA，不读GT normal/depth/material，零优化且不生成新checkpoint。预测材质组0先验角差约0.820–0.821°，其他项旋转梯度约先验的4.3–8.2倍，部分合成方向与先验近乎正交。分组来自训练RGB估计材质，不用GT布料标签。该诊断支持试验先验预热，并非证明法线会达到5°。

新增参数 geometry_rgb_prior_warmup 默认为0；仅新preset lambertian_geometry_area_rgb_warmup 默认2000。先验仍由94张训练RGBA、已知相机/灯光及模型预测位置/可见度推导；已有49k目标和valid缓存与真实52k状态逐位相同，仅把资产元数据绑定到真正52k初始化SHA，保留原49k推导来源。禁止诊断checkpoint作为训练来源或删除diagnostic标记。

开启后前2000步只反传 lambda_geometry_rgb_normal_prior × 原RGB先验loss，只有旋转使用学习率；XYZ/scale/opacity/material/light LR0且grad=None，连同其已有Adam状态冻结。代码prepare先用冻结父旋转建立子片中心偏移，再apply独立旋转；预热不会改变子片中心，但会改变表面片方向和投影覆盖，不能称所有表面几何固定。光度项只作监控。之后同Lambertian联合损失和调度从局部0重新开始。阶段由已完成iteration与来源52k之差计算，恢复不依赖外部初始化、先验或邻域资产；可在预热结束checkpoint停止再续训。

配对：z36_dense52_rgbwarmup_control 与 z37_dense52_rgbwarmup2000，同真实z35 52k、5000总优化步、52→57k，只 geometry_rgb_prior_warmup 0/2000不同。XYZ3e-6、rotation4e-5、scale5e-5、累计XYZ0.01extent/rotation20°/scale1.25、RGB权重2及其原5°目标截断均不变。控制5000联合步；处理2000先验预热+3000联合步；两者共同3000联合衰减调度，控制后2000步保持最终LR。不得把新增预算收益算作预热收益。

102单元测试PASS；旧4/16默认GPU输出和loss逐位一致、梯度容差PASS；真实52k初始化复制/新Adam/有效RGB缓存/旧联合RGB loss/GPU仅旋转梯度及两预测入口PASS。正式前每组40、20+20恢复、独立40重复；处理冒烟预热20、后联合20覆盖阶段边界，额外保存20k增量节点并逐位验证前20步XYZ/scale/material/light及其Adam未变化。原参数/Adam及13视角渲染界限不变，strict1e-6与独立重复范围单独报告；失败不启动正式，不放宽阈值。

正式评测55/56/57k仅内部11，选择其最低signed GS frame-MAE，并记录独立DN/覆盖及布料误差；每个正式模型自己四类图/视频。若出现新的实际最佳，导出该迭代自己HDR与105训练预测深度融合Mesh、独立GS/DN/Mesh及覆盖、三帧和ffprobe/linear目检。执行PASS不等于质量<5。当前短训练/恢复门控PENDING，正式未启动。

实验配置、命令及日志：`output/1007-03-static-geometry-stage2/rgb_warmup52_configs.json`、`run_rgb_warmup52_experiment.py`、`rgb_warmup52_queue.log`；GPU/恢复/短训练均在`output/smoke_test/1007-03-static-geometry`。


首次实际冻结验收FAILED并保留：前20预热Adam仅旋转state2，material/light/scale逐位冻结，但统一project对594个XYZ分量造成最多7.45058e-9变化。原门控拒绝正式启动。新增rotation_only投影，只在预热阶段跳过XYZ/scale/opacity；默认/联合投影保持原算法。不放宽逐位冻结要求，全部旧短训练与失败queue/checkpoint/log原位保留，快照在rgb_warmup52_initial_projection_failure。

新增投影/动量单元用例首次测试布局错误导致103测试1ERROR，保留该日志；修正仅测试布局后103全套PASS，当前证据tests_v28_rgb_warmup_v2_final.log。v2使用同一真实52k和完全相同config/固定bounds，仅修预热投影实现，重新检验两组新代码40/20+20/独立40与13视角及完整实际恢复。v1正式零启动，新GPU/门控/正式当前PENDING，不当质量通过。


v2冻结语义及8次实际短训练已通过；控制恢复max XYZ3.6135e-7/rotation5.0124e-6/scale4.8317e-6，13视角最大5.3093e-5；预热恢复XYZ8.6613e-8/rotation1.9707e-6/scale1.4901e-7，13视角最大8.3745e-6。均在原预声明参数/Adam/render界限内。两组strict1e-6 FAILED；控制自身重复包络PASS、预热自身重复包络FAILED，明确分别保留，不能称轨迹逐位可复现。

实际完整恢复工具由于复制残留将target设为旧46041而低于checkpoint52040，零优化退出，v2正式未启动。只将工具target改为state iteration+1，使用新fixed_target输出；核心103代码/config/源/8次短训练/固定阈值均不改，不重复短训练筛选。v3门控将复用已有正确全部8次v2结果和fixed_target完整恢复；原失败工具/queue日志/checkpoint与等待工具全部保留。当前正式和自己的产物PENDING。


v3新增启动SHA校验误把嵌套sha256清单当平铺键，工具KeyError退出、正式零启动；只修清单访问与README校验工具，新v4事先手工核对6个实际冒烟根启动四核心SHA。核心103/config/来源/8次短训练/恢复/bounds完全不改，无重跑筛选；全部失败日志/工具保留。

v4原界限完整门控PASS，严格1e-6两组FAILED、预热自身重复包络FAILED的限制单独保留。fixed_target实际恢复所有参数/material/light/图/Adam/相机/RNG逐位PASS，即使外部init/RGB/mask路径不可用。两个正式现已真正启动：z36_dense52_rgbwarmup_control 与 z37_dense52_rgbwarmup2000，均从真实52k SHA 629760f7ce15be018d4856432987938450d8399a936062a30b795224a9922bfa初始化，源学习参数/材质/灯光逐位复制、新Adam、RGB资产缓存、调度52起点及实际启动103四核心SHA均PASS。仍同5000总预算、只预热0/2000不同。自己的55/56/57指标、四类/HDR/Mesh均PENDING，不用52源图代替。

全局当前已完成最佳CV4 GS6.534843698°、CV4L GS6.144370784°，各自完整四类/HDR/105预测Mesh已目检，均未达5°。正式新处理模型已在52200进入RGB先验预热，XYZ/scale LR0，先验旋转微LR；这不是法线达标结果。原15仍未用。


根据实际模型预测palette核对颜色：0白/1红/2棕/3灰；此前称组0为红色的文字错误已更正。原数字组统计、训练资产/有效点/配置均不变；不是读取GT材质或调整点筛选。
