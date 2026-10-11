# 1011-15-LH-static-jumpingjacks_DL_V2-T09-深度统计median1k训练计划与执行记录

## 研究问题与配对约束

验证深度一致性中的统计量选择。初始方案拟从T06@12000继续训练，现已按下文修订采用T06@11000；使用同一新入口、源码快照、初始化、训练数据、seed=1011、route=control及loss_model=censored，除`normal_depth_statistic`分别为expected和median外完全一致。T08使用expected，T09使用median。此次仅比较统计量，不叠加新权重或其他损失变化。

## 配置计划

- 当前初始化：T06@11000，共享静态DL模型，独立新优化器，Adam动量从零开始；权重由已保存checkpoint加载。
- 数据：T06相同原135训练帧 `output/1011-09-静态DL-Goal训练总控/data_view`，仅RGB、原图Alpha和训练光照方向；不读取EL图像或GT法线、深度、材质、阴影、光强作为训练监督。
- 入口：`scripts.train_static_dl_depth_statistic`，通过native bootstrap加载原生栅格。快照及具体CLI待新代码稳定后写入`commands/train.json`。
- route=control，loss_model=censored；xyz LR=1e-5、rotation LR=1e-4、albedo LR=1e-3、light LR=1e-4；normal=.1、dist=1000、alpha=.1、shadow_bias=.03、shadow=.25、highlight=.5、residual_threshold=.08；seed=1011。
- steps=1000，save_steps=250/500/1000，对应全局iteration11250/11500/12000。除`normal_depth_statistic`外参数完全相同。xyz可按既定1e-5更新；点数、scale、opacity和变形分支冻结，不增密。
- 冒烟为20步、保存iteration11020；各自真实入口审计、native来源和fresh parity均PASS后再启动正式1k。


当前共同checkpoint：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-18-LH-static-jumpingjacks_DL_V2-T06-censored-单边残差-2k/model_mlp`，iteration11000；135帧训练source为`output/1011-09-静态DL-Goal训练总控/data_view`。训练只用原始RGB、图像自身Alpha和训练光照方向；GT法线、深度、材质、阴影、光强只允许用于固定评测。

新源码入口`scripts.train_static_dl_depth_statistic`已稳定，py_compile、CLI help和diffcheck通过；经仓库native bootstrap建立独立快照。正式输出预留`output/1011-21-LH-static-jumpingjacks_DL_V2-T09-深度统计median-1k`；20步冒烟预留`output/smoke_test/1011-21-staticDL-T09-median-20步深度统计冒烟`。先分别执行20步实际入口审计与fresh parity，再启动正式1000步。

## 输出与记录要求

正式保存全局iteration11250、11500、12000。每组记录完整训练和评测命令、native source与实际.so SHA、GPU UUID/PID/exit、训练日志与各checkpoint SHA。终点iteration13000执行完整15帧协议、四类150帧视频及代表帧目检；中间12250/12500检查点保留，按需要补完整15帧几何诊断，不预先承诺重复四类视频。定量或完整性失败均标记FAILED并保留输出。当前状态：正式训练尚未启动；新起点smoke及fresh parity待完成。


## 计划修订｜共同起点改为T06@11000

初始计划采用T06@12000。随后完整15帧复评发现T06@11000的GS/Depth/Mesh为5.987634313°/12.988064956°/8.589821770°，三项均优于T06@12000的6.080664354°/13.878983843°/8.852805157°；因此T06@11000支配T06@12000，后续两组采用T06@11000作为共同起点。预算仍为1000步、超参数不变；从iteration11000初始化并保存11250/11500/12000，smoke保存iteration11020。原smoke18/19已从T06@12000自然exit0，统计与公共审计均PASS；它们只证明旧起点入口/统计分支执行，不作为修订后正式门槛或fresh parity证据，全部保留。

新快照为`output/1011-09-静态DL-Goal训练总控/snapshots/stage2-depth-statistic-native-assets-20261011`（91个文件SHA）；新起点smoke分别为`output/smoke_test/1011-20-staticDL-T08-expected-20步深度统计冒烟`和`output/smoke_test/1011-21-staticDL-T09-median-20步深度统计冒烟`。修订后完整命令已在smoke output中准备；实际执行和审计结果待补。
