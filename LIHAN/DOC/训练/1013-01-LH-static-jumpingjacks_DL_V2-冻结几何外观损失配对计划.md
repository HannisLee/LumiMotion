# 冻结几何外观损失配对计划（T01／T02）

2026-10-12 UTC；服务器ushas；Conda lumimotion-ushas。对应新Goal `LIHAN/DOC/GOAL/1013-01-Stage2训练与外观优化迭代.md`。本轮已经独立授权训练与评测，满足实现、真实smoke和重载核验后执行正式实验，不需沿用历史收尾阶段的停止指令。

## 问题与假设

近期DL T13为Stage1 10000步；G2-T06累计16000步；F-T04各保存点累计16500–18000步。上一原生EL配置Stage1 35000、Stage2累计55000（新增20000）并不代表近期DL已跑Stage2；其首次Stage2因CUDA209失败，无55k有效模型。不得将参数中的iterations直接当成实际优化次数。

本轮假设：几何较优后，单独优化外观仍可能提升RGB；原鲁棒满码单边损失与标准PSNR目标不一致，固定几何的单因素损失比较可检验此影响。T01维持原censored，T02使用整张图sRGB MSE；两组共享同一基线、原相机序列、学习率与预算。不得把不同损失数值直接互相比大小。

## 冻结来源与训练边界

source：`output/1011-09-静态DL-Goal训练总控/data_view`。初始化整模型：`output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp`，iteration16000。原三MAE为2.6683778383745302／5.404577799172339／4.20848728697561°。F-T04@16500另列参考，不拼接最佳单项。

仅加载原135 TRAIN RGB及其Alpha、相机和GT方向。训练目录137输入SHA逐项匹配源记录；方向文件拒绝光强/颜色等字段。测试、GT几何/深度/法线/材质/阴影均不进入训练。所有基础几何、opacity、feature、形变和点集冻结，只优化原共享每点albedo DC与全局正光强；无增密、裁剪或透明度重置。不新增逐帧可学习颜色或阴影图。

原模型无Adam资产，首轮明确新建Adam；新入口保存外观Adam、随机状态和相机余栈，以便后续真实接续。没有BRDF/roughness优化，本轮仅为Stage2外观基线消融，不充作完整PBR方案完成。

## 预算与输出

先独立smoke检验真实梯度、前向来源、冻结不变性、保存和fresh重载；smoke不计入正式预算。

- T01 smoke：`output/smoke_test/1013-01-LH-static-jumpingjacks_DL_V2-T01-censored外观smoke`。
- T02 smoke：`output/smoke_test/1013-02-LH-static-jumpingjacks_DL_V2-T02-MSE外观smoke`。
- T01正式新增2000步：`output/1013-03-LH-static-jumpingjacks_DL_V2-T01-冻结几何censored外观2k`。
- T02正式新增2000步：`output/1013-04-LH-static-jumpingjacks_DL_V2-T02-冻结几何MSE外观2k`。
- 总控：`output/1013-02-LH-static-jumpingjacks_DL_V2-Stage2外观优化总控`。

正式重新从16000起点训练，保存相对500／1000／2000（累计16500／17000／18000）。GPU按显存和负载动态选，单卡本实验最多2个任务，禁止停止别人的任务。实际完整argv、快照manifest/source SHA、PID、GPU UUID、日志和真实exit code后续在本文件和输出README追加；尚未运行不得写PASS。

## 原协议评测与验收

完整原15测试10、20……150、原MAE像素掩膜/有效性/TSDF配置不变；GS／Depth／Mesh必须对应同一checkpoint。主PSNR为原15帧各全图sRGB PSNR的算术平均，并列前景辅助与原图满码比例。先CPU PNG复核历史参考（标注量化），正式候选以同一渲染前向float指标对照。图像保存截断不能隐式改变float指标；需要另报显示域clamp结果并说明采用何种PSNR口径。

冻结几何的实际张量hash和重新加载后的前向Normal／Depth／Alpha应与基线一致。三MAE以原基线不退化为门槛，只允许测量浮点容差并透明记录，不沿用足部旧Goal较宽容忍来宣布历史最佳保持。RGB改善需结合全部固定测试视角与四类图（RGB、Alpha、Normal/几何、Separation）目检，不只选择俯视。腿／足连续性、鞋尖短缩、黑边与阴影单列，固定几何无法填补缺失鞋尖，最终完整几何仍需独立判断。

按验证和收敛选择最佳保存点；若2k有收益且尚未平台，可接续至外观新增10k/20k并保留中间点；若损失改变不能改善外观，按阴性结果记录，转向BRDF／可见性／低学习率几何诊断。35k／55k不是硬性执行承诺，不以预算终点代替验收。

## 当前状态

PREPARED：新入口和独立评测实现中，尚无smoke、正式训练和新PSNR/MAE。所有旧成功/失败结果保留。本轮目标尚未达成。

## 预注册量化判据与冻结核查

基线三精确值作为严格不退化参考，另用1e-5°工程重复误差容差判断测量一致性；原始差值、strict_non_degradation及within_tolerance同时记录，不将微小正差称为科学改善。显示域clamp[0,1]后float全15逐帧PSNR均值提升>0.05dB作为首轮明确外观收益，前景辅助允许差值≥−0.05dB；raw float PSNR并列，以识别满码权衡。两阈值是工程筛选标准，非统计显著性。完全冻结的几何一致性及多视角目检仍单独验收。未达到0.05dB的小收益也完整记录，不将首轮筛选阈值当用户最终优化上限。

新训练保存appearance_state精确albedo/light以消除PLY序列化误差；fresh需从基线PLY重建全部冻结参数并应用state，另验证新PLY物理前向。核对scene/gaussian_model.py实际源码后确认：save_ply保存albedo_dc_stage1字段，load_ply优先重载该字段，缺字段才从当前DC派生。因此新PLY有字段时也须严格验证冻结副本；不能把派生fallback误称真实训练改动。该副本不参与本轮直接光着色。native三来源字段必须与T06保存记录完全一致。

## 实际110步smoke命令（执行前绑定）

快照 `/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-02-LH-static-jumpingjacks_DL_V2-Stage2外观优化总控/snapshots/appearance-fixed-geometry-v1`，manifest SHA 3552134cd9c8bf927aaea1c8f3ff267afcfc8bf52cf66f20f7eb8089860c5020，entry SHA a166553c2408bfcca7051dacad3403a2aefb1aba25de03c4a1137c948bf3075e；两个真实保存点16020/16110，物理GPU3。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal4_appearance_train_job.py --command '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-02-LH-static-jumpingjacks_DL_V2-Stage2外观优化总控/commands/t01_smoke.json' --output '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1013-01-LH-static-jumpingjacks_DL_V2-T01-censored外观smoke'
```

训练完整argv：

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-02-LH-static-jumpingjacks_DL_V2-Stage2外观优化总控/snapshots/appearance-fixed-geometry-v1' scripts.train_static_dl_appearance --init_model '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp' --init_iteration 16000 --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1013-01-LH-static-jumpingjacks_DL_V2-T01-censored外观smoke/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view' --loss_model censored --steps 110 --save_steps 20 110 --seed 1012 --smoke
```

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal4_appearance_train_job.py --command '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-02-LH-static-jumpingjacks_DL_V2-Stage2外观优化总控/commands/t02_smoke.json' --output '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1013-02-LH-static-jumpingjacks_DL_V2-T02-MSE外观smoke'
```

训练完整argv：

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1013-02-LH-static-jumpingjacks_DL_V2-Stage2外观优化总控/snapshots/appearance-fixed-geometry-v1' scripts.train_static_dl_appearance --init_model '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp' --init_iteration 16000 --model_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1013-02-LH-static-jumpingjacks_DL_V2-T02-MSE外观smoke/model_mlp' --source_path '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view' --loss_model mse --steps 110 --save_steps 20 110 --seed 1012 --smoke
```


## 首次smoke执行失败与修复方向（2026-10-13）

GPU3两child PID2974374/2974373，均真实exit1，GPU未分配且未创建model_mlp，0更新。严格来源断言正确拒绝已重编译的共享native.so；原439个源码条目SHA全部一致，但binary f9514476...→ead16554...。原失败目录与两完整命令保持，不覆盖。

独立复核build.ninja发现base Conda CXX/CC被继承；修复方向为监督器显式绑定ushas环境工具链、CUDA_HOME和私有TORCH_EXTENSIONS_DIR。在私有缓存先核是否恢复历史二进制身份；仅通过后重跑新retry1烟测/独立fresh。不盲目删除native检查，尚未正式训练或有新指标。

## 按用户要求收尾，后续计划停止

用户2026-10-13要求收尾当前目标。停止新增retry/正式训练和GPU评测；T01/T02两正式2k均UNEXECUTED，smoke仅两原FAILED、均0更新。未取得新checkpoint、原15三MAE、float PSNR或完整脚形改善。原预注册作为历史保留，不能据此自动开跑。已运行私有编译probe的终态由evaluation_closeout记录，不将编译通过等同恢复旧binary或训练成功。

CXX因果边界：当前build.ninja使用base编译器属实，历史另一个成功运行也曾使用base前缀，尚不能证明该前缀是唯一原因。核实事实是同439源码SHA、实际缓存二进制已替换；确切编译差异未定位完成。收尾报告将如实保留此局限。


### 2026-10-13最终审计收尾

用户要求停止新增实验并完成收尾。当前实际优化更新为0，无新checkpoint或新GPU评测；科学目标未达成。真实探针／smoke均在渲染前因native二进制身份变化退出1，私有重编译也未恢复历史SHA。所有失败与历史证据原位保留，不启动预留retry或正式训练。原会话现为idle，目标状态库已无本Goal行；本文旧ACTIVE／进行中描述只代表当时历史，不代表现在仍在运行。

运行审计和最终归档见`LIHAN/DOC/报告/1013-05-两个Goal长时低产出审计与最终收尾.md`。审计总控为`output/1013-05-两个Goal运行审计与最终收尾`；原位清单和复核凭据在两个原总控`archive_closeout/manifest.json`、`ARCHIVE_COMPLETED.json`，全分支同步凭据为`shared_sync_selected.json`。归档PASS不等于科学目标PASS。
