# 1011-13-LH-static-jumpingjacks_DL_V2-T07-two_sided配对控制2k训练计划与执行记录

## 研究问题与配对设计

T07作为T06满码RGB单边损失实验的同入口双边损失控制。两组从T03 C10k同一checkpoint开始，使用同一`train_static_dl_censored`入口、同一快照、同一训练source、seed、route、学习率、损失权重、采样和2000步预算。唯一科学变量为`loss_model`：T06=`censored`，T07=`two_sided`；模型输出目录各自独立。

原计划复用T04旧入口的two_sided正式结果，但新入口two_sided 20步与旧T04 smoke未满足预注册同义容差。失败项包括少量PLY位置、albedo与rotation字段，以及step20 normal loss；点数、opacity、scale、deform、light与大部分loss指标相同。完整逐字段数据保存在`output/smoke_test/1011-16-staticDL-T06-two_sided-20步旧控制对照/two_sided_equivalence.json`及独立复算JSON。门槛不放宽，旧T04不作为严格控制，因此增加同入口T07配对控制。

## 固定配置

- 初始化：T03 C10k，`output/1011-15-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH-GPU2重试/model_mlp`，iteration 10000。
- 数据：`output/1011-09-静态DL-Goal训练总控/data_view`，原135帧训练图像、训练方向、RGB和图像自身Alpha；不读取测试图像或GT几何、材质、光强。
- 源码快照：`output/1011-09-静态DL-Goal训练总控/snapshots/stage2-censored-loss-native-assets-20261011`，88项SHA清单；经`bootstrap_1011.py`加载仓库原生光栅。
- 参数：steps=2000，save_steps=500/1000/2000；xyz LR=1e-5、rotation LR=1e-4、albedo LR=1e-3、light LR=1e-4；normal=.1、dist=1000、alpha=.1、shadow_bias=.03、shadow=.25、highlight=.5、residual_threshold=.08；seed=1011、route=control、loss_model=two_sided。
- 固定点数、scale、opacity和形变；不增密，不引入额外曝光、镜面、饱和系数、光强锚定或遮挡XYZ因素。

## 启动门槛与执行状态

T07 20步smoke为`output/smoke_test/1011-16-staticDL-T06-two_sided-20步旧控制对照`，训练exit0、iteration10020稳定。旧T04同义比较FAILED，不放宽。T06 censored20步smoke为`output/smoke_test/1011-17-staticDL-T06-censored-20步审计冒烟`，训练exit0，真实12项审计PASS且零参数更新。两组fresh/fresh parity必须分别实际PASS后，才启动T06/T07正式训练。

正式T07输出为`output/1011-19-LH-static-jumpingjacks_DL_V2-T07-two_sided-配对控制-2k`，GPU计划物理GPU2；T06正式输出`output/1011-18-LH-static-jumpingjacks_DL_V2-T06-censored-单边残差-2k`，计划物理GPU1。标准launcher dry-run已验证T07目标UUID `GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`当时可用，正式启动前仍复查实时显存及占用。完整命令和配置保存在各输出目录`commands/train.json`与`commands/train.sh`。此记录编写时正式训练尚未启动。

正式T06/T07两组10020 fresh parity均实际PASS（八项最大绝对差均为0）后启动。T07于2026-10-11 10:20:48 UTC在物理GPU2启动，UUID `GPU-c64dda28-ea42-7b20-b990-06c1c8f61cd6`，PID 2144679、监督PID 2144666，10:23:29 UTC退出，训练/launcher exit0、`gpu_verified=true`，命令SHA256 `38cb564b2156c7f1bb171dba2aaf7685344878bed614ece73765f89879ce5dcc`。T06同步运行在GPU1。10500/11000/12000均保存PLY、deform和lighting，点数81323；`output/1011-19-LH-static-jumpingjacks_DL_V2-T07-two_sided-配对控制-2k/training_completion_audit.json`记录所有文件SHA并经10秒后二次核验稳定。step2000/image_0069末采样项loss=.0174628、photo=.0113767、alpha=.0111785、normal=.0492656、light=8.350029，非Goal评测指标。12000原协议15帧评测pipeline exit0、完整性PASS、数值FAILED：GS 6.918820°、Depth 13.664233°、Mesh 8.886867°，覆盖率.9994038/.9673233/.9965093。600张PNG和四个150帧视频完整解码，人工代表目检完整性PASS但脚周伪表面/长黑影、接触区外扩及平台皱纹等缺陷明显；分离视图不代表运动分离。fresh parity八项最大差全0；详见`output/1011-19-LH-static-jumpingjacks_DL_V2-T07-two_sided-配对控制-2k/evaluation_12000/README.md`与`FINAL_REVIEW_COMPLETED.json`。最终结论FAILED，完整评测命令、渲染、checkpoint、统计和日志均保留。
