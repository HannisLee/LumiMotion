# 1011-12-LH-static-jumpingjacks_DL_V2-T06-censored-满码RGB单边残差2k训练计划与执行记录

## 研究目标与单因素约束

检验原始训练RGB精确满码通道的单边残差，是否能减小T04 control短程Stage2的几何退化。沿用T04同一C10k checkpoint、route=control、135帧原训练source、seed、学习率、损失权重和采样；仅改 `loss_model` 从 `two_sided` 为 `censored`。算法定义、满码规则及边界见修改文档`LIHAN/DOC/修改/1011-21-静态DL满码RGB单边损失独立对照与数值审计.md`。未饱和观测仍惩罚过亮预测；满码只逐RGB通道停止上界惩罚，背景和完整图像分母保持不变。

共同初始化为T03 C10k `output/1011-15-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH-GPU2重试/model_mlp`，iteration10000。只读取原135训练图像、训练方向、RGB与原图Alpha；不读测试图像或GT几何/材质/光强。旧censored以外控制T04 `output/1011-16-LH-static-jumpingjacks_DL_V2-T04-静态分流control-2k` 的正式12000结果为配对参考，不再跑重复正式two_sided组。

## 固定正式参数

正式候选steps=2000，save_steps=500/1000/2000（全局10500/11000/12000）；xyz LR=1e-5、rotation=1e-4、albedo=1e-3、light=1e-4；normal=.1、dist=1000、alpha=.1、shadow_bias=.03、shadow=.25、highlight=.5、residual_threshold=.08；seed=1011、route=control、loss_model=censored。固定点数/scale/opacity/形变，不增密；独立无优化器动量。

## 启动前门槛

先运行two_sided20与旧T04 20步checkpoint同义检查；PLY浮点字段及光强要求atol2e-5/rtol1e-5，opacity/scale/点数/deform精确相同，step1/20 loss/photo/geometry为atol2e-6/rtol1e-5，超过即查因不放宽。再运行censored20，入口自动执行真实checkpoint12项路由/相机/满码残差审计且不得更新参数；检查满码且prediction>=1通道导数为零，其他通道逐元素等于旧函数导数。两组新smoke均经evaluator fresh parity后才可启动正式2k。训练参数唯一改变为loss_model；不引入镜面、曝光、饱和系数、光强锚定或遮挡XYZ。

## 核心快照与输出

新快照：`output/1011-09-静态DL-Goal训练总控/snapshots/stage2-censored-loss-native-assets-20261011`，manifest包含88个逐文件SHA；新入口`scripts.train_static_dl_censored`通过`bootstrap_1011.py`绑定仓库native rasterizer。两组smoke：`output/smoke_test/1011-16-staticDL-T06-two_sided-20步旧控制对照`、`output/smoke_test/1011-17-staticDL-T06-censored-20步审计冒烟`。正式输出预留：`output/1011-18-LH-static-jumpingjacks_DL_V2-T06-censored-单边残差-2k`。完整argv及launcher配置分别保存在对应目录`commands/train.json`。

## 训练前审计

CPU精确数值探针已由planner在`output/smoke_test/1011-15-staticDL-满码单边损失CPU数值审计`完成PASS；CLI help/bootstrap import检查通过。新的GPU快照/训练结果尚待运行；正式候选尚未启动。运行后记录GPU UUID、PID/退出码、native source与.so SHA、初始/结束checkpoint、三个稳定点SHA、censored诊断指标、完整15帧三项MAE、覆盖率、代表图/视频、四类目检与最终PASS/FAILED。失败输出保留不覆盖。
