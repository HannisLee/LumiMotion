# 1011-12-LH-static-jumpingjacks_DL_V2-T06-censored-满码RGB单边残差2k训练计划与执行记录

## 研究目标与单因素约束

检验原始训练RGB精确满码通道的单边残差。共同起点为T03 C10k checkpoint、route=control、135帧原训练source、seed、学习率、损失权重和采样相同；T06采用 `loss_model=censored`，T07采用同一新入口的 `loss_model=two_sided`，正式参数除loss_model与输出model_path外完全相同。算法定义、满码规则及边界见修改文档`LIHAN/DOC/修改/1011-21-静态DL满码RGB单边损失独立对照与数值审计.md`。未饱和观测仍惩罚过亮预测；满码只逐RGB通道停止上界惩罚，背景和完整图像分母保持不变。

共同初始化为T03 C10k `output/1011-15-LH-static-jumpingjacks_DL_V2-T03-方向Lambertian阴影BVH-GPU2重试/model_mlp`，iteration10000。只读取原135训练图像、训练方向、RGB与原图Alpha；不读测试图像或GT几何/材质/光强。原计划用旧T04作two_sided控制，但新入口two_sided20与旧T04 smoke在预注册容差下未通过：部分PLY坐标/材质/旋转字段及step20 normal项超限，虽点数、opacity、scale、deform、light和多数loss项一致。保留该FAILED证据，不放宽门槛，也不将旧T04当作可比控制；因此补做同一新入口、同一C10k起点的T07 two_sided正式控制，与T06形成严格配对。

## 固定正式参数

两组正式训练均steps=2000，save_steps=500/1000/2000（全局10500/11000/12000）；xyz LR=1e-5、rotation=1e-4、albedo=1e-3、light=1e-4；normal=.1、dist=1000、alpha=.1、shadow_bias=.03、shadow=.25、highlight=.5、residual_threshold=.08；seed=1011、route=control。T06 loss_model=censored，T07 loss_model=two_sided。固定点数/scale/opacity/形变，不增密；独立无优化器动量。

## 启动前门槛

先做two_sided20与旧T04 20步同义检查；预设门槛为PLY浮点字段及光强atol2e-5/rtol1e-5，opacity/scale/点数/deform精确相同，step1/20 loss/photo/geometry为atol2e-6/rtol1e-5。该检查已FAILED，门槛不放宽。censored20入口自动执行真实checkpoint12项路由/相机/满码残差审计且不得更新参数；满码且prediction>=1通道导数为零，其他通道逐元素等于旧函数导数，已PASS。T06/T07新入口smoke均须经evaluator fresh parity实际PASS后才启动正式2k。两正式组唯一差异为loss_model与model_path；不引入镜面、曝光、饱和系数、光强锚定或遮挡XYZ。

## 核心快照与输出

新快照：`output/1011-09-静态DL-Goal训练总控/snapshots/stage2-censored-loss-native-assets-20261011`，manifest包含88个逐文件SHA；新入口`scripts.train_static_dl_censored`通过`bootstrap_1011.py`绑定仓库native rasterizer。两组smoke：`output/smoke_test/1011-16-staticDL-T06-two_sided-20步旧控制对照`、`output/smoke_test/1011-17-staticDL-T06-censored-20步审计冒烟`。正式输出：T06 `output/1011-18-LH-static-jumpingjacks_DL_V2-T06-censored-单边残差-2k`，T07 `output/1011-19-LH-static-jumpingjacks_DL_V2-T07-two_sided-配对控制-2k`。完整argv及launcher配置分别保存在对应目录`commands/train.json`。

## 训练前审计

CPU精确数值探针已由planner在`output/smoke_test/1011-15-staticDL-满码单边损失CPU数值审计`完成PASS；CLI help/bootstrap import检查通过。smoke16/17训练均exit0，censored真实12项审计PASS；旧T04同义检查FAILED且保留。T06/T07 fresh parity均实际exit0/PASS，八项最大绝对差均为0。正式T06已于2026-10-11 10:20:48 UTC启动，物理GPU1 UUID `GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4`，PID 2144680、监督PID 2144665，`gpu_verified=true`，命令SHA256 `43ac75c22d9adb1dec781c2e906fb810b14d13eb74e9a47c54f2a7a67163b4b6`；与T07配对运行。训练尚在进行，退出码和三个稳定点SHA待补。继续记录native source与.so SHA、训练指标、完整15帧三项MAE、覆盖率、代表图/视频、四类目检与最终PASS/FAILED。失败输出保留不覆盖。
