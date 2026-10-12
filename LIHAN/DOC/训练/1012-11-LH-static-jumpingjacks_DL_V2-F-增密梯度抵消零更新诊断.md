# 足部增密梯度抵消零更新诊断

主机 `ushas`，环境 `lumimotion-ushas`。**收尾归档状态：未执行。** 按用户最新“继续，收尾实验，归档文档”要求，本项停留在代码实现与CPU静态检查，不启动GPU诊断或新训练。代码、命令和预注册规则仅保留供后续独立授权使用；没有真实135帧统计，没有T05候选科学结论，也不据此宣布足部成因已经证实。

本项设计为只诊断，不创建优化器、不增密／裁剪／重置、不更新模型，不产生新MAE。下述实现行为及产物均为未执行方案，不能视为实际运行证据。

## 固定起点和数据边界

模型为 `output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp`，iteration16000。使用原训练命令绑定的 `output/1012-01-静态DL-两度目标总控/snapshots/stage1-reset-audit-native-assets-20261012-retry1`，其12项科学源码SHA必须一致。原135张TRAIN RGB／自身Alpha、原相机与允许的GT方向均逐文件匹配原137项训练输入。

不读取测试图像、GT深度／法线／材质／阴影／能量或鞋语义。GT方向是原允许输入，不引入其他GT。G2-T06没有optimizer状态，因此本诊断首100窗口对应拟议新训练的零Adam／零增密累计、seed1012独立相机序列，不冒称恢复历史Adam或历史相机余栈。

## 预注册统计与决策

对冻结模型逐帧使用原损失：photo＋0.1×Alpha＋0.1×normal＋1000×distortion，normal→opacity固定detach。一次真实总损失backward后，光度渲染副本的屏幕余切 `go` 已包含photo、Alpha、distortion权重；几何副本的屏幕余切 `gn` 已包含normal权重。不得再次乘权重。

原每点每帧分数为 `norm((go+gn)[:2])`；候选为 `max(norm((go+gn)[:2]), norm(go[:2]))`。只在原visibility_filter内累加，分母每个可见点每视角加一次，阈值严格固定0.0002。候选不改变损失、前向、实际参数梯度或两支路余切本身。

使用与正式新起点相同的 `random.Random(1012)`，对135相机不放回采样。分别保存首100帧窗口和全135帧窗口的原／候选分数、每点可见支持、分数提高次数、额外越阈值点及clone／split分类。分类依据原 `percent_dense×camera_extent`，只预测选择，不执行拓扑；另报告当前opacity／世界scale过滤后可能保留的代理数和split期间预计点数上界。全135汇总是描述性诊断，不等同于实际100步重置累计后的训练过程。

固定模型没有参数演化，所以首100也只检验真实采样窗口下的候选机制能否触发，不冒称已经预测真实16100的点数。

- 两窗口均没有新增选择：否决T05，不通过临时降低阈值制造差异，仅保留同配置续训T04。
- 首100没有新增、全135才新增：不支持首个真实增密窗口，需单独判断，不能拿135均值宣布T05已获支持。
- 首100存在新增：报告数量、可见相机支持、clone／split构成和预计峰值；资源审核前不启动T05。大量新增尤其需要资源分析。
- 诊断执行PASS不等于修复效果PASS；真实训练仍须同一模型原15帧法线与多视角腿足完整联合验收。

## 实现与启动命令

helper为 `LIHAN/scripts/goal3_densification_floor_profile.py`。它调用真实 `bootstrap_1011.py` 完成原生CUDA载入，再只从原T06快照导入科学函数；临时Python模块finder仅定位本helper，不修改旧snapshot。诊断输出锁定 `output/1012-14-LH-static-jumpingjacks_DL_V2-F-增密梯度抵消零更新诊断`。已有plan／逐帧／summary时拒绝覆盖；失败另行登记重试，不删除旧结果。

归档的拟议GPU1命令如下，本轮未执行。后续独立授权后仍须重新确认资源，若换卡须追加实际完整命令与UUID；每卡最多2任务。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-b8e1473e-b813-9d70-aa90-5ebbf901aec4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/goal3_densification_floor_profile.py --model /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp --iteration 16000 --snapshot /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-01-静态DL-两度目标总控/snapshots/stage1-reset-audit-native-assets-20261012-retry1 --output /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1012-14-LH-static-jumpingjacks_DL_V2-F-增密梯度抵消零更新诊断 --seed 1012
```

stdout／stderr保存同目录 `logs/profile.log`，启动环境／命令／退出码由执行Agent保存。脚本在存活期间保存 `/proc/self/maps` 对应native／tracer SHA、实际GPU UUID、PID、环境与解释器，核对冻结native及tracer来源。

## 产物和核验

`plan.json` 保存原命令、实际bootstrap调用、输入／源码SHA和全部采样顺序；`per_frame/*.npz` 保存每帧全部点的两分数、两支路范数及可见mask；`per_frame.jsonl` 保存逐帧前向／参数梯度／屏幕余切不变核验。`first100_points.npz`／`all135_points.npz` 保存逐点累计、分母、分数、选择及新增点支持；`summary.json` 汇总两窗口、无更新不变性与候选决策；`runtime_active.json` 为真实活跃进程来源。

每帧评分前后验证实际RGB／Alpha／GS法线／expected深度／depth法线不变、全部已有参数梯度逐元素不变、两屏幕余切不变；结束验证参数、输入、源码、Python／NumPy／CPU／CUDA RNG前后不变，临时参数梯度全部清除，optimizer始终不存在。图片／视频和四类目检不适用，因为本项只保存统计，没有模型效果验收。

CPU已通过py_compile及11项公式／相机采样／禁止训练变更调用检查；另只读实核12项科学SHA、137项原训练输入SHA全部一致。六项G2-T06初始资产SHA与拟议训练入口一致，临时模块finder在CPU替代bootstrap下通过路由检查。该路由检查不证明真实native或GPU运行通过。

最终归档判定：**未执行（CPU静态检查PASS；科学验收不适用）**。本轮没有真实GPU诊断统计、没有新增模型、没有原15帧评测或四类可视化产物，T05增密下限方案保持未验证。
