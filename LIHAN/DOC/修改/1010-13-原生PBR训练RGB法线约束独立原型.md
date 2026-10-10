# 1010-17 dir 训练RGB法线约束独立原型

状态：PREPARED_SOURCE_RGB_PROTOTYPE；未改生产、未开始新完整训练，B09正式MAE仍待当前原后处理进程完成。2026-10-10T10:38:14.411234+09:00

已先读取GOAL，B09完整20k实际exit0但RGB/四类实际质量FAILED_EARLY；不能仅继续放大gate更新认定达到7度。已有135 TRAIN RGB线性可观测性只读诊断显示约40948点可拟合，source-local四元数运输与来源矩阵一致，但原additive大部分10度响应约9.1度，不能称原加法大范围失效。

本独立原型验证一个有观测来源的候选：只从原35k预测PLY/冻结来源NN、全部135 TRAIN RGB与自身Alpha/相机time/白色单位GT方向，估计来源canonical轴的RGB光照方向约束。GT albedo/normal/roughness/depth/mesh/blend/原光颜色能量不读。RGB估计方向是原训练图像派生，明确不称GT法线；全部99827点保留，可靠性权重不改变原photo loss/测试评分/帧。135个TRAIN键不含15 TEST；逐图像SHA/RGBA alpha和来源SHA绑定。

原型仅使用35k source XYZ/q/feature/scale/opacity与既有source-only plane，40k只作为固定来源NN/相机容器，学得40k XYZ/q/feature/material不用于构建。投影/来源方向运输逐点核验，source预测Z可见性.025m/ownAlpha≥.99/source rasterAlpha>.9。线性luminance最小二乘只作弱Lambertian方向候选，>=12观测、gram最小特征值>.01最大、系数norm>.01、relative residual<.3；原plane支持不重复施加该候选。原SHA/plane支持保持，所有完整点数组保存，未可靠点权重0而非删除点。此不是已通过GGX遮挡拟合或GT准确率。

必须先审计候选来源、有限性/维度/不含测试帧、confidence与原只读诊断逐点复算；后续若接入source-local渲染和训练，需独立Preset、全部入口/保存加载、CPU/GPU冒烟与完整20k/15帧七MAE/四类/HDR人工验收。B09生产474 SHA与当前后处理不改，不提前提交或宣称7度。无完整Stage2训练，无正式MAE结论。

## 首次生成失败保留

exec23962实际exit1并关闭；source PLY错误使用原型ROOT。未生成资产/未改变生产。初次脚本与logs/build_constraint.log保留，独立retry修正BASE并使用实际native_probability精确源gate。尚无完整训练/MAE。

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
