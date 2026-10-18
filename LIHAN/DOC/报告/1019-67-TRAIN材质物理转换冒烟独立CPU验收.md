# TRAIN材质物理转换冒烟独立CPU验收

日期：2026-10-19（Asia/Tokyo；主机运行日志为2026-10-18 UTC）。主机ushas，环境`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。

最终结论：**PASS_ENGINEERING_ONLY；精度未评**。本次只读smoke18—23、隔离135TRAIN来源、既有配对收据及实际源码，未使用GPU、GT或外部TEST，未训练、重跑模型、导出图像或作目检。该结论仅证明smoke的材质转换和物理训练接续工程，不能把20步smoke用于正式初始化，不能声称科学目标通过。

## 冻结输入与命令

- 短组：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1019-22-ELDL-V3-短SH预适配材质物理转换/model_mlp`，控制smoke18 actual0，输入smoke20导出DC actual20，点数100000。
- 长组：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1019-23-ELDL-V3-长SH预适配材质物理转换/model_mlp`，控制smoke19 actual0，输入smoke21导出DC actual20，点数32131。
- source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_train_eldl_raster_v1`，135个已见TRAIN；仅核查来源和metadata，未打开图像。
- 两处理均state exit0，`completed_lr_audit.json`均为PASS，实际20事件、7个物理保存（1/8/9/15/16/17/20）。0/1/20模型保留；训练输出与日志原地不变。
- 完整CPU命令：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/commands.txt`；独立环境：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/environment.json`。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/audit_material_physical.py' > '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/audit.log' 2>&1
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/supplement_v2/audit_dc_serialization.py' > '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/supplement_v2/audit.log' 2>&1
```

## 原始失败保留与分类修正

首轮审计共197项，195项PASS、2项FAILED，原脚本、日志和收据均保留，未覆盖：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/receipt.json`，SHA256 `82bec4e3e622a7942bb8e67076dd8b230cff11b0d2ac6f33c81445358cb7a791`。原审计仅将`albedo_dc_0..2`作为DC，误将另外6个序列化副本列视为非DC。两个失败都仅涉及`albedo_dc_stage1_0..2`及`f_dc_0..2`。

实际运行snapshot与当前源码逐字节相同。`GaussianModel.save_ply`将`f_dc`再次写为主DC；对象没有`_albedo_dc_stage1`时，stage1三列回退写主DC。GeometryWarmstart构造明确禁止独立stage1对象/优化器组，物理前向使用主DC的`get_albedo`。因此这6列是本构造的既有DC副本，不能要求它们在更换主DC时继续保留灰值。该解释只适用于已核实的此构造；真实独立stage1参数变化不能获得豁免。

按主控确认新增v2补充，不改变模型、训练、损失、数值阈值或任何科学协议。v2逐列证明：新6列全部等于对应输入DC，旧6列全部等于旧主DC，其余60个字段全部逐值相同。两组均没有独立stage1优化器组。v2还重新核对v1的119个来源文件SHA及既有生产重载记录，159项全部PASS。最终补充收据：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/supplement_v2/receipt.json`，SHA256 `730e046181b2647aef055d0862b1e6a69f8463512b2fee4755c79f604fa12d10`。

## 实际核查结果

| 项目 | 短SH22对照18/输入20 | 长SH23对照19/输入21 |
|---|---:|---:|
| actual0点数/点序及四几何effective包 | 100000，逐值相同 | 32131，逐值相同 |
| 除DC及其6个副本外的60个PLY字段 | 全部逐值相同 | 全部逐值相同 |
| 主DC与导出20/effective包 | 全部逐值相同 | 全部逐值相同 |
| 相对旧灰色DC变化元素 | 297402/300000 | 96393/96393 |
| 最大DC变化（原始SH系数） | 0.07801683247 | 0.07815981656 |
| zero Adam状态/参数组/相机栈 | 空/与旧控制相同/空 | 空/与旧控制相同/空 |
| Python/NumPy/Torch/CUDA四RNG零步 | 与旧控制逐值相同 | 与旧控制逐值相同 |
| xyz累积/denom/maxRadii增密统计零步 | 全0 | 全0 |
| MLP：新0对旧0、新0到1/8/9/15/16/17/20 | 全部逐值相同 | 全部逐值相同 |
| 首20次真实相机 | 等于各自控制且两处理互相相同 | 同左 |
| 终点真实Adam state.step | xyz/DC/opacity/scaling/rotation均20 | 同左 |
| 非活跃rest/roughness/feature及light状态 | 全空 | 全空 |
| normal透明度保护 | opacity梯度缺失，三几何梯度非零 | 同左 |
| 已保存actual0生产重载6条 | 全部最大差0 | 全部最大差0 |
| fixed_eldl_record_files真实CPU调用 | 新材质入口/SHA通过 | 同左 |
| eval.model_record真实CPU调用actual0/20 | eldl_fixed、父权重和材质身份通过 | 同左 |
| 正式评测入口拒绝smoke | 通过 | 通过 |

step1真实PLY更新最大绝对值：两组XYZ约0.000800014、scale约0.01000023、rotation约0.001000047、DC约0.010000002；opacity短组0.04999995、长组0.05000019。它们证明实际优化更新发生，不是梯度有限差分精度结论；既有几何exact-FD失败没有被本次颜色/接续检查改写。

生产重载证据来自训练运行时保存的`conversion_forward_audit.json`：TRAIN1/81/149各noV及fullBVH，共6个live/PLY重载比较（12次forward），RGB/Alpha/normal/depth最大差均0，记录声明无优化更新及RNG不变。本次CPU只独立验证其结构、来源和数值，未重新进行GPU渲染。`normal_opacity_audit.json`记录在正常工程运行step4实施透明度保护。

## 审计边界与复用

`no_gt_guard`只允许登记smoke、隔离TRAIN源、实际源码和审计输出；拒绝仓库data、eval_only及TEST transforms，所有检查无被拒读请求；CUDA未初始化，`CUDA_VISIBLE_DEVICES=''`。读取物理光资产的SHA和工程记录不构成GT读取，本报告不输出其光向/参数数值。

可复用初始比较函数：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/material_physical_smoke_audit_v1/supplement_v2/initial_material_checks.py::inspect_initial_stage`，SHA256 `5b74878b68d307d1aad1207c709d0b45a3f9d8b14598a51072e36eec99419080`。它接受model/control/DC文件、预期相机步数及调用者注入的受护栏约束CPU读函数，比较only-DC/60字段、四RNG、空Adam/组、增密零、MLP与实际相机。未来正式M08/M09可在独立授权后以对应M06/M07控制和135相机调用；当前只执行20步smoke，未创建正式实验目录。正式终点完整外评仍须另行就绪授权。

无新增渲染输出目录、代表图片或视频；四类RGB/Alpha/Normal/Separation未导出，未实际目检，不能标目检PASS。精度、外部几何、正式24k与联合科学目标均未由本次验收评估。
