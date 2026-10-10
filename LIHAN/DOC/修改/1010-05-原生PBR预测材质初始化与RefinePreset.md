# 1010-05 原生PBR预测材质初始化与RefinePreset

日期2026-10-10。状态：可选实现与151项CPU/真实35k单帧GPU接入检查通过；尚未执行新完整训练或取得新MAE。

## 来源审计和思路

已先读取GOAL。B02 40k RGB仍斑驳，平台棋盘纹理未恢复。只读预测35k DC发现43.5029%颜色通道在native get_albedo的[.03,.97]硬clamp范围外，梯度为零，高opacity通道54.5934%。候选仅将真实PBR_albedo_dc预测初值投影到[.031,.969]内部，让后续训练RGB有材质梯度；不使用GT albedo/normal/roughness/depth/mesh/blend或GT光强。原SH2RGB/RGB2SH函数用于转换，不改原生PBR/GGX公式或法线定义。

## 接入与持久化

scene/pbr_material.py提供有限性/shape/margin验证和初始化，拒绝与Stage1 SH来源共享存储；只写pc._albedo_dc，保留_albedo_dc_stage1/SH。默认pbr_albedo_interior_init=false；独立pbr_gt_directional_refine继承geometry preset并开启初始化，margin.001。NativeTrainer在加载35k之后、optimizer setup之前执行一次，pbr_albedo_initialization.json保存source文件SHA、初始化比例和外观最大差。其后真实参数保存回原生PLY，离线渲染直接读已训练PLY，绝不重做初始化。通用旧env/GT方向/geometry preset默认不启用。

## 验证

新增4项CPU检查：hard clamp梯度恢复与SH不变；NaN/shape/margin/共享存储拒绝；旧preset默认不启用、新refine启用；auto显式初始化选择refine。全量151项实际OK。

真实35k预测、训练frame0001、原始RGBA严格输入的PBR/GGX/BVH单帧GPU前后向PASS，32964个之前clamp外通道有非零光度梯度；所有梯度有限，SH来源/源PLY/PTH未改；0 optimizer steps，不是训练。原始RGBA150帧哈希和像素核验PASS。检查output/smoke_test/1010-06-dir-PBR材质初始化审计；初始独立脚本模块路径错误日志保留，修正sys.path后实际exit0。最大初始外观sRGB差约.001，不能据此预报MAE改善。

## 后续完整验收

B02运行中的代码已在02:10开训加载，源码快照保留，preset为geometry、没有此初始化开关；新候选没有改变B02渲染器/损失公式/运行配置。等待B02完整55k与七项MAE/人工审阅，再决定独立20k运行refine。每次实际训练需独立输出、全部15帧Albedo/Roughness/三类Normal/Expected与Median Z、RGB/覆盖率/共同像素以及四类150帧/转灯/HDR检查，全量落档。不提前commit/push或建立目标分支。

## 真实NativeTrainer构造补充

check_constructor.py实际正常exit0：从只读35k构造新refine Trainer、产生pbr_albedo_initialization.json、源SHA保持不变、RGBA150帧审核PASS；材质投影后clamp通道比例0、最大初始外观差.0010000467；优化组xyz/albedo_dc/albedo_rest/roughness/rotation/albedo_dc_stage1，opacity/scale冻结，几何/正光强优化已接入。iterations_executed=0，无新训练checkpoint；constructor_model_mlp仅保存复制来源35k和初始化/配置记录，不能作为训练结果评测。证据constructor_audit.json与logs/check_constructor.log。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-06-dir-PBR材质初始化审计/commands/check_constructor.py
```

## 更新后材质投影补充

审计实际native albedo_lr=.01，单步RGB变化可约.00282，大于初始化margin .001；只初始化可能再次越过硬clamp边界。Refine增加pbr_albedo_project_after_step=true，原生每4步Adam更新后把真实PBR DC投影回[.031,.969]，维持梯度；不修改shader/GGX/MAE公式。初始化只执行一次，更新后投影持续执行，离线渲染直接读保存PLY，不执行任一操作。

新增连续8次原生LR Adam越界后恢复梯度、投影保留SH/拒绝共享存储/NaN/margin等测试；全量153项实际OK，logs/unit_tests_project_after_step_guard.log。新增真实GPU越界后投影+原PBR单帧反向检查PASS，原先clamp通道中33951个得到非零光度梯度，源文件/SH未变、0 optimizer steps；此检查故意将一个材质通道扰动越界以测试投影，不是训练。证据audit_with_projection.json、check_gpu_with_projection_guard.log。初始化单独版本和投影安全护栏前版本的日志/源码快照保留，当前code_manifest/code_snapshot是最新护栏版本。

“SH不变”的检查范围是初始化/投影与预训练源文件；原生SH损失仍正常优化_stage1 SH DC与SH rest，不能把训练中SH参数说成冻结。B02在开训时已加载旧NativeTrainer与geometry preset，未启用新增初始化/投影，运行继续到55k；当前source renderer未改。
