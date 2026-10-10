# 1010-08 原生PBR几何评测共享Anchor与Plane运行时

## 来源与错误

修改前读取最新GOAL/B04+B05记录并审计实际脚本。GeometryRun旧init/rawrender独立加载PLY/PTH且直接以当前XYZ查询MLP，未消费Anchor/Plane。extract_geometry_mesh_dynamic也使用它，故“所有离线几何均与训练共享”声明不成立。旧CPU独立复算只能验证旧浮点数组算术，不能验证几何运行时；不能用自洽复算替代入口审计。B04旧3normal/2depth及common/missing180已标记SUPERSEDED，保留全部旧权重/图/数组/日志/数值，待全15重评。

## 修复

scripts/eval_geometry_gt.py：仅显式开启pbr_deformation_anchor或pbr_plane_projection的原生IR模型初始化共享FrozenScene，并使用它的pc/deform/deformation(camera)。默认与无flag旧SH/IR/Lambertian入口保持，非原生flag拒绝，runtime校验错误传播、不静默fallback。hash来源包含Anchor/Plane JSON/NPZ、source PLY/PTH快照、原source SHA和checkpoint marker。原来的raster/normal方向/Z/评测区域、normal/depth算法及Mesh TSDF参数不改。

动态Mesh仍每个目标time×全135train相机调用同一GeometryRun.render，camera.fid可变到目标time，source支持推导时仍全部真实train时刻；不使用test相机/GT深度或mesh重建。模型world几何本身经过训练时使用的Anchor/Plane组合，不是在输出后更换正常分数或normal图。

B05已有完整20k训练进程继续，训练/renderer/MLP/超参数不改，不重启。本次只修复尚未执行的离线几何入口。保存code_manifest_train_launch.json/commands/code_snapshot_train_launch和当前修复版本，inspection/offline_evaluator_revision.json记录旧新SHA与具体范围，不能声称整个代码从启动以来全部未改。

## 验证

179项CPU实际exit0，6项新增GeometryRun入口测试：Anchor共享实例/hash、Plane两资产与source快照/source_override、实际camera/targettime路由、legacy当前XYZ/iteration行为、非原生拒绝、loader资产错误传播。

真实GPU零优化步/不读GT标签：B04真实55k和B05 constructor源35k，0010/0080/0150逐帧 GeometryRun.render与原生render_ir(material_only=True) alpha/world-normal/Expected Z缓冲逐像素绝对max error全部0，Mesh timeoverride .375共享形变逐元素相同。Plane支持world距plane max<2.7e-8m、axis cosine min>.9999996。各自来源全部SHA和模型fingerprint未变。证明运行时相同，不代表MAE或图像质量通过。git CRLF-aware diff check PASS。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -m unittest discover -s tests
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-11-dir-PBR几何评测统一修复/commands/check_gpu_geometry.py
```

日志/证据output/smoke_test/1010-11-dir-PBR几何评测统一修复/{audit.json,logs/unit_tests.log,logs/gpu_check.log,commands/check_gpu_geometry.py}。

## B04完整重评

output/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评，不重训。模型/RGB/albedo/roughness引用B04同55k源产物，重新15time×135train Mesh→全15 geometry→独立复算7MAE及全部来源SHA。commands/planned_commands.json、logs、runtime_history.jsonl完整记录；原B04输出不覆盖。原生RGB/Albedo/Roughness原FrozenScene正确，复用并再次核验，不冒充重新训练/渲染。最终更正报告待完成，B05最终全15七MAE仍待训练结束。已有RGB四类质量FAILED不因运行时修复自动变PASS；未commit/push/新建目标分支。
