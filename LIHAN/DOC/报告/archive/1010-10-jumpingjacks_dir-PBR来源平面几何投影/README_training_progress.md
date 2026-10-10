
## 正式启动记录

{"status": "TRAINING", "training_pid": 245234, "process_start_ticks": "8927314", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_plane", "session_id": 42059, "observed_at": "2026-10-10T05:48:32.269187+09:00", "started_at": "2026-10-10T05:48:20.140000+09:00", "start_time_source": "proc_btime_plus_start_ticks"}

真实train session42059/PID245234/start ticks8927314；后处理session86603/PID245271绑定真实train；40k预览session2961同PID绑定，禁止退出后盲目重启。实际配置相比B04改变Plane Preset/projection、新增阈值.2/.01字段及本轮路径；其它既有参数相同，dir表字节相同。首次launch audit只列5个允许字段，漏列新阈值2项，断言失败；未修改训练，修正审计字段后PASS。150原RGBA来源、99827点全保留、source资产/cameraSHA/15187支持/代码manifest检查PASS，证据inspection/launch_configuration.json。最终MAE/质量待完成，未commit/push。

## 已核验实时进度

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T05:51:27.024252+09:00", "steps": 1395, "total_steps": 20000, "iteration": 36395, "training_pid": 245234, "process_start_ticks": "8927314", "postprocess": "WAITING_LIVE_TRAIN", "checkpoint_40000": false, "source_manifest_matches": true}

完整训练与七项MAE尚未完成，不提前提交/push，未标记目标完成。

## 2026-10-10 几何运行时遗漏更正（待完整重评）

后续源码审计确认旧scripts/eval_geometry_gt.py的GeometryRun仍直接用当前XYZ查询sourceMLP，未消费Anchor资产；逐time Mesh也使用这个入口。先前声明所有几何评测与训练一致不成立。旧GS18.91234950°/Depth27.51005704°/Mesh22.23676747°及2depth/common/missing180虽CPU复算自洽，但不是实际B04 Anchor模型结果，现标记SUPERSEDED。保留原权重/浮点图/原分数/日志，不用旧几何分数评价Anchor效果。RGB/Albedo/Roughness及原生四类渲染已用FrozenScene，受该遗漏影响的仅GeometryRun评测与Mesh；已有RGB四类质量FAILED不因此变PASS。新共享FrozenScene入口已改，179CPU PASS；真实GPU逐buffer验证和全15帧重新Mesh/几何评测待完成。B05训练运行中的source/renderer/参数不改，仅离线入口改并保存launch旧代码及hash记录，不重启训练。

## B05 40k真实检查点/完整15帧RGB预览

RGB：{"psnr": 18.393261800805856, "ssim": 0.8858578602472941, "lpips": 0.10931317210197448, "psnr_foreground": 11.360213718905596, "ssim_foreground_bbox": 0.7518338481585185, "lpips_foreground_bbox": 0.2303248941898346}

Canonical检查：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 40000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.3492876008828607, "max_displacement": 0.03536596521735191, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 5.298036098480225, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

全部99827 Gaussian保留，MLP32tensor原值、sourceXYZ anchor与marker绑定，额外15187支持plane资产/marker SHA及全time筛选统计核验PASS。canonical max位移.03536597m/rotation60.000038°，材质硬clamp0；这不是effective world总位移限制。真实光强5.29803610从RGB拟合，不取GT能量。

实际看0010/0080/0150对照B04同40k/GT：黑背景正常、主体存在；人物/头手/衣服斑驳稀疏、脚周平台伪结构/灰块/残影仍明显，FAILED_SO_FAR。PSNR18.39326 vsB04同40k18.51746略低，不能断言改善；GT背光本来暗，不凭该暗单独判失败。预览exec2961实际exit0，checkpoint CPU audit exec76226实际exit0。40k没有七MAE，不把此预览当最终55k成果；完整20000步/四类/HDR/15七MAE继续。证据inspection/preview_40000_review.json、preview_40000_rgb_comparison.png、checkpoint_40000_audit.json。

已核验实际/proc PID+start ticks：{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T06:05:17.355924+09:00", "steps": 7247, "total_steps": 20000, "iteration": 42247, "training_pid": 245234, "process_start_ticks": "8927314", "checkpoint_40000": true, "postprocess": "WAITING_LIVE_TRAIN", "offline_geometry_evaluator_fixed": true}

## B05运行中只读来源覆盖、法向运动和实际梯度审计

不修改运行中训练/renderer/MLP/参数，不读取GT材质/法线/几何，0optimizer步。所有脚本/log/JSON放本输出commands/logs/inspection；Source SHA及模型fingerprint保持，当前核心manifest仍匹配。

Source35k及既有full135统计：当前15187支持点未投影Alpha×scale0×scale1代理5.610998；同几何candidate条件的静态不稳定点1542（proxy.382483），动态candidate6836（proxy2.905280），全time距离≤.2m且axis abs cos≥.5。低Alpha静态near.2有29588点但proxy仅.047707，不据点数认定主导残影；proxy不是光栅实际贡献，不能据此判属于平台或删除/投影。

真实全部135time进一步查询sourceMLP：6836动态candidate沿plane法向运动span最小.01217797m、中位.12775928m、P99.31366219m，whole-coordinate boxspan中位.390175m。将稳定性从world改为normal span≤.01仍无动态candidate通过，不能声称它们只是沿plane移动而应全投影；推迟扩大支持，后续仍需RGB/来源几何证据。

首次诊断强求asset数组与native runtime重算逐bit一致失败（保留commands/*initial_failed.py及logs/audit_source_plane_normal_motion.log），未改训练或资产。GPU单独核验：asset sigmoid(2×fea)与native get_binary_feature先加减log(.5)路线有max1.19209e−7舍入、8152元素不同但static分类差0；实际span/maxdistance/mincos max差1.78814e−7/1.19209e−7/1.79163e−7。只读统计复查用1e−6有限容差实际PASS，训练资产严格重算校验没有放宽或修改。证据source_feature_roundoff.json、source_plane_normal_motion.json及*_retry.log。

B05实际40k三训练帧0001/0075/0149，真正PBR/BVH/GGX autograd分解；sourceMLP no_grad，Plane projection在其外求导，与Trainer相同，不误用FrozenScene no_grad投影当训练梯度。PBR+SH梯度等于分别求导之和；GS-side+depth-side等于完整ND梯度；全部finite、0optimizer、源SHA/fingerprint未改。动态旋转SH/PBR梯度L2比约2.9166%、1.2186%、2.4791%；ND/PBR约6.2180%、7.4410%、4.3123%。证据不支持SH为主要梯度干扰，也不能由量级推出GT法线准确；ND仍较弱但对自己的噪声深度一致不等于GT正确，不能随意加大后宣称7°。

额外原生预测depth distortion mean×现有Stage1 lambda_dist1000只读诊断：loss约2.64–3.10e−5，动态XYZ梯度/PBR约.0074%–.0193%，动态rotation约.0195%–.0223%；不是该项已用于本轮优化。只加默认1000不能预计明显修复当前geometry；若后续纳入或重标定必须新训练完整MAE验证，不把内部loss当GT normal改善。

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_unprojected_plane_groups.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_unprojected_plane_groups.log；实际exit0。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_source_plane_normal_motion.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_source_plane_normal_motion_retry.log；实际exit0。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_pbr_sh_geometry_gradients.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_pbr_sh_geometry_gradients.log；实际exit0。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_depth_distortion_gradients.py
```

日志：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/logs/audit_depth_distortion_gradients.log；实际exit0。

## B05 50k真实检查点

{"status": "PASS_CHECKPOINT_ONLY", "iteration": 50000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.3420851406266174, "max_displacement": 0.07999999821186066, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.189603805541992, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

99827点全保留、source32NNtensor/opacity/scale/feature原值、marker绑定anchor+plane资产；canonical位移/旋转有界、PBR DC可导区间，actual world额外投影不宣称canonical边界。最终55k全部15帧七MAE/四类/HDR待完成，50k不代替最终结果；未commit/push/新建分支。

## B05完整20000步结束，55k与后处理启动

{"status": "TRAIN_COMPLETED", "training_pid": 245234, "process_start_ticks": "8927314", "steps_planned": 20000, "source_iteration": 35000, "target_iteration": 55000, "loss_preset": "pbr_gt_directional_plane", "session_id": 42059, "observed_at": "2026-10-10T05:48:32.269187+09:00", "started_at": "2026-10-10T05:48:20.140000+09:00", "start_time_source": "proc_btime_plus_start_ticks", "finished_at": "2026-10-10T06:33:51.337112+09:00", "iteration": 55000, "steps_completed": 20000, "log_completed_at": "2026-10-10T06:33:46+09:00", "loop_duration": "45:06", "training_exit_code": 0, "verified_training_session_closed": true}

源35k→55k全部20k，log 55k保存06:33:45、Training complete06:33:46 JST，loop45:06；训练exec42059实际exit0并关闭。后处理真实/proc childPID252609命令scripts.render_stage2已核验，执行全量all渲染；完整15帧七MAE及四类/HDR最终质量仍待完成，不提前提交/7°。

55k真实canonical检查：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 55000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.34729749132666177, "max_displacement": 0.08000001311302185, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.492673397064209, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32, "deformation_input_exactly_source_xyz": true, "anchor_marker_bound": true, "source_files_unchanged": true, "final_mae": false}

## 来源预测的SUN像素预算只读GPU机制审计

实际B05 40k三TRAIN帧0001/0075/0149，不读测试RGB或GT材质/法线/几何，不优化或修改本轮训练。native_sampled（SH indirect256 enabled）选1024个photo像素，实际对应期望263168 rays（256 hemisphere+1SUN）；关闭间接光后原预算仍除256，direct_sampled只选1024像素/1024SUN rays。仅诊断通过独立args临时大cap40960000调用旧代码，使direct_dense选所有alpha>.9像素18927/15971/19186，SUN实际同数量，全部前/反向finite、source SHA/fingerprint不变，实际exit0。

动态点非零梯度覆盖 direct_sampled→direct_dense：14568→29613、3019→10757、14490→29873；native_sampled含indirect rays给其它Gaussians传播也有30077/30534/30436，不能声称direct_dense整体覆盖超过native_indirect。Forward+gradient时间sampled native约.19–.31s、direct_dense约.09–.13s（并行其它GPU任务环境，非正式benchmark）。只证明single SUN关闭SH间接光时可按实际光线数利用预算，并且全有效photo像素可跑，不能推断画质/7°通过。当前材料/光强已用indirect训练，offlinedirect loss与不同像素集L1不可作为重新训练后的质量比较。

若B05最终仍失败，候选是独立Direct Preset关闭间接光并按实际L个SUN预算训练photo像素，保留原生GGX/BVH/颜色、source和全部评分；改动需在读取完整B05结果后新训练/全七MAE验证。当前没有修改训练参数或新增训练。证据inspection/directional_ray_budget.json，完整命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/commands/audit_directional_ray_budget.py
```

日志logs/audit_directional_ray_budget.log；0optimizer、非最终MAE。

## B05 55k最终RGB与四类早期目检（全七MAE仍运行中）

RGB：{"psnr": 19.375063141094465, "ssim": 0.8948765238126118, "lpips": 0.11125242908795675, "psnr_foreground": 12.246959743787187, "ssim_foreground_bbox": 0.7720984180768331, "lpips_foreground_bbox": 0.22454677522182465}

已实际看全部测试集中首中尾0010/0080/0150及四类0001/0075/0150：背景黑且主体存在，人物头手/衣服斑驳稀疏、脚周平台伪几何/灰块仍明显；alpha基本覆盖但边缘残影，albedo光照烘入，normal身体斑驳。0075平台整体异常green/yellow正在用当前安装光栅器进行来源支持组只读核验，不先假定几何还是坐标问题。PSNR19.37506314 vsB04同55k19.47597408略低，未见足以通过的改善；GT背光本来暗，不凭暗单独失败。状态FAILED_SO_FAR，完整HDR/15帧七MAE仍待完成，未commit/push。证据inspection/final_55000_early_review.json及两张最终拼图。

## B05只读当前CUDA光栅来源与plane法线符号审计

真实installed diff_surfel_rasterization0.0.1的direct_url指向hbb1 f7e2b68a0f17d6d7bca3fa564a96c7678581ce5a，扩展SHA49de0cdbfae3b83f513c0c2a387eba317f9a159aaf69c115b1bff7c615a889e4；不是仓库LumiMotion内的computeTransMat实现。读取该commit本地Git源码可见forward/backward均以normal.view_z<0定向，仓库原生则以-dot(normal_view,position_view)>0定向。

55k 0步分组真实光栅验证：15187投影组world normal绝对plane夹角max.02798°，full alpha/worldnormal与nativePBR buffer maxerror0，证明轴与坐标一致但不证明朝向正确。0075组法线全部-plane而0001/0150全部+plane；其相机均在plane上方。全部150相机中0038/0075/0076的plane.view_z>0且camera-plane side>0，旧Z符号规则会错误朝下，三帧均TRAIN，不直接构成15测试GS MAE变化承诺；身体其它off-axis位置也可能有相同机制。分组独立渲染不是联合贡献，不能据此删除/扩大support或改变mask。

当前B05训练/模型/渲染/MAE仍使用同一旧installed包，保留其完整实际结果不覆盖；不把共享入口自洽误写为物理朝向正确。已启动output/smoke_test/1010-12-dir-PBR原生光栅方向核验独立namespace原仓库CUDA构建验证，不pip install、不改当前环境包/其它分支，不改本轮代码/参数。证据inspection/raster_kernel_provenance.json、plane_raster_normals.json、plane_sign_camera_diagnostic.json、plane_raster_group_comparison.png及commands/audit_plane_raster_normals.py；该实际exec26173 exit0，source权重fingerprint保持且无GT标签。
