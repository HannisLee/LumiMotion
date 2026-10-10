# 1010-07 jumpingjacks dir PBR材质初始化与投影修复完整Stage2计划

状态：TRAINING；B02完整评测及人工质量FAILED已归档，开始本轮独立完整20k。没有最终权重或MAE，不宣称质量/7°通过。

## 来源与修改

独立同1009-02 Stage1/35000，只训练Stage2 20000步至55000，135训练/15测试、400²、黑背景、真实相机/time。data_view指向已逐图审核的1010-05原始RGBA严格输入；只用原RGB/自带Alpha、相机/time、Stage1预测和GT方向。GT材质/normal/roughness/depth/mesh/blend与GT光能量/颜色不输入训练。继承源35k旧转换Alpha与原RGB全150帧数值相同的限制已披露；不重跑Stage1。

新增pbr_gt_directional_refine，只修复PBR DC硬clamp梯度：从预测值投影到[.031,.969]内部，并在每4步原生Adam更新后投影维持梯度；native get_albedo、sRGB/linear、GGX/BVH和MAE定义保持。源SH初始化/投影不污染；原生SH损失仍正常优化DC/rest。保留B02几何/材质学习率、warmup/ramp、信赖域、预测GS-Depth/来源RANSAC平面/轮廓/弱锚定、冻结opacity/scale/feature，光强白色初值π仅训练RGB拟合。没有GT材料初始化。

## 验证与完整评测

153项CPU实际OK，包含原生LR=.01连续越界后投影保持梯度；真实35k/frame0001 GPU forward/backward/投影与NativeTrainer构造接入通过，0 optimizer steps，无新候选权重。这不替代本轮完整训练。

计划四类150帧、转灯72、HDR150/2048、6视频全解码、15帧RGB和全部七项MAE：Albedo/Roughness、GS/Depth/Mesh Normal、Expected/Median Z、各覆盖率/共同像素/缺失180惩罚。Mesh按每测试time融合所有135 TRAIN位姿，禁止GT或测试视角重建。原协议/像素全集不变，不凭删除Gaussian达标。四类实际目检与前后景结论待55k，全部失败结果保留。

## 完整训练命令

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/model' --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 40000 55000 --render_mode original_ir --loss_preset pbr_gt_directional_refine --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-07-jumpingjacks_dir-PBR材质初始化与投影修复/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1
```

源码SHA/code_snapshot、train_argv.json、planned_commands.json与commands/*.sh留档；所有输出本目录model_mlp/renders_stage2/ours_55000及albedo/normal_depth/roughness/predicted_meshes，日志logs。完整渲染命令commands/render.sh（task all、HDR2048），全MAE/独立复算/归档共9项planned_commands.json。代表图和四类目检目前N/A，最终待实际产物回填。

未commit/push、未建目标分支。此为目标1分支质量修复候选，完整通过后才提交/push并从该SHA建立目标2分支。

## 实际启动

PID 219507，启动 2026-10-10T03:10:06.570000+09:00，train exec session26125；完整20k35k→55k。后处理与40k只读预览已绑定真实PID，训练异常退出不自动重启。

实际模型配置与初始化记录已核对PASS：refine/init+projection true，RGBA150帧PASS，初始化后clamp比例0、max外观差.0010000467。见inspection/launch_configuration.json；最终结果待完整训练。

## B03运行中只读来源平面形变诊断

原生“binary feature”实际是sigmoid(2×feature)软门，静态支持仍有少量MLP形变；实际只读35k三训练时刻0001/0075/0149检查，支持14556点中feature P90=.01290935、P99=.37615985。形变轴偏差P90≤.05595°、P99≤1.99525°；位置形变P90约.0005m、P99约.0193m、少数max.20857m。大多数支持的形变近零，不能归因全部平台误差于MLP。

同一固定支持上，35k canonical平面法线均角17.9940°，B02真实55k为.92838°（中位.24708°、P902.71599°），平面位置绝对均差.0184996→.0099137m。说明已支持的平面参数确实修正，但画面脚周/噪声仍可能由支持之外的来源片影响；不能盲目增加相同平面权重或把该预测一致性当GT法线MAE。source SHA不变、0 optimizer steps、GT输入false，B03训练未改。证据inspection/source_plane_deformation.json、commands/audit_source_plane_deformation.py与日志。最终结果待完整训练。

## B03 40k只读预览与真实检查点

15帧RGB：{"psnr": 18.469882165991756, "ssim": 0.8856826225916544, "lpips": 0.11016974796851477, "psnr_foreground": 11.435341690063554, "ssim_foreground_bbox": 0.7512827237447103, "lpips_foreground_bbox": 0.23224637508392335}

实看0010/0080/0150：黑背景正常，0080平台棋盘纹理开始出现；人体/脚周仍明显斑驳、头手边稀疏，正面平台多白及影周伪结构。FAILED_SO_FAR，完整55k与七项MAE待完成，不能commit。比B02同40k PSNR18.15228846、LPIPS.11477009略改善，不将不同迭代效果或早期预览当最终结论。

真实40k检查：{"status": "PASS_CHECKPOINT_ONLY", "iteration": 40000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "max_displacement": 0.03078901208937168, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 5.285261631011963, "final_mae": false}

99827全保留，材质DC仍[.031,.969]且clamp比例0，xyz/rotation信赖域内，opacity/scale/feature冻结；原生材质投影实际持续生效。光强为训练RGB拟合5.28526163，离线预览恢复同值。预览实际exit0，证据preview_40000_review.json、checkpoint_40000_audit.json及preview_40000_rgb_comparison.png。

## B03运行中来源局部几何与平面支持分布诊断

只读35k点云：按来源soft feature分组、32最近邻PCA，在opacity>.2、最小/次小协方差特征值<.2、邻域半径≤3倍来源最大scale条件下，静态1740/54244（3.2077%）、动态3265/45583（7.1628%）通过可靠平面筛选。来源轴与其PCA轴平均15.1489°/16.4298°，B02轴与同一来源局部轴15.6124°/20.4865°；这是来源自洽诊断，不是GT MAE，邻域含低opacity点、体积和边界会影响PCA，不能直接强制整个人体用这些PCA方向。未导出训练目标、0 optimizer steps、源SHA保持，B03未改。证据source_local_geometry.json与可复现commands/audit_source_local_geometry.py。

来源静态opacity>.05点19416，其中距来源RANSAC平面≤.08m15385、≤.12m16881、≤.2m17768；距平面P90=.14797m，当前support14556点还应用轴方向筛选。高opacity>.5点12603，≤.08m9786、≤.2m11589。当前几何信赖域.08m及support band.08不能校正较远的来源片；不能盲目只增已支持点的平面权重。此为下一步候选诊断，需结合B03完整真实MAE/质量后决定是否扩大源预测支持及有界位移；不读取GT几何定义支持，不删除Gaussian，不提前声称7°。证据source_plane_support_distribution.json。

## B03实际40k三训练帧梯度分解诊断

对0001/0075/0149真实40k权重，原生RGB+SH目标与lambda=.05的GS-Depth一致性梯度分别计算；所有梯度有限、full一致性梯度=GS侧+Depth侧（max差<1e-8）、源权重fingerprint/SHA不变，实际exit0、0 optimizer steps，未用GT材质/法线/深度或测试RGB。

rotation梯度L2：RGB+SH .10920/.69956/.12194，一致性full .00464/.01153/.00505（约1.65～4.25%）；GS侧 .00421/.01054/.00438，大于Depth侧 .00217/.00496/.00221。前两帧与RGB梯度cosine为-.10677/-.23217，最后帧+.09898。xyz同样RGB目标量级大于一致性。说明GS侧确实可受预测深度影响，但当前RGB目标整体主导，不能断言一致性项主导剩余噪声，也不能凭梯度冲突证明其GT准确性差。若后续尝试位置侧一致性，仍需完整训练/MAE验证。

首次独立脚本在最后校验误用verify_unchanged方法导致exit1，已保留失败script/log；改用真实FrozenScene.verify_frozen后重跑三帧正常exit0。当前训练未改、不重启。证据inspection/normal_depth_gradients.json、commands/audit_normal_depth_gradients.py、logs/audit_normal_depth_gradients_retry.log。

## B03 50k真实检查点

{"status": "PASS_CHECKPOINT_ONLY", "iteration": 50000, "gaussians": 99827, "all_finite": true, "opacity_scale_feature_frozen": true, "albedo_rgb_range": [0.030999988317489624, 0.968999981880188], "clamped_albedo_channel_fraction": 0.0, "channels_at_projected_bound_fraction": 0.3296035474704572, "max_displacement": 0.06572393327951431, "max_rotation_deg": 60.000038146972656, "actual_light_scale": 6.202476978302002, "final_mae": false, "deformation_state_tensors_unchanged": true, "deformation_tensor_count": 32}

材质DC真实保持[.031,.969]，native硬clamp比例仍0（约32.96%通道在投影边界，但位于native硬clamp内、仍可导）；同Gaussian全集、opacity/scale/feature冻结，XYZ与旋转有限且在信赖域内。MLP全部state tensor与35k源逐项完全相同；不能据此说不同XYZ下输出不变。光强6.20247698仅训练RGB拟合；不是GT能量。最终55k七项MAE/四类图像质量尚待完成。

## B03完整20k训练结束，后处理开始

启动03:10:06.570、55k保存及Training complete 03:54:31 JST；全部20000步，loop44:07，train exec实际exit0，真实55k PLY/PTH/lighting marker已保存。后处理绑定原PID结束后正常开始render，实际childPID226468。training_runtime原开始记录已保留，补充真实log完成时间/loop/exit0；完整四类、重光照、七项MAE及人工前后景质量仍待完成，不提前commit/push或声称7°。

## B03最终55k RGB实际评测/审閱（全量后处理继续）

{"psnr": 19.35354474812191, "ssim": 0.8932221968968709, "lpips": 0.11383797228336334, "psnr_foreground": 12.234762806258436, "ssim_foreground_bbox": 0.7692404826482137, "lpips_foreground_bbox": 0.22925928036371868}

已实际查看0010/0080/0150对照GT/B02同55k：背景正常，背面平台棋盘纹理有恢复；人物斑驳、头手边界稀疏及脚周伪结构仍明显，正面平台多白。PSNR19.35354 vs B02 18.91229略改善；最终RGB FAILED_SO_FAR，不能提交，完整四类与七项MAE尚待完成。不把GT背光本身较暗单独判失败。证据inspection/visual_review_early.json与rgb_final_55000_comparison.png。

## B03最终四类实际首中尾审閱

已查看RGB/Alpha/Normal/Albedo 0001/0075/0150，图inspection/four_categories_final_55000.png：Alpha主体平台基本完整，头手毛糙、平台边及下方薄雾/残影；Normal平台总体集中但脚周大块异常、人体斑驳；Albedo平台棋盘较B02恢复，但正背面光照残留、头手与脚周污染明显。四类质量FAILED，commit_allowed=false；材质投影确实持续生效仍不能修复来源几何与覆盖残影，完整MAE/HDR视频审计继续，不预填法线结果。仅首中尾实际人工审核，不声称150逐张。
