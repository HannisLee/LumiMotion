
## B07完整训练实际启动（2026-10-10T08:14:00.087990+09:00）

实际PID271624/startticks9801108，cwd与scripts.train_stage2 argv核验。固定35000→55000共20000步，pbr_gt_directional_plane_native_direct，仅GT方向/原RGB与自身Alpha，不续B06权重。源码471逐SHA及双snapshot一致、9步后处理计划备齐；尚无B07最终MAE或质量结论。

## B07真实启动只读审计与B06误差分解（2026-10-10T08:18:12.598836+09:00）

训练实际PID271624/startticks9801108持续存活。launch_audit_retry实际exit0：当前配置仅7个差异（3路径、Preset、actual-budget开关、wo_indirect与wo_indirect_relight），471核心/双snapshot逐SHA匹配；plane全部数组与B06逐bit一致、支持15187/99827、150帧whiteunit方向表快照与B06逐byte相同、原35k PLY/MLP SHA保持、原RGB/自身Alpha150帧provenance PASS。初次audit误认stage1_source.json为files嵌套对象而KeyError，保留失败源码/日志；只修正读取现有平铺path→SHA schema，retry PASS，不改训练。

仅离线读取B06全部15评测npz并重组原始GS17.29779705°、GT415433/有效390928完全一致。按GT法线与来源正plane轴有向夹角分箱（不是物体语义/训练掩码）：0–5°占有效75.63771334%，该箱MAE10.47368676°、贡献原MAE7.92205717°；5–30°占1.05722793%，MAE39.04751487°、贡献.41282123°；>30°占23.30505873%，MAE38.45911204°、贡献8.96291865°。所有分箱组成完整原域，不丢点/帧/难像素、不提出替代分数、不生成训练GT标签/prior。高误差30–60°占14.86283919%贡献6.30380581°，≥60°占6.16251586%贡献4.83946485°，仍是B06结果，不是B07指标。提示后续需同时改善不同方向的法线，不能只凭平台外观验收；B07核心与loss未据此修改。

证据inspection/launch_audit.json、b06_offline_error_decomposition.json；命令commands/audit_launch.py、diagnose_b06_offline_error.py；日志logs/launch_audit_retry.log、b06_offline_error_decomposition.log。诊断0优化、GT严格离线。完整B07训练、七MAE、四类/HDR最终质量仍待完成，未commit/push/新目标分支。

## B07早期训练事件快照（2026-10-10T08:20:17.353326+09:00）

固定事件字节快照保存至3929步，0额外优化、不读GT标签。100步平均photo L1和total loss持续下降，B07开头高于B06，约3k后接近；正E由原训练RGB拟合至约5.2，初值π，并非GT能量。B06每次sample1024，B07all-eligible alpha>.9且物理模型不同，不能把训练loss曲线直接当七项MAE/质量比较。图inspection/early_training/training_curves.png，原CSV、两份事件字节snapshot和来源SHA/时刻见同目录metadata.json；作图命令commands/plot_early_training.py实际exit0，日志logs/plot_early_training.log。已实际view图。最终B07训练/15帧七MAE仍PENDING。

## B07同40k检查点与实际RGB早期审核（2026-10-10T08:22:54.566226+09:00）

预览exec67658实际exit0、检查exec67311实际exit0。全15 TEST RGB PSNR18.05598159dB（B06同40k 18.51417174），SSIM0.89944883（B06 0.88792369），LPIPS0.11159734（B06 0.10788124）；前景PSNR10.84037981。PSNR/LPIPS变差、SSIM提高，不等于法线MAE变化。实际view GT/B06/B07首中尾0010/0080/0150拼图：背景正常/主体平台存在，但头手/衣服斑驳破碎、脚周平台伪凸起和黑色阴影污染明显；B07前景/脚周更暗更高对比，GT0080自身背光暗，不能仅以暗失败。总体未通过质量，未推断最终Normal MAE。 当前FAILED_SO_FAR_RGB_ONLY，完整55k七MAE仍PENDING，未commit/push/新目标分支。

40k全部99827点finite、opacity/scale/feature及32源NNtensor原值、source35k SHA保持。最大canonical XYZ位移0.03549213707m、旋转60.00005341°容差内，sourceXYZ anchor/plane/native marker通过；albedo可导DC [.031,.969]硬clamp0，更新后位于投影边界比例41.128820%；RGB拟合E5.49740267，非GT能量。源码471与launch仍匹配。

证据preview_40000/eval/metrics.json、inspection/rgb_40000_comparison.png、preview_40000_review.json、checkpoint_40000_audit.json；命令commands/preview_40000.py 271624（绑定live train后调用记录的preview_40000_argv.json），audit_checkpoint.py40000；日志logs/preview_40000_wait.log、preview_render_40000.log、checkpoint_40000_audit.log。全训练PID271624持续进行；50k只读检查PID/状态记录在inspection/audit_50000_wait_state.json，9步后处理PID271790绑定原训练等待终态，不重启本轮。

## B07 40k来源未投影near-plane组的实际联合贡献（2026-10-10T08:26:50.574185+09:00）

已读取GOAL与40k当前FAILED_SO_FAR；只读诊断exec73154实际exit0，未改live训练。固定35k PLY/MLP和既有全135时刻plane数组、alpha>.05/全时间距plane≤.2/axis abs cos≥.5筛选，不读任何GT。四组完整分区99827点：原投影15187、source-static近plane但motion>.01未投影1542、source-dynamic近plane6836、其余76262。只是诊断颜色，全原opacity/xyz/q保留；alpha/worldnormal/ExpectedZ实际native buffer误差0，两pass0，组累计权重之和与alpha max≤6.56e−7。原source/checkpoint/471源码SHA不变，0 optimizer。

实际view inspection/unprojected_plane_groups_40000/comparison.png（TRAIN0001/0075/0149）：蓝=原投影、红=static未投影、绿=dynamic near-plane、白=其它；动态near-plane占平台大部，原投影被其它组遮挡，static未投影仅零散区域。static未投影motion span中位.05101065m，放宽阈值到.02/.04/.08也仅新增349/664/968点（全1542）；对应主导alpha>.5像素422/498/298，远小于dynamic near-plane5117/2903/5310与其它12855/10966/13154。仅放宽既有static motion阈值不能据此推断足以解决整体伪几何。dynamic近plane全XYZ span中位.39017498m、P90.89354634m（并非B05较小的法线轴运动span），不能把这6836点未经验证当真实静态平台强制投影；source候选不是GT语义标签，也不作为删点/评分mask。尚未实施下一轮gate/NN/plane参数修改，须先完成本轮55k七MAE与验收。

命令commands/audit_unprojected_plane_groups.py，日志logs/unprojected_plane_groups.log；完整来源stats/实际joint weights与buffer误差见inspection/unprojected_plane_groups.json，3帧groups/normal和实际已看拼图在inspection/unprojected_plane_groups_40000。无GT训练先验保存。

## B07训练期间固定35k预测/原RGB/方向可观测性只读诊断（2026-10-10T08:35:38.311823+09:00）

新增候选调查，非新训练或Preset：仅源35k PLY与原值32tensor MLP、既有source-only plane、全135 TRAIN相机/time/原RGB自身Alpha、白色单位GT方向，未读GT normal/albedo/roughness/depth/mesh/blend/原GT能量/颜色。40k FrozenScene仅作相机/固定源NN容器，不用训练后40k XYZ/q/material。逐原Gaussian中心投影到135训练RGB，源自身预测alpha>.9、原RGBA alpha≥.99、点与源预测camera Z差<.025m来测可见性（非GT depth）。按固定源R与camera-facing符号将单SUN unitdir送回canonical基底，以原训练linear RGB亮度做三维Lambertian线性Gram/rhs拟合；只存数值可观测性统计，不保存拟合法线/监督/prior/评分mask。全99827点保留，0 optimizer。

首次完整exec17823与增加投影/运输核验的独立exec77927实际exit0，原记录与脚本/日志均保留。全部135帧source投影→相机ray逆回world max2.92062759e−6m，source light-dot-normal运输恒等max1.07288361e−6；471核心source/训练权重SHA不变。固定原35k PLY SHA8ebd88b6f847a20178458777d51aa191d5f67ba4bbc490fd1e99ef62f66d5529。

全点≥12次观测80388，Gram λmin>.01λmax且≥12次77904；再加g长度>.01、相对线性残差<.3后40948（static28716/dynamic12232，projected11327，为重叠subset，不相加）。dynamic筛选点残差中位.20139334，与源canonical轴无向夹角中位15.82878590°；全筛选点.18728082/11.02228355°。这些只是对原RGB线性近似的拟合筛选，不能据此叫方向正确或GT normal提升：原native GGX、硬阴影、RGB裁剪、源对应/transport误差都未由线性模型精确解释，source标签也不是GT物体分割。候选只说明部分点有RGB-direction信号，未来若构造RGB几何约束必须另行设计/source资产绑定/完整训练/全15原评分与目检；本轮不修改。

证据inspection/rgb_direction_observability{,_projection_verified}.json；命令commands/audit_rgb_direction_observability{,_projection_verified}.py，日志logs/rgb_direction_observability{,_projection_verified}.log。完整B07训练及七MAE/验收仍未完成，未commit/push/新目标分支。最初即时归档文本脚本因多行引号SyntaxError、尚未执行任何写入；仅修正归档字符串后完成，不涉及诊断/训练参数。

## B07 50k真实检查点审计（2026-10-10T08:36:34.141250+09:00）

50k实际保存、绑定原训练的audit exec54247实际exit0并关闭，不是等待超时或重启。99827点finite/全部保留，opacity-scale-feature及32source NNtensor保持，source35k SHA、anchor/sourceXYZ输入、plane/native marker核验。最大canonical XYZ.07999999821m/rotation60.00005341°容差内，额外world plane投影不等于canonical .08m。DC [.031,.969]硬clamp0，projection边界channel比例41.87477670%；RGB拟合E7.12501812（非GT强度）。当前471核心/launch仍匹配；本检查无GT标签/优化/七MAE。

命令commands/audit_50000_when_saved.py绑定process_handles里的原train，实际调用audit_checkpoint.py50000，日志logs/audit_50000_when_saved.log、checkpoint_50000_audit.log；证据inspection/audit_50000_wait_state.json、checkpoint_50000_audit.json、model_mlp/pbr_directional/iteration_50000/lighting.json。55k同类只读检查已排入audit_55000_when_saved.py，9步后处理仍等待原训练终态，最终15帧七MAE/coverage/common/missing180及四类/HDR目检待完成，未commit/push/新目标分支。

## B07完整20k训练实际完成与55k审计（2026-10-10T08:43:45.200906+09:00）

固定35000→55000共20000步完整运行，train exec44255实际exit0并关闭、原PID271624已退出；Training complete log 10/10 08:43:01，loop 28:42。55k只读审计exec20440实际exit0，完整99827点finite，opacity-scale-feature冻结、32源NN原值及源35k PLY/MLP SHA保持；sourceXYZ anchor、plane/native marker绑定通过。最大canonicalXYZ 0.08000001311m/rotation 60.00005341°，DC [.031,.969]硬clamp0，projection边界比例42.158267%，RGB拟合E7.94063759，非GT强度。471核心source/launch仍匹配，未改已完成训练参数/源。

后处理真实PID271790已进入全量render，child276537（实际cwd/argv可核验），all150×4/HDR150×2048/方向72/6视频；后续完整15七MAE与独立复算/目检仍PENDING。没有将checkpoint数值PASS当画质或7°。证据training_runtime.json、inspection/checkpoint_55000_audit.json、audit_55000_wait_state.json、model_mlp/pbr_directional/iteration_55000/lighting.json；日志train.log、audit_55000_when_saved.log、checkpoint_55000_audit.log、postprocess.log、render.log。未commit/push/新目标分支。

## B07最终RGB与四类早期实际目检（2026-10-10T08:48:01.127092+09:00，七MAE/HDR仍进行中）

全15 TEST RGB PSNR19.23561417（B06同55k 19.68220260），SSIM0.91049162（B06 0.89777244），LPIPS0.11079910（B06 0.11070925）；前景PSNR11.88265958、bbox SSIM0.80394393/LPIPS0.22023212。SSIM提高、PSNR下降，不能替代七MAE。

实际view GT/B06/B07首中尾0010/0080/0150及四类0001/0075/0150拼图：黑背景正常/人物平台存在，但头手/衣服和小腿稀疏斑驳、脚周伪凸起/灰黑及阴影污染仍明显，Direct更高对比；GT0080自身背光暗，不仅凭暗失败。 主体平台大部饱和覆盖，腰手边缘与平台/脚周薄雾、细碎残影仍有；Alpha饱和不能推断真实表面正确。 正背面光照/色差仍烘入材质，平台棋盘和脚周污染、身体破碎；source feature保持，未修正来源静动态误分。 平台整体上向、0075无旧kernel整个反向，但人体/脚周混合轴斑驳、伪起伏仍明显；Depth/Mesh与最终15七MAE待完成。 当前FAILED_SO_FAR_RGB_AND_FOUR_ONLY，未提前评定7°，未commit/push/新目标分支。HDR尚在生成，后处理真实render child276537仍活跃，实际SUN72完成但完整relight审核待HDR150。

代表图inspection/rgb_final_55000_comparison.png、four_categories_final_55000.png，early review JSON同inspection；命令commands/make_final_rgb_comparison.py、make_final_four_categories.py实际exit0；真实model_mlp/renders_stage2/ours_55000/eval/metrics.json、insights四类各150已生成，6视频全解码最终另审计。
