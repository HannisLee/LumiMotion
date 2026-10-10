# 1010-16 dir 原生PBR概率gate Preset功能冒烟

状态：211 CPU回归实际exit0；20步Trainer、GPU概率梯度、保存加载与共享buffer/四类/HDR/SUN待完成。

已先读取GOAL，B08完整20k/七MAE/人工质量FAILED，GS16.84133987°/覆盖95.01267352%、原7°目标FAILED，1055文件归档SHA3保持。55k只读梯度诊断显示sigmoid因子全点中位4.2676e−5，原raw Adam概率改变很小。本轮仅改gate优化坐标，不根据GT标签生成prior。

Preset pbr_gt_directional_plane_native_direct_probability_gate；source35000→35020（20步），源35k PLY/32 MLP、sourceXYZ、RGB/自身Alpha/相机time/GT白色单位方向唯一训练输入。禁GT材质/normal/roughness/depth/mesh/blend/原灯强度颜色。原NN32tensor、opacity/scale冻结，99827点全部保留；source平面15187继续由原35k全部135time推导。Direct SUN/GGX/BVH、原SH RGB辅助loss、DC投影与XYZ/q信赖域/loss保持。

概率叶子逐bit初始化为原native sigmoid gate，Adam feature组仅绑定概率叶子；前向逐bit使用真实PLY logit的native sigmoid值并对概率传递单位梯度。每4步更新后投影概率到来源±1及[0,1]、有限逆sigmoid写回PLY；端点使用±80有限raw，未改变的概率保留source raw逐bit值，然后同步概率叶子为实际native量化值。权重保存与所有FrozenScene/GeometryRun/材质/HDR/SUN均只读取实际PLY原native feature，另无训练概率资产；metadata v2与checkpoint SHA绑定，严格sourceNN/概率信赖域/协议验证。v1 raw gate和旧Preset保留。未保存完整Adam状态，不声明无损resume。

smoke warmup0、1000步ramp，仅验证20步功能；正式完整20k须恢复warmup2000，不从smoke/B08训练后权重继续。lr.001/max_probability_delta1/anchor.001，其余B08相同。CPU新增9项：来源初值/饱和单位梯度/有限端点/量化/Adam保存/边界/来源协议/legacy/auto，共211实际PASS。真实短训练与图像尚待检查；无正式MAE/7°结论，未commit/push/新目标分支。

## 命令

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-16-dir-PBR概率gate/data_view' --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-16-dir-PBR概率gate/model' --load_iter 35000 --iterations 35020 --save_iterations 35020 --test_iterations 35020 --render_mode original_ir --loss_preset pbr_gt_directional_plane_native_direct_probability_gate --pbr_gt_lights_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-16-dir-PBR概率gate/gt_directions_only.json' --pbr_light_intensity_scale 3.141592653589793 --depth_ratio 0 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1 --pbr_geometry_warmup 0
```

训练日志logs/train.log。功能诊断commands/{check_gate_checkpoint,check_pbr_gradients,check_shared_buffers,check_geometry_buffers,check_material_buffers}.py；输出inspection对应JSON，日志logs同名。四类150帧命令：python -m scripts.render_stage2 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-16-dir-PBR概率gate/model_mlp --load_iter 35020 --task insights --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1010-16-dir-PBR概率gate/insights; 日志logs/insights.log。代表图inspection/{four_categories,relight}.png；HDR16仅smoke，正式2048；四类/前后景与所有结果PENDING。

## 实际冒烟完成（2026-10-10T10:01:14.420317+09:00）

211 CPU、20步Trainer/GPU五参数梯度/保存/共享geometry-material/insights六会话均实际exit0且关闭；474核心和双snapshot逐SHA一致，未在训练后改生产源码。35000→35020保留99827点finite，opacity/scale和32来源NN tensor原值。raw feature变化70033点、实际native概率变化69971点，max概率delta 1.859664917e-05；raw最大偏移74.53421021是逆sigmoid/有限端点编码，不等于世界几何的大改变，不沿用raw±4的旧信赖域。

TRAIN0001/0075/0149 photo概率梯度L2 0.015394565,0.035415199,0.013251405，albedo/roughness/XYZ/q/概率全部finite非零；前向逐bit仍实际PLY native sigmoid，源码NN全冻结。单SUN实际18843/15904/19137光线，3SUN/间接/旧预算诊断都通过且0 optimizer，不称已独立训练。GeometryRun alpha/worldnormal/ExpectedZ最大误差0与目标time一致；albedo helper误差0，roughness≤5.96e−8。

概率全域[0,1]允许源gate修正，未强制任何source组静态、不删点/帧/难像素；canonicalXYZ .08m/q60°仍只限制canonical参数，gate改变原NN输出的world形变另需完整RGB/Alpha/覆盖与MAE验收。输入没有GT材质/法线/几何/灯强度颜色，GT方向白色单位，RGB拟合强度非GT。原生完整Adam未保存，不声明无损resume。

四类150×4、4视频全部解码150帧，实际看0001/0075/0150四类与HDR16/SUN代表图：

- 黑背景正常，人物/平台存在；20步初始材质人物斑驳/破碎、背面黑，脚周灰黑污染仍明显。
- 人物与平台主体覆盖完整/大部饱和，手臂/身体轮廓和平台边缘有薄雾及细碎残影。
- 正背面的光照色差/身体阴影烘入材质、平台脚周污染，未通过材质质量。
- Normal平台整体上向但脚周伪凸起、人体混合轴斑驳；GeometryRun仅buffer/time一致审计，不是GT精度，短训不作正式Depth/Mesh/7°评价。
- 实际view三帧独立替换SUN和HDR16，照明/暖色响应存在，身体斑驳/缺失和平台伪凸起仍明显；仅功能，最终全量HDR2048须重训后验收。

最终功能PASS、画质FAILED_SMOKE_ONLY、正式全15七MAE及7°未评；下一完整B09只从原35k独立训练，warmup恢复2000，所有评测口径/覆盖/common/missing180原值，不以20步smoke或B08训练后checkpoint继续。未commit/push/新目标分支。
