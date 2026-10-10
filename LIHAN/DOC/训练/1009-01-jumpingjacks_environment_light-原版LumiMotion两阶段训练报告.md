# 1009-01 jumpingjacks_environment_light 原版 LumiMotion 两阶段训练报告

日期：2026-10-09，Asia/Tokyo。状态：**数据转换、原版加载审计与首中尾目检PASS；用户已确认名字；35k/55k训练及六步流水线结束，完整视频最终目检待完成**。

用户要求转换 environment_light 与 direction_light，只训练 environment_light 的原版 LumiMotion，并核对报告名字。已确认报告名 `1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练报告.md`；用户已确认该名称并授权开始。本次输出目录下已有完整命令和顺序流水线，可直接审阅。

## 数据与转换

原始根目录 `/home/lihan/data/LH-data/danamic/`；转换根目录 `/home/lihan/data/LH-data/transfer-dynamic/`。两套数据名称为 `jumpingjacks_environment_light`、`jumpingjacks_direction_light`，各150帧、800×800，135 train/15 test，测试帧0010、0020至0150；相机、骨骼和时间对应一致。`time=frame/150`，物理秒数单独保留。未改动世界坐标、相机轨迹、场景尺度或灯光贡献。

environment_light 名称沿用源目录，实际照明为两盏 POINT（800/650 Blender energy参数），并非已知远场 HDR；direction_light 为两盏重叠 SUN，各强度3。完整多光源 `lights.json`、骨骼、对象姿态、法线、albedo与blend快照通过只读使用约定的软链接保留。训练不读取 GT albedo RGB、法线、灯位或骨骼，仅使用 RGB、albedo Alpha、相机和时间；direction_light 本次仅转换。

转换器：[prepare_jump_lh_dataset.py](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/prepare_jump_lh_dataset.py)，改编自既有 `prepare_lh_dynamic.py`。兼容新版 `lights.frames`，保留两个光源。16-bit sRGB PNG 通过 PIL 转为原版 reader 使用的8-bit RGBA，保留 soft Alpha。转换输出为 straight RGB，仅将Alpha=0的RGB置零；原版 reader 在 `no_bg=True` 时乘一次Alpha，防止转换与加载重复预乘。法线EXR和albedo原始16-bit文件保持不变。

转换命令：

```bash
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
/home/lihan/miniconda3/envs/lumimotion/bin/python LIHAN/scripts/prepare_jump_lh_dataset.py data/LH-data/danamic/jumpingjacks_environment_light data/LH-data/transfer-dynamic/jumpingjacks_environment_light --test-stride 10 --camera-extent 1.0
/home/lihan/miniconda3/envs/lumimotion/bin/python LIHAN/scripts/prepare_jump_lh_dataset.py data/LH-data/danamic/jumpingjacks_direction_light data/LH-data/transfer-dynamic/jumpingjacks_direction_light --test-stride 10 --camera-extent 1.0
```

两个转换集前景覆盖率均为10.775%–20.950%，计算相机半径5.661441。完整审计：`inspection/conversion_audit.json`；首/中/尾RGB与Alpha图：`inspection/conversion_first_middle_last.png`。全部300帧转换结果核对通过，两套各15个测试相机由原版reader读取核对通过，Alpha软边3335–6727像素/帧。实际打开首/中/尾检查：人物与平台完整，RGB和Alpha对齐，env灰背景正确转黑；direction背面暗部属于源照明。数据转换与输入目检PASS，见 `inspection/conversion_visual_review.json`。

## 原版训练与评测设置

执行代码 `/home/lihan/reproduce/LumiMotion`，main提交 `65287461075a1a4c0f54845325cc22039bd14e7d`；已有工作区FOVxy修复保留，源码状态与差异保存在 `code_commit.txt`、`code_status.txt`、`code_diff.patch`。服务器venus，Conda lumimotion，GPU0 RTX3090Ti。输出位于本仓库 `output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/`。

Stage1原版 SH 从头35000步；Stage2原版 IR 加载本次35000，累计55000步（新增20000）。默认MLP、hyper_dim=1、SH degree=3、默认10万点随机方块[-1.3,1.3)，不提供GT表面PLY。独立 `data_view` 链接转换后的transforms/images，由原版 loader 开训时生成随机PLY；转换后的公共数据目录保持不写入PLY。

JumpingJacks官方配置：Stage1 densify_until_iter=20000，lambda_separation=0.001，d_xyz_loss_weight=0.001，binarization_warm_up=1000，d_color_reg_loss_weight=0.01。Stage1 depth_ratio=1，Stage2/渲染depth_ratio=0；Stage2 diffuse_sample_num=512，HDR/NVS=2048。resolution=2（400×400）、黑背景、显式Alpha监督。其它参数沿用原版默认。

原版IR仍使用单个学习的环境光表达实际双点光源，不额外引入已知点灯或当前perlight训练模式；能否恢复对应照明与材质需要真实训练结果判断。学习环境光文件随55k checkpoint保存。

六步顺序：Stage1 → Stage1四类insights → Stage2 → materials → Golden Bay示例HDR重光照 → 原版dynamic NVS。Golden Bay只用于展示重光照，没有这套数据对应的Golden Bay真值，不报告重光照GT误差。训练/渲染 `train_light_folder=images`。

原版NVS直接读取PNG的RGB作GT，故单独生成 `eval_view/images` 的一次Alpha黑背景合成输入，保持评测GT与训练目标背景一致；该目录仅用于NVS，不参与训练。两套view共用相机划分和训练生成的随机PLY。原版NVS复用首帧BVH，按这一实现限制解释15测试帧PSNR/SSIM/LPIPS。代表视频和材质图分别在 `model_mlp/renders_stage1_insights/ours_35000/`、`trained_materials/ours_55000/`、`render_relight_with_hdr_golden_bay_4k_32x16_rot330/ours_55000/`；NVS在 `eval_nvs_dynamic/ours_55000/`。

## 完整训练与渲染命令

实际模型目录为 `model_mlp`，命令传 `model` 时原版自动加后缀。所有步骤用 `CUDA_VISIBLE_DEVICES=0`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；脚本保存在 `commands/`。

### stage1

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage1 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/logs/stage1.log`。

### stage1_insights

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage1_insights --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 35000 --depth_ratio 0.0
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/logs/stage1_insights.log`。

### stage2

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --load_iter 35000 --iterations 55000 --test_iterations 55000 --save_iterations 40000 50000 55000 --diffuse_sample_num 512 --depth_ratio 0.0
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/logs/stage2.log`。

### materials

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_materials --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/logs/materials.log`。

### relight_hdr

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_relight_with_hdr --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048 --hdr /home/lihan/reproduce/LumiMotion/example_envmaps/golden_bay_4k_32x16_rot330.hdr
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/logs/relight_hdr.log`。

### eval_nvs

```bash
cd /home/lihan/reproduce/LumiMotion
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_nvs_dynamic --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/eval_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/logs/eval_nvs.log`。

## 状态与验收

后台流水线脚本 `commands/run_pipeline.py`：每步失败即停，保留checkpoint、渲染和日志；检查35k/55k模型完整性和15测试帧NVS文件。用户已确认，`REPORT_NAME_CONFIRMED.txt` 已写入并正式启动。

训练正在进行，最终定量结果尚未产生。Stage1的RGB、Alpha、albedo/separation、normal四类训练结果初查已完成，Stage2结果待生成；最终PASS/FAILED待实际训练、渲染和审阅。数据转换的PASS不能替代训练验收。

## 用户确认与正式启动记录

用户已回复“可以的开始吧”，报告名确认。本次于2026-10-09T18:04:31.944774+09:00启动GPU0正式后台流水线，PID 120207。仅 environment_light 训练；direction_light 转换已完成。实际初始化100000点，seed=0坐标逐点匹配原版默认随机方块，法线全零；生成PLY与model/input.ply哈希一致：`f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`。证据：`inspection/initialization_check.json`。

核验时间2026-10-09T18:05:39.322631+09:00：Stage1已进入训练循环，迭代2494/35000，当前loss=0.0202244。后台状态为RUNNING，无FAILED标记；训练、后续渲染和最终验收尚未完成。本轮只报告成功启动，不记训练PASS。

启动器：`commands/run_pipeline.py`，后台输出`logs/pipeline.log`，各步骤日志`logs/stage1.log`等。运行状态以`state.json`和实际日志为准；后台自动衔接Stage1→insights→Stage2→materials→HDR→NVS，任何步骤失败时停止并保留现场。

## Stage1 结果目检与当前Stage2进度（2026-10-09T18:59:09.891736+09:00）

Stage1 35000步、81442高斯，checkpoint各属性有限；7视频各150帧全部实际解码通过。TensorBoard原始精度测试子集（最多4个相机）：PSNR=41.27235794 dB，SSIM=0.99748832，LPIPS=0.00946695。此为Stage1训练日志子集，不是最终15帧IR NVS。

实际查看同相机同时间GT对照，RGB人物、姿态和平台接近GT；固定测试相机中间t74有局部手臂/腿脚拖影与碎片，平台有局部伪影。Alpha主体完整，细肢体部分边缘较粗；SH基色仍有照明及褶皱痕迹，不等于物理albedo；平台红绿混合，动态静态分离不干净。平台法线相对平滑，人体法线明显斑驳、碎片化。未完成GT法线定量评分。

代表图：[首中尾四类目检](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/inspection/stage1_review.png)、[同相机同时间GT对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/inspection/stage1_aligned_gt.png)。证据：inspection/stage1_artifact_review.json。Stage1媒体验收PASS，仅说明checkpoint和媒体完整并完成代表图目检；存在上述质量限制。

当前原版IR仍在训练，新增15615/20000步；最终材质、HDR、15测试帧NVS和最终验收尚未完成，不能标记整套训练PASS。用户新授权direction_light Stage1，已另建1009-02输出并启动；不改变env自动两阶段计划。

## 55k全前景MAE与三类Normal评测完成

最终55k权重的全部15测试帧Albedo/Roughness/三法线/深度MAE与独立复算完成。Albedo 0.19101779；Roughness 0.19992533；GS/Depth/Mesh Normal 13.88986765°/18.92264049°/14.95726286°，覆盖率92.935564%/88.370929%/93.942224%；Expected/Median深度MAE 0.05899602/0.05912712米。材质与人体法线存在明显质量限制，MAE流程验收PASS。

完整[MAE报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1009-03-jumpingjacks_environment_light-各项MAE与三类Normal评测报告.md)，输出1009-03；本次没有修改来源训练权重。Stage2 55k和原六步流水线已结束。2026-10-09 19:31:37（Asia/Tokyo）状态为PIPELINE_COMPLETED_REVIEW_PENDING；最终15帧原版NVS：PSNR 33.86182582 dB、SSIM 0.97973040、LPIPS 0.02929334。完整视频最终目检待完成，MAE流程PASS不代替六步流水线整体验收。指标与代表图已归档至1009-03报告附件。
