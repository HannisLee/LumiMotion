# 1006-11 jumpingjacks150_v5_spec32 原版随机方块初始化默认两阶段训练计划

日期：2026-10-06（Asia/Tokyo）。状态：**COMPLETED（35k/55k、全部渲染与评测完成；流程及产物PASS，质量限制见最终结果）**。

用户已核对jumpingjacks及默认随机方块SH→IR方案，并于本轮明确回复“OK开始吧”。现已启动独立正式后台流水线；本次直接按确认的正式计划执行，未另行启动冒烟。

## 数据与历史记录

- 数据集：`jumpingjacks150_v5_spec32`，动态场景。
- 原始数据：`/home/lihan/data/d-nerf-relight-spec32/jumpingjacks150_v5_spec32`。
- 此前 source 路径：`/home/lihan/datasets/LumiMotion/d-nerf-relight-spec32/jumpingjacks150_v5_spec32`。
- 历史记录：[0930-01-jumpingjacks官方原生管线训练.md](0930-01-jumpingjacks官方原生管线训练.md)。0930 训练中止，未完成目标35k及正式评测；本次从头初始化，保留历史产物。
- 训练光照：`chapel_day_4k_32x16_rot0`；测试重光照：`golden_bay_4k_32x16_rot330`。
- 划分：135 train / 15 test，共150帧；原图800×800，resolution=2，训练400×400。

启动前检查完成：train/test JSON可解析、135/15划分且不重复；相机矩阵有限且可逆、FOV及time有效。两种光照300张图像均存在、800×800 RGBA、Alpha含背景0和有效前景。r_0001/r_0075/r_0149 RGB与Alpha真实目检通过，人物和平台轮廓对应。完整统计见实验inspection/data_preflight.json。

## 默认随机方块初始化

**已核实：之前原版jumpingjacks训练的初始化就是默认随机方块。** 原始数据目录的`points3d.ply`本身保存的就是10万点默认随机方块；文件名不代表它是物体表面或GT初始化。原版loader在该文件存在时读取它，不存在且无RGBD时先生成默认随机点云、写成同名文件，再读取。

本轮核验了两个数据路径下的`points3d.ply`与原版历史模型`/home/lihan/reproduce/LumiMotion/output/jumpingjack_chapelday_goldenbay_r2_mlp/input.ply`：三者SHA256均为`f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`；均100000点、零法线，坐标逐点精确匹配NumPy seed=0生成的默认[-1.3,1.3)方块（float32）。历史Stage1日志也记录初始化100000点。因此“读取已有PLY”与“默认随机方块初始化”在这次历史实验中同时成立；本次与原版历史训练的初始化方式一致。

为独立保存本次输入和避免修改原始目录，未来仍使用本实验独立`data_view`，只链接train/test transforms、训练与测试光照图像目录和对应HDR；不复制原始`points3d.ply`，也不包含`rgbd/`或`rgba/`覆盖目录。这是重新生成同样默认随机点云的产物隔离措施，不是将历史物体表面初始化改为随机初始化。

原版Blender reader在输入PLY不存在且没有RGBD时生成：

```python
num_pts = 100_000
xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
```

即10万点，各轴均匀分布于[-1.3, 1.3)，原点中心、零法线。沿用原版随机种子与高斯初始化，不调整方块范围、中心或场景尺度。原始目录与旧模型保持只读。

本轮已创建data_view，启动前没有随机PLY。正式启动时由reader生成`data_view/points3d.ply`，并核对模型`input.ply`的点数、范围、零法线及哈希；记录`Generating random point cloud (100000)`日志。独立新训练不加载0930或其他旧checkpoint。

## 默认管线与参数

执行代码拟使用`/home/lihan/reproduce/LumiMotion`原版main，与1006随机方块实验一致；本轮读取HEAD为`65287461075a1a4c0f54845325cc22039bd14e7d`。后续启动时重新记录实际提交及工作区改动。

配置依据：原版`bash_scripts/synthetic_results_from_paper.sh`中的`chapelday_goldenbay|jumping jack`项。仅选这一场景和光照组合，命令手工展开，不执行遍历全部场景的脚本。

| 项目 | 设置 |
| --- | --- |
| Stage1 | 原版SH，训练几何与形变，35000步 |
| Stage2 | 原版IR，从本实验35000步开始，累计55000步（新增20000步） |
| Stage1 densify_until_iter | 20000 |
| lambda_separation / d_xyz_loss_weight | 0.001 / 0.001（jumpingjacks官方组合配置） |
| d_color_reg_loss_weight / binarization_warm_up | 0.01 / 1000 |
| Stage1 depth_ratio | 1.0 |
| Stage2及离线渲染depth_ratio | 0.0 |
| Stage2 diffuse_sample_num | 512；HDR/NVS评测2048 |
| 输入及背景 | 原始RGBA，显式Alpha监督，黑背景，resolution=2 |
| 形变与SH | 原版默认MLP、hyper_dim=1、sh_degree=3 |

这里的“默认管线”指原版SH→IR。原版main训练入口没有当前perlight分支的`--render_mode`参数，因此下列原版命令不传该参数。其他优化参数沿用原版，不套用CV3有限densification实验参数。

服务器`venus`，环境`lumimotion`，Python为`/home/lihan/miniconda3/envs/lumimotion/bin/python`。GPU0 RTX3090Ti 24GB；启动前空闲24077MiB，无其他计算进程。本次启动正式训练，未另行执行冒烟。

## 输出与完整命令

- 实验输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段`。
- source：上述输出下的`data_view`。
- 命令model_path：上述输出下的`model`；原版自动追加后缀，实际目录为`model_mlp`。
- 日志：上述输出下的`logs/`；后续完整命令、渲染路径、结果和验收记录写入实验`README.md`。

以下为确认的完整命令；实际执行脚本已保存至实验commands/，逐步日志为logs/stage1.log、stage1_insights.log、stage2.log、materials.log、relight_hdr.log、eval_nvs.log。各bash脚本语法检查通过。后台commands/run_pipeline.py逐步记录state.json、检查Stage1/Stage2 checkpoint，失败时停止并保留产物。

```bash
cd /home/lihan/reproduce/LumiMotion
PYTHON=/home/lihan/miniconda3/envs/lumimotion/bin/python
EXPERIMENT=/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段
SOURCE="$EXPERIMENT/data_view"
MODEL="$EXPERIMENT/model"
TRAIN_LIGHT=chapel_day_4k_32x16_rot0
TEST_LIGHT=golden_bay_4k_32x16_rot330
COMMON=(--source_path "$SOURCE" --model_path "$MODEL" --is_blender --eval
        --load2gpu_on_the_fly --resolution 2
        --train_light_folder "$TRAIN_LIGHT" --test_light_folder "$TEST_LIGHT")
mkdir -p "$EXPERIMENT/logs"
set -o pipefail

# Stage1：原版SH，从随机初始化开始。
"$PYTHON" -u -m scripts.train_stage1 "${COMMON[@]}" \
  --gt_alpha_mask_as_scene_mask --iterations 35000 \
  --test_iterations 35000 --save_iterations 10000 20000 30000 35000 \
  --densify_until_iter 20000 --lambda_separation 0.001 \
  --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 \
  --depth_ratio 1.0 --d_color_reg_loss_weight 0.01 \
  2>&1 | tee "$EXPERIMENT/logs/train_stage1.log"

# Stage1：RGB、Alpha、SH基色/分离、normal全时序可视化。
"$PYTHON" -u -m scripts.render_stage1_insights "${COMMON[@]}" \
  --load_iter 35000 --depth_ratio 0.0 \
  2>&1 | tee "$EXPERIMENT/logs/render_stage1_insights.log"

# Stage2：加载本次Stage1，原版IR累计至55k。
"$PYTHON" -u -m scripts.train_stage2 "${COMMON[@]}" \
  --gt_alpha_mask_as_scene_mask --load_iter 35000 --iterations 55000 \
  --test_iterations 55000 --save_iterations 40000 50000 55000 \
  --diffuse_sample_num 512 --depth_ratio 0.0 \
  2>&1 | tee "$EXPERIMENT/logs/train_stage2.log"

# 材质与HDR重光照。
"$PYTHON" -u -m scripts.render_materials "${COMMON[@]}" \
  --load_iter 55000 --depth_ratio 0.0 \
  2>&1 | tee "$EXPERIMENT/logs/render_materials.log"
"$PYTHON" -u -m scripts.render_relight_with_hdr "${COMMON[@]}" \
  --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048 \
  --hdr "$SOURCE/$TEST_LIGHT.hdr" \
  2>&1 | tee "$EXPERIMENT/logs/render_relight_hdr.log"

# 动态NVS：15个测试相机，训练光照下的RGB指标。
"$PYTHON" -u -m scripts.eval_nvs_dynamic "${COMMON[@]}" \
  --load_iter 55000 --depth_ratio 0.0 --diffuse_sample_num 2048 \
  2>&1 | tee "$EXPERIMENT/logs/eval_nvs_dynamic.log"
```

以上每步成功并核验产物后才进入下一步。原版动态NVS脚本在首帧建立BVH后复用，后续报告应注明这一实现限制；不能直接把其指标当成已按逐时刻重建BVH的评测结果。

## 后续验收与当前状态

未来执行后保留35k/55k checkpoint、deformation和Stage2环境光文件；记录训练/渲染完整命令、实际source/model/iteration、日志、各类渲染目录和视频。报告RGB定量指标及RGB、Alpha、albedo/separation、normal四类首/中/尾代表帧真实目检；Stage1 SH基色代理与Stage2物理albedo分别解释。失败或中止产物均保留。

当前正式Stage1已启动；初始化证据由实际input.ply核验后回填，35k/55k、渲染、指标和最终验收仍待完成，不得沿用历史数字或标注整体PASS。

## 实际启动记录

- 启动：2026-10-06T18:03:30.903207+09:00（Asia/Tokyo）。
- 后台PID：412949；运行状态：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/state.json`。
- 六步顺序：Stage1 → Stage1 insights → Stage2 → materials → Golden Bay HDR → dynamic NVS。
- 源码提交、已有FOV兼容补丁及工作区状态保存于code_commit.txt、code_diff.patch、code_status.txt；本轮未改原版训练代码。
- 完整运行记录：[实验README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/README.md)。

初始化实证：**PASS_DEFAULT_RANDOM_CUBE**，100000点，seed=0坐标逐点核验通过、零法线；生成PLY与model input.ply哈希一致，SHA256 `f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`。Stage1已进入训练循环（核验时迭代2276）。

进度核验（2026-10-06T18:43:36.754799+09:00）：Stage1于18:41:28完成35000步，Stage1 insights完成150时刻；Stage2于18:42:00启动，新增步数739/20000（累计35739/55000）。Stage1日志测试PSNR39.66、训练44.62，日志SSIM舍入为1.00，均属原版训练日志子集指标；全量最终NVS指标待完成。GPU进程正常，无FAILED标记。

## 训练结果与代表帧初查

Stage1完成：2026-10-06 18:41:28；Stage2完成：19:21:36（Asia/Tokyo）。两阶段各146065高斯，35k/55k checkpoint完整，点云各属性有限。初查固定相机r_0010的t0/t74/t149；同时对齐r_0010、time=10/150的GT albedo与Golden Bay重光照图。

- RGB：人物、平台和服装完整；Alpha轮廓正常。
- normal：平台较平滑，人物躯干、手臂、头部明显斑驳和碎片化；尚无GT法线定量结论。
- albedo：服装/平台颜色与纹理可辨，但与GT对比仍有明暗和褶皱痕迹，材质与照明未完全分离。
- roughness：大部分区域较亮，平台砖块边缘和人物表面有结构化变化；未做GT粗糙度评分。
- Golden Bay：整体颜色与轮廓接近同视角同时间GT，衣袖、皮肤、平台边缘有斑驳和局部伪影。仅代表帧目检，完整HDR视频与NVS尚待完成。

Stage1原始精度测试子集：PSNR=39.655670，SSIM=0.99646771，LPIPS=0.01313273（最多4测试视角）。不能与Stage2最终15视角指标直接作同协议升降比较。

代表图：[Stage1四类目检](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/inspection/stage1_review.png)、[材质与粗糙度](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/inspection/materials_review.png)、[同相机同时间GT对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/inspection/aligned_gt_review.png)。

## 最终结果与验收

最终状态：**COMPLETED；流程与产物验收PASS，存在材质/法线质量限制**。完成时间：2026-10-06T19:29:13.744745+09:00（Asia/Tokyo）；六步流水线耗时1:25:42。Stage1 35000、Stage2累计55000，两阶段各146065高斯；10个视频各150帧全解码通过，所有计划的checkpoint、渲染、15帧NVS与指标完整。

| 评测 | 范围 | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | --- | ---: | ---: | ---: |
| Stage1 SH训练日志测试 | 最多4测试视角子集，TensorBoard原始精度 | 39.655670 | 0.99646771 | 0.01313273 |
| Stage2原版IR动态NVS | 15测试视角，GT Alpha遮罩后黑背景整图平均 | 27.342613 | 0.94895755 | 0.01941747 |

两行评测范围不同，不可据此计算同协议升降。Stage2原版NVS复用首帧BVH，以上按原版实现解释。未进行GT法线/深度/粗糙度全量定量评分，也未执行GT albedo尺度拟合；重光照未采用GT材质尺度修正。

真实目检：

- RGB与Alpha：人物、平台与动作完整，代表帧未见大面积背景泄漏；NVS存在局部模糊及明暗/纹理误差。
- albedo/separation：人物与平台分离可辨；albedo相对同相机同时间GT仍有明暗/褶皱残留，物理材质与照明未完全分离。
- normal：平台较平滑，人体躯干、头部与手臂明显斑驳和碎片化。
- HDR/roughness：Golden Bay整体颜色与轮廓接近对齐GT，但衣袖、皮肤与平台边缘有局部斑驳；roughness含结构化变化，不能据此宣称物理恢复准确。

PASS仅指流程、产物和代表帧检查完成，未设材质/几何质量阈值。全部10视频真实完整解码，人工目检固定相机首/中/尾，以及NVS首/中/尾（尾帧也是保存PNG PSNR最差），没有逐帧人工审阅所有视频。

证据：`inspection/artifact_checks.json`、`inspection/manual_review.json`、`inspection/initialization_check.json`、`model_mlp/results_nvs_dynamic.json`。

最终图片：[同相机同时间GT对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/inspection/aligned_gt_review.png)、[NVS首中尾/最差](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/inspection/nvs_review.png)。


新增MAE评测：[1006-13双阶段Albedo与Normal/Depth报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-13-jumpingjacks-双阶段Albedo与NormalDepth-MAE评测报告.md)。15测试帧，主值为GT Alpha≥0.5全前景线性RGB MAE：Stage1 SH代理0.08579548，Stage2 IR albedo0.06169353，下降28.09%；内区另报0.08566499/0.06209056。Normal真值不存在、150张Depth PNG全部纯白，几何MAE为N/A，不记为零。状态：ALBEDO_COMPLETED_GEOMETRY_GT_UNAVAILABLE。
