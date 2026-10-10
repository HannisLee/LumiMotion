# 1010-01 原生PBR支持GT方向光与Preset接入

日期：2026-10-10（Asia/Tokyo）。代码来源：本仓库1001-stage2-perlight，基准提交a9e4d1061fec9986d82362f78b42c0c269e6146e加本次工作区改动。

状态：**代码与preset接入完成，未启动训练、GPU渲染或测试。** 本次为实现修改，尚无真实权重运行、梯度/回归或MAE结果，不能标记训练或渲染验收PASS。用户要求原生PBR支持现有dir方向光渲染，并落实到preset。

## 对外选择方式

原生PBR继续使用render_mode=original_ir，新增光照分支pbr_light_mode=gt_directional；通过loss_preset=pbr_gt_directional选择。

```bash
--render_mode original_ir \
--loss_preset pbr_gt_directional \
--pbr_gt_lights_path /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/lights.json \
--pbr_light_intensity_scale 1.0
```

| 参数 | 定义 |
| --- | --- |
| loss_preset=pbr_gt_directional | 选原生PBR损失及GT方向光；关闭学习环境光所需的正则 |
| pbr_light_mode=gt_directional | 固定逐帧世界方向、颜色与强度；auto preset也能识别并保存为pbr_gt_directional |
| pbr_gt_lights_path | 原始多SUN元数据或已有顶层帧号单灯兼容表 |
| pbr_light_intensity_scale | 固定正倍率，默认1；不进行训练RGB自动标定 |
| wo_indirect | 关闭原生Stage1 SH间接光近似，供显式消融 |
| wo_indirect_relight | 方向替换时关闭来源照明的SH间接光近似 |

默认光照分支为envmap。显式CLI权重仍优先；GT分支若显式设置非零环境光正则会报错，因为没有环境光优化器。Lambertian配置与PBR光照参数交叉组合会拒绝。

## 数据与方向约定

scene/pbr_directional.py直接读取新版frames→帧号→lights数组，支持多盏方向不同、颜色不同的SUN，也接受已有单灯顶层帧号表。每盏光要求light_dir_world为世界单位向量、方向为表面指向光源，显式非负intensity、线性light_rgb；SUN角大小为0。按当前相机原帧号选择，不对测试光照做插值或学习，不从light_pos_world推断方向。

当前dir两盏SUN各intensity3，scale1时分别参与PBR直射求和，总强度为6；无需预先合并灯或转换lights.json。本实现将intensity按着色器的线性入射辐照度参数使用，不沿用1009-04 Lambertian的训练RGB标定3.09641359。Blender材质/颜色管理与原生BRDF的匹配仍需真实渲染验收。

## 原生PBR着色

复用原生Gaussian材质光栅化、世界Normal、视线、GGX_specular和动态BVH。对于每盏SUN，直接计算：

`Lo_direct = sum_lights[(albedo_linear / pi + native_GGX) * intensity * light_rgb * max(n·l, 0) * visibility]`

visibility通过原生pc.trace沿确切SUN方向取得，射线起点沿用light_t_min；没有把delta方向光塞入半球随机样本，也不乘半球面积。GGX/Fresnel等BRDF公式与原生PBR相同，roughness仍参与镜面着色并进入原生优化器。

训练期保留原生Stage1 SH间接光近似和相关detach/wo_indirect开关。方向光替换时，这一近似仍来源于训练照明，不声称重新求解多次反射；可用wo_indirect_relight关闭。HDR替换走原生EnvLight及原生PBR重光照路径，使用保存材质与BVH。

原生Stage2的材质/opacity/SH及颜色形变优化组、漫反射warmup和优化步频继续使用原来的Trainer；本次不套用Lambertian的仅albedo训练器。已知SUN不创建或更新EnvLight，环境光正则自动置0；TensorBoard跳过不存在的环境光展示。

## 保存和离线入口

首次训练提供独立stage1_model_path和正load_iter，输出目录为空；复制35k PLY与deform到独立输出、记录源哈希。GT灯光原文件只读，快照保存为model_mlp/pbr_gt_lights.json，stage2_config.json保存模式、倍率和快照路径。

保存PBR模型时写pbr_directional/iteration_<step>/lighting.json，记录同iteration、灯光快照SHA256与倍率。离线渲染和Roughness MAE使用此记录识别GT方向光PBR，拒绝复制的Stage1默认roughness，不要求不存在的envmap权重。

scripts.render_stage2的eval/insights/materials/directional/hdr/all均已接入这一PBR分支；每个真实相机/time更新BVH，GT方向按原帧号查询。替换time须同时指定替换方向或HDR。转灯默认使用选定帧所有GT灯的RGB强度和；HDR使用原生坐标变换、曝光及采样设置。保存的PBR HDR线性图为最终sRGB合成输出的逆sRGB值。

旧render_materials、render_relight_with_hdr和eval_nvs_dynamic简式入口根据保存配置分派至render_stage2；旧式入口使用保存的数据/配置，未识别的额外CLI参数会报错。推荐新统一入口。

Albedo与Normal/Depth/Mesh评测沿现有原生PLY/PTH入口。Roughness评测新增方向光checkpoint识别；有实际PBR材质后该MAE可评测，原Lambertian仍没有roughness输出。

## dir使用示例（尚未执行）

以下示例从已完成的1009-02/35k初始化，输出拟定1010-01；需使用本仓库代码。

```bash
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
CUDA_VISIBLE_DEVICES=0 /home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage2 \
  --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view' \
  --stage1_model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp' \
  --model_path 'output/1010-01-jumpingjacks_direction_light-原生PBR-GT方向光/model' \
  --load_iter 35000 --iterations 55000 --save_iterations 40000 50000 55000 --test_iterations 55000 \
  --render_mode original_ir --loss_preset pbr_gt_directional \
  --pbr_gt_lights_path /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/lights.json \
  --pbr_light_intensity_scale 1.0 --depth_ratio 0 --resolution 2 \
  --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask \
  --train_light_folder images --test_light_folder images --deform-type mlp --sh_degree 3 --hyper_dim 1

CUDA_VISIBLE_DEVICES=0 /home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.render_stage2 \
  --model_path 'output/1010-01-jumpingjacks_direction_light-原生PBR-GT方向光/model_mlp' \
  --load_iter 55000 --task all --hdr_samples 2048
```

实际训练后再使用eval_albedo_gt、eval_geometry_gt及eval_roughness_gt完整评测，动态Mesh仍由extract_geometry_mesh_dynamic提取；dir粗糙度GT需要从dir自己的blend导出，不能使用env roughness真值。

## 修改位置与证据

| 文件 | 修改 |
| --- | --- |
| arguments/stage2.py | PBR光照分支、GT路径和固定倍率参数 |
| scene/pbr_directional.py | 多SUN读取、方向/强度验证、模型记录识别 |
| gaussian_renderer/render_ir.py | 精确SUN直射GGX、BVH透射率、原生SH间接光衔接 |
| scripts/loss_stage2.py | pbr_gt_directional preset及组合约束 |
| scripts/train_stage2.py | 固定光照接入、快照与保存，环境光优化分支 |
| scripts/render_stage2.py | PBR GT评测/转灯/HDR与动态BVH |
| scripts/eval_roughness_gt.py | 实际方向光PBR权重及光照记录识别 |
| scripts/stage2_compat.py、scripts/eval_nvs_dynamic.py | 旧式简式入口分派 |
| utils/train_report_utils.py | 无EnvLight时的日志路径 |

本次进行源码阅读和变更人工检查，未新增或运行测试，未开始实验。代码在当前1001-stage2-perlight工作区，文档同步到本地相关checkout；使用其他checkout运行时需带入对应代码。

[实现补丁](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-01-原生PBR-GT方向光代码接入/implementation.patch)；[源码哈希和实现状态](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-01-原生PBR-GT方向光代码接入/provenance.json)。
