# 1009-02 jumpingjacks_direction_light 原版 LumiMotion Stage1 训练报告

日期：2026-10-09，Asia/Tokyo。状态：Stage1正式运行中。

用户已要求运行另一数据集的Stage1；使用已完成转换与原版reader核验的direction_light，仅训练原版SH Stage1 35000步，随后自动生成150时刻RGB、Alpha、SH基色、normal、动态/静态分离视频。

原始目录 `/home/lihan/data/LH-data/danamic/jumpingjacks_direction_light`；转换目录 `/home/lihan/data/LH-data/transfer-dynamic/jumpingjacks_direction_light`。150帧，135 train / 15 test，time=frame/150，800×800，resolution=2（训练400×400）；黑背景，soft Alpha来自albedo，原版reader合成一次。真实光照为两盏重叠SUN，各Blender强度3；SH训练不读取GT灯光或GT法线/albedo RGB。

服务器venus，Conda lumimotion，GPU0；与env Stage2并行，同卡最多2个训练进程。执行原版 `/home/lihan/reproduce/LumiMotion` main，提交及已有FOV修复见code_commit.txt/code_diff.patch。参数与env Stage1完全一致，独立data_view与model_mlp，从头默认seed=0、100000点随机方块[-1.3,1.3)初始化。

iterations=35000，densify_until_iter=20000，lambda_separation=0.001，d_xyz_loss_weight=0.001，d_color_reg_loss_weight=0.01，binarization_warm_up=1000，depth_ratio=1（离线insights为0），显式gt_alpha_mask_as_scene_mask。其它参数沿用原版。

输出：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1`；实际source：该目录data_view；实际model：model_mlp；iteration：35000。完整命令及每步日志：

```bash
cd /home/lihan/reproduce/LumiMotion
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.train_stage1 --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --iterations 35000 --test_iterations 35000 --save_iterations 10000 20000 30000 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/logs/stage1.log`。

```bash
cd /home/lihan/reproduce/LumiMotion
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage1_insights --source_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/data_view' --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model' --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --load_iter 35000 --depth_ratio 0.0
```

日志：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/logs/stage1_insights.log`。

渲染输出 `model_mlp/renders_stage1_insights/ours_35000/`，代表时刻t0/t74/t149，固定测试相机frame_0010。35k点云、deformation与初始化PLY均保留。Stage1日志测试指标属于原版少量相机子集，不能当成全15帧NVS或物理albedo精度。基色是SH代理，含光照痕迹。

状态见state.json；后台commands/run_stage1.py逐步检查checkpoint和7个完整150帧视频，失败停止并保留现场。训练未完成时无最终质量指标；四类目检与最终PASS/FAILED待实际完成后回填。

## 正式启动核验

用户已直接授权，2026-10-09 2026-10-09T18:56:40.886162+09:00启动后台，PID=129768。与env Stage2在GPU0并行，两个训练进程显存充足。核验时间2026-10-09T18:59:09.891736+09:00，Stage1已达2610/35000步，loss=0.0390584，状态RUNNING，无FAILED标记。

初始化验证PASS_DEFAULT_RANDOM_CUBE：100000点、seed=0坐标逐点精确匹配、全零法线，私有points3d.ply与model/input.ply哈希相同。训练与视频生成尚在进行，最终四类目检待完成。

Stage1 35000步完成：2026-10-09T19:46:07.255044+09:00；全时序可视化与目检待完成。

流水线完成：2026-10-09T19:46:35.708700+09:00；35k checkpoint完整，7个视频各150帧完整解码；真实目检与最终质量验收待执行。

## Stage1最终流程与质量检查

2026-10-09T20:02:12.161508+09:00实际查看固定测试相机0010、t0/t74/t149的RGB/Alpha/SH基色/Normal及静动态分离代表帧。99,827 Gaussian属性和32形变Tensor有限；7视频各150帧解码通过，流程与产物验收PASS（不表示达到质量阈值）。RGB主体和平台可辨认，中间时刻人体及腿脚有局部拖影/碎片；Alpha细肢体边界不规则，脚部残影；SH基色保留强照明、阴影和褶皱，平台红绿混杂，静动态分离不干净；人体法线明显斑驳。

[首中尾代表图](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/inspection/stage1_first_middle_last.png)，人工检查见inspection/visual_review.json。本次GT Lambertian Stage2使用该只读35k来源，几何及形变冻结，上述几何缺陷可能继续保留；见1009-04训练报告。
