# 静态ELDL-V3 S0原生Stage1从头训练计划

日期：2026-10-24；服务器 ushas，环境 lumimotion-ushas；所属GOAL：`LIHAN/DOC/GOAL/1024-02-静态ELDL从头Stage1与GT方向光GS法线突破.md` 阶段A（S0）。

## 1. 划分与输入（P0结论）

- 划分：沿用1019-01总控 `data_fit_v2`（I00内部fit117）既有划分，经P0核验后固定。
  - TEST15：10,20,…,150（原划分不变）。
  - DEV18：61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,79（61..79去TEST帧70）。
  - TRAIN117：其余帧。新S0与后续A0/G0/G1等所有阶段都只加载这117帧；DEV/TEST训练全程不可见。
  - 落档：`output/1024-17-LH-static-jumpingjacks-ELDL-V3-P0协议与净化输入/split_train117_dev18_test15.json`（SHA256 `6ce991aeff528743740dbaf1a1ab2bf320c8a3797f24922f967b7be2b2659e67`）。
- source：`output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_fit_v2`（no_gt_manifest SHA256 `beebcbb9d194cac0c533e6124ab89aebb48fb7717b5c10cbc92b075f960ef237`，transforms与117张16位RGBA逐文件SHA在manifest内并由入口逐一复核）。
- P0逐像素核验（`p0_audit.json`）：Alpha>0处RGB与原 `image/` 逐位相同；Alpha=0处RGB已清零（原图为不透明世界背景，16位码21742≈0.3318 sRGB）；Alpha通道与 `albedo/` Alpha逐位相同，未保留albedo RGB；原图Alpha恒为65535。加载器 `utils/rgba16.FloatRGBA`：uint16/65535、RGB预乘Alpha（黑背景）、INTER_AREA到400×400，满码掩膜在原分辨率取RGB==65535且Alpha>0。前景满码通道比例平均约32.0%（最大37.5%），后续censored单边约束必须保留。
- 初始化：`data_fit_v2/points3d.ply`（SHA256 `f9a2601f…e256c971`），即历史“原生随机100k立方体，seed 0”，与B00使用的 `data_train_v2/points3d.ply` 为同一文件，同规模随机点。`safe_state`：python/numpy seed 0、torch 5000000。
- 相机：transforms的c2w/内参与 `camera.json` 逐帧最大差<1e-6；时间=(frame-1)/149（静态配置下不被任何自由度使用）。

## 2. 静态配置（与B00的差别）

新入口 `scripts/train_static_eldl_stage1.py` 直接复用原生 `scripts/train_stage1.Trainer`（与B00快照逐文件一致），只做以下训练前设置：

1. 形变网络三个几何输出头 `gaussian_warp/rotation/scaling` 在优化器第一次更新前置零并 `requires_grad=False`，d_xyz/d_rotation/d_scaling 恒为0；
2. `pred_color=False`，不构造时变颜色头，d_color=None（无逐帧阴影调制/动态颜色）；
3. 训练中每1000步抽查形变输出严格为0，结束时核对几何头全零、形变网络state_dict哈希与初始化一致（冒烟中确认整网未变）。

因此S0是共享canonical Gaussian的真正静态模型，不是训练后截取某一帧。B00为135 TRAIN、时变MLP与时变颜色全开；两者数据量与自由度均不同，比较时须注明。

## 3. 预算、保存与真实更新数

- 名义 `--iterations 35000`，原生日程不变：densify 500–20000每100步，opacity reset每3000步（最后一次18000），normal/distortion自8000后启用（λ_normal=0.02，λ_dist=1000），SH每1000步升阶，XYZ LR按30000步指数衰减。
- 保存：10000、20000、35000（原生Trainer另会在warm_up-1=999自动保存一次，作诊断用）。
- 原生语义：iteration k 先保存后执行optimizer.step，且 iteration==35000 不更新。故 checkpoint k 含 k−1 次真实更新：@10000=9999，@20000=19999，@35000=34999。冒烟实测计数与此一致（200步→199次）。不追加补偿步，保持与B00同口径；S0@35000作为预定父模型（34999次更新），10000/20000为诊断点。

## 4. 命令（与B00一致的原生参数）

```bash
--deform-type mlp --is_blender --eval --load2gpu_on_the_fly --resolution 2 --train_light_folder images --test_light_folder images --gt_alpha_mask_as_scene_mask --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1 --d_color_reg_loss_weight 0.01 --data_device cpu --start_normal_reg 8000
```

源码快照：`<S0目录>/snapshot_code`（bootstrap_1011.py 从快照运行），完整命令、PID、源码与输入SHA见S0 README。

## 5. 冒烟

`output/smoke_test/1024-04-LH-static-jumpingjacks-ELDL-V3-S0原生Stage1冒烟`：200步，warm_up=10（使形变路径实际执行），densify/reset间隔缩短以覆盖一次真实拓扑与reset。exit0；191次形变零输出抽查全部为0；几何头全零；形变网络state未变；更新计数199；保存50/100/200均成功；拓扑100000→18908。冒烟只证明工程，不作质量结论。

## 6. 资源与验收

GPU1单任务；预计约1小时（B00同日程约56分钟）。完成后由独立评测Agent对S0@35000做同协议DEV/TEST评测（GS/固定分母/覆盖/Depth/Mesh及四类图），通过重载与基线评测后封存为父模型。不得以training loss判定质量。
