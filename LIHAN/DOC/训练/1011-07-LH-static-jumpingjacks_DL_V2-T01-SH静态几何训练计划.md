# 1011-07-LH-static-jumpingjacks_DL_V2-T01-SH静态几何训练计划

服务器：ushas；环境：`lumimotion-ushas`；静态DL原训练集135帧，测试10、20、…、150共15帧。

## 训练假设与约束

本轮固定时变几何为零，保留原生SH与`d_color`外观路径，以确认静态场景中明暗变化不会被转成伪几何。只允许训练RGB、image PNG自身Alpha、相机/时间和允许的训练方向信息；本SH控制入口不读取GT方向或任何GT法线、深度、材质、粗糙度、阴影、光强。初始化使用共享100000点随机点云，NumPy seed=0；SHA256：`f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`。

训练source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view`。source的RGB/Alpha只源自原始image RGBA，alpha=0处RGB清零以匹配既有loader；真实camera-to-world保留，135张训练帧与15张测试帧不变。评测source固定为旧协议`output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段/eval_view`。

核心代码快照：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/snapshots/T01-sh-static-heads-20261011`；SHA清单：`snapshot_manifest.json`。正式训练从快照导入，不依赖后续工作树修改。输出：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-10-LH-static-jumpingjacks_DL_V2-T01-SH零形变`；模型：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-10-LH-static-jumpingjacks_DL_V2-T01-SH零形变/model_mlp`；日志：`logs/train.log`；资源调度前后记录见`output/1011-09-静态DL-Goal训练总控/resource_snapshots/`。

## 完整训练命令

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/snapshots/T01-sh-static-heads-20261011/run_static_dl.py --source_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-09-静态DL-Goal训练总控/data_view --model_path /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1011-10-LH-static-jumpingjacks_DL_V2-T01-SH零形变/model --deform-type mlp --static_dl_mode sh --iterations 35000 --save_iterations 10000 20000 30000 35000 --test_iterations 35000 --densify_until_iter 20000 --lambda_separation 0.001 --d_xyz_loss_weight 0.001 --binarization_warm_up 1000 --depth_ratio 1 --d_color_reg_loss_weight 0.01 --resolution 2 --is_blender --eval --load2gpu_on_the_fly --gt_alpha_mask_as_scene_mask --train_light_folder images --test_light_folder images
```

## 检查点与状态

计划保存iteration 10000、20000、30000、35000；测试重建在35000执行。训练中追加每阶段检查点、日志尾部、运行PID、GPU占用与迭代状态。完整训练结束后由评测代理对完整15帧按既有协议评测GS／Depth／Mesh法线MAE；四类可视化RGB／Alpha／Normal／分离目检和最终PASS/FAILED在实验README及本计划回填。当前计划状态：PENDING。
