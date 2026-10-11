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

计划保存iteration 10000、20000、30000、35000；测试重建在35000执行。训练中追加每阶段检查点、日志尾部、运行PID、GPU占用与迭代状态。完整训练结束后由评测代理对完整15帧按既有协议评测GS／Depth／Mesh法线MAE；四类可视化RGB／Alpha／Normal／分离目检和最终PASS/FAILED在实验README及本计划回填。当前计划状态：RUNNING；正式重试PID 1972511，启动于2026-10-11T07:31:24Z，output/1011-13-LH-static-jumpingjacks_DL_V2-T01-SH静态零形变-nativeRasterizer。


## 启动协议修正及T01正式重试（2026-10-11T07:35:15.408272+00:00）

第一次#10启动日志止于`Reading Training Transforms`，无traceback、无退出码，PID已不存在，退出原因未知。该入口直接运行旧快照，未经过`bootstrap_1011.py`绑定仓库native rasterizer，故标记启动协议失败并保留日志/配置/模型目录，不用于本Goal结论。首轮smoke同样因未绑定native rasterizer而作废，零梯度审计及渲染文件完整保留。

验证后的SH smoke位于`output/smoke_test/1011-06-LH-static-jumpingjacks_DL_V2-T01-SH-native-assets冒烟`：bootstrap成功，exit 0；20/20步d_color梯度有限且非零；151个任意时间点输出几何严格为零；点云与共同初始化sha吻合。`static_dl_training.json`保存repository native source SHA及.so SHA。独立重载验证见同目录`reload_validation/`。

正式T01重试输出改用`output/1011-13-LH-static-jumpingjacks_DL_V2-T01-SH静态零形变-nativeRasterizer`，避免覆盖失败#10。启动命令先执行`LIHAN/scripts/bootstrap_1011.py <snapshot> scripts.train_static_dl`；使用带退出码监督的持久exec session。核心快照为`output/1011-09-静态DL-Goal训练总控/snapshots/T01-sh-native-assets-20261011`，包括训练入口依赖的`assets/bsdf_256_256.bin`。


## 正式训练与10k中期评测（2026-10-11）

实际正式输出为`output/1011-13-LH-static-jumpingjacks_DL_V2-T01-SH静态零形变-nativeRasterizer`，模型目录`model_mlp`，使用修正后的核心快照`output/1011-09-静态DL-Goal训练总控/snapshots/T01-sh-native-assets-20261011`。训练入口通过`LIHAN/scripts/bootstrap_1011.py`先绑定仓库native CUDA rasterizer再导入快照，完整参数命令保存在实验目录`commands/train.sh`；PID 1972511、GPU0、监督session 75000。初始直接run_static_dl.py及#10目录属于协议失败历史，不作为本次有效训练，均保留。当前正式训练继续至35000步。

iteration_10000的15帧Goal评测已完成：Mesh、normal_depth与独立audit三个步骤均exit 0，完整性PASS；数值阈值FAILED。GS-render MAE 11.257213802°（coverage 0.998600782）、depth-normal MAE 22.836470255°（coverage 0.966570124）、mesh MAE 16.853138010°（coverage 0.993374941），目标阈值为3.51°、7.35°、5.46°。原始预测网格、15帧数组、逐帧CSV、日志和审计结果保留在`output/1011-13-LH-static-jumpingjacks_DL_V2-T01-SH静态零形变-nativeRasterizer/evaluation_10000/`；详细结论见其中`goal_mae_summary.json`和`README.md`。此为10k中期检查，训练继续，35k最终评测未执行。
