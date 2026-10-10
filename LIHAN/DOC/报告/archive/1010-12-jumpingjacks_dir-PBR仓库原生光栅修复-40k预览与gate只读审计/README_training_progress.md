
## 正式启动：真实进程

{"status": "TRAINING_VERIFIED_LIVE", "training_pid": 258688, "process_start_ticks": "9357639", "session_id": 30746, "source_iteration": 35000, "target_iteration": 55000, "steps_planned": 20000, "loss_preset": "pbr_gt_directional_plane_native", "started_at": "2026-10-10T07:00:03.390000+09:00", "start_time_source": "proc_btime_plus_start_ticks", "observed_at": "2026-10-10T07:00:34.438075+09:00", "argv": ["/home/lihan/miniconda3/envs/lumimotion/bin/python", "-u", "-m", "scripts.train_stage2", "--source_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/data_view", "--stage1_model_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp", "--model_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/model", "--load_iter", "35000", "--iterations", "55000", "--save_iterations", "40000", "50000", "55000", "--test_iterations", "40000", "55000", "--render_mode", "original_ir", "--loss_preset", "pbr_gt_directional_plane_native", "--pbr_gt_lights_path", "/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/gt_directions_only.json", "--pbr_light_intensity_scale", "3.141592653589793", "--depth_ratio", "0", "--resolution", "2", "--is_blender", "--eval", "--load2gpu_on_the_fly", "--gt_alpha_mask_as_scene_mask", "--train_light_folder", "images", "--test_light_folder", "images", "--deform-type", "mlp", "--sh_degree", "3", "--hyper_dim", "1"]}

仅新Stage2，未重跑Stage1；完整MAE/最终质量尚无，不把source35000或20步smoke作本轮结果。

## B06启动配置/后处理核验

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T07:03:19.241733+09:00", "steps": 1470, "total_steps": 20000, "iteration": 36470, "source_cuda_sha256": "82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0", "process_handles": {"train": {"pid": 258688, "session_id": 30746, "start_ticks": "9357639"}, "postprocess": {"pid": 258864, "session_id": 47576, "start_ticks": "9362140"}}, "postprocess": "WAITING_LIVE_TRAIN", "final_mae": false}

正式配置仅loss_preset、backend和3本轮路径字段与B05不同，所有其它值一致；15187支持/所有来源plane数组逐元素与B05相同。源码manifest逐SHA核验PASS，实际仓库native CUDA综合SHA82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0绑定。首次inline审计误用默认vllm python缺smplx失败，未改训练；正确venus lumimotion执行归档commands/audit_launch.py实际exit0。真实后处理绑定同train PID/start ticks，完整9步将在train完成后执行，40k预览同PID绑定，不退出后盲重启。完整MAE/质量仍PENDING，未commit/push/新分支。

## B06实际进度与进程绑定更新

{"status": "TRAINING_VERIFIED_LIVE", "updated_at": "2026-10-10T07:04:56.276364+09:00", "steps": 2261, "total_steps": 20000, "iteration": 37261, "process_handles": {"train": {"pid": 258688, "session_id": 30746, "start_ticks": "9357639"}, "postprocess": {"pid": 258864, "session_id": 47576, "start_ticks": "9362140"}, "preview_40000": {"pid": 258868, "session_id": 25842, "start_ticks": "9362256"}}, "postprocess": "WAITING_LIVE_TRAIN", "final_mae": false, "commit_allowed": false}

TRAIN/POST/40k-preview实际PID/start ticks/cwd全部复核，40k预览cmdline为相对脚本路径，按resolve核验，不因绝对路径子串不匹配盲重启。核心469源码/GLM/快照SHA仍匹配launch。仅修正文档生成器末尾遗留1010-10引用为本轮1010-12，保留修正前脚本；运行中训练/renderer/MLP/参数未改。

## B06 40k真实检查点、全15RGB预览与只读gate诊断（2026-10-10T07:15:26.674133+09:00）

完整训练仍运行：6962/20000，实际iteration41962；train258688/startticks9357639与post258864/startticks9362140/cwd重新核验，核心469来源/GLM源码SHA匹配launch，未改变本轮训练参数或源码。preview exec25842、checkpoint审计exec1420、gate诊断exec90456均实际exit0并关闭。

检查点：全部99827 Gaussian保留，opacity/scale/feature与source32NNtensor冻结不变，XYZ最大位移0.03539922m、canonical旋转最大60.00002670°，DC RGB范围[.031,.969]、硬clamp0，训练RGB拟合光强5.28000259。native CUDA marker来源与anchor/plane绑定通过；这些只是保存完整性，不是法线7°或质量PASS。

15 TEST帧40k RGB：PSNR18.51417174、SSIM0.88792369、LPIPS0.10788124、前景PSNR11.46161042、前景bbox SSIM0.75713942/LPIPS0.22566057。B05同40k PSNR18.39326180/LPIPS0.10931317，略改善，不能由此声称GS法线改善。

已实际view inspection/preview_40000_comparison.png中GT/B05/B06的0010/0080/0150：黑背景正常、人物和平台存在，但头手细节欠完整、衣服斑驳、脚周平台伪凸起/灰黑残影仍明显，两个预测总体接近；0080 GT本身暗，不以暗单独判失败。中间质量FAILED_SO_FAR，55k完整七MAE/四类/HDR仍PENDING，未commit/push/新建分支。

只读gate诊断：source NN隐藏层与32参数固定，只开放当前feature logit的输出乘法gate自动求导；当前位置/旋转形变与原FrozenScene路径逐bit相等。0001/0075/0149三TRAIN帧PBR/GGX/BVH的RGB+SH、normal-depth和自身输入Alpha梯度均finite且存在非零梯度，source SHA/状态fingerprint不变；0optimizer、feature/NN值不变、未读GT标签/测试RGB、未生成新训练prior。本轮仍冻结gate。来源预测筛选的6836动态plane候选不是GT物体标签，梯度方向不证明它们全部属于平台。

| TRAIN帧 | RGB+SH gate梯度L2 | 非零点 | 来源动态plane候选RGB+SH L2 | 候选非零点 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.0005630466 | 34817 | 0.00024462515 | 4672 |
| 0075 | 0.01247051 | 36220 | 0.010364828 | 5064 |
| 0149 | 0.002293421 | 34628 | 8.5120133e-05 | 4629 |


该机制可作为本轮最终失败后的新候选，仅证明可导性，未实施新训练/未声称能到7°。不能从0075的大梯度推定正确几何或把候选全部静态化；需要重新读取本GOAL、独立完整训练与全量MAE验证。

完整诊断命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_feature_gate_gradients.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 40000
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_preview_comparison.py
```

证据：/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/inspection/feature_gate_gradients.json、checkpoint_40000_audit.json、preview_40000_review.json、preview_40000_comparison.png；logs/audit_feature_gate_gradients.log、checkpoint_40000_audit.log、preview_40000.log、preview_render_40000.log。预览完整命令另在commands/preview_40000_argv.json；所有失败轮保持原权重与统计，最终MAE尚无。
