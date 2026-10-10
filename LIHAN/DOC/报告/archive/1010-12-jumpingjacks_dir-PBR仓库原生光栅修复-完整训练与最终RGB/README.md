
## B06完整20000步结束、55k实际退出与最终RGB早期目检（2026-10-10T07:46:40.050962+09:00）

完整Stage2源35000→55000共20000步，07:00:03.390000 JST启动，log保存55k与Training complete均07:44:25，循环44:02；train exec30746实际exit0并关闭，真实PID258688已退出，不是观察超时。training_runtime.json已记录verified_training_session_closed/actual exit0，Stage1未重训。

55k独立检查exec88063实际exit0：全部99827 Gaussian保留、数值有限、opacity/scale/feature及32来源NNtensor原值、sourceSHA保持。最大canonical XYZ位移0.08000001311m、旋转60.00006485°容差内；DC [.031,.969]硬clamp0，原RGB拟合E6.61184454。source XYZ anchor、plane和native CUDA来源绑定通过，核心469文件/GLM hash仍匹配launch。该检查不是最终GT normal或质量验收。

55k最终15 TEST RGB：PSNR19.68220260、SSIM0.89777244、LPIPS0.11070925、前景PSNR12.48856820、前景bbox SSIM0.77829250/LPIPS0.22170198。B05同55k PSNR19.37506314→本轮19.68220260（+.30713946dB）；LPIPS.11125243→.11070925，稍改善，不等于normal7°或画质PASS。

已实际view inspection/rgb_final_55000_comparison.png中GT/B05/B06首中尾0010/0080/0150：黑背景/主体存在，衣服、头手稀疏斑驳，脚周平台伪凸起/灰黑污染仍明显；两轮总体接近，GT0080本身背光暗，不凭暗单独失败。当前FAILED_SO_FAR_EARLY_FINAL_RGB_ONLY，完整四类/HDR/方向光、七MAE仍PENDING，未commit/push/新建目标分支。

真实后处理PID258864/startticks9362140正在全量render；child265977/startticks9623907、cwd与scripts.render_stage2 argv实际核验存活，使用--task all --load_iter55000 --hdr_samples2048。后续albedo、15time×135TRAIN Mesh、normal_depth、roughness、独立七MAE复算、产物审核、报告仍须完成，不用RGB指标代替全部七MAE。没有重启训练或更改已运行参数/源。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_checkpoint.py 55000
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_rgb_comparison.py
```

证据training_runtime.json、inspection/checkpoint_55000_audit.json、final_55000_early_review.json、rgb_final_55000_comparison.png、model_mlp/renders_stage2/ours_55000/eval/metrics.json；日志logs/train.log、checkpoint_55000_audit.log、render.log、postprocess.log。完整训练/渲染命令在planned_commands.json和commands/train.sh/render.sh。
