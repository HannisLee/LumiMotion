
## B06完整渲染及HDR/转灯实际目检（2026-10-10T07:56:04.688664+09:00）

全量render子进程实际exit0（runtime_history.jsonl），结束07:54:59.887392 JST，用时632.9978秒；RENDER_COMPLETED=all。实际150×4四类、150 HDR2048样本、72转灯和6视频存在（完整视频解码另由最终产物审计执行，不提前标通过）；后处理进入材质/几何评测。实际view inspection/relight_final_55000.png：HDR0001/0076/0150和转灯0000/0036/0071，黑背景/主体平台存在、光照有响应，但人物正背面和脚周斑驳、伪起伏/材质污染仍明显，仍FAILED_SO_FAR；无重光照GT，不声明精度。方向光固定0001相机/time/geometry转灯，alpha逐bit保持，亮度全图均值范围.01817811653–.07329406589。

灯表重新核验150帧各SUN白色单位intensity1/light_rgb[1,1,1]，仅原GT方向；实际转灯强度是该白色单位表sum×保存的训练RGB拟合scale6.611844539642334。relight.json遗留generic字符串“frame GT RGB irradiance sum × saved scale”不是使用原GT color/intensity的证据，本轮保留原渲染metadata、另在报告澄清，不改已完成渲染/核心源码。最终结果后再读GOAL决定修正文案，当前输入边界未变。

命令python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_relight_comparison.py实际exit0；证据inspection/final_55000_relight_review.json、relight_final_55000.png、model_mlp/renders_stage2/ours_55000/directional/relight.json及hdr/hdr.json。全部15七MAE与Depth/Mesh/roughness审核仍PENDING，未commit/push/新建目标分支。
