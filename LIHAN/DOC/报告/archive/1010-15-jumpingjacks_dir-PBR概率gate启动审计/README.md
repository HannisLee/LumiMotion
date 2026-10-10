
## B09完整20k实际启动与只读审计（2026-10-10T10:05:44.168512+09:00）

真实PID291976/startticks10459825/exec91145，10:03:46.073168 JST独立35000→55000共20000步，当前实际tqdm约1285步/20000，正式warmup2000，Preset pbr_gt_directional_plane_native_direct_probability_gate。当前474核心及双snapshot SHA已锁定；未续smoke/B08权重。原NN32/opacity/scale冻结，所有99827点保留、原RGB/自身Alpha/源35k/相机time/白色单位GT方向唯一训练输入。

audit_launch实际exit0，cfg相较B08只有6差异（3路径、Preset、2新概率配置）；source35k PLY/MLP SHA保持，plane全部数组与B08逐bit一致/15187支持，150自身RGBA Alpha provenance PASS，方向表逐byte同B08白色单位，MLP optimizer0。0诊断优化、未读GT标签。真实train/postprocess/preview/audit四PID/ticks/argv/cwd核验；postprocess292132/exec16931、40k预览292130/exec1259、40/50/55k审计292131/exec19913绑定原训练，不自动重启。九步render/15七MAE/独立复算与文档已准备，尚无B09最终MAE/画质/7°结论，未commit/push/新目标分支。

smoke归档989文件SHA3，新增概率feature叶子仅优化坐标，所有渲染和保存仍真实native PLY sigmoid。训练文档、README、source manifest/train_argv/planned_commands和日志保存；commands/write_results.py在启动前纠正继承报告日期/继承B08表述，仅文档命令变化，生产核心与冒烟474 SHA保持。
