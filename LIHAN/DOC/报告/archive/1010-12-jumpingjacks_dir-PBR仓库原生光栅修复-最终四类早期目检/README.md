
## B06最终四类实际目检（2026-10-10T07:47:59.665924+09:00，完整七MAE仍进行中）

150×4=600张insights的RGB/Alpha/normal/albedo已实际生成，已view首中尾0001/0075/0150的inspection/four_categories_final_55000.png。RGB黑背景正常/主体平台存在，但头手、衣服和脚周平台仍斑驳伪影；Alpha大部饱和，腰手边缘/脚周薄雾与细碎残影。Albedo正背面照明依旧烘入，平台棋盘与脚周污染；GS Normal整体平台向上、0075不再旧包整个平台朝下，但人体/脚周混合轴与斑驳仍有。

本轮40k原生来源支持组正确符号与55k全图正常plane相互支持实际kernel修复生效，但不能据此推定所有Gaussian或GT normal准确；最终Depth/Mesh及15帧七MAE/coverage/common/missing180待完成。转灯/HDR正渲染，不能提前声明已审核。阶段质量FAILED_SO_FAR，仍未commit/push/新分支。证据inspection/final_55000_four_category_review.json、four_categories_final_55000.png；拼图命令python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_final_four_categories.py（实际exit0）。
