# T02-native：原生 Stage 2 思路（去 b 目标）训练脚本记录

日期：2026-11-14；服务器 ushas，环境 lumimotion-ushas。来源：管理者口头裁定（不开新 GOAL 文档），目的检验 T01 中用 b 当目标是否是问题所在。

## 与 T01 的唯一差异
1. 环境图 8×16 → 16(纬)×32(经)=512 texel（“32×16”按 2:1 经纬图读，宽32×高16）；仍 RGB 全局共享一张、不归一化、无平衡参数；固角 (2π/32)[cos(iπ/16)−cos((i+1)π/16)]，总和 4π（已数值核对）。
2. 删除 L_static：训练目标仅 L_RGB + 0.1·R_rev（R_rev 与 T01 相同，仅 C3 有梯度）。b 只作为 E4 先验输入：A 阶段 c0 的一维解析起点仍用 b（A 照旧，按新网格重算为 0.76533），静态 L1 仅在 no_grad 下记入日志，不进 total、不进梯度。
3. 删除 INIT 防护：`physics.guard` 在本进程内替换为恒通过（状态记 INFO_ONLY），能量占比与 b 残差照常写入 diagnostic，不触发停止。
4. 其余与 T01 相同：B/C1/C2/C3 步数与学习率、α0=0.5、K=1.0566、种子111301、float64、全批次、间接光关闭、body_fit 2683、冻结几何。

## 实现方式
冻结代码 `LIHAN/scripts/goal1113/` 未改动；新脚本 `LIHAN/scripts/goal1113_t02/native_run.py`（进程内替换 `P.grid`、`P.guard`、`O.objective_backward`、`F.setup`，其余复用 `fit.run`）与 `native_eval.py`（冻结 + 冻结后一次 GT 评测）。未做渲染、四类图与视频（管理者验收仅为方向光角度误差）。

## 启动命令（cwd=仓库根）
```bash
setsid nohup env CUDA_VISIBLE_DEVICES=2 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python -u -m LIHAN.scripts.goal1113_t02.native_run --device cuda --output output/1114-03-LH-static-ELDL-V3-T02-native-RGBonly-env16x32 > output/1114-03-LH-static-ELDL-V3-T02-native-RGBonly-env16x32/run.log 2>&1 &
```
启动 UTC 约 16:52:52，PID 582723。评测命令：`python -u -m LIHAN.scripts.goal1113_t02.native_eval --output <同目录>`。

## 补充：渲染与图（2026-11-14）
新增 `LIHAN/scripts/goal1113_t02/native_render.py`（进程内换 16×32 网格后复用冻结 `goal1113.render.run`，再调用 native_figures）与 `native_figures.py`（`goal1113/figures.py` 副本：环境图 reshape 16×32、setup 换为 native_run.setup、去掉 T01 防护阈值线）。冻结代码未改。渲染命令见产物 README，GPU3，exit 0。
