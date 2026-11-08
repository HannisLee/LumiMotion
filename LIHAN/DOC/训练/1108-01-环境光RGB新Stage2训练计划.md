# 1108-01 环境光RGB新Stage2训练计划

- 规划：LIHAN/DOC/规划/1108-01-T01环境光建模与RGB损失新Stage2规划.md；实现记录：LIHAN/DOC/修改/1108-01-环境光建模与RGB损失新Stage2实现.md
- 服务器 ushas，conda 环境 lumimotion-ushas，GPU 0。
- 步骤：① 冒烟（`--smoke`，输出 output/smoke_test/1108-02-…冒烟；未收敛仅允许 `--eps-lr 3e-4` 重做一次）→ ② 正式单一预注册配置（输出 output/1108-03-LH-static-ELDL-V3-T01环境光RGB新Stage2，日志 run.log 同目录）→ ③ 目检 5 张图写 visual_inspection.json → ④ freeze_r6_env 冻结 → ⑤ Academic-analysis 冻结后一次 GT 评测。
- 不做候选网格；GT 仅在 ⑤ 读取一次。
- 失败结果（含超差、FAILED）完整保留，不覆盖不删除。
