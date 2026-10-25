# 静态ELDL-R阶段：opacity lr 0.05与reset配对计划（Ra/Rb/Rc）

日期：2026-10-25；ushas / lumimotion-ushas；GOAL `1024-02` 第3节R阶段的另立版本。依据：报告1025-04第四节，在opacity lr 1e−3下reset无法在1000步内恢复，所以先把opacity lr恢复到原生的0.05。主Agent指示以冻结的G11@8100为共同父，继续争取1°目标。本计划在正式启动前冻结。

## 1. 设计（同父G11@8100，各2000次新增更新，固定拓扑）

| 运行 | opacity lr | reset | normal→opacity路由 | 输出 | GPU |
|---|---|---|---|---|---|
| **Ra** | 0.05 | 无 | full | `output/1025-36-LH-static-jumpingjacks-ELDL-V3-Ra-G11父opacity-lr0.05无reset` | GPU1 |
| **Rb** | 0.05 | stage步1000的optimizer更新之后执行原生reset：opacity←min(opacity,0.01)，其Adam动量清零 | 全程full | `output/1025-37-LH-static-jumpingjacks-ELDL-V3-Rb-G11父opacity-lr0.05-reset-full` | GPU3 |
| **Rc** | 0.05 | 同Rb | reset前full，**stage步1001起仅normal→opacity切为detach**（DL T06机制） | `output/1025-38-LH-static-jumpingjacks-ELDL-V3-Rc-G11父opacity-lr0.05-reset后detach` | GPU2 |

- **配对含义**：Ra对G11只改opacity LR；Rb对Ra只加reset；Rc对Rb只改reset后的路由。
- **共同条件**：
  - 父配置为G11（photo部分覆盖排除、阈值0.08、normal 0.1、linear）。父核验要求plan逐项一致，加载后8项参数＋照明哈希与G11 saves.json@8100逐位相同，PLY SHA也相同。
  - seed1012不放回相机序列，首相机image_0048，三组相同。
  - 新建零动量Adam；增密统计清零且不使用；无增密或裁剪。
  - LR：G11日程在2000步内重启，xyz 1e−5、scaling 1e−4、rotation 5e−5，按0.1^((r−1)/1999)衰减；opacity 0.05恒定；albedo 2e−4、照明1e−5。
- **保存（相对总步=8100+stage步）**：9000（900）、9100（1000，reset前）、9200（1100）、9300（1200）、10100（2000）。Rb/Rc另存 `model_mlp/post_reset/…/iteration_9100`（reset后副本，只作审计，不参与评测或选点）。
- **累计更新**：34999+1080+7020（G11）+2000。
- **reset后低opacity点的prune**：固定拓扑下不裁剪。原生路线会在下一次增密事件删除仍<0.01的点，本入口每10步记录该比例供核对。冒烟中reset后10步，<0.01的比例已回到15%（Rb）或3.4%（Rc），之后继续下降。
- **入口**：新文件 `scripts/train_static_eldl_reset.py`（SHA `2c4178dc…1800`）。marker脚本接受stage R。lighting文件和评测入口未改。快照 `output/1025-36-…/snapshot_code`（manifest `91aa72468f309f898d36abbeb476ee94dc4d46396a17d6b9bb5bb88641a63aa8`），三组共用。

## 2. 预登记判据

- 有效：相对G11@8100（DEV P 3.3905），**P改善≥0.03，即P<3.3605**，且全部DEV门槛通过（门槛父按冻结规则取直接父G11@8100）。
- 机制判读：
  - Ra与G11对比：只看opacity LR的效应。
  - Rb与Ra对比：reset本身的效应，用reset后的保存点9200/9300/10100比较。
  - Rc与Rb对比：reset后detach的效应。DL证据中T06相对T05降低0.27°。
  - 9000/9100为reset前的对照点，Rb/Rc应与Ra在这两点上近乎相同（同一代码路径、同一序列）。
- **恢复窗口诊断（重点）**：TRAIN视角缺失率与前景预测alpha均值在990–1300步每10步记录一次。若2000步时DEV coverage与IoU通过门槛，说明opacity lr 0.05在1000步窗口内足以恢复。
- 不跑TEST。

## 3. 冒烟（均PASS；父=正式G11@8100，300步，reset在第100步，保存8190/8200/8210/8220/8400）

| 冒烟 | GPU | 结论 |
|---|---|---|
| `output/smoke_test/1025-38-LH-static-jumpingjacks-ELDL-V3-Ra冒烟` | 1 | 三项零更新审计PASS；无reset；缺失率0.0003–0.0018；<0.01点比例从6.7%升到8.0%（opacity lr 0.05把更多点推低） |
| `output/smoke_test/1025-39-LH-static-jumpingjacks-ELDL-V3-Rb冒烟` | 3 | 审计PASS；**真实reset事件**：前景平均opacity 0.393→最大0.0100，opacity的Adam一阶/二阶动量由非零清为0（step计数保留，原生语义）；reset后路由审计PASS；训练视角缺失率在reset后10步为18.9%，20步8.9%，50步0.53%，150步0.04%（回到reset前水平） |
| `output/smoke_test/1025-40-LH-static-jumpingjacks-ELDL-V3-Rc冒烟` | 2 | 审计PASS；真实reset，**路由切换审计PASS**（reset后full/detach前向一致，normal→opacity非零变为0，其他路径保持）；第110步起route=detach_opacity；缺失率在10步时15.4%，20步3.6%，50步0.24%，150步0.035%，恢复比Rb更快 |

reset前（第1–100步）三组loss逐值相同（第1步0.00609）。

## 4. 执行记录

- 2026-10-25 14:41:47 UTC 同时启动。
  - Ra：GPU1，包装shell PID 1716090，python PID 1716099。
  - Rb：GPU3，包装shell PID 1716091，python PID 1716101。
  - Rc：GPU2，包装shell PID 1716089，python PID 1716100。
- 启动时GPU1/GPU3各有G13/G14，GPU2有1个评测进程，每卡不超过3个；GPU0未使用。
- 每路约1.4 GB显存；三卡满载，实测约2–3步/秒，预计约15:00–15:10 UTC完成；结束后自动补marker。
