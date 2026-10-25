# 静态ELDL物理续训：部分覆盖photo排除与opacity冻结选项

日期：2026-10-25；ushas / lumimotion-ushas；GOAL `1024-02` 阶段C第二轮。依据：报告 `LIHAN/DOC/报告/1025-02-静态ELDL-G0边缘覆盖损失机制诊断.md`。上位实现：`LIHAN/DOC/修改/1024-07-静态ELDL从头Stage1与GT方向混合光续训入口.md`。

两个新选项的默认值与G0数值一致。正式G0–G3使用的是旧快照（1024-20/1024-25），不受本次改动影响。

## 1. 改动文件

| 文件 | 改动 | 默认路径 |
|---|---|---|
| `scene/static_eldl_lighting.py` | 新增 `partial_alpha_keep_mask(alpha, upper=.99)`：GT Alpha∈(0,0.99)的像素记0，其余记1。`censored_loss` 新增可选参数 `pixel_keep=None`：非None时乘到逐像素权重上，分母仍取全部像素的mean，诊断中额外记录 `photo_pixel_excluded_fraction`。 | `pixel_keep=None` 时代码路径与原式完全相同（只多一次 `is not None` 判断） |
| `scripts/train_static_eldl_physical.py` | 新增 `--photo_partial_alpha {keep,exclude}`（默认keep）和 `--opacity_lr_multiplier {1.0,0.0}`（默认1.0），A0阶段禁止使用；`geometry_lr()` 的opacity项乘以 `opacity_multiplier`；新增零更新审计 `audit_photo_pixel_keep`（仅在exclude时执行，失败则不训练）；opacity倍率为0时，每个保存点断言opacity与父模型逐位相同；physical_plan/physical_finished 记录两个选项；每个日志步额外记录训练视角的 `train_view_alpha_missing_fraction` 和部分覆盖像素的预测/GT alpha均值（no_grad，只读，不影响RNG或计算图） | keep/1.0 时loss、lr日程与G0相同 |
| `LIHAN/scripts/register_static_eldl_eval_marker.py` | marker的args中写入 `photo_partial_alpha`、`opacity_lr_multiplier`（从physical_plan读取，旧plan缺省为keep/1.0） | 旧模型的marker字段只多两个缺省值 |

评测入口 `LIHAN/scripts/eldl_1024_eval.py` 与包装器 `LIHAN/scripts/eldl_1024_eval_static_gt.py` 都没有修改。两个选项只改变训练loss和lr，不改变前向渲染，评测时的RGB/几何渲染语义与G0相同。

## 2. 选项语义

- `--photo_partial_alpha exclude`：在GT Alpha∈(0,0.99)的像素上，photo（censored鲁棒RGB）的权重为0，其余像素的权重规则不变（0.08/0.25/0.5、背景权重1、满码单边）；分母不变，所以保留像素的梯度与G0逐元素相同。只排除photo项。这些像素上的Alpha BCE、normal一致性（其掩膜本来就是Alpha>0.99经5×5腐蚀，不含边缘）、distortion都照常计算。掩膜只由训练Alpha决定，不读取任何GT。
  - 理由：部分覆盖像素的target是 a·sRGB(含opaque世界背景的混合)，而物理前向是 sRGB(线性合成)，两者在合成域上不可比，导致边缘系统性偏亮。photo因此压低边缘opacity（报告1025-02第四、五节）。
- `--opacity_lr_multiplier 0`：G阶段opacity组的lr恒为0。参数仍留在Adam中并计算梯度，但不会更新。保存点断言opacity逐位不变。xyz/rotation/scaling/albedo/照明不受影响。

## 3. 冻结快照

`output/1025-04-LH-static-jumpingjacks-ELDL-V3-G4-full-部分覆盖photo排除/snapshot_code`（`scripts/snapshot_static_eldl_code.py` 流程；manifest SHA256 `e26e03f2cf39791dff21a4f63f6b23de5047be4d0c294769a22b6fa965233846`；328个文件；commit 9ed1e63）：
- `scripts/train_static_eldl_physical.py` SHA256 `0d4f08db3257a6b314d1fbc62733aad68e24981863bdb088333bb51362eb9af5`
- `scene/static_eldl_lighting.py` SHA256 `227dca035815601fc8b7be84ddd968e3620c6a59a688170037c21ecf16d76cff`
- 与G2/G3快照（`output/1024-25-…/snapshot_code`）的差异（忽略__pycache__和manifest）只有上述两个文件。`scene/gaussian_model.py` 23361f38…、`gaussian_renderer/__init__.py` c9c72097…、`scene/static_dl_photometric.py` 11eeb3f6… 均未变。
- G4与G5共用该快照。

## 4. 冒烟（工程，父=正式A0@1080，各200步，保存1180/1280，GPU3）

| 冒烟 | 结论 |
|---|---|
| `output/smoke_test/1025-07-LH-static-jumpingjacks-ELDL-V3-G默认选项与G0一致性冒烟` | 默认选项：初始化xyz哈希 `76cb5f9b…` 与G0相同，首相机image_0048；路线零更新审计PASS；第1步loss与G0 **逐位相同**（0.009566210210323334），第1步xyz位移相同。第100/200步loss相对差约1.4e−5/1.3e−5，xyz位移相对差<1e−5，各组lr逐值相同。这些差异来自光栅反向的原子加非确定性，不是选项本身引入的。比对见该目录 `parity_vs_G0.json`。 |
| `output/smoke_test/1025-08-LH-static-jumpingjacks-ELDL-V3-G4部分覆盖photo排除冒烟` | 路线审计PASS；`initial_photo_partial_alpha_audit.json` PASS：排除像素占1.09%且全部为GT Alpha∈(0,0.99)，保留像素的photo梯度与G0逐元素相同，排除像素梯度为0，G0在这些像素上原本有非零梯度，参数不变。 |
| `output/smoke_test/1025-09-LH-static-jumpingjacks-ELDL-V3-G5-opacity冻结冒烟` | 路线审计PASS；opacity组lr=0；两个保存点 `opacity_unchanged_vs_parent=true`（哈希 `1a6d3b2f…` 与父相同）；第1步loss与G0逐位相同（opacity第1步之前尚未更新）。 |

## 5. 第三轮候选选项（2026-10-25 09:28–09:33 UTC 追加；只实现与冒烟，未启动正式训练）

主Agent要求在G4配置上准备三组候选：G6 normal权重0.3、G7 几何LR×3、G8 sRGB域合成。三个新选项的默认值都与现状逐位一致。

| 文件 | 改动 | 默认路径 |
|---|---|---|
| `scripts/train_static_eldl_physical.py` | `--geometry_lr_multiplier` 的choices放开为{1,3,10}；新增 `--normal_weight {0.1,0.3}`（默认 `FROZEN['loss']['normal']`=0.1，进入训练loss和零更新路线审计的weights）；新增 `--composite {linear,srgb_per_gaussian}`（默认linear=`render_mixed`）；srgb模式用 `render_mixed_srgb`，起点额外做 `audit_composite`，失败则不训练；A0阶段禁止使用这三项；plan、physical_finished记录新选项；lighting.json只在非linear时才多一个 `composite` 字段（默认时文件内容与G4相同）；sources清单加入 `scene/static_eldl_composite.py` | 默认1/0.1/linear时loss、lr、前向与G4/G0相同 |
| `scene/static_eldl_composite.py`（新增） | `render_mixed_srgb`：照明、可见度、法线翻面与 `render_mixed` 共用（`shade_linear`），唯一差别是逐Gaussian颜色先 `linear_to_srgb(clamp_min(env+direct,0))` 再按alpha合成（黑背景）。components=True时另外渲染线性分量，只用于记录。`audit_composite`：几何map与linear模式逐元素相同、render等于原生光栅以逐点sRGB颜色合成、背景为0、部分覆盖像素不比linear更亮、photo对opacity/xyz/albedo有有限非零梯度、参数与RNG不变 | 不影响linear |
| `LIHAN/scripts/eldl_1024_eval_static_gt.py`（评测包装器，SHA `e6675246…`→`54091f32ed9daf64c34eda5e0f608812130d81079cc666657541a36ed12f265a`） | 从marker的args读取composite（缺省为linear）；linear时RGB仍用 `render_mixed`（行为不变），srgb_per_gaussian时用 `render_mixed_srgb` 并额外核对 `scene/static_eldl_composite.py` 的快照与运行时SHA；其他值报错；provenance中记录composite | linear模型（S0/A0/G0–G5）行为不变 |
| `LIHAN/scripts/register_static_eldl_eval_marker.py`（SHA `b90204e9…`） | marker的args中增加 `normal_weight`（缺省0.1）、`composite`（缺省linear） | – |

`scene/static_eldl_lighting.py` **没有再改**，SHA仍为 `227dca03…`，与G4/G5快照相同。原因：评测包装器会核对训练快照中的该文件与运行时是否一致，所以sRGB合成放在新文件里，G4/G5的DEV评测不受影响。

第三轮冻结快照：`output/1025-07-LH-static-jumpingjacks-ELDL-V3-第三轮G6G7G8待启动/snapshot_code`（manifest SHA256 `bc7056a4947c23f20a51026b8649dec04d71255595076d5a6aaf2e2b8be9115d`，329个文件）。
- 入口SHA `8d0b0eeb34f76dad3dc1a5cb777681ab29596ef58a2b4c00a16869701d92d16b`，composite模块SHA `a4e515439ae7c56a4de5800afda7ce414b5b089d5c19caa9809ad90820f93037`，lighting `227dca03…` 未变。
- 与G4快照的差异：入口文件、新增的composite模块，以及另一会话修改的 `gaussian_renderer/render_ir.py`（d_color为None时的保护）。本入口的导入链不使用render_ir，已用grep核对。

冒烟（父=正式A0@1080，各200步，保存1180/1280；汇总见 `output/smoke_test/1025-10-…/round3_smoke_summary.json`）：

| 冒烟 | GPU | 结论 |
|---|---|---|
| `output/smoke_test/1025-10-LH-static-jumpingjacks-ELDL-V3-第三轮快照G4配置一致性冒烟`（新快照、G4配置、新选项全默认） | 2 | 路线/物理/掩膜零更新审计PASS；第1步loss与正式G4 **逐位相同**（0.008904694579541683），第1步xyz位移相同；第100/200步loss相对差约3.7e−5/4.1e−5（原子加非确定性）；lr逐值相同 |
| `output/smoke_test/1025-11-LH-static-jumpingjacks-ELDL-V3-G6-G4加normal权重0.3冒烟` | 2 | 审计全PASS（路线审计以权重0.3执行）；第1步loss 0.017772 = G4的photo+Alpha+distortion+0.3·normal（normal 0.044337与G4相同） |
| `output/smoke_test/1025-12-LH-static-jumpingjacks-ELDL-V3-G7-G4加几何LR3倍冒烟` | 3 | 审计全PASS；首步lr为xyz 3e−6、rotation 1.5e−4、scaling 3e−5，opacity仍为1e−3；第1步loss与G4逐位相同；200步xyz平均位移1.33e−4（G4为4.62e−5，约2.9倍） |
| `output/smoke_test/1025-13-LH-static-jumpingjacks-ELDL-V3-G8-sRGB域合成冒烟` | 1 | 路线/物理审计PASS；`initial_composite_audit.json` PASS，几何map全部逐元素相同；部分覆盖像素的平均残差从linear的+0.063变为−0.024（不再偏亮，但略偏暗）；第1步loss 0.009409（G0为0.009566） |

注意：G8的零更新路线审计中，`rgb_opacity_preserved` 一项由linear前向计算，因为审计函数位于未改动的lighting模块。sRGB前向的photo→opacity梯度由 `audit_composite` 单独核验。评测包装器的srgb分支只经过代码审阅和编译，还没有做过评测冒烟。G8若启动，建议独立评测Agent先对它做一次3帧smoke-render，再跑DEV。
