# CV4 静态原版 PBR：Roughness MAE 实现与真实权重验收

日期：2026-10-08（Asia/Tokyo）；服务器 venus；Conda lumimotion。**评测流程 PASS**，仅表明公式、输入、渲染和复算一致，不代表材质恢复质量达标。没有重新训练，原 PBR 渲染器未修改。

## 结果

模型：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/model_mlp`，Stage2 **55000**。source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/data_view`。
GT：`/home/lihan/data/LH-data/static/only_clothV4/albedo`；源 Blender：`/home/lihan/data/LH-data/static/only_clothV4/chair_cloth_mvps_120views.blend`。
15测试帧0008、0016、……、0120，640×360；每帧使用真实相机和 time=(frame-1)/119，并调用原模型形变。

| 指标 | 结果 |
| --- | ---: |
| 全前景条件 roughness MAE | 0.18104681 |
| 四材质等权 bMAE | 0.24102391 |
| 全前景覆盖率 | 99.991658% |
| GT 可评分比例 | 87.784415% |
| 有效 / 可评分 GT 像素 | 251721 / 251742 |
| GT 内部前景像素 | 286773 |

| 材质 | GT roughness | 条件 MAE | 覆盖率 | 有效 / GT 像素 |
| --- | ---: | ---: | ---: | ---: |
| 红布 | 0.82 | 0.15885693 | 99.965774% | 52573 / 52591 |
| 椅面 | 0.62 | 0.19515487 | 100.000000% | 19331 / 19331 |
| 木框 | 0.38 | 0.44381684 | 99.976561% | 12796 / 12799 |
| 平台 | 0.50 | 0.16626699 | 100.000000% | 167021 / 167021 |

木框误差最高。全前景指标的平台像素占比较大，bMAE让四材质各占1/4。GT可评分比例的分母是GT alpha内部区域，不是整图；排除的12.2156%主要包含材质边界及无法稳定归类的像素。全局共21个GT评分像素未达到预测alpha阈值，未填零计误差。覆盖率是本协议区域内的alpha覆盖，不代表几何准确率。

## 真值和指标

从源 Blender 实际活动 Material Output 的 Surface 连接回溯，确认直接 Principled BSDF，Base Color/Roughness/Alpha均非连接输入，材质不透明。四材质在15个测试帧逐一核验为相同常量；对象关联和节点名称保存在material_manifest.json。源blend SHA256：`d3a901c14a9915fa2263de881f5fd5a71c64ee5decfe6e89fd75a43b0e32a9da`。清单中的值使用Blender原浮点数，不用表中四舍五入值评分。

GT albedo只用于材质分类。在线性RGB空间进行alpha加权双线性重采样，GT alpha≥0.99后内缩2像素；sRGB颜色最大分量距离≤3/255最近匹配材质，再内缩1像素。只在四材质区域并集G内赋予roughness真值。粗糙度本身不作sRGB/gamma变换、尺度拟合、平方或min-max归一化。

预测从PLY的roughness logit经sigmoid取r。原版几何、opacity、真实time形变后零背景光栅化得S=sum(T*alpha*r)、A=sum(T*alpha)，预测r=S/max(A,1e-6)。V=G且A≥0.9；MAE=sum_V(abs(pred-GT))/|V|，Coverage=|V|/|G|。全局对15帧像素汇总；bMAE对四个跨帧材质MAE等权。每帧结果单独保存，不与全局MAE混用。

有GT但无有效预测：MAE=null、coverage=0；没有GT可评分像素的材质：MAE和coverage均null；任一材质全测试集无有效预测：bMAE=null。NaN/Inf、缺帧、时间/分辨率错误、材质不支持和来源哈希变化均报错并保留FAILED。

原render_ir展示roughness=(S+bg*(1-A))*A。新入口不直接评分该alpha合成图。0008、0064、0120分别核对黑/白背景：黑背景差为0，白背景最大差5.9604645e-08（atol=2e-6, rtol=1e-5），alpha一致。

## 实现与产物

- 独立入口：[eval_roughness_gt.py](/home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/eval_roughness_gt.py)。参数包括model_path、iteration、source_path、gt_albedo_dir、material_manifest、output_path；前四项默认对应本次模型/55k/模型source/CV4 GT。固定测试集及阈值，拒绝Lambertian、缺少同迭代envmap的Stage1、缺失roughness列或冻结参考相机形变。
- 浮点渲染：[render_roughness.py](/home/lihan/reproduce/LumiMotion-stage2-perlight/gaussian_renderer/render_roughness.py)，返回单通道累积值、alpha、归一化r；未改原渲染行为。
- 公式及材质区域：[roughness_metrics.py](/home/lihan/reproduce/LumiMotion-stage2-perlight/utils/roughness_metrics.py)。清单导出：[export_cv4_roughness_manifest.py](/home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/export_cv4_roughness_manifest.py)。独立CPU复算：[verify_roughness_eval.py](/home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/verify_roughness_eval.py)，不导入渲染器或指标实现。
- [metrics.json](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation/metrics.json)、[frames.csv](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation/frames.csv)：全局、各材质及逐帧误差/覆盖率；[provenance.json](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation/provenance.json)：有效配置、代码和全部使用输入哈希；[material_manifest.json](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/inputs/material_manifest.json)：材质真值和出处。
- `evaluation/float/0008.npz`至`0120.npz`保存S、A、r、GT、GT alpha、core/评分/有效mask及material_labels。GT值0仅为mask外占位，不参与评分。
- [独立复算记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/independent_audit.json)：15帧NPZ重建S/A及GT、校验逐帧和聚合JSON/CSV与全部输入哈希通过，同时保存NPZ哈希。
- 日志：[材质核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/material_export.log)、[完整评测](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation.log)、[独立复算](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/independent_audit.log)；命令：[commands.json](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/commands.json)。输出存在则拒绝覆盖。

## 四类图像目检

已实际检查0008、0064、0120以及最差MAE帧0032，每帧四列均使用固定色标：GT roughness、有效区预测roughness、绝对误差、评分覆盖区域。前三列0–1；覆盖图灰=GT内部排除、红=评分区漏检、绿=有效。

1. GT图：四材质为核验常量，材料边界留白符合内缩规则。
2. 预测图：红布存在褶皱相关明暗，木框明显偏亮，平台有明显空间斑驳；同一材质未恢复为常量。
3. 误差图：木框和椅腿误差最明显，平台与红布也有结构性残差。0032为全前景MAE最差帧（0.21085785）。
4. 覆盖图：内部几乎全部绿色，末帧可见少量漏检；边界灰色为GT规定的排除区域，不能算作模型漏检。

未逐帧人工审阅其余11帧；全部15帧进行了数值检查。此次只产出静帧，不生成视频。该参数误差不表示Blender完整Principled BRDF与原版固定F0 GGX模型完全等价。

![最差帧0032](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation/previews/0032.png)

[首帧0008](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation/previews/0008.png) · [中帧0064](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation/previews/0064.png) · [末帧0120](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation/previews/0120.png)

## 完整命令与验收

以下均在`/home/lihan/reproduce/LumiMotion-stage2-perlight`执行；输出已存在，再跑需指定新的清单文件及评测目录。`commands.json`保存相同argv数组。评测命令未显式传递的默认值已在上方和provenance中展开。

### export

```bash
/home/lihan/reproduce/LumiMotion-stage2-perlight/output/Archive/1003-06-onlyclothV4-geometry_gt_eval/tools/blender-5.2.2-linux-x64/blender --background --factory-startup --disable-autoexec /home/lihan/data/LH-data/static/only_clothV4/chair_cloth_mvps_120views.blend --python /home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/export_cv4_roughness_manifest.py -- --output /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/inputs/material_manifest.json
```

### evaluate

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.eval_roughness_gt --material_manifest /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/inputs/material_manifest.json --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation
```

### verify

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m scripts.verify_roughness_eval /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/evaluation --output /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/independent_audit.json
```

### unit_tests

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m unittest tests.test_roughness_metrics tests.test_albedo_metrics -v
```

21项单元及albedo回归测试通过，覆盖恒定偏差、背景面积不变性、alpha归一化/阈值、材质等权、混合边界、缺失区域、NaN/Inf、错误模型及错误材质连接。15帧实际GPU评分和首/中/末黑白背景核对通过；CPU独立复算通过；输入文件及冻结张量未改变。验收结论：**PASS（评测流程）**，没有设置材质质量合格门槛。
