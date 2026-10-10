
## B09训练期间全135来源姿态旋转响应原型（2026-10-10T10:21:54.575849+09:00）

commands/audit_source_local_rotation_response.py实际exec33765 exit0并关闭，0 optimizer/无GT标签；40k容器仅source32 NN/原相机time，全部诊断使用原35k XYZ/q/feature与既有source-only plane，未用learned40k geometry/material/feature。全99827点、135 TRAIN姿态，RAM中人工10°canonical法线轴扰动，保留norm/yaw；不存candidate quaternion/normal/teacher/prior/评分mask，不改变B09生产或训练。

| source组（非GT语义） | 点数 | 平均时刻原加法响应中位 | 原加法最坏时刻P99 | source-local原型平均响应中位 |
| --- | ---: | ---: | ---: | ---: |
| source_projected | 15187 | 0.00000170° | 0.00001971° | 0.00000172° |
| source_dynamic_near_plane | 6836 | 9.25706546° | 13.59882556° | 10.00000078° |
| source_other_dynamic | 38747 | 9.09878417° | 15.11675295° | 10.00000070° |
| source_static_unprojected | 39057 | 9.99999483° | 10.16521196° | 10.00000074° |

投影15187点法线被原plane固定，不应响应，仍全部保留。其它动态38747点10°canonical扰动在原加法平均响应中位9.09878417°，near-plane动态9.25706546°：没有证明原加法存在大范围严重旋转响应抑制，不能据此盲目扩大角度或宣称仅更改旋转运输就可达7°。原加法时间响应有差异和极少大响应（全点最大88.05979930°）；source-local四元数原型未投影点响应约10°，和R_source_world R_source_canonical^T运输最大偏差4.22423871e−5°，自身初始化返回原raw deformation逐bit一致。它只是全来源的数学候选，不是已实现/已训练的Preset，也不是GT Normal准确率；若今后采用RGB方向约束，可能给固定source运动的局部修正提供一致表达，须另行接入全部入口/功能/完整训练验收。当前B09管线474 SHA保持且继续训练。
