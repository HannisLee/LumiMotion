
## B08 55k raw gate与概率梯度只读比较（2026-10-10T09:46:01.299518+09:00）

commands/audit_probability_gate_gradients_55000.py实际exec81335 exit0并关闭，0 optimizer、原训练已结束；同55k源NN/sourceXYZ/native/plane/材质固定，TRAIN0001/0075/0149只读比较∂L/∂raw和∂L/∂概率，严格验证链式因子2g(1−g)。全点导数中位4.26760162e−5、dynamic source-near-plane6836点中位.00051473238，显示大量sigmoid饱和。没有读取GT标签/生成prior或修改任何权重、点数、评分。

| TRAIN帧 | 全点 raw梯度L2 | 概率梯度L2 | >1e−12 raw/概率梯度点数 | 同lr.001假设首步概率平均偏移 raw/direct |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 0.000849484 | 0.010315651 | 34923/49210 | 0.000004960/0.000232780 |
| 0075 | 0.001164197 | 0.015449737 | 34522/49417 | 0.000007103/0.000243716 |
| 0149 | 0.000726180 | 0.012277407 | 35059/49329 | 0.000004998/0.000231180 |

首步比较仅RAM中以lr.001/无动量估计并投影[0,1]，没有执行optimizer，真实训练采用ramp/decay、多步动量，不能据此预测MAE或7°改善。备选方案是使用概率变量优化但前向/保存仍为原native sigmoid PLY feature，需新协议/Preset与真实完整训练；尚未实施。最终B08全量七MAE与人工审核仍待完成。证据inspection/probability_gate_gradients_55000.json与logs/probability_gate_gradients_55000.log，473核心SHA与FrozenScene fingerprint保持。
