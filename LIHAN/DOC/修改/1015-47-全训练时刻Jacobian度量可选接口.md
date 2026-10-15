# 全训练时刻Jacobian度量可选接口

新增--xyz-metric-times {nine,train}，默认nine。train模式要求xyz-preconditioner=jacobian，在全部selected训练时刻计算默认起点固定diag(JᵀJ)跨时间最大值，其余预条件/归一化/世界域和真实下降验收不变。用于T48发现的时间漏采问题，不能把局部线性度量当作非线性实际世界界。

训练计划1015-35，所有度量时刻和张量继续保存。与T45旧源码存在接口差异，配对需披露并核验共同输入、支持和warmup。
