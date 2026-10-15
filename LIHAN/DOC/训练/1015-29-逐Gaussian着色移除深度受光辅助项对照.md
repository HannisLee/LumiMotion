# 逐Gaussian着色移除深度受光辅助项对照

T41，ushas/lumimotion-ushas，GPU0一个自有训练。共同起点1011-03@35000、T05未知光、T02亮面轨迹支持、seed1015、expanded参数组、历史normal/Alpha/distortion权重、逐Gaussian着色、底项0.1、240warmup+120材质更新/40几何提议及T40实际位移回缩规则。唯一变化positive-weight由1变0；normal一致性等其余正则保留。

控制使用已完成T40 gaussian_retry1，源码逐SHA相同，不接续其权重；候选从同一默认起点重新训练，输入/支持/初值与warmup比较，配置只允许positive-weight和output差异。此控制复用不增加预算优势，新增候选360材质更新与40提议。输出output/smoke_test/1015-33-LH-dynamic-DL-T41移除受光辅助项。若发现控制来源不匹配，先重新配对，不直接比较。

训练无GT；终点独立400分辨率原15帧评测默认、旧frozen、T40控制及候选。主要门槛：候选主体expected/median同时优于默认和控制，覆盖及足部expected/median不恶化；不事后新增正容差。Depth Normal、缺失惩罚与共同有效区域并列。RGB改善或梯度不同不能代替几何成功。Mesh/三种子/对应增量仍缺则不判完整Goal完成。

四类图、真实深度、完整命令/源码/日志/checkpoint原位保留。若无收益，不重复沿该辅助项调权重或延长同配置；回到共享材质/表面对应能否给实际动态几何提供正确约束这一主问题。
