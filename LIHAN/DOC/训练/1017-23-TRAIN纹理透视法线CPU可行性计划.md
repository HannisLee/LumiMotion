# TRAIN纹理透视法线CPU可行性计划

2026-10-17，ushas / lumimotion-ushas。T14充分预算16k及完整评测已FAILED_NOT_ADOPTED，T08继续控制。Astra据完整613项和历史失败决定：仅检验固定XYZ下纹理透视形变是否含稳定局部方向信息，不直接安排新16k，不扫描T14权重/起始/邻域。

## 范围与隔离

输入仅原115静态RGB三角化XYZ、600 fit观测身份、135原TRAIN RGB/Alpha/相机及13既定held关联。tracks SHA1f73f399bc0fce74762a0ced282d8c4283b0efaed9172122f1389a27901936ca；相机7e76c3f5d4e591a9e44be9296b99745bd51efa8ab78d7569a8af6c1a52ac776c；完整来源冻结见原修改39。禁止模型、GS法线、深度缓存、灯向、GT及任何GT训权重输入。全115保留，不以Alpha/残差/语义/高度/LOO挑点。

full-fit、全部600 LOO与状态先冻结后，才解码held RGB/Alpha；预先hash held文件仅为来源核验，不用于选择算法。held13只预测，不重新对齐中心/参考/法向。原135 RGB仍是TRAIN来源，内部held只是本可行性拟合未用，不宣称整个模型未见数据。

## 唯一数学定义

所有相机运算float64；用冻结world-to-camera E求逆统一取C和ray，不混用单独舍入中心。v_i=(C_i−X)/norm；参考相机是v角距和medoid，≤1e−12rad平局选最小原source索引。n0=v_ref。以最小absdot世界轴（平局x/y/z）投影归一化建e1，e2=cross(n0,e1)。n=normalize(n0+a e1+b e2)，a,b∈[-3,3]，是斜率方框而非统一角锥，唯一初值(0,0)。

固定原生800分辨率7×7整数偏移[-3,3]，中心为XYZ在参考相机的精确投影，不用SIFT中心重新对齐。K边界坐标uv对应数组uv−0.5，双线性四邻全在图内。参考ray与过X的平面求交Q=C+n·(X−C)/n·ray *ray，再投其他相机，不优化XYZ/相机/中心/逐图平移。

先标准分段sRGB逆变换至linear RGB，Y=.2126R+.7152G+.0722B，再采样。49维Y分别去均值，NCC=clip(dot(a,b)/max(norm(a)norm(b),1e−12),−1,1)，cost=1−NCC。任一块中心化RMS≤1e−6为零纹理无效。任一49采样点越界、非finite、Z≤0、参考t≤0或双线性Alpha≤252/255，该相机对固定最大cost2；参考块无效则全配对2。分母始终原m−1，不因候选丢弃差视图。饱和仅报告，不筛选；Alpha不证明内部无阻挡。

SciPy1.10.1 / NumPy1.24.4 / Pillow9.5.0，单线程L-BFGS-B：maxiter80/maxfun2000/maxls20/ftol1e−12/gtol1e−6。梯度1e−4斜率中心差分，边界可行单侧；梯度内部F调用亦计实际2000预算。无重启、多起点、角度粗搜或事后延长。预算/非finite/异常/边界与全无效伪收敛均显式报告，不冒优化成功。

每个LOO删一fit观测，只按剩余相机重新medoid/n0/基，从0独立拟合，不用full结果warmstart。与full无向夹角acos(absdot)，固定XYZ使它只是条件敏感性，非完全独立几何LOO。失败角度N/A并列，不作为0；统计逐轨均值后等轨，并列原115/600固定数和有效/N/A数，NCC代价始终固定分母。

full终点法向切平面球面exp扰动h=1e−3rad，中心/轴±h/四对角共9次F作2×2Hessian。超原斜率域曲率N/A，配对有效mask变化标有效域切换，不宣称可辨识。lambda_min>1e−6标数值PD，<−1e−6非PD，其余退化；报两特征值/cond，单位cost/rad²，不当精度或置信区间。

## 准入、自检与停止

先纯合成CPU工程自检：半像素/独立投影与平面交点、NCC亮度不变性/零纹理、无效cost2固定分母、medoid平局/LOO隔离、实际函数预算及RNG保持，不读取真实GT。独立Sol核源码、原输入、fit/held边界、全失败/N/A及坐标梯度后，才允许唯一CPU全数据运行。

held13逐项对比full-fit平面与同参考初始平面，均值/P90/max/无效数全报。预测必要条件：mean下降>1e−7，P90/max/无效数不增加（tol1e−7）；否则关闭固定版本，不扩大patch或另加光度先验。即使通过，也须结合curvature/LOO/原覆盖另论证，不能直接转115法线监督或准入全身训练。

计划工程输出output/smoke_test/1017-20-staticDL-纹理平面CPU自检；唯一生产output/1017-01-静态DL未知光总控/1017-08-TRAIN纹理平面方向可行性，source/model/iteration/渲染均明确：原TRAIN派生源、模型N/A、更新0、GPU0。日志同输出目录，完整实际命令/源码SHA/运行退出/定量/全115与held图审/PASS或FAILED待真实执行登记。所有失败原件保留。当前PRE_REGISTERED_NOT_IMPLEMENTED，不伪称已实现或生产。


## 实现前独立评审补充锁定

Sol已独立审查规格，未运行真实RGB/GPU。2000硬限为每次full/LOO拟合的全部实际F调用，含初始、优化器fun及手写jac探针；终点评分优先复用已求值的同一精确参数记录，不伪造额外免费计算。九点曲率为额外独立诊断调用，逐轨单列，不在优化2000内。全无有效配对必须ZERO_VALID_SUPPORT，不能因L-BFGS success/零梯度/平坦Hessian称可辨识；失败LOO角度N/A不按0。

先整图sRGB→linearRGB→Y再插值，cost为1−NCC而非1−absNCC；连续投影不round，最后uv−.5。curvature超域/有效mask切换/零支持不能给辨识结论。held解码hook与全部115+600状态原子冻结作独立边界证据。拟实现LIHAN/scripts/diagnose_static_rgb_patch_planes.py，固定生产/合成自检输出目录，不接收科学超参扫描。
