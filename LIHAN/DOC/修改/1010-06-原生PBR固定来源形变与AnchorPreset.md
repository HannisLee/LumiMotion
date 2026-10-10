# 1010-06 原生PBR固定来源形变与Anchor Preset

## 定位与修改

B03完整20k/55k结果GS Normal18.89548239°、质量FAILED。只读真实来源诊断发现canonical移动中位约.00344m，通过固定MLP的空间输入可产生P99约.09～.10m的额外形变，少数法线轴额外改变70°。原训练MLP在no_grad下查询更新后的XYZ，位置梯度未包括该耦合；冻结权重不能等同来源形变不变。

新增pbr_gt_directional_anchor继承refine全部参数，仅开启pbr_deformation_anchor。MLP输入固定为真实复制的35k XYZ；每帧time/camera/feature与来源相同，实际原生渲染仍使用优化后的PLY XYZ/rotation加预测形变。训练、周期报告、FrozenScene及其RGB/Albedo/Roughness/GS/Depth/动态Mesh/HDR/转灯路径均一致接入。原env、directional、geometry、refine预设默认关闭此选项；auto在显式GT方向光+anchor时选择Anchor。拒绝非original_ir、非GT方向光、未开几何及相机/time static_anchor混用。

scene/pbr_deformation.py创建pbr_deformation_anchor.npz（仅来源XYZ）和JSON；源PLY/PTH副本匹配stage1_source.json的SHA，XYZ逐元素一致。禁止覆盖；加载重验真实来源副本、资产SHA/内容/count/dtype，无GT材质/几何/法线字段；每个真正Stage2 lighting marker绑定同一asset SHA。35k可用于构造诊断，但没有真正PBR marker，不得冒充训练结果评分。当前不支持完整Adam状态无损resume。

## 验证

163项CPU实际通过；新增10项覆盖来源不符/篡改/错误形状dtype、非有限XYZ/额外GT字段、资产伪改后重新SHA、marker缺失/不匹配、默认兼容及Preset。第一次测试2旧mock无新增属性异常保留，默认读取改为getattr后通过。

真实GPU来源99827点，三训练帧XYZ+0.002m后训练/FrozenScene的形变与真实来源逐元素一致；实际周期报告三次输入均为锚定XYZ。渲染位置仍真实移动0.002m。PBR/BVH/GGX前向和XYZ/rotation/albedo/roughness/光强反向有限非零，源文件SHA不变、0 optimizer steps，无GT材质/法线/几何。GPU首次独立检查BVH缺参exit1保留，独立retry正常exit0。详情output/smoke_test/1010-09-dir-PBR固定形变输入验证/README.md、audit.json、logs。

git diff --check在cr-at-eol配置下通过；CRLF源文件保留原格式。功能验证PASS，不等同训练质量/7°通过。B04只改Anchor并完整20k和原协议七项MAE，保留全部Gaussian与15测试帧，不使用GT几何/材质优化。

## 正式使用

在B03完整命令中改为--loss_preset pbr_gt_directional_anchor。独立输出：output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定；训练报告1010-09、全命令commands/train.sh与planned_commands.json。旧实验不覆盖。
