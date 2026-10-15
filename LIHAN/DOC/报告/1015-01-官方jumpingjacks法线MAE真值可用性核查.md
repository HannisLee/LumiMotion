# 官方 jumpingjacks 法线 MAE 真值可用性核查

日期：2026-10-15（Asia/Tokyo）；服务器ushas；环境lumimotion-ushas。对象为1015-01官方jumpingjacks150_v5_spec32训练，Stage1=35000、Stage2=55000。

## 结论

**FAILED_MISSING_GROUND_TRUTH：当前材料不足，未能计算真实法线MAE。指标为null，不是0，也不是预测结果质量失败。**

已核查本地动态数据、对应statictimestep75数据以及原始3,683,944,960字节tar包。tar包含10987项，其中jumpingjacks动态和静态共2287项，没有法线真值文件，也没有可用于重新导出GT的Blender/EXR/OBJ/FBX/GLB场景资产。

完整15个测试帧为0010、0020、……、0150。每个对应depth PNG均为800×800 RGBA、uint8，RGB所有像素都是255，最小值和最大值均255。这些图片不含可恢复的距离变化，不能拿来生成可信的GT法线。PNG元数据中的原作者.blend路径只是导出来源字符串，不表示本机有该文件。

模型自身已经有渲染法线，但预测法线图不能作为自己的真值；其它场景的normal、LH版本jumpingjacks的GT亦不能直接匹配本次官方场景。当前没有计算GS或Depth法线MAE，也没有用可视化颜色差、法线自一致性或纯白深度平面误差替代真实GT角度MAE。

## 计划评分口径

获得对应帧GT后，在相同测试相机、时间步、400×400评测分辨率和统一坐标系下，归一化法线，用acos(clip(dot(n_pred,n_gt),-1,1))转为度；对GT有效前景像素报告像素加权平均、逐帧值、有效像素数和覆盖率。不使用abs(dot)消除法线翻转。若从正确浮点GT深度导出参考法线，应另标注为GT-depth-derived，而不是原生GT法线。

需要补充同场景、对应15测试帧和相机的GT Normal EXR/PNG（含坐标/编码说明），或原始jumpingjacks_v5_specular32.blend及关联动画资产以重新导出。

## 可复查材料

- 数据：`/home/han.li/reproduce/LumiMotion/data/d-nerf-relight-spec32/jumpingjacks150_v5_spec32`。
- 模型：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1015-01-jumpingjacks150_v5_spec32-原版LumiMotion两阶段/model_mlp`。
- 核查结果：实验下`normal_mae/gt_availability.json`，含15帧深度逐文件SHA256、像素范围和tar目录统计。
- 日志：实验下`logs/normal_gt_audit.log`。
- 完整命令：

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1015-01-jumpingjacks150_v5_spec32-原版LumiMotion两阶段/commands/audit_normal_gt.py' > '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1015-01-jumpingjacks150_v5_spec32-原版LumiMotion两阶段/logs/normal_gt_audit.log' 2>&1
```

这是GT可用性核查命令，不是已经运行的MAE渲染命令。因缺GT，未额外启动GPU法线评测。已有代表法线图为inspection/normal_75.png，视频为model_mlp/renders_stage1_insights/ours_35000/Normals_camr_0010.mp4；颜色目检不能提供角度MAE。本次没有修改训练、数据或模型权重，已有渲染、评测结果和失败日志保留。
