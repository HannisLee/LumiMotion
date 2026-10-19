# 完整TRAIN平台平面支撑零优化诊断

2026-10-19，ushas／lumimotion-ushas；计划1019-38。source：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_train_v2`；model：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-53-LH-static-jumpingjacks-ELDL-V3-A45原生PBR末端联合衰减日程/model_mlp`，iteration4050，0 optimizer update，实际135TRAIN，退出0。

完整实际命令：

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python /home/han.li/reproduce/LumiMotion-stage2-perlight/LIHAN/scripts/bootstrap_1011.py '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-18-LH-static-jumpingjacks-ELDL-V3-三阶段总控/snapshots/three_stage_v15_floor_resume' diagnose_train_plane_support_v1 --output '/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-55-LH-static-jumpingjacks-ELDL-V3-TRAIN平台平面支撑零优化诊断'
```

输出/日志：`/home/han.li/reproduce/LumiMotion-stage2-perlight/output/1019-55-LH-static-jumpingjacks-ELDL-V3-TRAIN平台平面支撑零优化诊断`，pipeline.log、job_state.json、canonical_plane.json、逐135帧metrics.json、receipt.json、三张train_plane_{0001,0075,0149}.png及实际manual_review.json。父checkpoint/PLY/32MLP/marker前后SHA逐文件不变。

canonical支撑73136/146272点，中心=[0.028669208195904208, 0.035350185825316835, -0.015220119778691666]，朝+Z法向=[0.0002113606514983576, -0.00035748761641051906, 0.9999999137646357]，XY参考半径95分位=1.023501488614888m；点距均值=0.0025672258529094313m，p95=0.0061253088062508025m。各TRAIN有效几何拟合相对canonical法向偏离最大=0.003972780610401712°，中心漂移最大=3.183863036198369e-05m。该支撑拟合稳定不等于可见表面或GT平面。

所有135TRAIN均有相应区域；下表角度都是预测内部或预测拟合平面的一致性，并非GT误差。GS/Expected与平面为朝向相机的有向角，Gaussian轴角另为无向描述。像素加权和等帧明确分开：

|区域|有效像素|GS-Expected像素加权°|GS-Expected等帧°|GS-平面像素加权°|Expected-平面像素加权°|
|---|---:|---:|---:|---:|---:|
|interior|4502562|6.109181674665048|6.125829944183021|11.43310632625307|15.561230951481257|
|plane_band|588356|5.311595339000028|5.931158460188079|1.8716285883129797|5.415548184778223|
|plane_annulus|274125|3.5060079907833614|4.154461989792825|1.4669662810962807|3.58398871893443|
|outside_plane_band|3914206|6.22906926006718|6.231549954708048|12.870320615941322|17.086258844720994|

高度带仅占全部内部有效域0.1306713822041762，外环高度带占0.06088200451209778。这是相对内部有效域的比例，不是GT平台覆盖率。canonical支撑Gaussian轴角等帧平均约14.500167036579189°，与实际渲染像素总体不同，不能直接把此数解释成可见GS误差。

实际目检：三视角GS和Expected保留人体与平台，Expected平台法线更细碎。按canonical对应点拟合的1cm高度带在可见平台上只形成斑块，外环高度带更稀疏，明显不能替代完整可见平台；部分近鞋/中心支撑可能含脚边，不能当语义平台标签。平台Expected点距图大范围未落入窄带，GS-Expected误差在人体轮廓和平台纹理均存在。该反例不支持直接把canonical支撑索引作为完整可见平台训练约束。。范围：实际查看TRAIN1/75/149三张3x3几何/支撑图，非RGB/PBR或GT精度验收。；代表图和SHA见manual_review.json。RGB/Separation未重新渲染，本诊断仅Alpha/GS/Expected及支持/角度/点距；原A45完整RGB/Alpha/Normal/Separation和物理分量的已实际验收见报告1019-116，不冒称本次完成PBR/BVH核验。没有新视频或TEST/GT评价。

最终 **PASS（零优化执行、父资产不变及实际三TRAIN查看）**；**FAILED（canonical窄带支撑可替代完整可见平台的假设）**。所有结果保留。该反例要求先从TRAIN可见表面检查支撑，不能直接据此增加平面训练约束；当前A46预算/参数保持原计划，尚未判断其最终精度。
