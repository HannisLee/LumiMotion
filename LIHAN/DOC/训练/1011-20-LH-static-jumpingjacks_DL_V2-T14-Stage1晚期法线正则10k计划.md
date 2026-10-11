# 1011-20-LH-static-jumpingjacks_DL_V2-T14-Stage1晚期法线正则10k计划

## 研究问题与控制变量

T14是T13的配对控制：两组从相同静态DL随机初始化、使用相同135帧、相同10000步预算和相同所有物理/损失/优化器设置。唯一变量是法线一致性正则的开始步数：T14在step8000启用，T13在step2000启用。该设计只检验normal-reg启用日程，distortion日程两组完全相同。

## T14配置

- 输出：`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal-reg-8000-Stage1-10k`。
- 入口：`scripts.train_static_dl_early_geometry`；实现稳定前不创建快照或训练输出，完整CLI待入口稳定后追加。
- 初始化：从头训练，使用与T13相同的100000点确定性随机`points3d.ply`（seed=0）及完全相同SHA；不从Stage2 checkpoint加载。
- 数据及监督：`output/1011-09-静态DL-Goal训练总控/data_view`原135帧RGB、Alpha、训练光照方向与正确相机。禁止使用GT法线、深度、材质、阴影、光强作训练监督；测试图不加载。expected统计量只进入独立几何副本，RGB前向depth_ratio=1。
- 训练预算与保存：10000步，保存5000、8000、9000、10000。T13/T14共享增密截止20000及原Stage1其余训练设置。
- 正则日程：`start_normal_reg=8000`；两组共用`dist_start=8000`、dist系数1000、normal权重0.1。
- 物理与RGB设置：physical start=1000，visibility start=5000，shadow bias=0.03，BVH interval=1；censored RGB损失，shadow weight=0.25、highlight weight=0.5、residual threshold=0.08、alpha weight=0.1。
- 其他参数严格沿用T03 Stage1：deform-type mlp、resolution2、is_blender/eval/load2gpu_on_the_fly/gt_alpha_mask_as_scene_mask、train/test light folder images；lambda separation=0.001，d_xyz loss weight=0.001，binarization warm-up=1000，d_color regularizer=0.01；XYZ LR日程沿用原30k长度不压缩，rotation=0.001、albedo=0.01、light=0.001；seed、方向JSON、初始化文件及来源SHA由稳定入口和prelaunch审核记录。

## 启动门槛与状态

入口稳定后，T13/T14分别用独立模型目录、同一快照和共享初始化SHA。smoke必须实际越过物理、T14 normal-reg及dist-reg启用边界，并由梯度/状态检查确认，而非仅执行SH预热。首帧和边界审计、checkpoint来源、native实现及fresh parity均通过后才可正式运行；停止在T14专属normal启用前的共同行为及启用后差异均需如实记录。所有日志、checkpoint、配置和SHA留存，失败不覆盖。

T14计划输出目录为`output/1011-28-LH-static-jumpingjacks_DL_V2-T14-normal-reg-8000-Stage1-10k`。当前仅为预注册，训练入口未stable，尚未创建快照、冒烟或正式输出目录。
