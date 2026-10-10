# 1010-09 原生PBR显式仓库CUDA光栅与Native Preset

已读取GOAL及B05全量结果：GS17.73081984°，较B04更正值改善1.04661466°，但RGB/四类/HDR质量FAILED和7°FAILED，全部177归档文件SHA三处同步核验通过。新修改只修复运行时实际CUDA来源/法线朝向，不同时关闭SH间接光或改ray预算。

## 实际证据与修改

环境installed diff_surfel_rasterization0.0.1的Git来源hbb1 f7e2b68a，以normal.view_z<0确定方向；本仓库LumiMotion原始CUDA以-dot(n_view,p_view)>0确定方向，forward/backward一致。世界plane与相机已证明0038/0075/0076在plane上方但旧包朝下，均是TRAIN帧；不是某个GT normal错误或评分mask问题。独立原生六相机0步GPU核验全部朝向正确，合成normal梯度finite difference最大误差.00035429；不根据旧权重替换渲染宣称新训练质量。

scene/pbr_rasterizer.py独立按CUDA/GLM来源hash构建namespace，原LumiMotion kernel不改、不pip install；GLM优先native嵌套目录，缺失时用仓库stable目录的已追踪同库头文件，缺失则拒绝构建。不会静默回退installed同名包。pbr_rasterizer.json保存每文件SHA/综合SHA、.so SHA、torch/CUDA版本；checkpoint绑定综合SHA，加载和各评测核验，不允许不同native源码被当同checkpoint。旧Preset与Stage1/env默认仍installed；新pbr_gt_directional_native和pbr_gt_directional_plane_native显式pbr_rasterizer_backend=lumimotion。仅支持scale/rotation CUDA路径，compute_cov3D_python=True拒绝。

训练render_ir、周期测试、FrozenScene、全四类/方向光/HDR、GeometryRun/逐time Mesh、原始材质albedo/roughness helper全部同后端；原有评分/GT域/coverage/common/missing180不改，不把最终混合normal后翻转。eval source包含kernel与GLM每文件SHA。render provenance保存本次真实构建记录。

## 验证与限制

187 CPU最终discover PASS，真实20步35000→35020 Trainer保存加载marker；三TRAIN帧PBR/GGX/BVH对材质与真实geometry梯度非零finite；source权重保持。GeometryRun三个test frame alpha/normal/Z逐buffer0、Mesh timeoverride一致，453来源SHA通过。材质helper albedo0、roughness黑0/白5.96046e−8。四类150与HDR16/方向光三帧已渲染且首中尾实际看：背景/主体存在、0075平台定向修复，人物/脚周伪影仍有，只FUNCTIONAL PASS；不代替完整20000步/15七MAE/7°验收。

初次独立build缺GLM头文件失败、首次CPU指定两个不存在测试名失败，日志均保留；补本地GLM快照retry/实际discover通过。详见两个smoke README及命令/log/inspection。B05所有权重/指标仍对应旧实际kernel，不覆盖、不改分数定义。

## 完整训练计划

B06 output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复：固定35k→55k全20000步，仅Stage2，B05所有参数一致只换Native backend/preset，从原始来源独立训练。完整all与15测试帧七MAE及独立复算/每time135train Mesh、四类/HDR目检全部必须完成。原RGB/自身RGBA/相机/time/source预测/GT direction-only；禁止GT材质/normal/geometry/GT强度与颜色用于训练。最终质量/7°PENDING，未commit/push/新分支。

完整命令/source/model/iteration/log/产物及验收计划见LIHAN/DOC/训练/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复训练报告.md和输出README/planned_commands.json。修改仅当前1001-stage2-perlight代码，LIHAN文档/index三处同步。
