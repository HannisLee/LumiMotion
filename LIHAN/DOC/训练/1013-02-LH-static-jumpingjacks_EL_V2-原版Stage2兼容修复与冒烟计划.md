# 原版 Stage2 兼容修复与冒烟计划

日期：2026-10-13。服务器 ushas；环境 lumimotion-ushas；GPU 2（A40）。

使用 1011-02 的 stage1_frozen/35000 复制件，从 original 分支导出独立源码，运行 original-source/scripts/train_stage2.py 原版入口。source 沿用 1011-02/data_view；模型输出为 output/smoke_test/1013-05-LH-static-jumpingjacks_EL_V2-原版Stage2兼容修复/model_mlp。

先复现 nvdiffrast CUDA 209，再以原安装对应提交 253ac4fcea7de5f396371124af597e6cc957bfae 为 A40 的 sm_86 重编译，验证纹理前向和反向后安装。保留原二进制、编译日志及失败记录。

原版训练从 35001 到 37100，共 2100 步，保留原版 2000 步 diffuse 预热与后续 100 步完整 PBR 损失，diffuse_sample_num=512、resolution=2、depth_ratio=0。原版源码、损失及更新频率不修改。采用原有 bootstrap 显式加载仓库原生光栅器，构建产物放本实验私有缓存。

验收：原版入口正常退出；2100 条 loss 有限；跨过 diffuse 预热；保存 Gaussian、变形与环境光检查点；roughness 参数发生有效更新；重新加载检查点渲染 RGB、Alpha、Normal、分离及材质代表帧。记录量化结果与四类目检。PASS 只表示运行冒烟通过，不等于 55000 步收敛或材质质量达标。完整命令与最终结果回填实验 README。

## 执行结果

原版 Stage2 运行冒烟 **PASS**。2100 步全部完成（35001–37100），覆盖 2000 步 diffuse 预热与 100 步完整 PBR；loss 全有限，87401 个 Gaussian 的 roughness 更新，保存并重新加载检查点成功。原版四类可视化、材质渲染、15 帧 PBR NVS 均 exit=0。PSNR 38.26438039、SSIM 0.99494872、LPIPS 0.00575557。

四类代表帧目检通过运行可视化检查，但 Albedo 有阴影残留、roughness 近 0.75，未判定材质收敛；未运行完整 55000 步训练。附加 roughness GT MAE 因现有 EXR 去预乘后越界而 FAILED，无有效 MAE，失败产物保留。

完整命令、source/model/iteration、渲染目录、日志、四类目检及代表图片/视频见 [1013-05-LH-static-jumpingjacks_EL_V2-原版Stage2兼容修复 实验 README](/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1013-05-LH-static-jumpingjacks_EL_V2-原版Stage2兼容修复/README.md)。
