# 原版 Stage2 的 nvdiffrast A40 兼容修复

服务器 ushas，Conda 环境 lumimotion-ushas。原环境在 A40（sm_86）执行 nvdiffrast.texture 时复现 CUDA 209。

使用原安装对应的 NVlabs/nvdiffrast 提交 253ac4fcea7de5f396371124af597e6cc957bfae，以 TORCH_CUDA_ARCH_LIST=8.6、CUDA_HOME=/home/han.li/miniconda3/envs/lumimotion-ushas 重编译 0.4.0 wheel。先在实验私有目录验证，再安装到 lumimotion-ushas。没有升级 PyTorch、CUDA 或修改 Stage2 算法。

输出：output/smoke_test/1013-05-LH-static-jumpingjacks_EL_V2-原版Stage2兼容修复。原安装备份 original_install，wheel 位于 wheels，编译和安装日志位于 logs。回滚可从 original_install 恢复原 nvdiffrast 包、dist-info 与 _nvdiffrast_c 二进制；原版失败结果保留。

候选版本纹理前向、tex/uv 反向均 PASS，输出和为 12，纹理梯度和为 12。安装后验证与 2100 步原版训练验收均已 PASS。

光栅器重新构建采用本次实验私有 native_cache，记录实际源码和二进制 SHA。首次构建因 shell PATH 未包含 Conda bin 而找不到 ninja；补全 PATH 后重编译成功，两个日志均保留。此前其他实验的二进制身份校验失败与本次 nvdiffrast CUDA 209 属于不同问题。

为确认原版身份，本次从 original 分支直接导出 original-source。训练、渲染、光照、GaussianModel 与参数文件均与该分支逐字节一致，source_identity.json 记录哈希。原有 bootstrap 只选择仓库原生 CUDA 光栅器并设置编译环境。

## 执行结果

原版 Stage2 运行冒烟 **PASS**。2100 步全部完成（35001–37100），覆盖 2000 步 diffuse 预热与 100 步完整 PBR；loss 全有限，87401 个 Gaussian 的 roughness 更新，保存并重新加载检查点成功。原版四类可视化、材质渲染、15 帧 PBR NVS 均 exit=0。PSNR 38.26438039、SSIM 0.99494872、LPIPS 0.00575557。

四类代表帧目检通过运行可视化检查，但 Albedo 有阴影残留、roughness 近 0.75，未判定材质收敛；未运行完整 55000 步训练。附加 roughness GT MAE 因现有 EXR 去预乘后越界而 FAILED，无有效 MAE，失败产物保留。

完整命令、source/model/iteration、渲染目录、日志、四类目检及代表图片/视频见 [1013-05-LH-static-jumpingjacks_EL_V2-原版Stage2兼容修复 实验 README](/home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1013-05-LH-static-jumpingjacks_EL_V2-原版Stage2兼容修复/README.md)。
