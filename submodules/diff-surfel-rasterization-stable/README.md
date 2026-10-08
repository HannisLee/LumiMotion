# 数值稳定的深度畸变实验模块

独立包 `diff_surfel_rasterization_stable`，不替换原 rasterizer。基于当前 Conda 安装包的精确 commit `f7e2b68a0f17d6d7bca3fa564a96c7678581ce5a`，由本地 git 对象读取，无远程下载。

保留原 RGB、alpha、法线与深度公式，仅以首个有效贡献的 ray-splat 深度为参照计算 NDC 深度成对平方差，消除接近 1 的二阶矩相减。反向采用同一参照；平移不变性使参照项导数抵消。

版权与使用范围继承 LICENSE.md；来源散列见 upstream_manifest.json。GLM 头文件来自当前仓库 surfel_tracer。首次使用仓库 Lumimotion CUDA 源码的构建未通过真实 RGB 回归，已完整保留于 output/1007-03-static-geometry-stage2/stable_rasterizer_build/failed_lumimotion_source。当前构建产物位于 stable_rasterizer_build_hbb。
