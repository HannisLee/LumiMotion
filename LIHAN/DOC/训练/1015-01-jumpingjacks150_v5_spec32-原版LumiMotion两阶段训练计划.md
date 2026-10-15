# jumpingjacks150_v5_spec32 原版 LumiMotion 两阶段重跑

日期：2026-10-15（Asia/Tokyo）。沿用1015-01编号，从头训练，不加载旧20000步权重。

服务器ushas；Conda环境lumimotion-ushas；GPU 2（A40）。动态source：`/home/han.li/reproduce/LumiMotion/data/d-nerf-relight-spec32/jumpingjacks150_v5_spec32`。800×800 RGBA，135训练/15测试相机，共150时间步；resolution=2，实际400×400。训练光chapel_day_4k_32x16_rot0，测试光golden_bay_4k_32x16_rot330。模型为本目录`model_mlp`。

原始资料：作者仓库 https://github.com/joaxkal/LumiMotion ，作者提交65287461075a1a4c0f54845325cc22039bd14e7d；论文与补充 https://arxiv.org/html/2604.10994v1 ，重点核对附录5、6；场景配置见references/upstream/bash_scripts/synthetic_results_from_paper.sh。已有数据来自官方合成基准，未使用LH转换数据。

本次源码为baseline仓库提交396215f217c875b20415fafc456fac370adde860的私有导出`source/`。本地与作者差异见references/local_vs_upstream.patch；哈希见references/source_identity.json；完整参数见references/effective_hyperparameters.json；包版本见references/pip_freeze.txt。环境沿用Python3.10.20/PyTorch2.11.0+cu128/NumPy1.24.4，与官方安装示例版本不同，不升级已有native扩展。

## 训练参数与差异

Stage1=35000步，Stage2从35000至累计55000步；动态MLP、8层×256、SH degree3、hyper_dim1、Concrete温度0.5。GT alpha作为场景mask，黑背景，逐相机搬运GPU。训练不输入GT HDR，环境光由Stage2学习。

- Stage1：lambda_separation=0.001，d_xyz_loss_weight=0.001，d_color_reg_loss_weight=0.01；warm_up与binarization_warm_up均1000；depth_ratio=1.0。
- 增密：从>500步开始，每100步，直到20000；梯度阈值0.0002，percent_dense=0.01；每3000步reset opacity，min_opacity=0.01。
- 损失：DSSIM=0.2，alpha BCE=0.1；normal/dist在>8000步启用，代码normal=0.02、dist=1000。
- Stage1学习率：xyz基础0.00016→0.0000016，空间倍率5；feature0.004；opacity0.05；scaling基础0.002×5；rotation0.001；albedo0.01；形变初始0.0008，调度末值参数0.0000016、max_steps40000，保留实际代码的参数组更新方式。
- Stage2：冻结位置、旋转、尺度、分离和形变主体；训练albedo、roughness、环境光，微调opacity、Stage1颜色与颜色头。albedo由canonical颜色初始化，roughness初值0.75。前2000步diffuse，之后18000步完整PBR。
- Stage2采样：512入射方向，2^18条光线/步，即512个像素；depth_ratio=0；保留镜面和间接光。环境光32×16，exp激活，init_value1.5，下半球正则0.00001。
- Stage2实际基础学习率：env0.1、albedo0.01、roughness0.002；opacity0.005、Stage1颜色0.001、颜色头初始化0.000075。额外roughness/light/base-color平滑正则均0。NVS与重光照评测采样2048。

论文与作者代码存在差别：论文normal=0.002而代码0.02；论文env/roughness LR=0.2/0.005而代码0.1/0.002；形变末端LR描述0.00008而代码调度参数为0.0000016。补充材料指向代码获取精确设置，本次以公开场景脚本及代码为准，不混合两套数值。

本地baseline已有影响轨迹的修复：opacity重置上限由作者0.01改为max(0.01,2*min_opacity)，本次为0.02，避免重置后浮点值低于prune阈值。这是沿用上次方式，不能声称全部算法文件与作者逐字节相同。额外prune、Gaussian上限和续训开关均不启用。其他已有补丁包括RGBA uint8、相机FOV读取、Open3D延迟导入和Matplotlib兼容。Stage2入口、render_ir、EnvLight与形变网络保持作者实现。

官方总控shell含未定义START_BIN和W_BIN解析表达式问题；本次直接把场景表里的0.001传给模块，不执行该整表shell。现装rasterizer的Python wrapper与baseline bundled版本哈希相同；已存在A40 nvdiffrast兼容构建。完整环境来源另存JSON。

## 输出、调度与验收

流程：Stage1 → 完整时序insights → Stage2 → materials → 官方static scale/material/relight/NVS → dynamic同套评测 → HDR展示。static对照使用同级jumpingjacks150_v5_spec32_statictimestep75，仅用于评测。训练恢复官方默认测试/保存日程，追加最终步，不再只在最终步评测。

Stage1渲染目录：model_mlp/renders_stage1_insights/ours_35000；Stage2材质目录：model_mlp/trained_materials/ours_55000；NVS：model_mlp/eval_nvs_dynamic/ours_55000；HDR：model_mlp/render_relight_with_hdr_golden_bay_4k_32x16_rot330/ours_55000。完整阶段命令在commands，每步日志在logs，总控日志logs/pipeline.log，状态status.json。TensorBoard保存在model_mlp。

独立tmux会话1015-01-lumimotion，启动器每30秒记录心跳，每阶段记录PID、退出码和时间。失败保留所有checkpoint、渲染、统计和日志。执行成功只表示COMPLETED_PENDING_REVIEW，仍须实际目检RGB、Alpha、基色/动静分离、Normal；训练期间不伪造最终PASS。

旧中断结果已整体归档，准确路径见references/archive.json；原目录由本次重建。最终阶段前定量结果、代表图片/视频和四类目检结论均为待生成/待检查。

启动：`bash /home/han.li/reproduce/LumiMotion-stage2-perlight/output/1015-01-jumpingjacks150_v5_spec32-原版LumiMotion两阶段/commands/run_pipeline.sh`。准确完整环境和命令见实验README。


## 已有结果与代表帧目检（2026-10-15 Asia/Tokyo）

Stage1与Stage2均exit=0，55000步模型、deform和envmap已保存。Stage1用时49分43秒，Stage2用时47分13秒（均含初始化/保存）。静态材质评测：Albedo PSNR=34.88827006、SSIM=0.9840739131、LPIPS=0.0219776845；roughness MSE=0.00790291696，指标文件model_mlp/results_material_static.json。

代表图：inspection/four_classes.png、inspection/materials.png。RGB/Alpha/Normal/Separation及Albedo/Roughness六个视频均报告150帧，抽取并成功解码索引0、75、149；不声称逐帧人工目检全部视频。

- RGB：人物和圆盘在三张代表帧中完整，姿态变化合理，服装与棋盘格可辨；画面上方绿色字为原版视频注释。
- Alpha：人物、四肢、圆盘轮廓完整，背景黑；未见大面积空洞或漂浮条带。
- Normal：圆盘整体平滑，人物轮廓对齐；头部、抬起的手臂等细节存在颗粒噪声。
- 基色/动静分离：人物主要红色动态、圆盘主要绿色静态；足部邻近圆盘有少量混合。Stage2 Albedo保留衣服颜色和棋盘纹理，代表帧地面没有明显大片阴影。
- 材质限制：roughness图出现明显棋盘纹理/凹凸状变化；目前没有据此认定材质完全恢复。

Stage1代表帧四类检查PASS（仅限所述抽样范围，包含法线细节噪声）。全流程最终结论待重光照、NVS等剩余评测完成；原静态重光照因CUDA编译缺头文件FAILED，补跑记录见evaluation_retry_01，不能把训练完成等同于整套评测通过。

补跑只调整编译子进程环境：CPATH指向Conda targets/x86_64-linux/include，并设置对应CC/CXX与实验私有扩展缓存。原失败日志、初始状态、已有模型和结果保留。


## 扩展加载兼容的补充修复

头文件路径补齐后CUDA扩展编译成功，但原版ops.py丢弃cpp_extension.load的返回值再import同名模块，在本环境PyTorch2.11下触发ModuleNotFoundError。仅修改实验私有source/scene/renderutils/ops.py：把load返回值直接存为_cached_plugin，删除重复import。CUDA源文件、训练代码、模型权重和渲染计算均未改动。补丁与修改后哈希在实验references/renderutils_compatibility.patch及.json；训练时source_identity.json保留原值，不追改训练来源记录。

前两次失败日志和状态均保留，第二次恢复目录为evaluation_retry_02，根status.json镜像当前恢复状态。新增代表帧检查仅抽样0/75/149，不声称完整视频人工逐帧检查。
