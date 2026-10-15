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
