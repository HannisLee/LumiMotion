# SH预热135已见TRAIN诊断协议

2026-10-19（Asia/Tokyo；主机日志UTC2026-10-18），ushas/lumimotion-ushas。本协议在新诊断结果前固定，仅检查共同静态SH预热P00的工程与已见TRAIN变化。尚未执行GPU，所有图像目检、完整端点数值和执行状态PENDING。训练使用完整135TRAIN，原61至79除70的18图也已参与训练，不能称held或开发外推验证。

## 输入和读取边界

独立入口`scripts/diagnose_eldl_v3_sh_warmup.py`，不修改训练、旧评测、旧验证器或任何光资产。模型为`output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热/model_mlp`；source为总控`data_train_v2`。只接收`actual_entrypoint=scripts/train_eldl_v3_sh_warmup.py`、baseline static_sh、无GT/无旧权重、135帧白名单和post_optimizer_post_native_topology保存语义。

精确登记模型marker/cfg、选定端点PLY/deform/geometry.pt/receipt、完成后的geometry_parent_manifest及TRAIN manifest/135张16位RGBA/transforms/随机起始points3d；不得读eval_only、原外部TEST、GT灯/几何/材质或冻结物理灯资产。正式run安装no_gt_guard和额外精确路径audit hook，记录实际打开的模型/源文件；文件SHA首尾不变。CPU计划只读这些元数据和字节SHA，不解码图像/加载checkpoint，不触发CUDA构建；尚未完成的文件明确PENDING。正式run必须完整16000 optimizer calls、parent manifest和全部六端点，不可只见到PLY就开跑。

## 固定端点与保存解释

固定1000、12000、13000、14000、15000、16000。每点完整135个已见TRAIN、原400×400图像与相机，固定代表1/81/149。不按诊断数字选择终点。实际PLY与独立geometry.pt的xyz/scaling/rotation/opacity逐张量相等；逐点核验receipt的requested_step、completed_optimizer_calls、optimizer_audit、PLY/deform/geometry SHA和parent manifest绑定。

原native拓扑先于optimizer，不把调用次数等同各参数有效更新次数。12000和15000均为opacity reset后的post-step保存；这两点覆盖可能较低，应保留，不能当加载错误后丢弃。六点仍处于densify_until_iter=20000之前，满足间隔时会增密/裁剪；1000也处于warm_up边界及SH升阶时刻。记录每点reset/增密/活跃SH阶/点数，端点间跨拓扑变化不做Gaussian ID匹配。末五点指12000—16000五个预登记保存，并非相邻一步；1000→12000也报告但不称局部收敛率。

## 真实SH颜色和目标

加载训练快照一致的GaussianModel/DeformModel/原生renderer；active_sh_degree=min(max_degree,iteration//oneupSHdegree_step)。恢复`iteration<warm_up`才旁路deform的边界，1000即真实SH+MLP d_color。d_color的前3通道经生产render调制DC，其余SH保留；不以纯SH或物理ELDL前向代替。检查静态warp/rotation/scaling头权重全零，并逐相机断言实际d_xyz/d_rotation/d_scaling严格零、d_opacity=None。

相机读取显式传`no_bg=True`，与Scene→readNerfSyntheticInfo默认一致：原RGBA先以Alpha合成黑背景再AREA缩小，保留16位RGB和饱和通道记录；不能调用readCamerasFromTransforms默认no_bg=False而意外取消边缘合成。Alpha为既有容器只保留的通道；不读取材质RGB作统计或初始化。

逐帧报告原SH训练目标相关量：全图sRGB L1、1−SSIM、(1−lambda_dssim)L1+lambda_dssim(1−SSIM)、全图原Alpha BCE及Alpha L1、原lambda_alpha加权项、normal/dist、严格零几何项、d_color正则、binary separation正则及总和。保留normal正则原depth_ratio=1的median伪表面语义。该目标**不是**物理linear censored损失，不作线性化或饱和删失；饱和通道仅计数。冻结post-step检查点重算全135目标，不声称复现训练更新前单个随机camera日志。

原训练get_binary_feature(eval=False)采用随机Concrete gate，T固定.5。诊断在每step/frame的fork_rng内固定seed=1000×step+frame，分别保留deform调用和separation正则原随机调用，结束恢复外部RNG。d_color前3阴影调制与feature无乘积，静态几何头严格零；不能因此谎称原训练完全无随机gate，或将可重复诊断正则当原随机batch值。

## 几何变化、覆盖与缺失罚

通过仅在独立进程局部、finally恢复的Rasterizer包装截取**同一次**真实SH生产forward的allmap，不改公共renderer；同次Alpha逐值一致。先以rawZ=allmap[0]/allmap[1]保留有限性mask，再将NaN/+Inf/−Inf显式置0供存储；有效支持必须同时满足rawZ有限且Z>0。expected camera-Z独立于保留原median训练正则的surf_depth，非有限点不能借nan_to_num成为有效支持，逐帧记录原Alpha内非有限数。GS法线沿用renderer世界坐标和面向相机有向语义，归一化后求acos(dot)，不取abs、不静默翻转。

原缩小Alpha≥.5逐帧固定支持，不根据预测支持或几何变化筛选。逐点记录Alpha TP/FP/FN/IoU，expected-Z与GS各自有效覆盖；有效条件为预测Alpha≥.5、有限正Z或有限非零法线。相邻预登记端点深度与法线各用各自共同有效支持，避免互相剔除。报告像素聚合/等帧有效MAE和共同覆盖，同时固定原Alpha分母计罚：缺失法线180°；深度相对变化min(|Znew−Zold|/Zold,1)，任一端缺失计1。绝对Z变化只在共同有效像素报告，不伪造绝对缺失米值。零共同支持输出null并保留固定分母缺失罚，不伪记0。

相邻变化仅预测稳定性，不是GT准确性。脚/鞋语义没有TRAIN标签，不能在本脚本宣称分区准确率；不读GT分区。暂不增加任何通过阈值或据稳定性调光、改loss/学习率、选几何终点。

## 输出与执行顺序

CPU计划：`output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热/train_diagnostic_plan_v1/plan.json`；准备时缺失保持PENDING。模型正式完成后用**新计划目录**重新冻结，不能覆盖第一次PENDING收据。待根任务审阅本协议并明确GPU ready后，才可运行；本次代码准备不占GPU。

```bash
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python scripts/diagnose_eldl_v3_sh_warmup.py --self-check
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python scripts/diagnose_eldl_v3_sh_warmup.py --action plan --model 'output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热/model_mlp' --source 'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_train_v2' --output 'output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热/train_diagnostic_v1' --plan 'output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热/train_diagnostic_plan_v1/plan.json'
```

未来run经bootstrap_1011绑定训练同源CUDA，CUDA_VISIBLE_DEVICES由根任务释放的GPU显式指定；OMP/MKL/OPENBLAS各4，supervisor完整环境/源码/命令/退出日志放同一新诊断目录。严格使用ready时新计划：

```bash
CUDA_VISIBLE_DEVICES=<根任务已释放GPU> OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_EXTENSIONS_DIR='<仓库>/output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/native_cache' /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/bootstrap_1011.py /home/han.li/reproduce/LumiMotion-stage2-perlight scripts.diagnose_eldl_v3_sh_warmup --action run --model '<上述模型绝对路径>' --source '<上述TRAIN源绝对路径>' --output '<新诊断目录绝对路径>' --plan '<正式ready新计划绝对路径>'
```

输出plan/environment、逐端点iteration_*.json、summary、输入SHA/实际读取路径、四类代表RGB/Alpha/世界Normal/binary gate图片。分离仅gate灰度图，不是EL/DL。实际四类目检在执行后补manual_review/README；PENDING不能记精度PASS。失败保留全部产物，新目录重试；本协议未授权训练、外部GT或灯资产导出。
