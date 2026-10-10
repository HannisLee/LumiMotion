
## B06原生40k联合几何贡献审计（2026-10-10T07:19:17.158800+09:00）

完整训练继续：8684/20000。0optimizer、无GT材质/normal/geometry、未改本轮源码/参数。只读实际40k六帧0001/0038/0075/0076/0149/0150的仓库native CUDA光栅，三原坏TRAIN视角0038/0075/0076的projected支持组都朝向camera所在的plane上方，所有signed dot最小≥.99999988。PBR的alpha/worldnormal/Expected Z与原始native光栅逐buffer maxerror0，当前NN和权重fingerprint不变、469核心/GLM SHA保持launch。

联合贡献核验：全部99827点每个pass均用真实原opacity/position/rotation，输入组one-hot颜色而非只渲单组，得到真实联合alpha合成权重；四组贡献总和与全alpha最大误差6.56e−7，重复pass几何buffer0。下表为全图alpha>.5内最大权重来源组的像素数量，不是GT物体分区、不是评测mask，不用于丢点/丢像素。

| 帧 | 投影plane主导像素 | 动态plane候选主导 | 其它动态主导 | 静态非投影主导 |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 1990 | 4886 | 12057 | 1047 |
| 0038 | 4006 | 3861 | 8057 | 1150 |
| 0075 | 2610 | 2573 | 8774 | 2808 |
| 0076 | 2459 | 2688 | 8885 | 2742 |
| 0149 | 1893 | 5098 | 12314 | 954 |
| 0150 | 1922 | 4943 | 12090 | 1096 |


已实际view inspection/native_joint_geometry_comparison.png首中尾0001/0075/0150：投影组单独形成完整向上的平面，但联合图中平台被其它来源组大量遮盖；来源动态plane候选红组主要出现在平台，其它动态绿组也占据脚周/平台，人体主要绿组，衣服含白色静态非投影贡献。因此native符号修复真实生效，但它不会自动把未支持组的geometry变成平面。不能将source候选当真实平台标签或直接删除它们，不能以投影组单独的平整图片验收完整结果。

候选审计补充（尚未修改/启动）：gate只读梯度证明可通过原RGB/自身Alpha/normal-depth自洽调整时间形变幅度；若最终B06失败，可独立评估可导gate微调并绑定来源gate信赖域/先验，保持source NN与所有点、统一训练/评测入口及marker。当前Plane asset重算用source35k feature，不要求改变来源选择；需要明确当前可学习gate与固定source支持组语义。不能根据该分组图预设所有candidate必须静态、不得读取GT正常向量训练。其它候选actual SUN ray预算/Direct Preset仍未实施；需最终完整七MAE和读取GOAL后选择一个独立新实验。

第一次exec96086因诊断import误写get_rasterizer_types（真实API rasterizer_classes）exit1，未渲染/未改训练；原脚本commands/audit_native_joint_geometry_before_api_name_fix.py与logs/audit_native_joint_geometry.log保留。只修正诊断API后retry exec17809实际exit0并关闭，生成18张来源贡献/normal图和inspection/native_joint_geometry.json。实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/audit_native_joint_geometry.py
python /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/commands/make_joint_geometry_comparison.py
```

日志logs/audit_native_joint_geometry_retry.log；证据inspection/native_joint_geometry.json、native_joint_geometry_review.json、native_joint_geometry_comparison.png与native_joint_geometry_40000。只读机制PASS、最终质量/七MAE/7°仍PENDING，未commit/push/新建目标分支。
