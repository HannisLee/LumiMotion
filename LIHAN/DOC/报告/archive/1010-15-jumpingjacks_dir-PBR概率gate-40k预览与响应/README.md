
## B09同40k检查点/全15 RGB与实际响应（2026-10-10T10:16:35.143803+09:00）

预览exec1259与gate-world等待诊断exec14352均实际exit0并关闭，检查点父exec19913已完成40k子审计exit0、仍等待50k/55k。全部99827点finite/保留，opacity/scale与32NNtensor原值，max概率delta .60847592；raw max79.74610138是有限端点/逆sigmoid编码，不沿用旧raw±4或声称同量世界变化。CanonicalXYZ .03549813m/q60.00006485°容差内，DC硬clamp0/边界40.58320895%、RGB拟合E5.50689125非GT强度，source/plane/native/v2 marker绑定。

全15 RGB PSNR18.35268777（B08同40k18.05530376）、SSIM.90386886（.89959086）、LPIPS.10786089（.11140581），前景PSNR10.98395868。实际view GT/B08/B09 TEST0010/0080/0150：黑背景正常，人物/平台存在；B09头/衣服/腿略更完整，平台脚周黑灰污染稍有变化，但人物斑驳/手臂与腿稀疏、平台脚周伪凸起仍明显，未通过质量。GT0080自身背光暗，不单以暗失败。没有正式40k Normal/七MAE，不推断最终7°。

同一40k canonical/material/sourceNN/XYZ/plane，仅比较saved概率与source35k native概率，0 optimizer/noGT。全点概率delta中位.00163957/P99.35333336（B08对应4.34e−8/.02914502），source raw阈值跨越227；动态near-plane6836点概率均值.97179770→.89238799、delta中位.05443400。TRAIN1/75/149全点axis响应P99 3.53469853/4.60136210/3.66612515°（B08约.139/.167/.130），动态near-plane axis响应P99 2.37774202/2.08524452/2.21833535°、XYZ P99 .06728367/.02283315/.07268968m。概率方案的实际响应比raw更大，但不等于GT法线改善/7°；source组不是GT物体，全99827点完整分组、不改评分。生产474核心/SHA保持，当前完整train291976持续，最终七MAE/四类/HDR/质量待完成，无commit/push/新分支。

命令commands/{preview_40000,make_rgb_40000_comparison,audit_gate_effect_when_40000_saved,audit_gate_world_effect_40000}.py，检查audit_checkpoint.py40000；证据preview_40000/eval/metrics.json、inspection/{preview_40000_review,rgb_40000_comparison,checkpoint_40000_audit,gate_world_effect_40000,gate_effect_wait_state}.json/png与对应logs。
