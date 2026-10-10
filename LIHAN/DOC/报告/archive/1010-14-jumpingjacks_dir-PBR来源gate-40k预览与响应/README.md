
## B08同40k检查点、全15 RGB与实际审核（2026-10-10T09:22:17.610367+09:00）

预览exec88932实际exit0并关闭，checkpoint等待审计exec3502已执行40k检查实际exit0（后续50k/55k仍绑定原训练等待）。99827点finite/全部保留，opacity/scale及32来源NNtensor保持；71553点raw feature变化，最大source logit偏移.92840385437（边界4），canonicalXYZ .03547000885m、q60.00005341°容差内；sourceXYZ/plane/native/gate marker与source35k SHA绑定，DC硬clamp0、边界41.16488191%，RGB拟合E5.49645996，非GT光强。

全部15 TEST RGB PSNR18.05530376dB（B07同40k18.05598159）、SSIM.89959086（B07.89944883）、LPIPS.11140581（B07.11159734），前景PSNR10.83801726。量化变化很小；实际view GT/B07/B080010/0080/0150图：GT/B07/B08实际首中尾拼图：黑背景正常，人物平台存在；人物头手/衣服/腿依旧破碎斑驳，脚周大块灰黑阴影污染和平台伪凸起明显。B08同40k外观几乎与B07相同，未通过质量。GT0080本身背光暗，失败不单凭暗。 当前FAILED_SO_FAR_RGB_ONLY，未测全15 Normal，不推断最终7°；正式训练持续进行，无commit/push/新分支。

命令commands/preview_40000.py282299、make_rgb_40000_comparison.py；日志logs/preview_40000_wait.log、preview_render_40000.log、checkpoint_40000_audit.log；证据preview_40000/eval/metrics.json、inspection/{rgb_40000_comparison.png,preview_40000_review.json,checkpoint_40000_audit.json}。

## B08同40k gate真实世界响应只读审计

commands/audit_gate_world_effect_40000.py实际exec29785 exit0并关闭；0 optimizer，未读GT标签/材质/法线/几何。固定同40k canonical XYZ/q/material/原32 MLP/sourceXYZ，仅比较saved gate与source35k gate的Deform+既有plane输出；不是B07另训权重，也不是GT normal准确率。4个source组完整分区全部99827点，没删点或改任何评分/训练参数、没生成标签/prior；source/checkpoint和473核心SHA保持。

概率偏移全点中位4.33647074e−8、P99.02914502、最大.25986132；source raw静动态阈值跨越91点（投影0、dynamic near-plane34、other dynamic14、static未投影43）。source dynamic near-plane6836点概率均值.97179770→.96945214、中位偏移2.45571136e−5、P99.08227499；组标签不是GT物体。TRAIN0001/0075/0149全点axis无向响应P99.13932712/.16727745/.12982376°、最大5.07943976/3.94833914/4.91278177°；近plane动态组P99.39499720/.41688162/.36503260°、XYZ P99.00792922/.00451554/.00697693m。多数source gate原已饱和，40k真实世界响应大多小；不能用raw feature变化数量宣称整体法线已修复，仍须本轮55k完整MAE/目检。完整JSON inspection/gate_world_effect_40000.json，日志logs/gate_world_effect_40000.log。
