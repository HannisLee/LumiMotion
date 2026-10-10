
## 2026-10-10 几何运行时遗漏更正（待完整重评）

后续源码审计确认旧scripts/eval_geometry_gt.py的GeometryRun仍直接用当前XYZ查询sourceMLP，未消费Anchor资产；逐time Mesh也使用这个入口。先前声明所有几何评测与训练一致不成立。旧GS18.91234950°/Depth27.51005704°/Mesh22.23676747°及2depth/common/missing180虽CPU复算自洽，但不是实际B04 Anchor模型结果，现标记SUPERSEDED。保留原权重/浮点图/原分数/日志，不用旧几何分数评价Anchor效果。RGB/Albedo/Roughness及原生四类渲染已用FrozenScene，受该遗漏影响的仅GeometryRun评测与Mesh；已有RGB四类质量FAILED不因此变PASS。新共享FrozenScene入口已改，179CPU PASS；真实GPU逐buffer验证和全15帧重新Mesh/几何评测待完成。B05训练运行中的source/renderer/参数不改，仅离线入口改并保存launch旧代码及hash记录，不重启训练。
