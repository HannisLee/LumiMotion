请使用中文撰写所有文档
所有文档名字为日期-序号-修改内容.md(如0829-01-CV3s直射光数据集基础训练.md)
如果有对代码上的修改，请更新到LIHAN/DOC/修改 文件夹下
如果有新的训练，请更新训练计划到LIHAN/DOC/训练 文件夹下
## 训练输出规定
1. 输出请统一输出到 output 文件夹下。
2. 如果是冒烟测试请统一输出到 `output/smoke_test` 文件夹内。
3. 请输出为“日期-当日计数-数据集-特征”，如 `0712-01-CV3-GTligth_fixed_geo`。CV3为clothV3数据集，目前主要都训练这个
4. 如果有 log 文件也请放在同目录内。
5. 实验 `README.md` 必须记录：完整渲染命令、source/model/iteration、渲染输出目录、日志、定量指标、代表图片/视频路径、四类可视化的目检结论以及最终 `PASS`/`FAILED` 结论。验收失败时必须保留 checkpoint、渲染、统计和日志，不得覆盖或删除失败结果。

## 文档入口

LIHAN/README.md

## 运行环境约定

本项目在多个服务器上运行。执行命令前，应先确认当前服务器名称，并使用对应的 Conda 环境。

| 服务器名称 | Conda 环境            |
| ---------- | --------------------- |
| `mahadevi` | `lumimotion-mahadevi` |
| `minakshi` | `lumimotion-minakshi` |
| `parvati`  | `lumimotion-parvati`  |
| `ushas`    | `lumimotion-ushas`    |
| `garuda`   | `lumimotion-garuda`   |