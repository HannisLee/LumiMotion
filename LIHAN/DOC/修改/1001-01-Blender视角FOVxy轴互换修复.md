# Blender 视角 FOVxy 轴互换修复

> 归档来源：本记录对应的训练或代码修改发生在原始项目 `/home/lihan/reproduce/LumiMotion`（`main` 分支），不是在 `/home/lihan/reproduce/LumiMotion-perlight` 中执行。2026-10-02 将记录复制到本仓库；1001-01 的完整产物已复制至本仓库 `output/1001-01-onlyclothV4-FOVxy修复重训/`。原始绝对路径和命令保留用于追溯。0930 输出未在本次复制范围内。

## 原因与修改

`scene/dataset_readers.py` 的 `readCamerasFromTransforms` 正确读取水平角 `camera_angle_x`，并用图像宽高计算垂直角，但随后将水平角赋给 `FovY`、垂直角赋给 `FovX`。下游相机和投影矩阵按正确轴向消费这些字段，导致非正方形图片的投影比例错误。

修复为 `FovX = fovx`、`FovY = fovy`。此次仅修正轴向赋值，保留原始读取器的其他行为。

## 验证

使用实际 Blender reader 读取 only_clothV4 的全部 15 个测试相机，逐一核对 FOV 与 JSON 中的 `camera_angle_x`、`camera_angle_y`，并反算焦距与 `fl_x`、`fl_y` 比较，断言全部通过。1280×720 图像的水平视角为 58.7155°、垂直视角为 35.1154°；两个轴的焦距均为 1137.7778 像素。

## 对已有实验的影响

`output/0930-01-onlyclothV4-LumiMotion原始管线` 使用了错误的 FOV。其检查点和指标保留供追溯；原先 PASS 仅说明流程执行成功，不能作为正确相机投影下的训练验收。需要重新训练与评估后才能获得修复后的指标。
