# 1002-07-minakshi历史Conda环境

归档日期：2026-10-02。

## 归档来源与适用范围

- 分支 `PS-stage1-V1`，提交 `bfa2c4675b99367f9ae84b0339142293b24be78a`，原路径 `DOC/Archive/minakshi-conda环境.md`。
- 分支 `PS-stage1-V2`，提交 `9021a8d8eb0d2ac8e8149aa29403544bb0278f20`，原路径 `DOC/Archive/minakshi-conda环境.md`。
- 分支 `PS-stage1-V3`，提交 `ff08af104359177bcb9f758209578ebd7420c9e0`，原路径 `DOC/Archive/minakshi-conda环境.md`。
- 分支 `PS-stage1-V5`，提交 `668e83e1ed0894381abd76c28d1dce9ab97b6a12`，原路径 `DOC/Archive/minakshi-conda环境.md`。
- 分支 `PS-tools-common`，提交 `b14cf4193c9a3952703b57533d976191bef848fe`，原路径 `DOC/Archive/minakshi-conda环境.md`。

本文件为历史资料，参数、环境、绝对路径及实验结论对应来源提交，不能直接视为当前分支的训练配置。现行服务器环境以根目录 `LIHAN.md` 为准。原始 Git 对象：`b128550caf865f1563598217b098154d38e6b4f8`；原始内容可用 `git show <来源提交>:<原路径>` 追溯。正文命令未执行。

---

在 `minakshi` 上重编 CUDA/native 扩展时，conda 激活脚本可能默认设置 `NVCC_PREPEND_FLAGS` 到 base conda 的 GCC 14，CUDA 12.1 不兼容。编译前必须覆盖为系统 g++ 11：

```bash
export NVCC_PREPEND_FLAGS=" -ccbin=/usr/bin/g++-11"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.9"
```

从仓库根目录执行脚本，并优先使用模块方式：

```bash
python -m scripts.train_stage1 ...
python -m scripts.train_stage2 ...
```

除非已确认导入路径不会出问题，不要优先使用 `python scripts/foo.py`。
