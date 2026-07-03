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