#!/usr/bin/env python3
"""为本批实验显式载入仓库 CUDA 光栅器，再执行指定仓库的模块。"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = Path('/home/han.li/miniconda3/envs/lumimotion-ushas')
os.environ['PATH'] = str(ENV / 'bin') + ':' + os.environ.get('PATH', '')
os.environ['CPATH'] = str(ENV / 'targets/x86_64-linux/include') + ':' + os.environ.get('CPATH', '')
os.environ.setdefault('CUDA_HOME', str(ENV))
os.environ.setdefault('CXX', str(ENV / 'bin/x86_64-conda-linux-gnu-g++'))
os.environ.setdefault('CC', str(ENV / 'bin/x86_64-conda-linux-gnu-gcc'))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
os.environ.setdefault('MAX_JOBS', '4')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
sys.path.insert(0, str(ROOT))
from scene.pbr_rasterizer import native_module
module = native_module()
sys.modules['diff_surfel_rasterization'] = module
target, entry = sys.argv[1:3]
for name in list(sys.modules):
    if name.split('.')[0] in {'scene', 'utils', 'arguments', 'scripts', 'gaussian_renderer'}:
        del sys.modules[name]
sys.path = [str(Path(target).resolve())] + [p for p in sys.path if p not in {str(ROOT), str(Path(__file__).parent)}]
os.chdir(target)
sys.argv = [entry] + sys.argv[3:]
runpy.run_module(entry, run_name='__main__')
