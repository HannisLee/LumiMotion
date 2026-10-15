#!/usr/bin/env python3
"""隔离降低Alpha剔除和透射率终止阈值；前反向Alpha阈值一致。"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
import shutil
import json
import scene.pbr_rasterizer as backend
out = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
copy_root = Path(os.environ.get('DL_LOW_THRESHOLD_SOURCE', str(out.parent / 'low_threshold_source'))).resolve()
assert copy_root.is_relative_to(ROOT / 'output')
if not copy_root.exists():
    shutil.copytree(backend.SOURCE, copy_root, ignore=shutil.ignore_patterns('third_party', 'build', '*.so', '__pycache__'))
    for name in ['forward.cu', 'backward.cu']:
        path = copy_root / 'cuda_rasterizer' / name
        text = path.read_text()
        assert text.count('1.0f / 255.0f') == 1
        text = text.replace('1.0f / 255.0f', '0.00001f')
        if name == 'forward.cu':
            assert text.count('test_T < 0.0001f') == 1
            text = text.replace('test_T < 0.0001f', 'test_T < 0.000001f')
        path.write_text(text)
# 复用已编译源码必须核对全部参与编译的源文件，不能静默加载任意修改。
for original in backend.SOURCE.rglob('*'):
    if not original.is_file() or original.suffix not in {'.cu', '.cpp', '.h', '.hpp', '.inl', '.py'}:
        continue
    relative = original.relative_to(backend.SOURCE)
    if any(part in {'third_party', 'build', '__pycache__'} for part in relative.parts):
        continue
    expected = original.read_text()
    if str(relative) in {'cuda_rasterizer/forward.cu', 'cuda_rasterizer/backward.cu'}:
        expected = expected.replace('1.0f / 255.0f', '0.00001f')
        if original.name == 'forward.cu':
            expected = expected.replace('test_T < 0.0001f', 'test_T < 0.000001f')
    assert (copy_root / relative).read_text() == expected, str(relative)
backend.SOURCE = copy_root
module = backend.native_module()
(out.parent / 'low_threshold_provenance.json').write_text(json.dumps(module.provenance, ensure_ascii=False, indent=2))
sys.modules['diff_surfel_rasterization'] = module
target, entry = sys.argv[1:3]
for name in list(sys.modules):
    if name.split('.')[0] in {'scene', 'utils', 'arguments', 'scripts', 'gaussian_renderer'}:
        del sys.modules[name]
sys.path = [str(Path(target).resolve())] + [p for p in sys.path if p not in {str(ROOT), str(Path(__file__).parent)}]
os.chdir(target)
sys.argv = [entry] + sys.argv[3:]
runpy.run_module(entry, run_name='__main__')
