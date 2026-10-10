"""显式选择仓库原生LumiMotion光栅器，避免同名pip包替换实际CUDA实现。"""
from functools import lru_cache
from pathlib import Path
from types import ModuleType
import hashlib
import importlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'submodules/2dgs_rasterizer_lumimotion/diff-surfel-rasterization'
ASSET='pbr_rasterizer.json'
METHOD='repository_lumimotion_cuda_rasterizer'
TRANSLATION_UNITS=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/forward.cu',
                   'cuda_rasterizer/backward.cu','rasterize_points.cu','ext.cpp']


def validate_backend(args):
    backend=getattr(args,'pbr_rasterizer_backend','installed')
    if backend not in {'installed','lumimotion'}:
        raise ValueError('未知PBR光栅器：'+str(backend))
    if backend=='lumimotion':
        if getattr(args,'render_mode','original_ir')!='original_ir':
            raise ValueError('仓库原生PBR光栅器仅适用于original_ir。')
        if getattr(args,'compute_cov3D_python',False):
            raise ValueError('仓库原生PBR需要真实scale/rotation CUDA路径；关闭compute_cov3D_python。')
    return backend


def source_manifest():
    """绑定真实CUDA/Python与GLM头文件；嵌套原生GLM缺失时使用仓库已有的同库头文件。"""
    glm=SOURCE/'third_party/glm'
    if not (glm/'glm/glm.hpp').is_file():
        glm=ROOT/'submodules/diff-surfel-rasterization-stable/third_party/glm'
    if not (glm/'glm/glm.hpp').is_file():
        raise FileNotFoundError('缺少GLM头文件，不能构建仓库原生PBR CUDA光栅器。')
    files=[]
    for folder in [SOURCE/'cuda_rasterizer',glm/'glm']:
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.h','.hpp','.inl','.cu'})
    files.extend(SOURCE/x for x in ['rasterize_points.cu','rasterize_points.h','ext.cpp',
                                  'diff_surfel_rasterization/__init__.py'])
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return {'format_version':1,'method':METHOD,'backend':'lumimotion','source_sha256':digest,
            'source_files':hashes,'glm_include':str(glm.relative_to(ROOT)),
            'normal_orientation':'camera_ray_dot_normal','gt_geometry_normal_material_inputs':False}


@lru_cache(maxsize=1)
def native_module():
    import torch
    from torch.utils.cpp_extension import load
    record=source_manifest()
    name='lumimotion_pbr_cuda_'+record['source_sha256'][:16]
    os.environ.setdefault('MAX_JOBS','4')
    extension=load(name=name,sources=[str(SOURCE/x) for x in TRANSLATION_UNITS],
                   extra_include_paths=[str(ROOT/record['glm_include'])],verbose=False)
    wrapper=SOURCE/'diff_surfel_rasterization/__init__.py'
    source=wrapper.read_text()
    if source.count('from . import _C')!=1:
        raise ValueError('仓库原生光栅Python wrapper格式不匹配，拒绝静默回退pip实现。')
    source=source.replace('from . import _C','')
    module=ModuleType(name+'_wrapper');module.__file__=str(wrapper);module._C=extension
    sys.modules[module.__name__]=module
    exec(compile(source,str(wrapper),'exec'),module.__dict__)
    module.provenance=dict(record,extension_path=extension.__file__,
                           extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                           torch_version=torch.__version__,torch_cuda_version=torch.version.cuda)
    return module


def rasterizer_classes(args):
    module=(native_module() if validate_backend(args)=='lumimotion'
            else importlib.import_module('diff_surfel_rasterization'))
    return module.GaussianRasterizationSettings,module.GaussianRasterizer


def prepare_backend(model,args):
    if validate_backend(args)=='installed':return None
    record=native_module().provenance
    path=Path(model)/ASSET
    if path.exists():raise FileExistsError('原生光栅来源资产禁止覆盖。')
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


def validate_backend_model(model,args):
    if validate_backend(args)=='installed':return None
    record=json.loads((Path(model)/ASSET).read_text())
    expected=source_manifest()
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('仓库原生PBR光栅CUDA/GLM来源或协议与训练记录不匹配。')
    return record
