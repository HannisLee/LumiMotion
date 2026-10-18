"""在已冻结真实接收点上核查生产全BVH点着色语义，不代表完整图像或GT精度。"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','fit','validation','source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name,value in vars(a).items():setattr(a,name,value.resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache,a.cache.parent/'manifest.json',a.fit,a.validation,a.source,a.output])
    data=torch.load(a.cache,map_location='cpu',weights_only=False)
    meta=json.loads((a.cache.parent/'manifest.json').read_text())
    fit=json.loads((a.fit/'summary.json').read_text())
    report=json.loads((a.validation/'summary.json').read_text())
    if (meta.get('receiver_representation')!='surfel' or meta['gt_read'] or report['gt_read']
            or report['external_test_read'] or not report['exact_bvh']):
        raise ValueError('需要无GT真实接收点的全BVH验证')
    if sha(a.cache)!=fit['cache_sha256'] or sha(a.cache)!=meta['cache_sha256']:
        raise ValueError('缓存来源绑定失败')
    state=torch.load(a.fit/f"init_{fit['selected_init']}/fit.pt",map_location='cpu',weights_only=False)
    exact_path=a.validation/'arrays/exact_bias_1.pt'
    if sha(exact_path)!=report['output_sha256'][str(exact_path)]:raise ValueError('精确BVH数组变化')
    exact=torch.load(exact_path,map_location='cpu',weights_only=False)
    spec=importlib.util.spec_from_file_location('production_eldl',ROOT/'scene/eldl_v3_photometric.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames=json.loads((a.source/'transforms_train.json').read_text())['frames']
    assert [frame['source_frame'] for frame in frames]==[frame['frame'] for frame in data['frames']]
    disagreements=0;maximum=0.;valid_maximum=0.
    for i,frame in enumerate(frames):
        center=torch.tensor(frame['transform_matrix'],dtype=torch.float32)[:3,3]
        normal=data['receiver_normals']
        sign=torch.where((normal*(center-data['points'])).sum(-1,keepdim=True)>=0,1.,-1.)
        disagreements+=int(((sign[:,0]<0)&data['valid'][i]).sum())
        _,_,prediction=module.diffuse_components(state['albedo'],normal*sign,
            state['directions'][i:i+1],torch.tensor([math.pi]),state['environment'],exact['visibility'][i,:,None])
        delta=(prediction-exact['prediction'][i]).abs()
        maximum=max(maximum,float(delta.max()));valid_maximum=max(valid_maximum,float(delta[data['valid'][i]].max()))
    passed=disagreements==0 and valid_maximum<2e-6 and abs(exact['bias']-meta['shadow_bias'])<1e-12
    result=dict(status='PASS_FULL_BVH_REAL_RECEIVER_SEMANTICS' if passed else 'FAILED',
        gt_read=False,updates=0,valid_view_facing_disagreements=disagreements,
        valid_max_abs_color_difference=valid_maximum,all_points_max_abs_difference=maximum,
        shadow_bias=meta['shadow_bias'],light_parameters_frozen=True,
        numerical_light_validation=report['status'],
        scope='仅比较已追踪全BVH在真实接收点的生产颜色方程/朝向/bias；不声称完整光栅图像或最终几何精度通过',
        input_sha256={str(path):sha(path) for path in [a.cache,a.fit/'summary.json',a.validation/'summary.json',exact_path,
            a.source/'transforms_train.json',ROOT/'scene/eldl_v3_photometric.py',Path(__file__)]})
    (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
