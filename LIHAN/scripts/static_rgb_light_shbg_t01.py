"""T01 操作入口：开发中，禁止启动真实数据处理。"""
import argparse
import json
from pathlib import Path
import sys
import time
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest

OBS_HASH='eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47'
PLAN_HASH='a71d513f797465e7509a3b5144aef600adf3152e902986dd1b56cb52bd30d5c5'


def inventory(a):
    root=Path(__file__).resolve().parents[2]; baseline=a.baseline.resolve(); model=a.model.resolve(); source=a.source.resolve()
    files=[a.plan.resolve(),a.observations.resolve(),source/'transforms_all.json',model/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply',model/'deform'/f'iteration_{a.iteration}'/'deform.pth',baseline/'freeze_manifest.json',root/'utils'/'sh_utils.py']
    files += list((baseline/'fitting').rglob('*'))+list((baseline/'frozen_code').rglob('*'))
    for folder in ('images','rgba'):
        if (source/folder).exists(): files += [p for p in (source/folder).rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.exr')]
    files += list((root/'LIHAN'/'scripts').glob('static_rgb_light_shbg_t01*.py'))
    files += [root/'LIHAN'/'scripts'/'static_rgb_light_fit_r3.py',root/'LIHAN'/'scripts'/'static_rgb_light_fit_r5b.py']
    return sorted(set(p.resolve() for p in files if p.is_file()))


def lock(a):
    if digest(a.observations)!=OBS_HASH or digest(a.plan)!=PLAN_HASH: raise ValueError('观测或规划 SHA256 不匹配')
    baseline=a.baseline.resolve(); root=Path(__file__).resolve().parents[2]
    for script in ('static_rgb_light_fit_r3.py','static_rgb_light_fit_r5b.py'):
        if digest(root/'LIHAN'/'scripts'/script)!=digest(baseline/'frozen_code'/script): raise ValueError('导入的 E4 代码与冻结基线不一致')
    files=inventory(a)
    for required in (a.observations.resolve(),a.plan.resolve(),a.source.resolve()/'transforms_all.json',a.model.resolve()/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply',a.model.resolve()/'deform'/f'iteration_{a.iteration}'/'deform.pth',baseline/'freeze_manifest.json'):
        if required not in files: raise FileNotFoundError(required)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    rec={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'argv':sys.argv,'files':{str(p):{'sha256':digest(p),'size':p.stat().st_size} for p in files},'gt_read':False}
    (out/'input_lock.json').write_text(json.dumps(rec,indent=2)); return rec


def verify(a):
    rec=json.loads((a.output/'input_lock.json').read_text()); current=inventory(a)
    if set(map(str,current))!=set(rec['files']): raise ValueError('输入清单发生变化')
    for p in current:
        if digest(p)!=rec['files'][str(p)]['sha256'] or p.stat().st_size!=rec['files'][str(p)]['size']: raise ValueError(f'输入已变化: {p}')
    if digest(a.observations)!=OBS_HASH or digest(a.plan)!=PLAN_HASH: raise ValueError('观测或规划 SHA256 不匹配')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['lock','extract','fit','choose','figures','freeze']); p.add_argument('--output',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--observations',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--iteration',type=int,required=True); p.add_argument('--plan',type=Path,required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    # 完整合成测试、冻结及 CLI 联调尚未完成，不能绕过门槛执行真实流程。
    raise RuntimeError('T01 执行链尚未通过完整自检，禁止真实数据运行')

if __name__=='__main__': main()
