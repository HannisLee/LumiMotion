"""使用原几何评测器只读渲染中间检查点，全15测试相机，不训练。"""
from pathlib import Path
import hashlib,json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
PYTHON='/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
source=next((ROOT/'output').glob('1011-01*/eval_view'))
gt_depth=ROOT/'output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/gt_depth'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
plans=[]
for prefix,tag in [('1011-27','T13'),('1011-28','T14')]:
    model=next((ROOT/'output').glob(prefix+'*/model_mlp'))
    for step in (999,5000,8000,9000):
        destination=OUT/'checkpoint_renders'/f'{tag}_{step}'
        command=[PYTHON,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),'scripts.eval_geometry_gt',
                 '--model_path',str(model),'--source_path',str(source),'--load_iter',str(step),
                 '--gt_root','/mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2',
                 '--gt_depth_dir',str(gt_depth),'--split','test','--resolution','2',
                 '--alpha_threshold','.5','--interior_radius','2','--output_dir',str(destination),'--save_arrays']
        plan=dict(tag=tag,iteration=step,command=command,output=str(destination),log=str(OUT/'logs'/f'{tag}_{step}.log'))
        plans.append(plan)
(OUT/'commands/checkpoint_render_plan.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
executed=[]
for plan in plans:
    if Path(plan['output']).exists():raise RuntimeError('拒绝覆盖已有渲染：'+plan['output'])
    start=time.time()
    with open(plan['log'],'w') as log:
        code=subprocess.call(plan['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    executed.append(dict(**plan,exit_code=code,elapsed_seconds=time.time()-start))
    (OUT/'commands/checkpoint_render_executed.json').write_text(json.dumps(executed,ensure_ascii=False,indent=2)+'\n')
    print(plan['tag'],plan['iteration'],'exit',code,flush=True)
    if code:raise RuntimeError('渲染失败，现场保留：'+plan['log'])
