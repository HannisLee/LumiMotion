"""中期DEV不足后的单因素小参数对照；原始四组训练保持不变。"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from LIHAN.scripts.stage3_el_eldl_four import ROOT, CONTROL, ARMS, PYTHON, env, write
TUNES={'dynamic_ELDL':('1026-01-LH-dynamic-ELDL-GT方向光-Stage3材质截断修正',0),
       'static_ELDL':('1026-02-LH-static-ELDL-GT方向光-Stage3环境初值0.3',2)}

def prepare():
 dest=CONTROL/'snapshot_tune';shutil.copytree(CONTROL/'snapshot',dest,symlinks=True)
 p=dest/'LIHAN/scripts/stage3_el_eldl_entry.py';s=p.read_text()
 s=s.replace("runpy.run_module('scripts.train_stage2', run_name='__main__')", """# 原实现的区间投影与照明类型无关；为本轮显式混合光放开已有门禁。
import copy
import scene.pbr_material as material
validate = material.validate_albedo_initialization
def validate_mixed(args):
    if getattr(args, 'pbr_mixed_lights_path', ''):
        checked = copy.copy(args)
        checked.pbr_light_mode = 'gt_directional'
        return validate(checked)
    return validate(args)
material.validate_albedo_initialization = validate_mixed
runpy.run_module('scripts.train_stage2', run_name='__main__')""")
 p.write_text(s)
 write(CONTROL/'tune_manifest.json',{'snapshot':str(dest),'code_sha256':{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*.py')},'differences':'动态：颜色初值投影到clamp内部并每次更新后保持；静态：环境初值1.5改0.3；初始预算8000，不读TEST'})

def run(arm,smoke):
 name,gpu=TUNES[arm];original,_,start,_=ARMS[arm]
 base=ROOT/'output'/original
 state=json.loads((base/'state.json').read_text());cmd=list(state['command'])
 snapshot=CONTROL/'snapshot_tune'
 cmd[1]=str(snapshot/'LIHAN/scripts/bootstrap_1011.py');cmd[2]=str(snapshot)
 output=ROOT/'output'/('smoke_test' if smoke else '')/(name+('-冒烟' if smoke else ''));output.mkdir(parents=True,exist_ok=False)
 end=start+(24 if smoke else 8000)
 cmd[cmd.index('--model_path')+1]=str(output/'model_mlp');cmd[cmd.index('--iterations')+1]=str(end)
 first=cmd.index('--save_iterations')+1;last=cmd.index('--is_blender');cmd[first:last]=[str(end)]
 if arm=='dynamic_ELDL':cmd+=['--pbr_albedo_interior_init','--pbr_albedo_project_after_step']
 else:cmd+=['--envmap_init_value','0.3']
 record={'arm':arm,'status':'RUNNING','command':cmd,'iteration':end,'parent_iteration':start,'gpu':gpu,'started':time.time(),'steps':end-start,'reference':str(base),'single_factor':'材质clamp可导区间修正' if arm=='dynamic_ELDL' else '环境初始幅度1.5→0.3','smoke':smoke}
 write(output/'state.json',record)
 (output/'README.md').write_text('# Stage3小参数对照\n\n'+record['single_factor']+'；中期DEV未胜出后启动。完整训练命令：\n```bash\n'+shlex.join(cmd)+'\n```\n\n日志train.log；source/model/iteration见命令和state.json；RGB与四类内部目检待完成；当前结论PENDING。\n')
 with (output/'train.log').open('x') as f:
  child=subprocess.Popen(cmd,cwd=snapshot,env=env(gpu),stdout=f,stderr=subprocess.STDOUT);record['pid']=child.pid;write(output/'state.json',record);code=child.wait()
 record.update(status='TRAIN_COMPLETED' if code==0 else 'FAILED',exit_code=code,finished=time.time());write(output/'state.json',record)
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=TUNES);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 else:raise SystemExit(run(a.arm,a.smoke))
