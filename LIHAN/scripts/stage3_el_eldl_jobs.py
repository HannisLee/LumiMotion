"""本轮四组统一进程记录、评价命令和退出码。"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from LIHAN.scripts.stage3_el_eldl_four import ROOT, CONTROL, ARMS, PYTHON, env, write

def evaluate(arm,smoke,split,label,samples=128,bias=.1,iteration=None,experiment=None,gpu_override=None,direct_multiplier=1.):
 name,parent,start,gpu=ARMS[arm]
 out=Path(experiment) if experiment else ROOT/'output'/('smoke_test' if smoke else '')/(name+('-冒烟' if smoke else ''))
 state=json.loads((out/'state.json').read_text());it=(iteration or state['iteration'])
 gpu=state['gpu'] if gpu_override is None else gpu_override
 # 两快照物理前向相同；训练差异只在初始化，统一评测来源避免JIT反复重编译。
 snapshot=CONTROL/'snapshot'
 command=[PYTHON,str(snapshot/'LIHAN/scripts/bootstrap_1011.py'),str(snapshot),'LIHAN.scripts.stage3_el_eldl_evaluate','--model',str(out/'model_mlp'),'--iteration',str(it),'--source',str(CONTROL/f'{arm}_{"train" if smoke else split}_source'),'--output',str(out/label),'--samples',str(samples),'--bias',str(bias),'--direct-multiplier',str(direct_multiplier)]
 if smoke:command+=['--smoke']
 elif split=='test':command+=['--audit']
 if arm.endswith('ELDL'):command+=['--directions',str(CONTROL/f'{arm}_{"train" if smoke else split}_directions.json')]
 record={'command':command,'started':time.time(),'status':'RUNNING','gpu':gpu}
 with (out/'README.md').open('a') as f:f.write('\n完整渲染命令：\n```bash\n'+shlex.join(command)+'\n```\n日志：'+label+'.log；渲染输出：'+label+'。\n')
 write(out/(label+'_state.json'),record)
 with (out/(label+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=snapshot,env=env(gpu),stdout=log,stderr=subprocess.STDOUT);record['pid']=child.pid;write(out/(label+'_state.json'),record);code=child.wait()
 record.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,finished=time.time());write(out/(label+'_state.json'),record)
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--label',required=True);p.add_argument('--samples',type=int,default=128);p.add_argument('--bias',type=float,default=.1);p.add_argument('--iteration',type=int);p.add_argument('--experiment');a=p.parse_args();raise SystemExit(evaluate(a.arm,a.smoke,a.split,a.label,a.samples,a.bias,a.iteration,a.experiment))
