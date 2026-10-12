#!/usr/bin/env python3
"""已有T04中间点全原协议回溯；复用原六命令，只新增严格中间点准入。"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal3_checkpoint_audit import audit,destination
from goal_eval_protocol import ROOT,read,write,digest,verify,CONTROL
import goal3_densify_run

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fresh',type=Path,required=True);p.add_argument('--gpu',type=int);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if subprocess.check_output(['hostname'],text=True).strip()!='ushas':raise AssertionError('必须ushas/lumimotion-ushas。')
 output=destination(a.output);fresh_path=destination(a.fresh)/'parity.json';before=audit(a.model,a.iteration)
 if output.exists():raise FileExistsError('中间点回溯禁止覆盖。')
 if a.execute:
  if a.gpu is None:raise AssertionError('GPU必须已协调显式指定。')
  proof=read(fresh_path)
  if proof['status']!='PASS' or not proof.get('checkpoint_retrospective') or proof['actual_training_terminal_iteration']!=18000 or proof['iteration']!=a.iteration or Path(proof['model']).resolve()!=a.model.resolve() or proof['source']!=before['source'] or proof['checkpoint_four_asset_sha256']!=before['checkpoint_four_asset_sha256']:raise AssertionError('须对应中间点真实fresh，不可复用终点结果。')
  if any(digest(p)!=s for p,s in proof['sources'].items()):raise AssertionError('中间点fresh后来源已变。')
 # 原run仅生成计划（不会执行18000门禁，也未删除/修改它）；六命令仍来自原冻结入口。
 old_argv=sys.argv
 sys.argv=[old_argv[0],'--model',str(a.model.resolve()),'--iteration',str(a.iteration),'--output',str(output)]+(['--gpu',str(a.gpu)] if a.gpu is not None else [])
 try:goal3_densify_run.main()
 finally:sys.argv=old_argv
 write(output/'checkpoint_cpu_evidence.json',before)
 plan=read(output/'plan.json');plan['checkpoint_retrospective']=True;plan['actual_training_terminal_iteration']=18000;plan['fresh']=str(fresh_path);plan['checkpoint_four_asset_sha256']=before['checkpoint_four_asset_sha256'];plan['new_entry_sha256']={str(Path(__file__).resolve()):digest(Path(__file__)),str(ROOT/'LIHAN/scripts/goal3_checkpoint_audit.py'):digest(ROOT/'LIHAN/scripts/goal3_checkpoint_audit.py')}
 write(output/'retrospective_plan.json',plan)
 with (output/'README.md').open('a') as f:f.write(f'\n这是原T04@{a.iteration}保存点的独立回溯；原训练仍真实结束于18000/2000。新包装先验原归档/四资产/Adam/RNG/LR和对应fresh，旧六命令与指标数学不变。当前'+('已准入实际执行，结果按真实退出记录。' if a.execute else '只有CPU计划，GPU/指标/人工验收NOT_RUN。')+'\n')
 if not a.execute:return
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',TORCH_CUDA_ARCH_LIST='8.6');lib=ROOT/'output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu';env['LD_LIBRARY_PATH']=str(lib)+':'+env.get('LD_LIBRARY_PATH','')
 outcomes=[]
 for item in plan['steps']:
  verify(read(CONTROL/'protocol.json'))
  for group in ('helper_sha256','new_entry_sha256'):
   if any(digest(p)!=s for p,s in plan[group].items()):raise AssertionError('活跃评测helper变化。')
  for path,sha in before['checkpoint_four_asset_sha256'].items():
   if digest(path)!=sha:raise AssertionError('活跃回溯中间资产变化。')
  name,argv=item['name'],item['command'];log=output/'logs'/(name+'.log')
  with log.open('x') as f:
   f.write(item['shell']+'\n');f.flush();process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   write(output/('running_'+name+'.json'),dict(pid=process.pid,argv=argv,started_unix=time.time(),log=str(log),gpu=a.gpu))
   code=process.wait()
  outcomes.append(dict(step=name,pid=process.pid,exit_code=code,log=str(log),actual_helper_sha256={p:digest(p) for p in plan['helper_sha256']}));write(output/'executed.json',outcomes);print(name,'exit=',code,flush=True)
  if code:
   write(output/'FAILED.json',dict(step=name,exit_code=code,retained=True));raise SystemExit(code)
 after=audit(a.model,a.iteration)
 if before['sources']!=after['sources']:raise AssertionError('完整回溯前后来源/资产发生变化。')
 write(output/'checkpoint_execution_after.json',dict(status='PASS',unchanged=True,files=after['checkpoint_four_asset_sha256']))
 target=read(output/'goal3_numeric/goal3_acceptance.json');write(output/'PIPELINE_COMPLETED.json',dict(pipeline='PASS',original_measurement_complete=True,checkpoint_iteration=a.iteration,actual_training_terminal_iteration=18000,goal3_numeric=target['numeric_status'],geometry_review='PENDING',visual_review='PENDING'))
if __name__=='__main__':main()
