from pathlib import Path
import json,os,time,sys,subprocess,datetime
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
train=json.loads((ROOT/'process_handles.json').read_text())['train'];proc=Path('/proc')/str(train['pid'])
def live():
 try:return (proc/'stat').read_text().split()[21]==train['start_ticks'] and (proc/'cmdline').read_bytes().split(b'\x00')[:-1]==[x.encode() for x in train['argv']]
 except FileNotFoundError:return False
history=[]
for step in [40000,50000,55000]:
 marker=ROOT/f'model_mlp/pbr_directional/iteration_{step}/lighting.json'
 state=dict(status='WAITING_CHECKPOINT',iteration=step,train_pid=train['pid'],completed=history)
 (ROOT/'inspection/checkpoint_wait_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
 while not marker.exists():
  if not live():raise RuntimeError('原train在检查点保存前退出，禁止重启。')
  time.sleep(5)
 with (ROOT/f'logs/checkpoint_{step}_audit.log').open('w') as log:
  result=subprocess.run([sys.executable,'-u',str(ROOT/'commands/audit_checkpoint.py'),str(step)],cwd=CWD,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,step
 history.append(dict(iteration=step,exit_code=0,completed_at=datetime.datetime.now().astimezone().isoformat()))
 print(json.dumps(history[-1]),flush=True)
(ROOT/'inspection/checkpoint_wait_state.json').write_text(json.dumps(dict(status='COMPLETED_CHECKPOINT_AUDITS',completed=history),ensure_ascii=False,indent=2)+'\n')
