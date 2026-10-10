"""绑定同一live训练，等50k实际保存后核验，不重启训练。"""
from pathlib import Path
import os,sys,json,time,subprocess,datetime
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
h=json.loads((ROOT/'process_handles.json').read_text())['train'];p=Path('/proc')/str(h['pid'])
assert p.exists() and (p/'stat').read_text().split()[21]==h['start_ticks']
cmdline=(p/'cmdline').read_bytes();assert b'scripts.train_stage2' in cmdline and str(ROOT).encode() in cmdline
statepath=ROOT/'inspection/audit_50000_wait_state.json'
def state(status,**extra):
 d=dict(status=status,pid=os.getpid(),start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().split()[21],training_pid=h['pid'],training_start_ticks=h['start_ticks'],updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),final_mae=False,**extra)
 statepath.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d,ensure_ascii=False),flush=True)
state('WAITING_LIVE_TRAIN_50000')
marker=ROOT/'model_mlp/pbr_directional/iteration_50000/lighting.json'
while not marker.exists():
 assert p.exists() and (p/'stat').read_text().split()[21]==h['start_ticks'] and (p/'cmdline').read_bytes()==cmdline,'50k前真实训练已退出，拒绝自动重启'
 time.sleep(5)
state('AUDITING_50000')
argv=[sys.executable,'-u',str(ROOT/'commands/audit_checkpoint.py'),'50000']
with (ROOT/'logs/checkpoint_50000_audit.log').open('w') as log:subprocess.run(argv,cwd=CWD,stdout=log,stderr=subprocess.STDOUT,check=True)
record=json.loads((ROOT/'inspection/checkpoint_50000_audit.json').read_text());assert record['status']=='PASS_CHECKPOINT_ONLY'
state('PASS_50000_CHECKPOINT_ONLY',audit_subprocess_exit_code=0,iteration=50000)
