"""绑定原训练与40k已通过检查记录，等待世界gate响应只读诊断，不重启。"""
from pathlib import Path
import json,sys,subprocess,os,time,datetime
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
train=json.loads((ROOT/'process_handles.json').read_text())['train']
proc=Path('/proc')/str(train['pid'])
def live():
    try:return (proc/'stat').read_text().split()[21]==train['start_ticks'] and (proc/'cmdline').read_bytes().split(b'\x00')[:-1]==[x.encode() for x in train['argv']]
    except FileNotFoundError:return False
assert live()
state=ROOT/'inspection/gate_effect_wait_state.json'
state.write_text(json.dumps(dict(status='WAITING_VERIFIED_CHECKPOINT_40000',pid=os.getpid(),train_pid=train['pid'],started_at=datetime.datetime.now().astimezone().isoformat()),ensure_ascii=False,indent=2)+'\n')
checkpoint=ROOT/'inspection/checkpoint_40000_audit.json'
while not checkpoint.exists():
    if not live():raise RuntimeError('原训练在40k审计完成前退出；不自动重启。')
    time.sleep(5)
assert json.loads(checkpoint.read_text())['status']=='PASS_CHECKPOINT_ONLY'
with (ROOT/'logs/gate_world_effect_40000.log').open('w') as log:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'commands/audit_gate_world_effect_40000.py')],cwd=CWD,stdout=log,stderr=subprocess.STDOUT)
    state.write_text(json.dumps(dict(status='RUNNING_READ_ONLY_GATE_EFFECT',pid=os.getpid(),child_pid=child.pid,train_pid=train['pid']),ensure_ascii=False,indent=2)+'\n')
    code=child.wait()
assert code==0
state.write_text(json.dumps(dict(status='COMPLETED_READ_ONLY_GATE_EFFECT',exit_code=code,pid=os.getpid(),finished_at=datetime.datetime.now().astimezone().isoformat()),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(exit_code=code,diagnostic='READ_ONLY_GATE_WORLD_RESPONSE',optimizer_steps=0),ensure_ascii=False),flush=True)
