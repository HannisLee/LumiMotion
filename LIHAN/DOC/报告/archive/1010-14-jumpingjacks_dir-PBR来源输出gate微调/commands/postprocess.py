"""等待已确认的训练PID，完整离线渲染/评测；不重启或改动训练。"""
import json,os,sys,time,subprocess,traceback,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parent.parent
CWD=ROOT.parent.parent
PID=int(sys.argv[1])
PROC=Path('/proc')/str(PID)
START=(PROC/'stat').read_text().split()[21]
EXPECTED=(PROC/'cmdline').read_bytes()
assert b'scripts.train_stage2' in EXPECTED and str(ROOT).encode() in EXPECTED
STATE=ROOT/'postprocess_state.json'
JST=timezone(timedelta(hours=9))
def state(status,step,**extra):
 d=dict(status=status,step=step,pid=os.getpid(),training_pid=PID,updated_at=datetime.now(JST).isoformat(),**extra)
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(STATE)
 print(json.dumps(d,ensure_ascii=False),flush=True)
def live():
 try:return (PROC/'stat').read_text().split()[21]==START and (PROC/'cmdline').read_bytes()==EXPECTED
 except FileNotFoundError:return False
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(CWD))
try:
 state('WAITING_LIVE_TRAIN','train',process_start=START)
 while live(): time.sleep(5)
 assert 'Training complete.' in (ROOT/'logs/train.log').read_text(errors='replace'), '训练没有正常完成'
 model=ROOT/'model_mlp'
 assert (model/'point_cloud/iteration_55000/point_cloud.ply').is_file()
 assert (model/'pbr_directional/iteration_55000/lighting.json').is_file()
 (model/'TRAIN_COMPLETED').write_text('55000\n')
 runtime_path=ROOT/'training_runtime.json'
 runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
 runtime.update(status='TRAIN_COMPLETED',finished_at=datetime.now(JST).isoformat(),iteration=55000,training_pid=PID,steps_completed=20000)
 runtime_path.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
 for cmd in json.loads((ROOT/'planned_commands.json').read_text()):
  if cmd['name']=='roughness':
   gt=Path(cmd['argv'][cmd['argv'].index('--gt_roughness_dir')+1])/'manifest.json'
   if not gt.is_file(): raise RuntimeError('Roughness GT导出尚未完成，需要人工核对，不静默跳过。')
  state('RUNNING',cmd['name'],argv=cmd['argv'])
  started=time.time()
  with open(cmd['log'],'w') as log:
   child=subprocess.Popen(cmd['argv'],cwd=CWD,env=env,stdout=log,stderr=subprocess.STDOUT)
   state('RUNNING',cmd['name'],child_pid=child.pid,started_at=datetime.now(JST).isoformat())
   code=child.wait()
  if code:raise RuntimeError('{} exited {}'.format(cmd['name'],code))
  with (ROOT/'runtime_history.jsonl').open('a') as f:
   f.write(json.dumps(dict(step=cmd['name'],exit_code=code,elapsed_seconds=time.time()-started,finished_at=datetime.now(JST).isoformat()))+'\n')
 state('PIPELINE_COMPLETED_REVIEW_PENDING','finished',iteration=55000,frames=15)
except Exception:
 detail=traceback.format_exc();state('FAILED','failed',error=detail);(ROOT/'POSTPROCESS_FAILED').write_text(detail);raise
