from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,os,subprocess,time,traceback
root=Path(__file__).resolve().parent.parent;cwd=root.parent.parent
state=root/'evaluation_state.json';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(cwd))
def record(status,step,**kwargs):
 d=dict(status=status,step=step,pid=os.getpid(),updated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),**kwargs);state.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d,ensure_ascii=False),flush=True)
try:
 for c in json.loads((root/'planned_commands.json').read_text()):
  started=time.time()
  with Path(c['log']).open('w') as log:
   child=subprocess.Popen(c['argv'],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
   record('RUNNING',c['name'],child_pid=child.pid,argv=c['argv'])
   code=child.wait()
  with (root/'runtime_history.jsonl').open('a') as f:f.write(json.dumps(dict(step=c['name'],exit_code=code,elapsed_seconds=time.time()-started))+'\n')
  if code:raise RuntimeError('{} exited {}'.format(c['name'],code))
 record('COMPLETED_REVIEW_PENDING','finished',frames=15,iteration=55000)
except Exception:
 record('FAILED','failed',error=traceback.format_exc());raise
