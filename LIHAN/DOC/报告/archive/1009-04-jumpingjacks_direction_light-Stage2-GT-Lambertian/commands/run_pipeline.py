import json,os,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parent.parent
CWD=ROOT.parent.parent
GROUP=sys.argv[1] if len(sys.argv)>1 else 'main'
STATE=ROOT/('gt_state.json' if GROUP=='gt' else 'state.json')
def state(status,step,**extra):
 d=dict(status=status,step=step,updated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),pid=os.getpid(),**extra)
 temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temp.replace(STATE);print(d,flush=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(CWD),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
try:
 for cmd in json.loads((ROOT/'planned_commands.json').read_text()):
  if cmd['group']!=GROUP:continue
  if cmd['name']=='normal_depth':
   while not (ROOT/'gt_depth/manifest.json').is_file():
    gs=json.loads((ROOT/'gt_state.json').read_text()) if (ROOT/'gt_state.json').is_file() else {}
    if gs.get('status')=='FAILED':raise RuntimeError('GT preparation failed')
    state('WAITING_GT','normal_depth');time.sleep(5)
  state('RUNNING',cmd['name'])
  with open(cmd['log'],'w') as log:
   child=subprocess.Popen(cmd['argv'],cwd=CWD,env=env,stdout=log,stderr=subprocess.STDOUT);state('RUNNING',cmd['name'],child_pid=child.pid);code=child.wait()
  if code:raise RuntimeError(f"{cmd['name']} exited {code}")
 if GROUP=='gt':state('COMPLETED','finished');(ROOT/'GT_COMPLETED').write_text('PASS\n')
 else:state('PIPELINE_COMPLETED_REVIEW_PENDING','finished',iteration=55000,frames=15);(ROOT/'PIPELINE_COMPLETED').write_text('review pending\n')
except Exception:
 detail=traceback.format_exc();state('FAILED','failed',error=detail);(ROOT/('GT_FAILED' if GROUP=='gt' else 'FAILED')).write_text(detail);raise
