"""真实20步Trainer子进程，记录PID/argv/退出；不自动重试。"""
from pathlib import Path
import subprocess,os,json,datetime,time
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parents[2]
argv=json.loads((ROOT/'train_argv.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4')
started=datetime.datetime.now().astimezone().isoformat()
with (ROOT/'logs/train.log').open('w') as log:
    child=subprocess.Popen(argv,cwd=CWD,env=env,stdout=log,stderr=subprocess.STDOUT)
    proc=Path('/proc')/str(child.pid)
    record=dict(status='RUNNING',pid=child.pid,start_ticks=(proc/'stat').read_text().split()[21],
        argv=argv,cwd=str(CWD),source_iteration=35000,target_iteration=35020,steps_planned=20,
        started_at=started,preset='pbr_gt_directional_plane_native_direct_probability_gate')
    path=ROOT/'training_runtime.json';path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    code=child.wait()
record.update(status='TRAIN_COMPLETED' if code==0 else 'FAILED',exit_code=code,
    finished_at=datetime.datetime.now().astimezone().isoformat(),verified_session_closed=False)
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False),flush=True)
raise SystemExit(code)
