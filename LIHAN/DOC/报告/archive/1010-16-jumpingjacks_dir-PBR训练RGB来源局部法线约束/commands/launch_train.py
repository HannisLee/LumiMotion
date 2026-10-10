"""真实独立20k启动，记录准确PID/ticks并等待其实际退出；不自动重试。"""
from pathlib import Path
import json,hashlib,os,subprocess,datetime
root=Path(__file__).resolve().parent.parent;cwd=root.parent.parent
for rel,sha in json.loads((root/'code_manifest_train_launch.json').read_text()).items():
 assert hashlib.sha256((cwd/rel).read_bytes()).hexdigest()==sha
argv=json.loads((root/'train_argv.json').read_text())
assert not (root/'model_mlp').exists() and not (root/'process_handles.json').exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4')
started=datetime.datetime.now().astimezone().isoformat()
with (root/'logs/train.log').open('w') as log:
 child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
 proc=Path('/proc')/str(child.pid)
 ticks=(proc/'stat').read_text().split()[21]
 assert str(cwd)==os.readlink(proc/'cwd')
 assert (proc/'cmdline').read_bytes().split(b'\x00')[:-1]==[x.encode() for x in argv]
 handle=dict(pid=child.pid,start_ticks=ticks,argv=argv,cwd=str(cwd),started_at=started)
 (root/'process_handles.json').write_text(json.dumps(dict(train=handle,launcher_pid=os.getpid()),ensure_ascii=False,indent=2)+'\n')
 runtime=dict(status='TRAIN_RUNNING',started_at=started,training_pid=child.pid,process_start_ticks=ticks,
  source_iteration=35000,target_iteration=55000,steps_planned=20000,preset='pbr_gt_directional_source_rgb_normal',
  gt_training_inputs=['light_direction_only'],quality='PENDING',normal_goal='PENDING')
 (root/'training_runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(status='TRAIN_RUNNING',**handle),ensure_ascii=False),flush=True)
 code=child.wait()
runtime=json.loads((root/'training_runtime.json').read_text())
runtime.update(training_exit_code=code,verified_training_session_closed=False,child_finished_at=datetime.datetime.now().astimezone().isoformat())
if code==0:
 assert 'Training complete.' in (root/'logs/train.log').read_text()
 runtime.update(status='TRAIN_COMPLETED',iteration=55000,steps_completed=20000)
else:runtime['status']='TRAIN_FAILED'
(root/'training_runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(training_exit_code=code,training_pid=child.pid),ensure_ascii=False),flush=True)
raise SystemExit(code)
