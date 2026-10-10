"""同一训练40k checkpoint保存完毕后只读预览，不修改或重启训练。"""
from pathlib import Path
import os,sys,time,json,subprocess
root=Path(__file__).resolve().parent.parent
cwd=root.parent.parent
pid=int(sys.argv[1]);proc=Path('/proc')/str(pid)
start=(proc/'stat').read_text().split()[21]
cmdline=(proc/'cmdline').read_bytes()
assert b'scripts.train_stage2' in cmdline and str(root).encode() in cmdline
marker=root/'model_mlp/pbr_directional/iteration_40000/lighting.json'
while not marker.exists():
 try:assert (proc/'stat').read_text().split()[21]==start and (proc/'cmdline').read_bytes()==cmdline
 except FileNotFoundError:raise RuntimeError('40k保存前训练退出，禁止自动重启')
 time.sleep(5)
argv=[sys.executable,'-u','-m','scripts.render_stage2','--model_path',str(root/'model_mlp'),'--load_iter','40000','--task','eval','--output_path',str(root/'preview_40000')]
(root/'commands/preview_40000_argv.json').write_text(json.dumps(argv,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'RENDERING_40000','argv':argv},ensure_ascii=False),flush=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(cwd))
with (root/'logs/preview_render_40000.log').open('w') as log:
 subprocess.run(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(json.dumps({'status':'PREVIEW_COMPLETED','path':str(root/'preview_40000'),'not_final_evaluation':True},ensure_ascii=False),flush=True)
