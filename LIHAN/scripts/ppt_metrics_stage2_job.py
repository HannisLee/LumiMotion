"""研究展示的两组Stage2补训：隔离输出，保留命令、来源和退出码。"""
import argparse,json,os,sys,subprocess,time,hashlib,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
CONTROL=ROOT/'output/1024-02-静态ELDL-展示数值补全总控'
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['EL','DL'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--retry',default='');a=p.parse_args()
 assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path(PYTHON).resolve()
 kind=a.kind;start=35000 if kind=='EL' else 16500;end=start+(24 if a.smoke else 20000)
 name=f'1024-{1 if kind=="EL" else 2:02d}-static{kind}-展示Stage2冒烟' if a.smoke else f'1024-{3 if kind=="EL" else 4:02d}-static{kind}-展示Stage2补训'
 name+=a.retry
 out=ROOT/'output'/('smoke_test' if a.smoke else '')/name;out.mkdir(parents=True,exist_ok=False)
 parent=ROOT/('output/1011-02-LH-static-jumpingjacks_EL_V2-原生LumiMotion两阶段/model_mlp' if kind=='EL' else 'output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp')
 source=(ROOT/'output/1011-02-LH-static-jumpingjacks_EL_V2-原生LumiMotion两阶段/data_view' if kind=='EL' else CONTROL/'source_DL')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_EXTENSIONS_DIR=str(CONTROL/'native_cache'),CUDA_HOME=str(Path(PYTHON).parents[1]),TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='4',PYTHONPATH=str(ROOT/'output/archive/1011-07-六组训练总控/tools/nvdiffrast-sm86/extension'))
 boot=[PYTHON,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT)]
 cmd=boot+['scripts.train_stage2','--source_path',str(source),'--model_path',str(out/'model_mlp'),'--stage1_model_path',str(parent),'--load_iter',str(start),'--iterations',str(end),'--test_iterations',str(end),'--save_iterations',str(end),'--is_blender','--eval','--load2gpu_on_the_fly','--resolution','2','--train_light_folder','images','--test_light_folder','images','--gt_alpha_mask_as_scene_mask','--diffuse_sample_num','512','--depth_ratio','0','--pbr_rasterizer_backend','lumimotion']
 if kind=='DL':cmd+=['--pbr_light_mode','gt_directional','--pbr_gt_lights_path',str(CONTROL/'known_directions_fitted_intensity.json'),'--pbr_learn_light_intensity']
 state=dict(status='RUNNING',kind=kind,smoke=a.smoke,parent=str(parent),source=str(source),source_iteration=start,iteration=end,loop_steps=end-start,expected_optimizer_updates=sum(i%4==0 for i in range(start+1,end)),gpu=a.gpu,started=time.time(),command=cmd,environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','TORCH_EXTENSIONS_DIR','PYTHONPATH','CUDA_HOME']},parent_sha256={str(parent/r):hashlib.sha256((parent/r).read_bytes()).hexdigest() for r in [f'point_cloud/iteration_{start}/point_cloud.ply',f'deform/iteration_{start}/deform.pth']})
 write(out/'state.json',state)
 import shlex
 (out/'README.md').write_text('# 展示用Stage2补训\n\n'+f'来源模型：{parent}，iteration={start}；source={source}。输出模型model_mlp，终点{end}。\n\n'+f'新增循环{end-start}次，原生每4步更新且终点保存早于更新，预计真实optimizer更新{state["expected_optimizer_updates"]}次。\n\n完整命令：\n```bash\n'+shlex.join(cmd)+'\n```\n\n环境：'+json.dumps(state['environment'],ensure_ascii=False)+'\n\n日志train.log；渲染、指标、四类目检待完成。最终状态PENDING。\n')
 with (out/'train.log').open('x') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);state['pid']=child.pid;write(out/'state.json',state);code=child.wait()
 state.update(status='TRAIN_COMPLETED' if code==0 else 'FAILED',exit_code=code,finished=time.time());write(out/'state.json',state)
 with (out/'README.md').open('a') as f:f.write(f'\n训练退出码{code}；状态{state["status"]}。原有模型与所有失败现场保留。\n')
 if code==0 and not a.smoke:
  evaluation=boot+['LIHAN.scripts.ppt_metrics_stage2_eval','--model',str(out/'model_mlp'),'--iteration',str(end),'--output',str(out/'evaluation/rgb')]
  albedo=boot+['scripts.eval_albedo_gt','--model_path',str(out/'model_mlp'),'--iterations',str(end),'--gt_albedo_dir',str(ROOT/f'data/LH-data/static/LH-static-jumpingjacks_{kind}_V2/albedo'),'--output_path',str(out/'evaluation/albedo'),'--source_path',str(source),'--split','test','--save_float']
  for name,command in [('rgb',evaluation),('albedo',albedo)]:
   (out/'evaluation').mkdir(exist_ok=True);state['status']='EVALUATING_'+name;state[name+'_command']=command;write(out/'state.json',state)
   with (out/(name+'.log')).open('x') as log:code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   state[name+'_exit_code']=code;write(out/'state.json',state)
   with (out/'README.md').open('a') as f:f.write('\n完整评测命令：\n```bash\n'+shlex.join(command)+'\n```\n退出码：'+str(code)+'\n')
   if code:break
  state['status']='READY_FOR_REVIEW' if code==0 else 'EVALUATION_FAILED';write(out/'state.json',state)
 print(json.dumps(state,ensure_ascii=False));sys.exit(code)
if __name__=='__main__':main()
