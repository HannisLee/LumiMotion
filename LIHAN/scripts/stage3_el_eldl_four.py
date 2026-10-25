"""静动态EL和GT方向光ELDL四组Stage3：准备、冒烟、训练、隔离评价。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
CONTROL = ROOT/'output/1025-33-静动态EL与GT方向光ELDL-Stage3总控'
ARMS = {
 'static_EL': ('1025-34-LH-static-EL-Stage3外观', '1011-02-LH-static-jumpingjacks_EL_V2-原生LumiMotion两阶段', 35000, 0),
 'static_ELDL': ('1025-35-LH-static-ELDL-GT方向光-Stage3外观', '1025-16-LH-static-jumpingjacks-ELDL-V3-G9-full-部分覆盖photo排除-xyz与scaling-LR3倍-rotation1倍', 8100, 1),
 'dynamic_EL': ('1025-36-LH-dynamic-EL-Stage3外观', '1011-04-LH-dynamic-jumpingjacks_EL_V2-原生LumiMotion两阶段', 35000, 2),
 'dynamic_ELDL': ('1025-37-LH-dynamic-ELDL-GT方向光-Stage3外观', '1024-23-LH-dynamic-jumpingjacks-ELDL-GT方向十倍强轴先验', 1500, 3),
}
def write(p,d):
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 CONTROL.mkdir(exist_ok=False)
 base_static=ROOT/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/eval_only/eval_source_v1'
 dynamic=ROOT/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控'
 fit=[i for i in range(1,151) if i%10 and not 61<=i<=79]
 for arm,(name,parent,start,gpu) in ARMS.items():
  parent=ROOT/'output'/parent/'model_mlp'
  if arm.endswith('_EL'):
   n=ast.parse((parent/'cfg_args').read_text(),mode='eval').body
   cfg={x.arg:ast.literal_eval(x.value) for x in n.keywords}
   source=Path(cfg['source_path'])
   train=json.loads((source/'transforms_train.json').read_text())
   test=json.loads((source/'transforms_test.json').read_text())
   held={**train,'frames':[f for f in train['frames'] if 61<=int(f['source_frame'])<=79]}
   # 原生父见过135 TRAIN；本轮Stage3统一FIT117。历史已见DEV的限制保留。
   train={**train,'frames':[f for f in train['frames'] if int(f['source_frame']) in fit]}
  elif arm.startswith('static'):
   source=base_static
   train=json.loads((source/'transforms_fit.json').read_text())
   test=json.loads((source/'transforms_test.json').read_text())
   held=json.loads((source/'transforms_held.json').read_text())
  else:
   source=dynamic/'data_train_v1'
   train=json.loads((source/'transforms_train.json').read_text())
   held=json.loads((dynamic/'data_dev_v1/transforms_dev.json').read_text())
   test={**train,'frames':[]}
   metadata=json.loads((ROOT/'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3/camera.json').read_text())
   # 相机表格式独立适配，所有测试输入仅用于最后冻结评价。
   for i in range(10,151,10):
    rec=metadata['frames'][f'{i:04d}']['extrinsics']
    template=dict(train['frames'][0]);template.update(source_frame=i,file_path=f'image_{i:04d}.png',time=(i-1)/149)
    template['transform_matrix']=rec.get('camera_to_world',rec.get('transform_matrix',rec.get('matrix_world')))
    if template['transform_matrix'] is None:
     template['transform_matrix']=rec['c2w']
    test['frames'].append(template)
  # 各split沿用自身RGB和相机/time；不同照明的GT分别保留，禁止把EL图作ELDL目标。
  for mode,targets in [('train',[]),('dev',held['frames']),('test',test['frames'])]:
   dest=CONTROL/f'{arm}_{mode}_source';dest.mkdir()
   (dest/'images').mkdir()
   for f in train['frames']+targets:
    fn=f['file_path'];target=dest/'images'/fn
    if target.exists():continue
    image=source/'images'/fn
    if not image.exists() and arm=='dynamic_ELDL':
     dev_image=dynamic/'data_dev_v1/images'/fn
     if dev_image.exists():
      image=dev_image
     else:
      import cv2
      raw_root=ROOT/'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3'
      image=CONTROL/'dynamic_heldout_images'/fn
      if not image.exists():
       image.parent.mkdir(exist_ok=True)
       rgba=cv2.imread(str(raw_root/'image'/fn),-1)
       alpha=cv2.imread(str(raw_root/'albedo'/fn.replace('image_', 'albedo_')),-1)[...,3]
       rgba[...,3]=alpha;rgba[alpha==0,:3]=0
       assert cv2.imwrite(str(image),rgba)
    target.symlink_to(image.resolve())
   points=source/'points3d.ply'
   if points.exists():(dest/'points3d.ply').symlink_to(points.resolve())
   write(dest/'transforms_train.json',train)
   write(dest/'transforms_test.json',{**test,'frames':targets})
  if arm.endswith('ELDL'):
   raw=ROOT/('data/LH-data/static/LH-static-jumpingjacks-ELDL-V3' if arm.startswith('static') else 'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3')
   lights=json.loads((raw/'lights.json').read_text())
   for mode,ids in [('train',fit),('dev',fit+[i for i in range(61,80) if i%10]),('test',fit+list(range(10,151,10)))]:
    frames={}
    for i in ids:
     r=lights[f'{i:04d}'];d=r['light_dir_world']
     import math
     length=math.sqrt(sum(x*x for x in d))
     frames[f'{i:04d}']={'light_dir_world':[x/length for x in d],'intensity':1.,'type':'directional','source_type':'SUN'}
    write(CONTROL/f'{arm}_{mode}_directions.json',{'coordinate_space':'world','frames':frames,'gt_direction_only':True,'gt_intensity_used':False,'intensity_source':'单位初值，真实幅度只由TRAIN RGB学习'})
 snapshot=CONTROL/'snapshot';snapshot.mkdir()
 for name in ['arguments','scene','scripts','utils','gaussian_renderer','lpipsPyTorch','LIHAN/scripts']:
  shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
 for name in ['submodules','assets','example_envmaps']:(snapshot/name).symlink_to((ROOT/name).resolve(),target_is_directory=True)
 path=snapshot/'scripts/train_stage2.py';s=path.read_text()
 needle='        self.iters_only_diffuse = 2000'
 assert s.count(needle)==1
 s=s.replace(needle,"        from LIHAN.scripts.stage3_el_eldl_mixed import attach\n        attach(self)\n\n"+needle)
 needle="        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'],"
 assert s.count(needle)==1
 s=s.replace(needle,"        if getattr(self.args, 'pbr_static_parent', False):\n            d_values = dict(d_xyz=0., d_rotation=0., d_scaling=0., d_opacity=None, d_color=None)\n"+needle)
 # 内置training_report会自动将终点评价加入TRAIN/TEST；此处只允许独立评价进程。
 s=s.replace('self.testing_iterations = testing_iterations','self.testing_iterations = []')
 path.write_text(s)
 path=snapshot/'scripts/stage2_runtime.py';s=path.read_text();needle='    def deformation(self,camera,time_override=None):'
 assert s.count(needle)==1
 s=s.replace(needle,needle+"\n        if getattr(self.args, 'pbr_static_parent', False):\n            return dict(d_xyz=0., d_rotation=0., d_scaling=0., d_opacity=None, d_color=None)")
 path.write_text(s)
 write(CONTROL/'manifest.json',{'server':socket.gethostname(),'environment':'lumimotion-ushas','arms':ARMS,'fit':fit,'snapshot':{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')}})

def env(gpu):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_EXTENSIONS_DIR=str(ROOT/'output/1025-02-Stage3-PPT配对总控/native_cache'),CUDA_HOME=str(Path(PYTHON).parents[1]),TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='4',PYTHONPATH=str(ROOT/'output/archive/1011-07-六组训练总控/tools/nvdiffrast-sm86/extension'))
def run(a):
 name,parent,start,gpu=ARMS[a.arm]
 output=ROOT/'output'/('smoke_test' if a.smoke else '')/(name+('-冒烟' if a.smoke else '')+a.suffix)
 output.mkdir(parents=True,exist_ok=False)
 snapshot=CONTROL/'snapshot';boot=[PYTHON,str(snapshot/'LIHAN/scripts/bootstrap_1011.py'),str(snapshot)]
 end=start+(24 if a.smoke else a.steps)
 source=CONTROL/f'{a.arm}_train_source'
 cmd=boot+['LIHAN.scripts.stage3_el_eldl_entry','--source_path',str(source),'--model_path',str(output/'model_mlp'),'--stage1_model_path',str(ROOT/'output'/parent/'model_mlp'),'--load_iter',str(start),'--iterations',str(end),'--save_iterations',*[str(v) for v in ([end] if a.smoke else sorted(set([start+min(4000,a.steps),start+min(12000,a.steps),end])))],'--is_blender','--eval','--load2gpu_on_the_fly','--resolution','2','--train_light_folder','images','--test_light_folder','images','--gt_alpha_mask_as_scene_mask','--diffuse_sample_num','128','--trace_num_rays','65536','--depth_ratio','1','--pbr_rasterizer_backend','lumimotion']
 if a.arm.endswith('ELDL'):cmd+=['--pbr_mixed_lights_path',str(CONTROL/f'{a.arm}_train_directions.json')]
 if a.arm=='static_ELDL':cmd+=['--pbr_static_parent']
 state={'arm':a.arm,'status':'RUNNING','command':cmd,'iteration':end,'parent_iteration':start,'steps':end-start,'gpu':gpu,'started':time.time()}
 write(output/'state.json',state)
 readme=output/'README.md';readme.write_text('# Stage3四组外观实验\n\n完整训练命令：\n```bash\n'+shlex.join(cmd)+'\n```\n\n日志train.log；输出model_mlp；初始状态PENDING。GT方向仅单位向量；不使用GT光强和环境。\n')
 with (output/'train.log').open('x') as log:
  child=subprocess.Popen(cmd,cwd=snapshot,env=env(gpu),stdout=log,stderr=subprocess.STDOUT);state['pid']=child.pid;write(output/'state.json',state);code=child.wait()
 state.update(status='TRAIN_COMPLETED' if code==0 else 'FAILED',exit_code=code,finished=time.time());write(output/'state.json',state)
 with readme.open('a') as f:f.write(f'\n训练退出码{code}，状态{state["status"]}。未读TEST评分。\n')
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=ARMS);p.add_argument('--smoke',action='store_true');p.add_argument('--steps',type=int,default=20000);p.add_argument('--suffix',default='');a=p.parse_args()
 assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path(PYTHON).resolve()
 if a.prepare:prepare()
 else:sys.exit(run(a))
