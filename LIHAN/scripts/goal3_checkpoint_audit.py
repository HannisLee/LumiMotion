#!/usr/bin/env python3
"""只读核验已完成T04训练的真实中间保存点；不把保存点伪称训练终点。"""
import argparse
import math
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal_eval_protocol import ROOT, read, digest, verify, CONTROL
from goal3_densify_render import actual_sources, initialization_evidence
MODEL=ROOT/'output/1012-16-LH-static-jumpingjacks_DL_V2-F-T04-原增密续训2k/model_mlp'
OUTPUT=ROOT/'output/1012-21-LH-static-jumpingjacks_DL_V2-F-T04中间检查点回溯'
OLD=ROOT/'output/1012-20-LH-static-jumpingjacks_DL_V2-F-T04-T05执行总控'
BINDINGS={str(OLD/'root_formal_completed_checkpoints.json'):'6c09629c05a8a538a7642374d3122b78e73466d109e8022e1caa47aaeb35d901',str(OLD/'archive_T04_T05/manifest.json'):'681348b2a702d9b1e3721a3f19ec14f080a6727ed570866a16b50b0e2d58b01c'}
HELPERS={'goal3_densify_render.py':'0c34101ab102df56ead9dba6528a4a7dc6519eb42dc282559b5d778debe4c241','goal3_densify_parity.py':'7bf490b23298b78f3ff34a51694cb0ce25069e17cb7b319b52e4721b4a42d6da','goal3_densify_target.py':'428f71468b7d5db0694cb8dd2e56fdd95655631b94b9ccf5fd4fa14de193d3c8','goal3_densify_run.py':'2eb0900beb8b09c7f3bbebab56905fdd5b5639a1b5d8971f2cffff96d6d2fd13','goal3_densify_visuals.py':'3eccbf2d3e69d50d99ca4e54bf3f9d67203ee0ca2a581fa6defbb94927a039f0'}
def destination(path):
 path=Path(path).resolve()
 if not path.is_relative_to(OUTPUT) or path==OUTPUT:raise AssertionError('回溯产物只能新output21的独立子目录。')
 return path

def audit(model,iteration):
 import torch
 from plyfile import PlyData
 model=Path(model).resolve()
 if model!=MODEL or iteration not in (16500,17000,17500):raise AssertionError('只回溯原T04三个预存点，不启动/伪装新训练。')
 protocol=verify(read(CONTROL/'protocol.json'))
 files={str(CONTROL/'protocol.json'):digest(CONTROL/'protocol.json')}
 for name,expected in HELPERS.items():
  path=ROOT/'LIHAN/scripts'/name
  if digest(path)!=expected:raise AssertionError('冻结旧helper变化：'+name)
  files[str(path)]=expected
 for path,expected in BINDINGS.items():
  if digest(path)!=expected:raise AssertionError('原归档/保存点清单变化。')
  files[path]=expected
 marker,sources=actual_sources(model);files.update(sources)
 initialization=initialization_evidence(model,marker);files[initialization['path']]=initialization['sha256']
 if marker['smoke'] or marker['update_budget']!=2000 or marker['densification_route']!='original':raise AssertionError('须原正式T04轨迹。')
 finished_path=model/'foot_densify_finished.json';finished=read(finished_path)
 if finished['status']!='TRAINING_FINISHED' or finished['iteration']!=18000 or finished['optimizer_updates']!=2000 or finished['opacity_resets_performed']!=0 or not finished['initial_sources_unchanged']:raise AssertionError('原实际训练必须完成18000/2000，不能改写为中间终点。')
 # 保留原fresh全部存活runtime/反向/初始化门禁：原18000已验证记录先完整SHA复核，新GPU fresh仍再次执行旧parity门禁。
 terminal=model.parent/'fresh_parity_18000/parity.json';proof=read(terminal)
 if proof['status']!='PASS' or proof['iteration']!=18000 or Path(proof['model']).resolve()!=model or proof['source']!=protocol['source_path'] or proof['densification_route']!='original':raise AssertionError('缺对应原18000独立fresh来源证据。')
 for path,expected in proof['sources'].items():
  if digest(path)!=expected:raise AssertionError('原终点fresh来源变化：'+path)
 files.update(proof['sources']);files[str(terminal)]=digest(terminal);files[str(finished_path)]=digest(finished_path)
 for name in ('fresh18000','evaluation18000'):
  path=model.parent/'commands'/(name+'.json')
  if read(path)['actual_exit_code']!=0:raise AssertionError('原真实命令未成功。')
  files[str(path)]=digest(path)
 assets=[model/s for s in (f'point_cloud/iteration_{iteration}/point_cloud.ply',f'deform/iteration_{iteration}/deform.pth',f'static_dl_lighting/iteration_{iteration}/lighting.json',f'optimizer/iteration_{iteration}.pth')]
 saved=next(j for j in read(OLD/'root_formal_completed_checkpoints.json') if j['run']==model.parent.name)['checkpoint_assets'];archived={j['path']:j['sha256'] for j in read(OLD/'archive_T04_T05/manifest.json')['files'] if j['type']=='file'}
 hashes={str(p):digest(p) for p in assets}
 if any(saved.get(p)!=s or archived.get(p)!=s for p,s in hashes.items()):raise AssertionError('四资产必须同时匹配原保存点和封存清单SHA。')
 files.update(hashes);point_count=len(PlyData.read(str(assets[0]))['vertex'])
 light=read(assets[2]);updates=iteration-16000
 for k,v in {'iteration':iteration,'initial_iteration':16000,'continuation_step':updates,'optimizer_updates':updates,'ambient_fraction':0.,'densification_route':'original','reset_policy':'skip_all','clock_start':16000,'clock_shift_diagnostic':False,'smoke':False}.items():
  if light[k]!=v:raise AssertionError('中间lighting实际元数据不同：'+k)
 if light['actual_entrypoint']!=marker['actual_entrypoint'] or light['directions_sha256']!=digest(marker['inherited_config']['static_dl_directions']) or any(not math.isfinite(v) or v<=0 for v in light['intensity_from_train_rgb']):raise AssertionError('实际方向/训练光强/入口错误。')
 state=torch.load(assets[3],map_location='cpu',weights_only=False)
 for k,v in {'iteration':iteration,'optimizer_updates':updates,'reset_policy':'skip_all','densification_route':'original','clock_shift_diagnostic':False}.items():
  if state[k]!=v:raise AssertionError('中间optimizer实际元数据不同：'+k)
 steps={}
 for group in ('gaussian','light'):
  if not state[group]['state']:raise AssertionError('真实中间Adam为空。')
  steps[group]=[float(v['step']) for v in state[group]['state'].values()]
  if any(s!=updates for s in steps[group]):raise AssertionError('Adam真实step不等于实际保存更新数。')
  if any(not torch.isfinite(v[k]).all() for v in state[group]['state'].values() for k in ('exp_avg','exp_avg_sq')):raise AssertionError('Adam状态非有限。')
 for k in ('xyz_gradient_accum','denom','max_radii2D'):
  if len(state[k])!=point_count or not torch.isfinite(state[k]).all():raise AssertionError('中间保存统计/PLY点数或有限性不同。')
 stack=[];rng=random.Random(marker['args']['seed'])
 for _ in range(updates):
  if not stack:stack=list(range(135))
  stack.pop(rng.randrange(len(stack)))
 if state['remaining_camera_indices']!=stack or state['camera_rng']!=rng.getstate():raise AssertionError('独立原seed相机序列与保存状态不同。')
 config=marker['inherited_config'];t=iteration/config['position_lr_max_steps'];lr=math.exp((1-t)*math.log(config['position_lr_init']*5)+t*math.log(config['position_lr_final']*5));actual_lr=next(g['lr'] for g in state['gaussian']['param_groups'] if g['name']=='xyz')
 if not math.isclose(lr,actual_lr,rel_tol=1e-12,abs_tol=1e-15):raise AssertionError('保存XYZ LR与原30k日程不一致。')
 if any(digest(p)!=s for p,s in files.items()):raise AssertionError('CPU核验时来源发生变化。')
 return dict(status='PASS_RECORDED_CHECKPOINT_CPU_EVIDENCE',model=str(model),source=protocol['source_path'],checkpoint_iteration=iteration,checkpoint_optimizer_updates=updates,actual_training_terminal_iteration=18000,actual_training_terminal_optimizer_updates=2000,point_count=point_count,adam_steps=steps,remaining_camera_count=len(stack),xyz_lr=actual_lr,checkpoint_four_asset_sha256=hashes,initialization=initialization,original_six_source_sha256=protocol['code_sha256'],sources=files,fresh_status='NOT_RUN',formal_metrics_status='NOT_RUN',geometry_status='PENDING',scope='只核已真实保存的中间资产与原完整训练证据，不伪称中间终点、不恢复Adam做训练。')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=destination(a.output);report=audit(a.model,a.iteration);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:__import__('json').dump(report,f,ensure_ascii=False,indent=2)
 print(report['status'],a.iteration,report['point_count'],flush=True)
if __name__=='__main__':main()
