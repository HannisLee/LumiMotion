"""等待完整训练后执行预登记DEV选择、冻结、TEST、独立核验及外观导出。"""
import concurrent.futures
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from LIHAN.scripts.stage3_el_eldl_four import ROOT,CONTROL,ARMS,PYTHON,env,write
from LIHAN.scripts.stage3_el_eldl_jobs import evaluate
from LIHAN.scripts.stage3_el_eldl_tune import TUNES
LOCKS={i:threading.Lock() for i in range(4)}
PATHS={k:ROOT/'output'/v[0] for k,v in ARMS.items()}
TUNEPATHS={k:ROOT/'output'/v[0] for k,v in TUNES.items() if k!='static_ELDL'}
PATHS['static_ELDL']=ROOT/'output/1026-03-LH-static-ELDL-GT方向光-Stage3保存点续训'
STATUS=CONTROL/'finish_state.json'

def status(phase,**extra):write(STATUS,{'phase':phase,'time':datetime.now(timezone(timedelta(hours=9))).isoformat(),**extra})
def run_command(command,log,gpu=0,cwd=ROOT):
 with log.open('x') as f:
  code=subprocess.run(command,cwd=cwd,env=env(gpu),stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise RuntimeError(f'退出码{code}；命令{shlex.join(command)}；日志{log}')
def render(arm,path,split,label,bias=.1,gpu=None,direct_multiplier=1.):
 state=json.loads((path/'state.json').read_text());gpu=state['gpu'] if gpu is None else gpu
 cached=path/label/'metrics.json'
 if cached.exists():
  data=json.loads(cached.read_text())
  assert data['model']==str(path/'model_mlp') and data['iteration']==state['iteration'] and data['samples']==512
  assert data['source']==str(CONTROL/f'{arm}_{split}_source') and data['light_t_min']==bias and data.get('direct_multiplier',1.)==direct_multiplier
  assert len(data['frames'])==(18 if split=='dev' else 15) and data['checkpoint_unchanged']
  return data
 with LOCKS[gpu]:
  code=evaluate(arm,False,split,label,512,bias,None,str(path),gpu,direct_multiplier)
 if code:raise RuntimeError(str(path)+' '+label+'失败')
 return json.loads((path/label/'metrics.json').read_text())

def main():
 status('WAITING_TRAIN')
 while True:
  states=[]
  try:states=[json.loads((p/'state.json').read_text()) for p in [*PATHS.values(),*TUNEPATHS.values()]]
  except (FileNotFoundError,json.JSONDecodeError):pass
  if any(s['status']=='FAILED' for s in states):raise RuntimeError('训练出现FAILED，停止后处理，保留日志')
  if len(states)==len(PATHS)+len(TUNEPATHS) and all(s['status']=='TRAIN_COMPLETED' for s in states):break
  time.sleep(10)
 # 两训练快照的物理前向必须完全相同；初始化/投影不进入渲染路径。
 for relative in ['gaussian_renderer/render_ir.py','scene/light.py','scene/gaussian_model.py','scripts/stage2_runtime.py','LIHAN/scripts/stage3_el_eldl_mixed.py']:
  assert (CONTROL/'snapshot'/relative).read_bytes()==(CONTROL/'snapshot_tune'/relative).read_bytes(),relative
 status('EVALUATING_DEV_512')
 tasks=[(arm,path) for arm,path in PATHS.items()]+[(arm,path) for arm,path in TUNEPATHS.items()]
 dev={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(render,arm,path,'dev','dev_final_512'):path for arm,path in tasks}
  for future in concurrent.futures.as_completed(futures):dev[str(futures[future])]=future.result()
 write(CONTROL/'all_final_dev.json',dev)
 selected=dict(PATHS)
 for arm,path in TUNEPATHS.items():
  candidate=dev[str(path)]['metrics'];base=dev[str(PATHS[arm])]['metrics']
  if all(candidate[k]>base[k] for k in ['psnr','ssim']):selected[arm]=path
 status('DEV_BIAS_DIAGNOSTIC',selected={k:str(v) for k,v in selected.items()})
 biases={arm:.1 for arm in ARMS}
 multipliers={arm:1. for arm in ARMS}
 for arm in ['static_ELDL','dynamic_ELDL']:
  path=selected[arm];baseline=dev[str(PATHS[arm.replace('ELDL','EL')])]['metrics'];value=dev[str(path)]['metrics']
  if all(value[k]>baseline[k] for k in ['psnr','ssim']):continue
  s=json.loads((path/'state.json').read_text());out=CONTROL/('bias_'+arm);snapshot=CONTROL/'snapshot'
  command=[PYTHON,str(snapshot/'LIHAN/scripts/bootstrap_1011.py'),str(snapshot),'LIHAN.scripts.stage3_el_eldl_bias_grid','--model',str(path/'model_mlp'),'--iteration',str(s['iteration']),'--source',str(CONTROL/f'{arm}_dev_source'),'--directions',str(CONTROL/f'{arm}_dev_directions.json'),'--output',str(out)]
  write(CONTROL/('bias_'+arm+'_command.json'),{'command':command,'test_read':False})
  if not (out/'result.json').exists():run_command(command,CONTROL/('bias_'+arm+'.log'),s['gpu'],snapshot)
  bias=json.loads((out/'result.json').read_text());assert bias['model']==str(path/'model_mlp') and bias['iteration']==s['iteration'] and bias['samples']==512;biases[arm]=bias['selected']['bias'];multipliers[arm]=bias['selected']['direct_multiplier']
 selection={}
 for arm,path in selected.items():
  s=json.loads((path/'state.json').read_text());model=path/'model_mlp';it=s['iteration']
  files=[model/'stage2_config.json',model/f'point_cloud/iteration_{it}/point_cloud.ply',model/f'deform/iteration_{it}/deform.pth',model/f'envmap/iteration_{it}/envmap.pth']
  if (model/f'envmap/iteration_{it}/mixed.json').exists():files.append(model/f'envmap/iteration_{it}/mixed.json')
  selection[arm]={'experiment':str(path),'model':str(model),'iteration':it,'parent_iteration':s.get('lineage_parent_iteration',s['parent_iteration']),'new_loops':s.get('lineage_loops',s['steps']),'optimizer_updates':s.get('optimizer_updates_lineage',sum(i%4==0 for i in range(s['parent_iteration']+1,s['iteration']))),'optimizer_restarted':s.get('optimizer_restarted',False),'light_t_min':biases[arm],'direct_multiplier':multipliers[arm],'samples':512,'evaluation':str(path/'evaluation_final_512'),'checkpoint_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'dev':dev[str(path)]['metrics'],'frozen_before_test':True}
 write(CONTROL/'final_selection.json',selection)
 status('EVALUATING_TEST_512',selection_frozen=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(render,arm,selected[arm],'test','evaluation_final_512',biases[arm],i,multipliers[arm]):arm for i,arm in enumerate(ARMS)}
  tests={}
  for future in concurrent.futures.as_completed(futures):tests[futures[future]]=future.result()
 write(CONTROL/'all_final_test.json',tests)
 status('INDEPENDENT_VALIDATION')
 for arm,path in selected.items():
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in selection[arm]['checkpoint_sha256'].items())
  run_command([PYTHON,'-m','LIHAN.scripts.stage3_el_eldl_validate','--experiment',str(path),'--evaluation',str(path/'evaluation_final_512')],path/'validation_final_512.log')
 status('EXPORTING')
 run_command([PYTHON,'-m','LIHAN.scripts.stage3_el_eldl_export'],CONTROL/'export.log')
 target=ROOT/'LIHAN/DOC/报告/1026-01-静动态EL与GT方向光ELDL四组Stage3外观结果.md'
 labels={'static_EL':'静态EL','static_ELDL':'静态GT方向光ELDL','dynamic_EL':'动态EL','dynamic_ELDL':'动态GT方向光ELDL'}
 lines=['# 静动态EL与GT方向光ELDL四组Stage3外观结果','','日期：2026-10-26（Asia/Tokyo）；服务器ushas；环境lumimotion-ushas。','','| 组别 | TEST PSNR(dB) | TEST SSIM | 新增循环 | 射线偏移 |','|---|---:|---:|---:|---:|']
 for arm in ARMS:
  m=tests[arm]['metrics'];e=selection[arm];lines.append(f"| {labels[arm]} | {m['psnr']:.6f} | {m['ssim']:.6f} | {e['new_loops']} | {e['light_t_min']} |")
 verdicts={}
 for kind in ['static','dynamic']:
  b=tests[kind+'_EL']['metrics'];c=tests[kind+'_ELDL']['metrics'];passed=all(c[k]>b[k] for k in ['psnr','ssim']);verdicts[kind]='PASS' if passed else 'FAILED'
  lines+=['',f"{kind}：ΔPSNR={c['psnr']-b['psnr']:+.6f}dB，ΔSSIM={c['ssim']-b['ssim']:+.6f}；GT方向光同时超过EL目标 **{verdicts[kind]}**。"]
 lines+=['','固定15 TEST，400×400黑底完整浮点sRGB，逐帧计算后均值；SSIM11×11 sigma1.5。训练128半球样本，最终DEV/TEST512样本。每4循环更新，未中断20k实际4999次、8k实际1999次；静态GT主实验从12k保存点恢复，权重链20k共4998次更新，重建Adam且丢失6393步未保存计算，不能声称完全连续或总计算成本相同；终点先保存后更新。独立NumPy/SciPy核验和父参数继承核验均通过。','', 'EL与ELDL原始照明、输入与父训练历史不同；本结果是跨数据集完整方法比较，不是同观测单因素照明消融。EL父曾见过DEV，历史TEST已多轮查看，不称新盲测。两侧GT分别展示。','', '四组Stage3均有终点；静态GT主实验包含12k＋8k的中断恢复；静态环境初值对照收到SIGTERM且没有终点，保留为工程失败，不进入选模；动态颜色clamp修正8k只有DEV两项均优于默认20k才替换；偏移只由完整18DEV选取，选择与文件SHA在任何本轮TEST评分前冻结。完整来源、命令、日志、默认/失败候选均保留；没有通过降低EL预算、替换GT或掩膜制造胜出。','', '实际最终四类目检：PENDING，由主执行者查看代表帧后补记；不能以独立指标复算冒充人工目检。','', '## 完整DEV记录','', '| 实验 | PSNR(dB) | SSIM |','|---|---:|---:|']
 for path,r in dev.items():lines.append(f"| {Path(path).name} | {r['metrics']['psnr']:.6f} | {r['metrics']['ssim']:.6f} |")
 lines+=['','交付：LIHAN/exports/1025-02-静动态EL与GT方向光ELDL-Stage3外观对比；仅指标表、同视角RGB、CSV、PDF和来源说明，不含几何覆盖、材质诊断或固定相机。','', '总控：'+str(CONTROL)+'；所有训练/渲染完整命令、source/model/iteration、日志、代表图在各实验README。']
 target.write_text('\n'.join(lines)+'\n')
 for arm,path in tasks:
  with (path/'README.md').open('a') as f:
   r=dev[str(path)]['metrics'];f.write(f"\n最终DEV512：PSNR {r['psnr']:.9f}dB，SSIM {r['ssim']:.9f}；统计dev_final_512/metrics.json，日志dev_final_512.log。\n")
   if selected[arm]==path:
    m=tests[arm]['metrics'];f.write(f"\n最终TEST512：PSNR {m['psnr']:.9f}dB，SSIM {m['ssim']:.9f}；渲染evaluation_final_512/rgb与gt，原始数组arrays；代表视角0010/0080/0150.png。独立验证independent_validation.json；最终四类目检待补。\n")
   else:f.write('\n该候选未满足DEV双指标替换规则，FAILED（DEV未入选）；未运行TEST，全部检查点/统计/日志保留。\n')
 write(CONTROL/'numeric_verdict.json',verdicts)
 status('READY_FOR_MANUAL_REVIEW',verdicts=verdicts,export='LIHAN/exports/1025-02-静动态EL与GT方向光ELDL-Stage3外观对比')
if __name__=='__main__':
 try:main()
 except Exception as exc:
  status('FAILED',error=str(exc));raise
