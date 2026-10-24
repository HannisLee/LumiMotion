"""等待已授权的静态Stage2补训评测结束，自动更新展示数值表。"""
from pathlib import Path
import json,time,subprocess,sys
R=Path(__file__).resolve().parents[2];C=R/'output/1024-02-静态ELDL-展示数值补全总控'
paths=[R/'output/1024-03-staticEL-展示Stage2补训重试1',R/'output/1024-04-staticDL-展示Stage2补训']
while True:
 states=[json.loads((p/'state.json').read_text()) for p in paths]
 if any(s['status'] in ['FAILED','EVALUATION_FAILED'] for s in states):
  (C/'自动导出状态.json').write_text(json.dumps(dict(status='FAILED',states=states),ensure_ascii=False,indent=2));sys.exit(1)
 if all(s['status']=='READY_FOR_REVIEW' for s in states):break
 time.sleep(30)
for p,s in zip(paths,states):
 rgb=json.loads((p/'evaluation/rgb/metrics.json').read_text());albedo=json.loads((p/f'evaluation/albedo/iteration_{s["iteration"]}/metrics.json').read_text());assert len(rgb['frames'])==len(albedo['frames'])==15
 assert albedo['checkpoint_unchanged'] and albedo['calibration'] is None
with (C/'表格自动导出.log').open('w') as log:code=subprocess.run([sys.executable,str(R/'exports/1024-02-静态EL与DL-数值对比/build_tables.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT).returncode
(C/'自动导出状态.json').write_text(json.dumps(dict(status='EXPORTED' if code==0 else 'FAILED',exit_code=code,visual_review='PENDING',exports='exports/1024-02-静态EL与DL-数值对比'),ensure_ascii=False,indent=2))
