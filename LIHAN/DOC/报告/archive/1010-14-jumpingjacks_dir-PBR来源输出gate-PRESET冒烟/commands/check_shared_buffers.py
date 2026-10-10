from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parent.parent
python='/home/lihan/miniconda3/envs/lumimotion/bin/python'
for name in ['geometry','material']:
 with (r/('logs/'+name+'_buffers.log')).open('w') as log:
  result=subprocess.run([python,'-u',str(r/('commands/check_'+name+'_buffers.py'))],stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,name
print(json.dumps({'shared_buffer_checks':'PASS'}),flush=True)
