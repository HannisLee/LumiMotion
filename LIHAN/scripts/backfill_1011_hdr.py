#!/usr/bin/env python3
"""用隔离的A40兼容nvdiffrast，为05现有55k检查点补渲染HDR。"""
import argparse,json,os,subprocess
from pathlib import Path
import cv2
import batch_1011 as b
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--attempt',type=int,default=1)
attempt=parser.parse_args().attempt
name='hdr_retry' if attempt==1 else f'hdr_retry_{attempt:02d}'
exp=b.read_json(b.CONFIG)['experiments'][4]
out=Path(exp['output']);directory=out/'review_1011'/name
modules=b.CONTROL/'tools/nvdiffrast-sm86/extension'
if len(list(modules.glob('_nvdiffrast_c*.so')))!=1:raise RuntimeError('兼容模块尚未编译完成。')
command=b.cmd(b.ROOT,'scripts.render_stage2','--model_path',exp['model'],'--load_iter',55000,
              '--task','hdr','--hdr_samples',2048,'--hdr_filepath',b.ROOT/'example_envmaps/golden_bay_4k_32x16_rot330.hdr','--output_path',directory)
gpu=b.gpu_acquire(exp, update_waiting=False)
try:
    environment=b.env(gpu);environment['PYTHONPATH']=str(modules)+':'+environment.get('PYTHONPATH','')
    logfile=out/'review_1011'/(name+'.log')
    with logfile.open('x') as log:
        log.write('开始：'+b.now()+'\n命令：'+b.shell_line(command)+'\nPYTHONPATH：'+environment['PYTHONPATH']+'\nGPU：'+str(gpu)+'\n');log.flush()
        code=subprocess.run(command,cwd=b.ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
        log.write('\n结束：'+b.now()+'；退出码：'+str(code)+'\n')
    record=dict(time=b.now(),command=command,environment=dict(PYTHONPATH=environment['PYTHONPATH']),exit_code=code,gpu=gpu,log=str(logfile),output=str(directory),training_restarted=False)
    if code==0:
        cap=cv2.VideoCapture(str(directory/'hdr/hdr.mp4'));count=0
        while cap.read()[0]:count+=1
        cap.release();record['decoded_frames']=count
        record['png_count']=len(list((directory/'hdr').glob('*.png')))
        record['status']='PASS' if count==150 and record['png_count']==150 else 'FAILED'
    else:record['status']='FAILED'
    b.write_json(out/'review_1011'/(name+'.json'),record)
finally:b.gpu_release()
