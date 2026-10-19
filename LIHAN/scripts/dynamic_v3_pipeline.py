"""固定预算动态V3流水线：等待原生终点、估光、四组、DEV与同父灯向对照。

不执行TEST/GT评测，不依据目标分数停止训练。独立子任务保留退出码和失败现场。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time


PYTHON='/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Pipeline:
    def __init__(self,plan):
        self.plan_path=plan.resolve();self.plan=read(plan)
        self.root=Path(self.plan['root']).resolve();self.snapshot=Path(self.plan['snapshot']).resolve()
        self.control=Path(self.plan['control']).resolve();self.out=self.control/'pipeline_v1'
        assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path(PYTHON).resolve()
        assert self.out.is_relative_to(self.root/'output') and not self.out.exists()
        self.out.mkdir()
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(self.plan['gpu']),PYTHONUNBUFFERED='1',
                      DYNAMIC_V3_REPO_ROOT=str(self.root))
        self.state=dict(status='STARTING',pid=os.getpid(),plan_sha256=sha(plan),stages={},
                        started_unix=time.time(),test_or_gt_evaluation=False)
        for relative,digest in read(self.snapshot/'snapshot_manifest.json')['files'].items():
            assert sha(self.snapshot/relative)==digest,relative
        for path,digest in self.plan['engineering_gates'].items():
            assert sha(path)==digest,path
        for path,digest in self.plan['input_sha256'].items():assert sha(path)==digest,path
        self.persist()

    def persist(self):
        self.state['updated_unix']=time.time();write(self.out/'state.json',self.state)

    def command(self,module,args):
        return [PYTHON,str(self.root/'LIHAN/scripts/bootstrap_1011.py'),str(self.snapshot),module]+list(map(str,args))

    def start(self,name,command,folder):
        for path,digest in self.plan['external_runtime_sha256'].items():assert sha(path)==digest,path
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        assert not (folder/'job.json').exists(),'禁止覆盖子任务'
        write(folder/'command.json',command)
        job=dict(status='STARTING',command=command,command_shell=shlex.join(command),started_unix=time.time(),
                 pipeline_pid=os.getpid(),snapshot=str(self.snapshot),gt_read=False,test_read=False)
        log=(folder/'run.log').open('x')
        process=subprocess.Popen(command,cwd=self.root,env=self.env,stdout=log,stderr=subprocess.STDOUT)
        job.update(status='RUNNING',pid=process.pid);write(folder/'job.json',job)
        self.state['stages'][name]=dict(job,folder=str(folder));self.persist()
        (folder/'README.md').write_text('# 动态V3固定预算子任务\n\n完整命令：\n\n```bash\n'+shlex.join(command)+
            '\n```\n\nsource/model/iteration及完整参数以上述命令、plan.json与最终marker为准。\n'
            '日志：`run.log`；训练资产：`model_mlp`；渲染与定量指标：后续DEV及隔离评测目录，不将训练loss当GT指标。\n'
            '代表图/视频：尚未生成。四类RGB/法线/Depth-Mesh/静动态目检均待生成后填写。\n'
            '最终结论：PENDING，不能将进程启动视为PASS。\n')
        return process,log,job,folder

    def finish(self,name,task):
        process,log,job,folder=task
        code=process.wait();log.close()
        job.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'FAILED',finished_unix=time.time())
        write(folder/'job.json',job);self.state['stages'][name]=dict(job,folder=str(folder));self.persist()
        with (folder/'README.md').open('a') as stream:
            stream.write('\n进程退出码：'+str(code)+'。最终模型验收仍PENDING；进程失败则本子任务FAILED，保留全部现场。\n')
        return code

    def run(self,name,module,args,folder):
        task=self.start(name,self.command(module,args),folder)
        code=self.finish(name,task)
        assert code==0,name+'失败，禁止覆盖重试'

    def resource(self):
        uuid=subprocess.check_output(['nvidia-smi','--id',str(self.plan['gpu']),
            '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'],text=True)
        for line in processes.splitlines():
            pid,gpu=line.split(',')
            if gpu.strip()!=uuid:continue
            status=Path('/proc')/pid.strip()/'status'
            if status.exists():
                uid=next(row.split()[1] for row in status.read_text().splitlines() if row.startswith('Uid:'))
                if int(uid)!=os.getuid():return 0
        raw=subprocess.check_output(['nvidia-smi','--id',str(self.plan['gpu']),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        return int(raw.strip())

    def train_args(self,variant,output,direction_lr=0.):
        args=['--source',self.plan['source'],'--parent',self.plan['parent'],'--parent-iteration',35000,
              '--lighting',str(Path(self.plan['light_output'])/'run_v1/lighting.json'),
              '--stats',self.plan['stats'],'--output',output,'--path',variant[0],
              '--direction-lr',direction_lr]
        if variant[1]=='1':args.append('--ps')
        return args

    def score(self,name,model,kind,output):
        self.run(name,'LIHAN.scripts.dynamic_v3_dev',['score','--source',self.plan['dev_source'],
            '--model',model,'--kind',kind,'--iteration',35000 if kind=='stage1' else 8100,'--output',output],
            self.out/(name+'_job'))

    def execute(self):
        parent=Path(self.plan['parent']);stage1_job=parent.parent/'job.json'
        self.state['status']='WAITING_NATIVE_STAGE1_35000';self.persist()
        while True:
            if (parent/'training_finished.json').exists():
                marker=read(parent/'training_finished.json')
                assert marker['args']['iterations']==35000 and marker['actual_optimizer_updates']==34999
                assert not marker['weights_loaded'] and not marker['gt_supervision']
                for relative,digest in marker['checkpoint_sha256'].items():assert sha(parent/relative)==digest
                break
            job=read(stage1_job)
            assert job['status']!='FAILED','原生Stage1失败，禁止复用旧模型'
            pid=job.get('training_pid');assert pid and Path('/proc',str(pid)).exists(),'Stage1进程消失且无终点'
            time.sleep(15)
        self.state['status']='NATIVE_RELOAD_AND_TRAIN_LIGHT_ESTIMATION';self.persist()
        while self.resource()<16000:time.sleep(15)
        reload_out=self.control/'native_reload_35000_v1'
        self.run('native_reload','LIHAN.scripts.dynamic_v3_reload',['--source',self.plan['source'],'--model',parent,
            '--iteration',35000,'--output',reload_out],self.out/'native_reload_job')
        assert read(reload_out/'reload_audit.json')['status']=='PASS_RELOAD_AND_DYNAMIC_GRADIENT_ENGINEERING'
        self.score('native_dev',parent,'stage1',self.control/'dev_native_35000_v1')
        light_root=Path(self.plan['light_output'])
        self.run('estimate_light','LIHAN.scripts.dynamic_v3_estimate_light',['--source',self.plan['source'],
            '--parent',parent,'--iteration',35000,'--stats',self.plan['stats'],'--output',light_root/'run_v1'],light_root)
        light=read(light_root/'run_v1/lighting.json')
        assert light['parent_iteration']==35000 and not light['smoke'] and len(light['train_frames'])==117
        self.state['status']='FOUR_CONTROLLED_8100_RUNS';self.persist()
        pending=list(self.plan['variants']);running={};failed=[]
        while pending or running:
            while pending and len(running)<3 and self.resource()>16000:
                variant=pending.pop(0);out=Path(self.plan['variants'][variant])
                running[variant]=self.start(variant,self.command('LIHAN.scripts.dynamic_v3_train',self.train_args(variant,out)),out)
            for variant,task in list(running.items()):
                if task[0].poll() is None:continue
                code=self.finish(variant,task);del running[variant]
                if code or not (Path(self.plan['variants'][variant])/'training_finished.json').exists():failed.append(variant)
            time.sleep(10)
        assert not failed,'四组失败：'+str(failed)+'；其他任务已完成且现场保留'
        for variant,output in self.plan['variants'].items():
            marker=read(Path(output)/'training_finished.json')
            assert marker['actual_optimizer_updates']==8100
            self.score('dev_'+variant,Path(output)/'model_mlp','stage2',Path(output)/'dev_8100_v1')
        baseline=self.control/'dev_native_35000_v1/summary.json'
        selection=self.control/'dev_four_selection_v1'
        args=['select','--baseline',baseline,'--mode','four','--output',selection]
        for variant,output in self.plan['variants'].items():args+=['--candidate',variant+'='+str(Path(output)/'dev_8100_v1/summary.json')]
        self.run('select_four','LIHAN.scripts.dynamic_v3_dev',args,self.out/'select_four_job')
        chosen=read(selection/'selection.json')['selected'];direction_out=Path(self.plan['direction_output'])
        self.state.update(status='SAME_PARENT_DIRECTION_LR_1E5',selected_variant=chosen);self.persist()
        self.run('direction_lr','LIHAN.scripts.dynamic_v3_train',self.train_args(chosen,direction_out,1e-5),direction_out)
        assert read(direction_out/'training_finished.json')['actual_optimizer_updates']==8100
        self.score('dev_direction',direction_out/'model_mlp','stage2',direction_out/'dev_8100_v1')
        final_out=self.control/'dev_final_selection_v1'
        self.run('select_final','LIHAN.scripts.dynamic_v3_dev',['select','--baseline',baseline,'--mode','final',
            '--candidate','freeze='+str(Path(self.plan['variants'][chosen])/'dev_8100_v1/summary.json'),
            '--candidate','directionlr='+str(direction_out/'dev_8100_v1/summary.json'),'--output',final_out],self.out/'select_final_job')
        self.state.update(status='READY_FOR_ISOLATED_EVALUATION',final_selection=str(final_out/'selection.json'),
                          model_quality_pass=False,finished_unix=time.time());self.persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();pipeline=Pipeline(args.plan)
    try:pipeline.execute()
    except Exception as error:
        pipeline.state.update(status='FAILED',error=repr(error));pipeline.persist();raise


if __name__=='__main__':main()
