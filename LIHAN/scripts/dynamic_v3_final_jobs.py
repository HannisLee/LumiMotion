"""等待全部固定训练/DEV终点后，一次性冻结七模型并隔离评价；不反馈训练。"""
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def live(pid):
    path=Path('/proc')/str(pid)/'status'
    if not path.is_file():return False
    return not any(row.startswith('State:') and row.split()[1] in ['Z','X'] for row in path.read_text().splitlines())


class FinalJobs:
    def __init__(self,control):
        self.root=Path(__file__).resolve().parents[2];self.control=control.resolve()
        assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path(PYTHON).resolve()
        assert self.control.is_relative_to(self.root/'output')
        self.plan=read(self.control/'formal_pipeline_plan_v1.json')
        self.out=self.control/'eval_only/final_evaluation_jobs_v1';self.out.mkdir(parents=True,exist_ok=False)
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(self.plan['gpu']),PYTHONUNBUFFERED='1',
                      OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
        self.state=dict(status='WAITING_FIXED_TRAINING_AND_DEV',pid=os.getpid(),started_unix=time.time(),
            jobs={},training_or_model_selection_modified=False,source_code_sha256=sha(__file__))
        self.locked={str(self.control/'formal_pipeline_plan_v1.json'):sha(self.control/'formal_pipeline_plan_v1.json')}
        self.locked[str(Path(__file__).resolve())]=self.state['source_code_sha256']
        self.persist()

    def persist(self):
        self.state['updated_unix']=time.time();write(self.out/'state.json',self.state)

    def check_locks(self):
        for path,digest in self.locked.items():assert sha(path)==digest,path

    def start(self,name,command,expected):
        self.check_locks();folder=self.out/'jobs'/name;folder.mkdir(parents=True,exist_ok=False)
        log=(folder/'run.log').open('x')
        process=subprocess.Popen(list(map(str,command)),cwd=self.root,env=self.env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(status='RUNNING',pid=process.pid,command=list(map(str,command)),shell=shlex.join(list(map(str,command))),
                    expected_output=str(expected),started_unix=time.time(),
                    environment={key:self.env[key] for key in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
                        'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        write(folder/'job.json',record);self.state['jobs'][name]=record;self.persist()
        return process,log,record,folder

    def finish(self,name,task):
        process,log,record,folder=task;code=process.wait();log.close()
        record.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'FAILED',finished_unix=time.time())
        write(folder/'job.json',record);self.state['jobs'][name]=record;self.persist()
        return code

    def run(self,name,command,expected):
        assert self.finish(name,self.start(name,command,expected))==0,name+'进程失败，现场保留'
        assert Path(expected).is_file(),name+'缺预定产物'

    def resource(self):
        uuid=subprocess.check_output(['nvidia-smi','--id',str(self.plan['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
        for row in rows.splitlines():
            pid,gpu=row.split(',');status=Path('/proc')/pid.strip()/'status'
            if gpu.strip()!=uuid or not status.exists():continue
            uid=next(line.split()[1] for line in status.read_text().splitlines() if line.startswith('Uid:'))
            if int(uid)!=os.getuid():return 0
        return int(subprocess.check_output(['nvidia-smi','--id',str(self.plan['gpu']),'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())

    def execute(self):
        while True:
            state=read(self.control/'pipeline_v1/state.json')
            if state['status']=='READY_FOR_ISOLATED_EVALUATION':break
            assert state['status']!='FAILED','训练/DEV流水线失败，禁止提前评价'
            assert live(state['pid']),'训练流水线已消失且没有完整终点'
            time.sleep(15)
        four_path=self.control/'dev_four_selection_v1/selection.json'
        final_path=self.control/'dev_final_selection_v1/selection.json'
        four,final=read(four_path),read(final_path)
        assert four['mode']=='four' and final['mode']=='final'
        assert all(not r['gt_read'] and not r['test_read'] for r in [four,final])
        assert four['selected']==state['selected_variant']
        selected=final['candidates'][final['selected']]['model']
        for path in [four_path,final_path]:self.locked[str(path)]=sha(path)
        # 评价前确保当前几何前向与实际训练快照一致；允许独立评测脚本改进归档。
        snapshot=Path(self.plan['snapshot']);snapshot_files=read(snapshot/'snapshot_manifest.json')['files']
        dependency_parity={}
        for relative,digest in snapshot_files.items():
            if relative.startswith(('scene/','utils/','gaussian_renderer/')) or relative=='LIHAN/scripts/dynamic_v3_surface.py':
                path=self.root/relative
                assert sha(path)==digest,'训练/评价前向代码不同：'+relative
                dependency_parity[relative]=digest;self.locked[str(path)]=digest
        self.locked[str(self.root/'LIHAN/scripts/dynamic_v3_evaluate.py')]=sha(self.root/'LIHAN/scripts/dynamic_v3_evaluate.py')
        self.locked[str(self.root/'LIHAN/scripts/dynamic_v3_pair_audit.py')]=sha(self.root/'LIHAN/scripts/dynamic_v3_pair_audit.py')
        self.state.update(status='AUDITING_CONTROLLED_PREFIX',selected_model_from_dev=selected);self.persist()
        pair_out=self.control/'four_pair_audit_v1'
        command=[PYTHON,'-m','LIHAN.scripts.dynamic_v3_pair_audit','--output',pair_out]
        for name,path in self.plan['variants'].items():command += ['--'+name.lower(),Path(path)/'model_mlp']
        self.run('pair_audit',command,pair_out/'audit.json')
        pair_status=read(pair_out/'audit.json')['status']
        self.locked[str(pair_out/'audit.json')]=sha(pair_out/'audit.json')
        old_model=self.root/'output/1011-04-LH-dynamic-jumpingjacks_EL_V2-原生LumiMotion两阶段/stage1_frozen'
        entries=[dict(name='ELV2',dataset='elv2',model=str(old_model),stage='stage1',step=35000),
                 dict(name='N00',dataset='v3',model=self.plan['parent'],stage='stage1',step=35000)]
        entries += [dict(name=name,dataset='v3',model=str(Path(path)/'model_mlp'),stage='stage2',step=8100)
                    for name,path in self.plan['variants'].items()]
        entries.append(dict(name='DIR',dataset='v3',model=str(Path(self.plan['direction_output'])/'model_mlp'),stage='stage2',step=8100))
        self.state['status']='FREEZING_ALL_SEVEN_BEFORE_TEST';self.persist()
        for number,entry in enumerate(entries,5):
            key=entry['name'];dataset=entry['dataset']
            source=self.control/'eval_only'/('unified_source_elv2_v3' if dataset=='elv2' else 'unified_source_v3_v3')/'source.json'
            proof=self.control/'eval_only'/('1019-02-旧动态GT持久顶点对应审计' if dataset=='elv2' else '1019-03-新动态GT持久顶点对应审计')/'correspondence.json'
            freeze_out=self.out/'freezes'/key
            command=[PYTHON,'-m','LIHAN.scripts.dynamic_v3_evaluate','--action','freeze','--source',source,
                '--model',entry['model'],'--iteration',entry['step'],'--stage',entry['stage'],
                '--output',freeze_out,'--confirm-frozen','--track-protocol','persistent_vertices_v1','--correspondence',proof]
            self.run('freeze_'+key,command,freeze_out/'freeze.json')
            freeze=freeze_out/'freeze.json';self.locked[str(freeze)]=sha(freeze)
            frozen=read(freeze)
            for path,digest in frozen['code_sha256'].items():self.locked[path]=digest
            label='LH-dynamic-jumpingjacks_EL_V2' if dataset=='elv2' else 'LH-dynamic-jumpingjacks-ELDL-V3'
            entry.update(freeze=str(freeze),freeze_sha256=sha(freeze),output=str(self.control/f'eval_only/1019-{number:02d}-{label}-{key}统一最终评价'))
        write(self.out/'selection_lock.json',dict(selected_model_from_dev=selected,four_selection=str(four_path),
            final_selection=str(final_path),selection_status=final['status'],pair_audit_status=pair_status,
            all_seven_frozen_before_test=True,candidate_gt_feedback_allowed=False,entries=entries,
            dependency_parity=dependency_parity,locked_files=self.locked))
        self.state.update(status='RUNNING_ISOLATED_EVALUATION',entries=entries);self.persist()
        pending=entries.copy();running={};failed=[]
        while pending or running:
            while pending and len(running)<2 and self.resource()>16000:
                entry=pending.pop(0);name='evaluate_'+entry['name']
                command=[PYTHON,self.root/'LIHAN/scripts/bootstrap_1011.py',self.root,
                    'LIHAN.scripts.dynamic_v3_evaluate','--action','run','--freeze',entry['freeze'],'--output',entry['output']]
                running[name]=(self.start(name,command,Path(entry['output'])/'COMPLETED.json'),entry)
            for name,(task,entry) in list(running.items()):
                if task[0].poll() is None:continue
                code=self.finish(name,task);del running[name]
                completed=Path(entry['output'])/'COMPLETED.json'
                if code or not completed.is_file():failed.append(entry['name'])
                elif read(completed)['status']!='PASS_EXECUTION_ONLY':failed.append(entry['name'])
            if pending or running:time.sleep(15)
        self.check_locks()
        self.state.update(status='FAILED_EVALUATION_JOBS' if failed else 'READY_FOR_MANUAL_REVIEW_AND_REPORT',
            failed_models=failed,scientific_goal='NOT_DECIDED',finished_unix=time.time());self.persist()
        assert not failed,'部分评价失败；全部尝试已完成，所有现场保留：'+str(failed)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--control',type=Path,required=True)
    args=parser.parse_args();runner=FinalJobs(args.control)
    try:runner.execute()
    except Exception as error:
        if runner.state['status']!='FAILED_EVALUATION_JOBS':runner.state['status']='FAILED'
        runner.state['error']=repr(error);runner.persist();raise


if __name__=='__main__':main()
