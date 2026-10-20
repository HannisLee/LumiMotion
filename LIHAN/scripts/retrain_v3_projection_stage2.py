"""投影灯向替换后的单次Stage2同父重训与终点持久监督。"""
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
ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控'
PROJECTION=ROOT/'output/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-T05动态投影估光'
PARENT=ROOT/'output/1019-59-LH-dynamic-jumpingjacks-ELDL-V3-N00原生Stage1/model_mlp'
OLD_LIGHT=ROOT/'output/1019-60-LH-dynamic-jumpingjacks-ELDL-V3-共同TRAIN估光/run_v1/lighting.json'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def run(args):
    assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path(PYTHON).resolve()
    out=args.experiment.resolve();assert out.is_relative_to(ROOT/'output')
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'state.json').exists() and not (out/'model_mlp').exists(),'禁止覆盖已有运行'
    snapshot=CONTROL/'snapshots/formal_stage2_v1';source=CONTROL/'data_train_v1'
    stats=CONTROL/'input_audit_v1/train_statistics.json'
    manifest=read(source/'no_gt_manifest.json');freeze=read(PROJECTION/'freeze.json')
    projected=PROJECTION/'ambient_contrast/result.json';prediction=read(projected);old=read(OLD_LIGHT)
    assert freeze['primary']=='ambient_contrast/result.json' and not freeze['gt_read']
    assert sha(projected)==freeze['artifact_sha256']['ambient_contrast/result.json']
    assert prediction['primary'] and not prediction['gt_read']
    assert prediction['train_frames']==manifest['train_frames']==old['train_frames']
    assert prediction['times']==old['times'] and len(prediction['directions'])==117
    for path,digest in freeze['input_sha256'].items():assert sha(path)==digest,path
    for relative,digest in read(snapshot/'snapshot_manifest.json')['files'].items():assert sha(snapshot/relative)==digest,relative
    for relative in ['LIHAN/scripts/dynamic_v3_surface.py','LIHAN/scripts/dynamic_v3_tracks.py',
                     'LIHAN/scripts/dynamic_v3_train.py','LIHAN/scripts/dynamic_v3_ps_math.py']:
        assert sha(ROOT/relative)==sha(snapshot/relative),'当前训练实现与冻结快照不一致：'+relative
    bound=[source/'no_gt_manifest.json',source.parent/'input_audit_v1/radiometry_v1.json',
           PARENT/'point_cloud/iteration_35000/point_cloud.ply',PARENT/'deform/iteration_35000/deform.pth',stats]
    for path in bound:assert old['source_sha256'][str(path)]==sha(path)
    parent_marker=read(PARENT/'training_finished.json')
    for relative,digest in parent_marker['checkpoint_sha256'].items():assert sha(PARENT/relative)==digest
    lighting=dict(format_version=1,status='TRAIN_PROJECTION_DIRECTIONS_INHERITED_TRAIN_INTENSITY',
        source_split='TRAIN',gt_read=False,source=str(source),parent=str(PARENT),parent_iteration=35000,
        train_frames=prediction['train_frames'],times=prediction['times'],directions=prediction['directions'],
        intensity=old['intensity'],intensity_source='沿用原共同TRAIN估光的全局强度，非投影算法新估计',
        world_fixed=False,global_intensity=True,per_frame_exposure=False,smoke=False,
        direction_source=str(projected),projection_freeze_sha256=sha(PROJECTION/'freeze.json'),
        direction_postprocessing='无GT删帧/修复/额外平滑；117原始投影细化结点完整保留',
        source_sha256={str(p.resolve()):sha(p) for p in bound+[projected,PROJECTION/'freeze.json',OLD_LIGHT,Path(__file__)]})
    write(out/'lighting.json',lighting)
    used,total,util=map(int,subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    assert used<1024 and total-used>16000,'需要空闲GPU，禁止干预其他任务'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1',DYNAMIC_V3_REPO_ROOT=str(ROOT),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    state=dict(status='PREPARED',supervisor_pid=os.getpid(),hostname=socket.gethostname(),gpu=args.gpu,
        started_unix=time.time(),jobs={},fixed_updates=8100,lighting_sha256=sha(out/'lighting.json'),
        source_code_sha256=sha(__file__),snapshot_manifest_sha256=sha(snapshot/'snapshot_manifest.json'),
        gt_supervision=False,checkpoint_selection='唯一预先固定8100终点，不按DEV/TEST更换')
    def persist():
        state['updated_unix']=time.time();write(out/'state.json',state)
    persist()
    def command(module,params,target=snapshot):
        return [PYTHON,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(target),module]+list(map(str,params))
    train=command('LIHAN.scripts.dynamic_v3_train',['--source',source,'--parent',PARENT,'--parent-iteration',35000,
        '--lighting',out/'lighting.json','--stats',stats,'--output',out,'--path','A','--direction-lr',1e-5,'--ps'])
    dev=command('LIHAN.scripts.dynamic_v3_dev',['score','--source',CONTROL/'data_dev_v1','--model',out/'model_mlp',
        '--kind','stage2','--iteration',8100,'--output',out/'dev_8100_v1'],ROOT)
    freeze_command=[PYTHON,'-m','LIHAN.scripts.dynamic_v3_evaluate','--action','freeze',
        '--source',str(CONTROL/'eval_only/unified_source_v3_v3/source.json'),'--model',str(out/'model_mlp'),
        '--iteration','8100','--stage','stage2','--output',str(out/'eval_only/freeze_v1'),'--confirm-frozen',
        '--track-protocol','persistent_vertices_v1','--correspondence',str(CONTROL/'eval_only/1019-03-新动态GT持久顶点对应审计/correspondence.json')]
    evaluation=command('LIHAN.scripts.dynamic_v3_evaluate',['--action','run','--freeze',out/'eval_only/freeze_v1/freeze.json',
        '--output',out/'eval_only/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-投影灯向Stage2统一评价'],ROOT)
    commands=dict(train=train,dev=dev,freeze=freeze_command,evaluation=evaluation)
    write(out/'commands.json',commands)
    readme=out/'README.md'
    readme.write_text('# 投影灯向Stage2同父重训\n\n'
        '从N00 Stage1@35000重新初始化，A+PS、共享SH、8100新增更新，灯向lr=1e-5自1351开放。'
        '只替换初始方向，光强初值沿用1.448094367980957；不修复已知灯向异常结点。\n\n'
        'source：`'+str(source)+'`；父model：`'+str(PARENT)+'`；iteration=35000。'
        '本次model：`model_mlp`；固定终点iteration=8100。\n\n'
        '完整训练/DEV/冻结/渲染命令（仓库根目录，环境CUDA_VISIBLE_DEVICES='+str(args.gpu)+
        '，DYNAMIC_V3_REPO_ROOT='+str(ROOT)+'，OMP/MKL/OPENBLAS/NUMEXPR线程均4）：\n\n'+
        '\n\n'.join('```bash\n'+shlex.join(cmd)+'\n```' for cmd in commands.values())+
        '\n\n日志：supervisor.log、train.log、dev.log、freeze.log、evaluation.log。'
        'TRAIN更新记录：train_audit.jsonl；PID/状态/退出码：state.json。\n\n'
        '渲染目录：`dev_8100_v1/renders`与`eval_only/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-投影灯向Stage2统一评价`。'
        '量化指标/代表图/视频：尚未生成，完成后由终点产物更新。'
        '四类RGB、法线、Depth-Mesh、静动态目检均PENDING，不能由启动推断图像质量。\n\n'
        '**最终结论：PENDING。训练、评价或科学失败均保留checkpoint、图像、统计、日志，不覆盖旧结果。**\n')
    def job(name,cmd,expected):
        state['status']='RUNNING_'+name.upper();persist()
        record=dict(command=cmd,shell=shlex.join(cmd),started_unix=time.time(),status='STARTING')
        state['jobs'][name]=record;persist()
        with (out/(name+'.log')).open('x') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record.update(status='RUNNING',pid=process.pid);persist()
            code=process.wait()
        record.update(exit_code=code,finished_unix=time.time(),status='PROCESS_FINISHED' if code==0 else 'FAILED');persist()
        with readme.open('a') as stream:stream.write('\n'+name+'退出码：'+str(code)+'。\n')
        assert code==0 and Path(expected).is_file(),name+'失败，保留现场，禁止覆盖'
    try:
        job('train',train,out/'training_finished.json')
        marker=read(out/'training_finished.json');assert marker['actual_optimizer_updates']==8100
        assert marker['config']==read(ROOT/'output/1019-65-LH-dynamic-jumpingjacks-ELDL-V3-主线灯向微调同父对照/training_finished.json')['config']
        for relative,digest in marker['checkpoint_sha256'].items():assert sha(out/'model_mlp'/relative)==digest
        job('dev',dev,out/'dev_8100_v1/summary.json')
        job('freeze',freeze_command,out/'eval_only/freeze_v1/freeze.json')
        job('evaluation',evaluation,out/'eval_only/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-投影灯向Stage2统一评价/COMPLETED.json')
        state.update(status='READY_FOR_MANUAL_REVIEW',actual_optimizer_updates=8100,finished_unix=time.time(),scientific_result='PENDING_MANUAL_REVIEW');persist()
        with readme.open('a') as stream:stream.write('\n8100真实更新及DEV/冻结/统一评价完成，资产已保存。四类实际目检与最终科学PASS/FAILED待填写。\n')
    except Exception as error:
        state.update(status='FAILED',error=repr(error),finished_unix=time.time());persist()
        with readme.open('a') as stream:stream.write('\n最终结论：FAILED（监督流程异常），错误：'+repr(error)+'；全部现场保留。\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
    run(p.parse_args())
