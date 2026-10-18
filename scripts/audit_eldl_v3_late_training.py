"""只读核验已完成晚期步长训练的实际学习率、保存状态和固定灯光。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from LIHAN.scripts.eldl_v3_model_record import fixed_eldl_record_files
from utils.general_utils import get_expon_lr_func


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def audit(experiment,output):
    require(not output.exists(),'禁止覆盖已有审计。')
    model=experiment/'model_mlp';state=read(experiment/'state.json')
    require(state['exit_code']==0,'仅审计完整正常终止的实验。')
    marker=read(model/'eldl_v3_training.json');late=read(model/'late_lr_training.json')
    extra=fixed_eldl_record_files(model,marker)
    require(extra is not None and extra,'需要严格登记的晚期入口。')
    args=marker['args'];budget=args['iterations'];start,end=late['start'],late['end']
    active=set(late['active_groups']);base=late['original_lrs'];enabled=late['case']=='anneal'
    # 独立写出预登记数学式，不调用训练乘子函数。
    def factor(k):return 10**(-max(0,min(1,(k-start)/(end-start)))) if enabled else 1.
    scale=base['xyz']/args['position_lr_init']
    xyz=get_expon_lr_func(lr_init=base['xyz'],lr_final=args['position_lr_final']*scale,
                         lr_delay_mult=args['position_lr_delay_mult'],max_steps=args['position_lr_max_steps'])
    def expected(k):
        nominal=dict(base);nominal['xyz']=base['xyz'] if k==1 else xyz(k-1)
        return {name:value*(factor(k) if name in active else 1.) for name,value in nominal.items()}
    def same(actual,wanted):
        require(actual.keys()==wanted.keys(),'学习率组集合不同。')
        require(all(math.isclose(actual[k],wanted[k],rel_tol=1e-12,abs_tol=1e-16) for k in wanted),'实际学习率与独立公式不同。')
    events=[json.loads(line) for line in (model/'late_lr_events.jsonl').read_text().splitlines()]
    steps=[row['iteration'] for row in events]
    require(steps==sorted(set(steps)) and steps[-1]==budget,'事件顺序/终点缺失。')
    require(set((start-1,start,start+1,end-1,end,end+1,budget)).issubset(steps),'关键边界日志缺失。')
    for row in events:
        k=row['iteration'];post=row['post_update_scheduler']
        require(row['applied_multiplier']==factor(k) and post['multiplier']==factor(k+1),'本步/下一步乘子不同。')
        require(post['completed_step']==k and post['next_step']==k+1,'下一步编号错误。')
        same(row['applied_lrs'],expected(k));same(post['next_lrs'],expected(k+1))
        require(all(x==0 for x in row['light_lrs']),'光照学习率非零。')
    files=[experiment/'state.json',experiment/'plan.json',model/'eldl_v3_training.json',model/'late_lr_events.jsonl',*extra,Path(__file__).resolve()]
    saved=[]
    for k in sorted(set(args['save_iterations'])):
        if k>budget:continue
        opt=model/f'optimizer_{k}.pt';meta_path=model/f'static_dl_lighting/iteration_{k}/lighting.json'
        data=torch.load(opt,map_location='cpu',weights_only=False);meta=read(meta_path)
        require(data['iteration']==meta['iteration']==k,'保存步数不符。')
        require(meta['frozen_light_check']=='PASS' and meta['lighting_frozen'] and meta['gt_supervision'] is False,'固定无GT灯光检查失败。')
        post=meta['late_lr_next_step'];require(post['completed_step']==k and post['next_step']==k+1,'保存下一步编号不符。')
        same(post['next_lrs'],expected(k+1))
        groups=data['gaussian_optimizer']['param_groups'];same({g['name']:g['lr'] for g in groups},expected(k+1))
        require(all(g['lr']==0 for g in data['light_optimizer']['param_groups']),'保存光照学习率非零。')
        files.extend([opt,meta_path]);saved.append(k)
    require(budget in saved,'缺少真实预算终点。')
    receipt=dict(status='PASS_COMPLETED_LR_EXECUTION_AUDIT',case=late['case'],budget=budget,
                 event_steps=steps,saved_iterations=saved,gt_read=False,external_test_read=False,
                 accuracy_claim=False,input_sha256={str(p):sha(p) for p in files})
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='input_sha256'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.experiment.resolve(),a.output.resolve())
