"""固定旧EL+DL资产的24k晚期步长对照入口；其余物理/损失/拓扑保持原实现。"""
import hashlib
import json
from pathlib import Path
from scene.eldl_v3_trainer import ELDLTrainer
from utils.eldl_v3_late_lr import LateLearningRate, ACTIVE_GROUPS
from scripts import train_eldl_v3 as entry


class LateAnnealTrainer(ELDLTrainer):
    def __init__(self,*args,**kwargs):
        config=kwargs['args']
        if not config.smoke_schedule and config.iterations!=24000:
            raise ValueError('本次预登记固定24000实际更新。')
        enabled=config.late_lr_case=='anneal'
        config.late_lr_schedule=('16001到20000指数下降至0.1，后续保持；smoke8到16' if enabled else '同代码control，乘子恒1，保留原日程')
        super().__init__(*args,**kwargs)
        self.late_start,self.late_end=(8,16) if self.args.smoke_schedule else (16000,20000)
        self.lr_policy=LateLearningRate(self.gaussians,self.late_start,self.late_end,enabled=enabled)
        self.gaussians.update_learning_rate=self.lr_policy
        root=Path(__file__).resolve().parents[1];model=Path(self.args.model_path)
        sources={str(root/p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                 ('scripts/train_eldl_v3_late_anneal.py','utils/eldl_v3_late_lr.py')}
        record=dict(actual_entrypoint='scripts/train_eldl_v3_late_anneal.py',start=self.late_start,end=self.late_end,
            case=config.late_lr_case,final_multiplier=.1 if enabled else 1.,active_groups=ACTIVE_GROUPS,original_lrs=self.lr_policy.base,
            semantics='原本步step后XYZ scheduler保持；乘子作用下一次更新的lr；非XYZ每次从固定base算，不累乘',
            adam_moments_changed=False,lighting_unchanged=True,loss_unchanged=True,topology_policy_unchanged=True,
            gt_supervision=False,sources=sources)
        (model/'late_lr_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        for name in ('eldl_v3_training.json','unknown_light_training.json','static_dl_training.json'):
            path=model/name;meta=json.loads(path.read_text());meta['actual_entrypoint']=record['actual_entrypoint']
            meta['late_learning_rate']=record
            if 'new_sources' in meta:meta['new_sources'].update(sources)
            path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')

    def train_step(self):
        current=self.iteration
        before={g['name']:float(g['lr']) for g in self.gaussians.optimizer.param_groups}
        factor=self.lr_policy.factor(current)
        if current>self.late_start:
            assert self.lr_policy.last['next_step']==current
            assert before==self.lr_policy.last['next_lrs']
        super().train_step()
        if self.args.smoke_schedule or current%1000==0 or current in (self.late_start-1,self.late_start+1,self.late_end-1,self.late_end+1):
            row=dict(iteration=current,applied_multiplier=factor,applied_lrs=before,
                post_update_scheduler=self.lr_policy.last,light_lrs=[g['lr'] for g in self.light_optimizer.param_groups])
            with (Path(self.args.model_path)/'late_lr_events.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')

    def _save_physical(self):
        assert self.lr_policy.last['completed_step']==self.iteration
        assert self.lr_policy.last['next_step']==self.iteration+1
        assert self.lr_policy.last['next_lrs']=={g['name']:float(g['lr']) for g in self.gaussians.optimizer.param_groups}
        super()._save_physical()
        path=Path(self.args.model_path)/f'static_dl_lighting/iteration_{self.iteration}/lighting.json'
        meta=json.loads(path.read_text());meta['actual_entrypoint']='scripts/train_eldl_v3_late_anneal.py'
        meta['late_lr_next_step']=self.lr_policy.last
        path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')


def main():
    original_parser,original_trainer=entry.ArgumentParser,entry.ELDLTrainer
    class LateParser(original_parser):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.add_argument('--late_lr_case',choices=('control','anneal'),default='anneal',
                              help='同代码固定乘子控制或16k起统一退火；不改变其他训练策略。')
    entry.ArgumentParser,entry.ELDLTrainer=LateParser,LateAnnealTrainer
    try:
        entry.main()
    finally:
        entry.ArgumentParser,entry.ELDLTrainer=original_parser,original_trainer


if __name__=='__main__':main()
