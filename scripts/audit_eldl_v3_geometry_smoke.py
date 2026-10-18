"""纯几何转换冒烟的零更新、更新后和实际相机顺序独立验收。"""
import argparse,json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,m):
    if not v:raise ValueError(m)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiments',type=Path,nargs=2,required=True);args=parser.parse_args()
    camera_orders=[]
    for experiment in args.experiments:
        e=experiment.resolve();m=e/'model_mlp';state=read(e/'state.json');initial=read(m/'geometry_initialization.json');forward=read(m/'conversion_forward_audit.json');lr=read(e/'completed_lr_audit.json');finish=read(m/'training_finished.json')
        require('smoke_test' in e.parts and state['exit_code']==0 and finish['actual_updates']==20,'仅验收完整20步冒烟。')
        require(forward['status']=='PASS_LIVE_VS_PRODUCTION_RELOAD' and len(forward['rows'])==6 and forward['rng_unchanged'],'真实生产重载未通过。')
        require(lr['status']=='PASS_COMPLETED_LR_EXECUTION_AUDIT' and lr['budget']==20,'实际学习率未验收。')
        zero=torch.load(m/'optimizer_0.pt',map_location='cpu',weights_only=False);last=torch.load(m/'optimizer_20.pt',map_location='cpu',weights_only=False)
        require(zero['iteration']==0 and not zero['gaussian_optimizer']['state'] and not zero['light_optimizer']['state'] and not zero['camera_stack'],'零步须空Adam/相机栈。')
        require(all(not v.any() for v in zero['densification'].values()),'零步增密统计非零。')
        final_groups={g['name']:g for g in last['gaussian_optimizer']['param_groups']};states=last['gaussian_optimizer']['state'];active={'xyz','albedo_dc','opacity','scaling','rotation'};steps={}
        for name,group in final_groups.items():
            ids=group['params'];require(len(ids)==1,'多参数组不符。');entry=states.get(ids[0])
            if name in active:
                require(entry is not None and int(entry['step'])==20 and all(torch.isfinite(v).all() for v in entry.values() if torch.is_tensor(v)),'活跃组20实际更新未完成。');steps[name]=int(entry['step'])
            else:require(entry is None,'非活跃组发生更新。')
        require(not last['light_optimizer']['state'],'固定光照optimizer意外更新。')
        start_mlp=torch.load(m/'deform/iteration_0/deform.pth',map_location='cpu',weights_only=True);last_mlp=torch.load(m/'deform/iteration_20/deform.pth',map_location='cpu',weights_only=True)
        require(start_mlp.keys()==last_mlp.keys() and all(torch.equal(v,last_mlp[k]) for k,v in start_mlp.items()),'MLP发生更新。')
        audits=list((m/'static_dl_gradient_audit').glob('*.json'));require(len(audits)>=3,'真实物理梯度分支不足。')
        require(read(m/'normal_opacity_audit.json')['opacity_gradient_absent'],'normal透明度保护失败。')
        order=[read_line for read_line in map(json.loads,(m/'first_cycle_cameras.jsonl').read_text().splitlines())]
        require([x['physical_step'] for x in order]==list(range(1,21)),'相机顺序日志缺步。');camera_orders.append(order)
        files=[e/'plan.json',e/'state.json',m/'geometry_initialization.json',m/'conversion_forward_audit.json',e/'completed_lr_audit.json',m/'optimizer_0.pt',m/'optimizer_20.pt',m/'first_cycle_cameras.jsonl',m/'normal_opacity_audit.json',*audits,Path(__file__).resolve()]
        result=dict(status='PASS_PURE_GEOMETRY_PHYSICAL_ENGINEERING',scope='仅工程，不是法线/深度准确性或正式父模型验收',points=initial['points'],parent_step=initial['geometry_binding']['geometry_step'],actual_physical_updates=20,active_group_steps=steps,zero_adam_empty=True,inactive_and_mlp_frozen=True,light_optimizer_empty=True,production_reload_views=forward['rows'],gt_read=False,external_test_read=False,accuracy_claim=False,input_sha256={str(p):sha(p) for p in files})
        target=e/'engineering_receipt.json';require(not target.exists(),'禁止覆盖验收。');target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    require(camera_orders[0]==camera_orders[1],'同seed转换后实际相机序列不同。')
    for e in args.experiments:
        with (e/'README.md').open('a') as f:f.write('\n## 最终工程验收\n\n**PASS（工程）**：20实际物理更新、五活跃组Adam step均20，非活跃组/MLP/灯光不更新；零步新Adam/增密统计/相机栈清空；三TRAIN视角无V/全BVH共6次live与生产重载逐像素核验通过。两组20条实际相机顺序相同，独立lr/保存/normal透明度保护通过。详见engineering_receipt.json与completed_lr_audit.json；无外部精度与四类视觉PASS声明。\n')
    print('PASS：两组纯几何物理转换、20实际步、生产重载、共同相机序列。')
if __name__=='__main__':main()
