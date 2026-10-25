"""静态主实验收到SIGTERM后，从保留的12k检查点恢复颜色与环境。"""
import hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
from LIHAN.scripts.stage3_el_eldl_four import ROOT,CONTROL,PYTHON,env,write
OUTPUT=ROOT/'output/1026-03-LH-static-ELDL-GT方向光-Stage3保存点续训'

def main():
 base=ROOT/'output/1025-35-LH-static-ELDL-GT方向光-Stage3外观';old=json.loads((base/'state.json').read_text());snapshot=CONTROL/'snapshot_resume'
 shutil.copytree(CONTROL/'snapshot',snapshot,symlinks=True)
 helper=snapshot/'LIHAN/scripts/stage3_el_eldl_resume_attach.py'
 helper.write_text('''"""恢复已保存PBR环境与强度，优化器状态未被旧训练保存，重新建立。"""
import json
from pathlib import Path
import torch
from LIHAN.scripts.stage3_el_eldl_mixed import attach as mixed_attach
def attach(tr):
    mixed_attach(tr)
    source=Path(tr.args.stage1_model_path)
    it=tr.load_iter
    data=torch.load(source/f"envmap/iteration_{it}/envmap.pth",map_location="cpu",weights_only=False)
    tr.env_light.load_weights(str(source),it)
    assert all(torch.equal(v.detach().cpu(),data["state_dict"][k]) for k,v in tr.env_light.state_dict().items())
    (Path(tr.args.model_path)/"stage3_resume_audit.json").write_text(json.dumps({"source":str(source),"iteration":it,"environment_and_intensity_equal":True,"optimizer_restored":False,"diffuse_only_warmup_restarted":False,"test_read":False},ensure_ascii=False,indent=2)+"\\n")
''')
 p=snapshot/'scripts/train_stage2.py';s=p.read_text();s=s.replace('from LIHAN.scripts.stage3_el_eldl_mixed import attach','from LIHAN.scripts.stage3_el_eldl_resume_attach import attach').replace('self.iters_only_diffuse = 2000','self.iters_only_diffuse = 0');p.write_text(s)
 output=OUTPUT;output.mkdir(exist_ok=False);cmd=list(old['command']);cmd[1]=str(snapshot/'LIHAN/scripts/bootstrap_1011.py');cmd[2]=str(snapshot)
 cmd[cmd.index('--model_path')+1]=str(output/'model_mlp');cmd[cmd.index('--stage1_model_path')+1]=str(base/'model_mlp');cmd[cmd.index('--load_iter')+1]='20100';cmd[cmd.index('--iterations')+1]='28100'
 i=cmd.index('--save_iterations')+1;j=cmd.index('--is_blender');cmd[i:j]=['24100','28100']
 state={'arm':'static_ELDL','status':'RUNNING','command':cmd,'iteration':28100,'parent_iteration':20100,'lineage_parent_iteration':8100,'lineage_loops':20000,'steps':8000,'optimizer_updates_phase':1999,'optimizer_updates_lineage':4998,'gpu':1,'started':time.time(),'resume_parent':str(base),'original_parent':'G9@8100','optimizer_restarted':True,'unsaved_lost_loops':6393}
 write(output/'state.json',state)
 (output/'README.md').write_text('# 静态GT方向光ELDL的Stage3中断恢复\n\n原主实验收到SIGTERM退出143，保留现场。从原12k检查点@20100恢复，继续8000循环至28100；原始G9@8100父保持。环境、SUN强度、颜色、粗糙度、opacity、MLP从保存点继承，旧代码未保存optimizer状态，因此重新建立Adam；不重启2000步diffuse-only。\n\n可追溯权重链12k＋8k=20k循环，真实更新2999＋1999=4998次（比未中断组少1次）；另有中断前6393步未保存，不能算入最终权重链，额外计算开销如实记录。\n\n完整命令：\n```bash\n'+shlex.join(cmd)+'\n```\n\n日志train.log；渲染与最终指标待完成；当前PENDING。\n')
 write(CONTROL/'resume_manifest.json',{'snapshot':str(snapshot),'code_sha256':{str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*.py')},'signal':143,'optimizer_state_available':False})
 with (output/'train.log').open('x') as f:
  child=subprocess.Popen(cmd,cwd=snapshot,env=env(1),stdout=f,stderr=subprocess.STDOUT);state['pid']=child.pid;write(output/'state.json',state);code=child.wait()
 state.update(status='TRAIN_COMPLETED' if code==0 else 'FAILED',exit_code=code,finished=time.time());write(output/'state.json',state);return code
if __name__=='__main__':raise SystemExit(main())
