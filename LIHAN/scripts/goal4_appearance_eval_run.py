#!/usr/bin/env python3
"""外观新入口原15三MAE、离线分区、float RGB与150四类全流程。"""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal4_appearance_eval_common import ROOT,audit,destination,unchanged,checked
from goal_eval_protocol import CONTROL,FRAMES,PYTHON,read,write,verify,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fresh',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--gpu',type=int);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if subprocess.check_output(['hostname']).decode().strip()!='ushas':raise AssertionError('仅ushas/lumimotion-ushas。')
    if a.execute and a.gpu is None:p.error('执行必须显式已协调GPU。')
    out=destination(a.output);e=audit(a.model,a.iteration);sources=e['sources'];protocol=e['protocol'];fresh=read(a.fresh/'parity.json')
    if fresh['status']!='PASS' or fresh['iteration']!=a.iteration or fresh['point_count']!=67081 or fresh['parameter_updates']!=0:raise AssertionError('独立真实fresh尚未PASS。')
    unchanged(fresh['sources']);sources.update(fresh['sources']);checked(a.fresh/'parity.json',sources)
    for comparison in fresh['comparisons'].values():
        if comparison['status']!='PASS' or len(comparison['checks'])!=11 or not all(x['passed'] for x in comparison['checks'].values()):raise AssertionError('fresh11maps未完整通过。')
    reference=read(a.reference)
    if not reference['baseline'] or reference['iteration']!=16000 or reference['frames']!=FRAMES:raise AssertionError('须先同法完整15float基线。')
    unchanged(reference['sources'])
    checked(a.reference,sources);checked(Path(__file__),sources)
    shared=['--model_path',str(a.model.resolve()),'--source_path',protocol['source_path'],'--load_iter',str(a.iteration)]
    boot=[PYTHON,str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT)];frames=['--frames']+list(map(str,FRAMES))
    steps=[
      ('predicted_meshes',boot+['scripts.extract_geometry_mesh_dynamic']+shared+frames+['--split','test','--resolution','2','--voxel_size','.02','--sdf_trunc','.08','--depth_trunc','20','--alpha_threshold','.5','--output_dir',str(out/'predicted_meshes')]),
      ('normal_depth',boot+['scripts.eval_geometry_gt']+shared+frames+['--gt_root',protocol['gt_root'],'--gt_depth_dir',protocol['gt_depth_dir'],'--split','test','--resolution','2','--alpha_threshold','.5','--interior_radius','2','--output_dir',str(out/'normal_depth'),'--pred_mesh_manifest',str(out/'predicted_meshes/manifest.json'),'--save_arrays']),
      ('audit',[PYTHON,str(ROOT/'LIHAN/scripts/goal_eval_audit.py'),str(out),str(a.iteration),'--protocol',str(CONTROL/'protocol.json')]),
      ('spatial_diagnostic',[PYTHON,str(ROOT/'LIHAN/scripts/goal2_error_spatial_diagnostic.py'),'--evaluation',str(out),'--gt_depth_dir',protocol['gt_depth_dir'],'--output',str(out/'spatial_diagnostic'),'--figures']),
      ('rgb',boot+['LIHAN.scripts.goal4_appearance_eval_rgb']+shared+['--output_dir',str(out/'rgb'),'--reference',str(a.reference.resolve())]),
      ('four_classes',boot+['LIHAN.scripts.goal4_appearance_eval_visuals']+shared+['--output_dir',str(out/'four_classes')]),
      ('numeric',[PYTHON,str(ROOT/'LIHAN/scripts/goal4_appearance_eval_target.py'),'--evaluation',str(out),'--reference',str(a.reference.resolve()),'--output',str(out/'numeric')])]
    for name in ('goal4_appearance_eval_common.py','goal4_appearance_eval_render.py','goal4_appearance_eval_parity.py','goal4_appearance_eval_rgb.py','goal4_appearance_eval_visuals.py','goal4_appearance_eval_target.py','goal_eval_visuals.py','goal_eval_render.py','goal_eval_audit.py','goal_eval_protocol.py','goal2_error_spatial_diagnostic.py'):
        checked(ROOT/'LIHAN/scripts'/name,sources)
    out.mkdir(parents=True);(out/'logs').mkdir()
    plan=dict(model=str(a.model.resolve()),iteration=a.iteration,source=protocol['source_path'],gpu=a.gpu,protocol_sha256=digest(CONTROL/'protocol.json'),sources=sources,
        steps=[dict(name=n,command=c,shell=shlex.join(c)) for n,c in steps],frozen_normal_baseline={'gs_render':2.6683778383745302,'depth_normal':5.404577799172339,'mesh':4.20848728697561},repeat_tolerance_deg=1e-5)
    write(out/'plan.json',plan);write(out/'protocol.json',protocol)
    lines=['# 独立外观阶段完整评测计划','',f'source：{protocol["source_path"]}；model：{a.model.resolve()}；iteration：{a.iteration}。','原完整15帧和六固定源码、原掩膜、像素权重、TSDF不变；GT只离线。freeze新入口门禁独立，旧foot阈值不用于本轮。','',f'输出：{out}；日志：{out / "logs"}。代表图/四视频待four_classes产生；RGB原float为rgb/arrays/NNNN.npz。','']
    for name,command in steps:lines += ['## '+name,'','```bash',shlex.join(command),'```','']
    lines += ['三MAE精确值及strict不退化/工程tol双报；PSNR同法float clipped全图均值提升>0.05dB，前景辅助Δ≥−0.05dB；raw并列。四类与脚/黑边人工目检PENDING，最终PASS/FAILED由真实终审追加，数值完成不等于完整足形成功。','']
    (out/'README.md').write_text('\n'.join(lines))
    if not a.execute:print('评测计划已保存：'+str(out/'plan.json'));return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',TORCH_CUDA_ARCH_LIST='8.6')
    env['LD_LIBRARY_PATH']=str(ROOT/'output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    outcomes=[]
    for name,command in steps:
        verify(protocol);unchanged(sources);log=out/'logs'/f'{name}.log'
        with log.open('x') as stream:
            stream.write(shlex.join(command)+'\n');stream.flush();code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        outcomes.append(dict(step=name,exit_code=code,command=command,log=str(log)));write(out/'executed.json',outcomes);print(name+' exit='+str(code),flush=True)
        if code:write(out/'FAILED.json',dict(step=name,exit_code=code,retained=True));raise SystemExit(code)
    unchanged(sources);write(out/'PIPELINE_COMPLETED.json',dict(status='PASS_PIPELINE',scientific_acceptance='PENDING_VISUAL',iteration=a.iteration,steps=outcomes,sources=sources))
if __name__=='__main__':main()
