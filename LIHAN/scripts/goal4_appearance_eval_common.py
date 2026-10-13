#!/usr/bin/env python3
"""独立外观评测来源门禁；不修改旧计量脚本。"""
import argparse, ast, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_eval_protocol import CONTROL, FRAMES, PYTHON, digest, read, verify
ENTRY = 'scripts/train_static_dl_appearance.py'
MAPS = ('render','render_linear','rend_alpha','rend_normal','surf_depth','surf_normal','rend_dist','radii','visibility_filter','mean_visibility','light_intensity')
BASE = ROOT/'output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp'
BASE_FILES = {
 'static_dl_training.json':'8e5d3c826a55b20a1510f3821e741463fea45624eda20fe092b3198f7c32c4c7',
 'reset_audit_training.json':'5180de352826dc791f3581e35238f99dbb7538fdeaacc2eed6480a4e201996d0',
 'reset_audit_finished.json':'8008a1185004668e4db77ad86f97d0726676cc737f70429977778a822bb81504',
 'point_cloud/iteration_16000/point_cloud.ply':'346cc983263fb5d6243b5b276e73a636e865de9be38662d9db6636c2bdd4fccc',
 'deform/iteration_16000/deform.pth':'343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0',
 'static_dl_lighting/iteration_16000/lighting.json':'85490b1dcce17306f4bf228a945e6950c89efbfa1fe071a1671523073ada3586'}

def destination(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'output') or path==ROOT/'output':raise AssertionError('结果必须独立output子目录。')
    if path.exists():raise FileExistsError('禁止覆盖，失败重试须另建目录：'+str(path))
    return path

def checked(path,sources,expected=None):
    path=Path(path).resolve(); value=digest(path)
    if expected is not None and value!=expected:raise AssertionError('SHA不符：'+str(path))
    sources[str(path)]=value;return value

def frozen_hashes(pc):
    result={}
    import torch
    for name,value in vars(pc).items():
        if isinstance(value,torch.Tensor) and value.is_leaf and name!='_albedo_dc':
            value=value.detach().cpu().contiguous();h=hashlib.sha256()
            h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes());result[name]=h.hexdigest()
    return result

def audit(model,iteration,require_runtime=True):
    model=Path(model).resolve();sources={}
    if not model.is_relative_to(ROOT/'output') or model.name!='model_mlp':raise AssertionError('必须output内真实model_mlp。')
    protocol=verify(read(CONTROL/'protocol.json'));checked(CONTROL/'protocol.json',sources)
    marker=read(model/'appearance_training.json');finished=read(model/'appearance_finished.json');static=read(model/'static_dl_training.json')
    args,cfg=marker['args'],marker['inherited_config']
    if (marker['actual_entrypoint']!=ENTRY or marker['format_version']!=1 or marker['base_iteration']!=16000
        or Path(args['init_model']).resolve()!=BASE or args['init_iteration']!=16000 or args['seed']!=1012
        or args['audit_only'] or marker['loss_model'] not in ('censored','mse') or args['loss_model']!=marker['loss_model']
        or marker['base_sources']!=BASE_FILES or marker['initial_sources']!={str(BASE/p):h for p,h in BASE_FILES.items()}
        or marker['initial_point_count']!=67081 or marker['train_images']!=135
        or marker['test_images_loaded'] or marker['forbidden_gt_supervision'] or marker['shared_deformation_forward_called']
        or not marker['geometry_heads_zero'] or not marker['geometry_frozen']
        or marker['densification'] or marker['pruning'] or marker['opacity_reset']
        or set(marker['optimizer_groups'])!={'albedo_dc','light'} or marker['prediction_upper_clamp']
        or args['smoke']!=marker['smoke'] or ('smoke_test' in model.parts)!=marker['smoke']):raise AssertionError('独立appearance训练边界不符。')
    if (static['actual_entrypoint']!=ENTRY or static['args']!=cfg or not static['geometry_frozen']
        or cfg['depth_ratio']!=1 or cfg['white_background'] or cfg['static_dl_mode']!='lambertian_visibility'
        or cfg['normal_depth_statistic']!='expected' or cfg['loss_model']!=marker['loss_model']
        or Path(cfg['model_path']).resolve()!=model or cfg['test_iterations']):raise AssertionError('static/inherited物理配置不一致。')
    prior=read(BASE/'reset_audit_training.json')
    if marker['training_inputs']!=prior['training_inputs'] or len(marker['training_inputs'])!=137:raise AssertionError('须严格原137TRAIN输入。')
    inherited=read(BASE/'static_dl_training.json')['args']
    allowed={'model_path','source_path','iterations','data_device','load2gpu_on_the_fly','loss_model','save_iterations','test_iterations','appearance_geometry_frozen','ambient_mode','ambient_fraction'}
    if (set(cfg)-set(inherited)-allowed or set(inherited)-set(cfg)
        or any(cfg[k]!=v for k,v in inherited.items() if k not in allowed)
        or Path(cfg['source_path']).resolve()!=Path(inherited['source_path']).resolve()
        or not cfg['appearance_geometry_frozen']):raise AssertionError('未经授权的物理/原配置改变。')
    before=marker['appearance_updates_before'];budget=marker['update_budget'];total=before+budget
    if (finished['status']!='TRAINING_FINISHED' or finished['actual_entrypoint']!=ENTRY
        or finished['steps']!=budget or finished['optimizer_updates']!=budget or args['steps']!=budget
        or finished['appearance_updates']!=total or finished['iteration']!=16000+total
        or finished['point_count']!=67081 or not finished['geometry_unchanged']
        or finished['initial_frozen_sha256']!=marker['initial_frozen_sha256']
        or finished['final_frozen_sha256']!=marker['initial_frozen_sha256']
        or any(finished[k] is not True for k in ('initial_sources_unchanged','training_inputs_unchanged','source_files_unchanged'))
        or finished['smoke']!=marker['smoke'] or finished['loss_model']!=marker['loss_model']):raise AssertionError('真实finished/更新/冻结证据不符。')
    saves={16000+before+n for n in args['save_steps']+[budget]}
    if iteration not in saves:raise AssertionError('未登记保存步不得评测。')
    for group in ('initial_sources','training_inputs','new_sources'):
        for p,h in marker[group].items():checked(p,sources,h)
    for relative,h in marker['source_hashes'].items():
        checked(ROOT/relative,sources,h)
        matches=[p for p in marker['new_sources'] if Path(p).as_posix().endswith('/'+relative)]
        if len(matches)!=1 or marker['new_sources'][matches[0]]!=h:raise AssertionError('真实新source与相对SHA不符。')
    node=ast.parse((model/'cfg_args').read_text(),mode='eval').body
    actual_cfg={v.arg:ast.literal_eval(v.value) for v in node.keywords}
    if actual_cfg!=cfg:raise AssertionError('cfg_args和marker不同。')
    initial=read(model/'initial_checkpoint_gradient_audit.json')
    if (initial['status']!='PASS' or initial['parameter_updates']!=0 or initial['frozen_sha256']!=marker['initial_frozen_sha256']
        or {k for k,v in initial['requires_grad'].items() if v}!={'_albedo_dc'}
        or not all(initial['zero_deform_heads'].values())):raise AssertionError('真实起点冻结审计失败。')
    for k in ('baseline_to_zero_ambient','before_to_frozen'):
        if initial[k]['status']!='PASS' or set(initial[k]['checks'])!=set(MAPS) or not all(x['passed'] and x['max_abs']==0 for x in initial[k]['checks'].values()):raise AssertionError('起点11map同义检查失败。')
    gradient=read(model/'first_training_gradient_audit.json')
    if (gradient['status']!='PASS' or not gradient['frozen_gradients_absent'] or set(gradient['allowed_optimizer_groups'])!={'albedo_dc','light'}
        or gradient['parameter_updates']!=0 or set(gradient['gradients'])!={'albedo_dc','light'}
        or not all(x['norm']>0 for x in gradient['gradients'].values())):raise AssertionError('真实首步允许梯度缺失。')
    meta=read(model/f'appearance_checkpoint/iteration_{iteration}.json')
    if (meta['status']!='SAVED' or meta['iteration']!=iteration or meta['appearance_updates']!=iteration-16000
        or meta['run_updates']!=iteration-16000-before or meta['frozen_sha256']!=marker['initial_frozen_sha256'] or meta['point_count']!=67081):raise AssertionError('保存点真实元数据不符。')
    checked(model/f'appearance_state/iteration_{iteration}.pth',sources,meta['state_sha256'])
    for p,h in meta['assets'].items():
        if not (model/p).resolve().is_relative_to(model):raise AssertionError('保存资产越界。')
        checked(model/p,sources,h)
    for label in ('initial','iteration_'+str(iteration)):
        forward=read(model/f'appearance_forward/{label}.json')
        if (forward['actual_entrypoint']!=ENTRY or forward['dataset_split']!='train' or forward['camera_index']!=0 or forward['parameter_updates']!=0
            or forward['maps']!=list(MAPS) or forward['frozen_sha256']!=marker['initial_frozen_sha256']):raise AssertionError('实际训练前向记录不符。')
        checked(model/f'appearance_forward/{label}.npz',sources,forward['npz_sha256']);checked(model/f'appearance_forward/{label}.json',sources)
    if digest(model/f'deform/iteration_{iteration}/deform.pth')!=BASE_FILES['deform/iteration_16000/deform.pth']:raise AssertionError('冻结deform资产改变。')
    commands=[p for p in (model.parent/'commands/train.json',model.parent/'commands.json') if p.is_file()]
    if len(commands)!=1:raise AssertionError('必须唯一实际启动命令。')
    command=read(commands[0]);argv=command['argv']
    if len(argv)<4 or Path(argv[0])!=Path(PYTHON) or Path(argv[1]).resolve()!=ROOT/'LIHAN/scripts/bootstrap_1011.py' or argv[3]!='scripts.train_static_dl_appearance':raise AssertionError('必须ushas原bootstrap独立appearance模块。')
    parser=argparse.ArgumentParser()
    for key in ('init_model','model_path','source_path','loss_model','resume_state'):parser.add_argument('--'+key)
    for key,default in (('init_iteration',16000),('seed',1012),('steps',None)):parser.add_argument('--'+key,type=int,default=default)
    parser.add_argument('--save_steps',type=int,nargs='+');parser.add_argument('--smoke',action='store_true');parser.add_argument('--audit_only',action='store_true')
    actual=vars(parser.parse_args(argv[4:]))
    for k,v in actual.items():
        expected=args[k]
        if k in ('init_model','model_path','source_path','resume_state'):
            if (v is None)!=(expected is None) or (v is not None and Path(v).resolve()!=Path(expected).resolve()):raise AssertionError('真实CLI与marker路径不同：'+k)
        elif v!=expected:raise AssertionError('真实CLI与marker不同：'+k)
    if marker['optimizer_state_restored']!=(args['resume_state'] is not None) or (args['resume_state'] is None and before!=0):raise AssertionError('恢复语义或累计更新不符。')
    if args['resume_state'] is not None:
        checked(args['resume_state'],sources)
        resume=read(model/'appearance_resume_audit.json')
        if resume['status']!='PASS' or resume['appearance_updates']!=before or resume['state_sha256']!=digest(args['resume_state']) or any(v!=before for v in resume['optimizer_steps'].values()):raise AssertionError('真实完整状态恢复未通过。')
        checked(model/'appearance_resume_audit.json',sources)
    snapshot=Path(argv[2]).resolve();manifest_path=snapshot/'snapshot_manifest.json';manifest=read(manifest_path)
    if not snapshot.is_relative_to(ROOT/'output') or manifest['status']!='READY' or Path(manifest['destination']).resolve()!=snapshot:raise AssertionError('真实snapshot未READY。')
    if command.get('snapshot_manifest_sha256')!=digest(manifest_path) or command.get('entry_sha256')!=manifest['file_sha256'][ENTRY]:raise AssertionError('实际命令未绑定snapshot/entry SHA。')
    for p,h in manifest['file_sha256'].items():checked(snapshot/p,sources,h)
    for p in marker['new_sources']:
        if not Path(p).resolve().is_relative_to(snapshot):raise AssertionError('实际训练source不在启动快照。')
    checked(commands[0],sources);checked(manifest_path,sources)
    if require_runtime:
        state=read(model.parent/'state.json');runtime=read(model.parent/'runtime_module_audit.json')
        shared=read(model.parent/'runtime_shared_library_audit.json');launch=read(model.parent/'launch_provenance.json');environment=read(model.parent/'child_environment_audit.json')
        capture_code=int((model.parent/'runtime_capture_exit_code.txt').read_text().strip())
        if (state['status']!='TRAINING_COMPLETED_PENDING_EVAL' or state['exit_code']!=0 or state['training_exit_code']!=0 or not state['gpu_verified']
            or state['argv']!=argv or state['command_sha256']!=digest(commands[0]) or state['launcher_error'] is not None
            or runtime['status']!='PASS_CAPTURED' or not runtime['process_alive_during_hash'] or runtime['actual_module']!='scripts.train_static_dl_appearance'
            or runtime['pid']!=state['pid'] or runtime['argv']!=argv
            or capture_code!=0 or shared['status']!='CAPTURED' or shared['expected_training_module']!='scripts.train_static_dl_appearance' or shared['training_exit_code']!=0
            or launch['host']!='ushas' or launch['conda_environment']!='lumimotion-ushas' or launch['argv']!=argv
            or launch['gpu_uuid']!=state['gpu_uuid'] or launch['gpu_physical']!=state['gpu_physical']
            or environment['pid']!=state['pid'] or environment['expected']!=environment['observed']
            or environment['expected']!=launch['child_env'] or launch['child_env']!=state['child_env']
            or state['child_env']['CUDA_VISIBLE_DEVICES']!=state['gpu_uuid']):raise AssertionError('实际退出/native/tracer/GPU/子环境证据不完整。')
        if not any(r[0]==str(state['pid']) and r[1]==state['gpu_uuid'] for r in state['actual_gpu_rows']):raise AssertionError('真实训练GPU PID行缺失。')
        captured=shared['snapshots']
        if not captured:raise AssertionError('真实存活期shared library映射为空。')
        from datetime import datetime
        start,end=datetime.fromisoformat(state['started_at_utc']),datetime.fromisoformat(state['ended_at_utc'])
        if not start<=datetime.fromisoformat(runtime['captured_at_utc'])<=end:raise AssertionError('native/tracer采集时点不在训练生存区间。')
        for snapshot_record in captured:
            if (snapshot_record['pid']!=state['pid'] or snapshot_record['start_ticks']!=state['start_ticks'] or snapshot_record['argv']!=argv
                or snapshot_record['expected_training_module']!='scripts.train_static_dl_appearance' or snapshot_record['gpu_uuid']!=state['gpu_uuid']
                or not any(r[0]==str(state['pid']) and r[1]==state['gpu_uuid'] for r in snapshot_record['gpu_rows'])
                or not start<=datetime.fromisoformat(snapshot_record['captured_at_utc'])<=end):raise AssertionError('shared library真实PID/时间/GPU不符。')
        if len(runtime['actual_runtime_modules'])!=2:raise AssertionError('实际native/tracer模块不齐。')
        for module_path,h in runtime['actual_runtime_modules'].items():checked(module_path,sources,h)
        if set(runtime['actual_runtime_modules'].values())!={'f9514476baec5854c88862404f266242e62b1f828304f541745260fb012d43f3','5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e'}:raise AssertionError('真实native/tracer与原冻结二进制不同。')
        checked(launch['native_tracer_capture_script'],sources,launch['native_tracer_capture_sha256'])
        proof=launch['snapshot_verification']
        if proof['manifest_sha256']!=digest(manifest_path) or proof['entry_sha256']!=manifest['file_sha256'][ENTRY] or Path(proof['snapshot']).resolve()!=snapshot:raise AssertionError('launcher实际快照绑定不符。')
        for p in ('state.json','runtime_module_audit.json','runtime_shared_library_audit.json','launch_provenance.json','child_environment_audit.json','runtime_capture_exit_code.txt'):checked(model.parent/p,sources)
    for p in ('appearance_training.json','appearance_finished.json','static_dl_training.json','cfg_args','initial_checkpoint_gradient_audit.json','first_training_gradient_audit.json',f'appearance_checkpoint/iteration_{iteration}.json'):
        checked(model/p,sources)
    return dict(status='PASS_CPU_SOURCE_AUDIT',model=str(model),iteration=iteration,marker=marker,sources=sources,protocol=protocol)

def unchanged(sources):
    for p,h in sources.items():
        if digest(p)!=h:raise AssertionError('评测期间来源改变：'+p)
