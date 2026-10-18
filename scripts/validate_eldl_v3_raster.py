"""TRAIN全图冻结终点独立验收；仅显式统一epoch和fit初值后才运行held。"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import shlex
import socket
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts import calibrate_eldl_v3_raster as forward

NAMES=('init_1','init_2','init_3','constant')
RATIOS=(1/3,1.,3.,1.)
DOC=ROOT/'LIHAN/DOC/报告/1019-24-全图光度冻结终点验收协议.md'
RULES=dict(bias_scales=[.5,1.,2.],max_relative_held_degradation=.25,
    all_scales_strictly_better_than_constant=True,all_fit_frames=117,all_held_frames=18,
    foreground='原输入AREA后Alpha>0.5，固定完整分母×3',
    objective='linear_censored_mse',forward_absolute_tolerance=2e-5,
    material_roundtrip_tolerance=3e-7,geometry_tolerance=0.,
    exact_visibility_cache_tolerance=2e-6,checkpoint_fit_mean_tolerance=1e-6,
    environment_over_pi_interior=[.0101,29.9],budget_improvement_threshold=.01,
    v1_budget='50到60完整fit正向改善不得大于1%',
    v2_budget='25到30完整fit绝对相对变化不得大于1%',
    fixed_fit_representatives=[1,81,149],fixed_held_representatives=[61,71,79],
    direction_profile='PENDING_SEPARATE_PROTOCOL',internal_validation_passed=False)
ANNEAL=[dict(first=1,last=10,lr=.003),dict(first=11,last=20,lr=.001),dict(first=21,last=30,lr=.0001)]


def directory_for(calibration,name):
    choices=[path for path in (calibration/name,calibration/name.replace('init_','init')) if path.is_dir()]
    choices=list(dict.fromkeys(choices))
    if len(choices)!=1:raise ValueError('运行目录缺失或命名冲突：'+name)
    return choices[0]


def same_tree(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(same_tree(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(same_tree(x,y) for x,y in zip(left,right))
    return left==right


def statistics(pred,target,saturated,foreground,environment=None):
    """分层计数在原观察上固定；完整前景永不按预测支持缩小。"""
    fg=foreground.bool().expand_as(target)
    sat=saturated.bool()&fg;unsat=(~saturated.bool())&fg
    delta=pred-target;censored=torch.where(saturated.bool(),delta.clamp_max(0),delta)
    result={}
    for name,mask in [('all',fg),('saturated',sat),('unsaturated',unsat),('background',~fg)]:
        n=int(mask.sum());s=float(censored[mask].double().square().sum());raw=float(delta[mask].double().square().sum())
        result[name]=dict(channels=n,squared_error=s,mse=s/n if n else None,raw_squared_error=raw,
                         raw_mse=raw/n if n else None,contribution_to_foreground_mse=s/int(fg.sum()) if name!='background' else None)
    if environment is not None:
        direct=pred-environment
        result['contributions']={name:dict(sum=float(value[fg].double().sum()),mean=float(value[fg].double().mean()),
            rgb_mean=[float(value[c][foreground[0]].double().mean()) for c in range(3)])
            for name,value in [('EL',environment),('DL',direct),('total',pred)]}
    return result


def aggregate(rows):
    result={}
    for split in ('fit','held'):
        chosen=[r for r in rows if r['split']==split];groups={}
        for group in ('all','saturated','unsaturated','background'):
            n=sum(r['statistics'][group]['channels'] for r in chosen)
            error=sum(r['statistics'][group]['squared_error'] for r in chosen)
            frame_values=[r['statistics'][group]['mse'] for r in chosen if r['statistics'][group]['channels']]
            groups[group]=dict(frame_mean=sum(frame_values)/len(frame_values) if frame_values else None,
                pixel_mean=error/n if n else None,channels=n,frames=len(frame_values),squared_error=error)
        contributions={}
        if chosen and 'contributions' in chosen[0]['statistics']:
            for kind in ('EL','DL','total'):
                contributions[kind]=dict(frame_mean=sum(r['statistics']['contributions'][kind]['mean'] for r in chosen)/len(chosen),
                    pixel_mean=sum(r['statistics']['contributions'][kind]['sum'] for r in chosen)/groups['all']['channels'])
        result[split]=dict(frames=len(chosen),groups=groups,contributions=contributions)
    return result


def decide(metrics,constant,environment,budget_ok):
    center=metrics['1']['held']['groups']['all']['frame_mean'];baseline=constant['held']['groups']['all']['frame_mean']
    checks=dict(budget_sufficient=budget_ok,all_fit_supported=metrics['1']['fit']['frames']==117,
        all_held_supported=metrics['1']['held']['frames']==18,
        environment_not_bound=bool(((environment/math.pi>.0101)&(environment/math.pi<29.9)).all()))
    relative={}
    for key in ('0p5','1','2'):
        value=metrics[key]['held']['groups']['all']['frame_mean']
        checks[key+'_strictly_beats_constant']=value<baseline
        if key!='1':
            relative[key]=(value-center)/max(center,1e-8)
            checks[key+'_degradation_within_25pct']=relative[key]<=.25
    return checks,relative


def inputs(a):
    """CPU阶段验证统一预算、源、mask及仅fit选择，不计算held损失。"""
    from utils.no_gt_guard import install_no_gt_guard
    configs_before={name:v.read(directory_for(a.calibration,name)/'configuration.json') for name in NAMES}
    parent_paths=[]
    for name,config in configs_before.items():
        if 'parent_checkpoint' in config:
            parent=Path(config['parent_checkpoint']).resolve()
            if parent.parent.parent!=a.calibration.parent/'raster_calibration_v1' or parent.parent.name not in (name,name.replace('init_','init')):
                raise ValueError('退火父checkpoint不在明确的同批v1运行。')
            parent_paths.append(parent.parent)
    install_no_gt_guard([a.calibration,a.raster_cache.parent,a.cache.parent,a.fit,a.model,a.source,a.output,*parent_paths])
    data,state,old_summary,old_result,hashes=v.load_inputs(a)
    meta=v.read(a.raster_cache.parent/'summary.json')
    if meta['gt_read'] or meta['status']!='PASS_FORWARD_CACHE_NOT_LIGHT_ACCEPTANCE' or v.sha(a.raster_cache)!=meta['cache_sha256']:
        raise ValueError('全图缓存来源失败。')
    packed=torch.load(a.raster_cache,map_location='cpu',weights_only=False)
    if packed['frames']!=v.TRAIN or not torch.equal(packed['directions'],state['directions']) or packed['bias']!=data['shadow_bias'] or packed['scales']!=RULES['bias_scales']:
        raise ValueError('全图缓存方向/时序/bias不一致。')
    hashes[str(a.raster_cache)]=v.sha(a.raster_cache)
    hashes[str(a.raster_cache.parent/'summary.json')]=v.sha(a.raster_cache.parent/'summary.json')
    checkpoints={};configs={};scores={};budgets={};masks=None;first=None;first_rng=None;first_schedule=None;first_parent_rng=None
    for name,ratio in zip(NAMES,RATIOS):
        directory=directory_for(a.calibration,name);config=configs_before[name];annealed='parent_checkpoint' in config
        complete=v.read(directory/f'train_epoch_{a.epoch}.json')
        checkpoint_path=directory/f'checkpoint_epoch_{a.epoch}.pt'
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        fit=v.read(directory/f'fit_epoch_{a.epoch}.json')
        if config['gt_read'] or config['external_test_read'] or not config['geometry_frozen'] or not config['directions_frozen']:
            raise ValueError('非冻结TRAIN校准。')
        if config['geometry_fit_frames']!=v.FIT or config['held_frames']!=v.HELD or config['loss']!='linear_censored_mse_original_alpha_gt_0p5_fixed_denominator':
            raise ValueError('校准分母/划分/损失不一致。')
        if checkpoint['epoch']!=a.epoch or complete['epochs']!=a.epoch or len(checkpoint['history'])!=a.epoch or not complete['geometry_bitexact_unchanged']:
            raise ValueError('统一终点尚未完整完成。')
        if checkpoint['constant']!=(name=='constant') or checkpoint['initial_ratio']!=ratio or not torch.equal(checkpoint['directions'],state['directions']):
            raise ValueError('checkpoint身份/固定方向不一致。')
        if [r['epoch'] for r in checkpoint['history']]!=list(range(1,a.epoch+1)) or any(r['updates']!=r['epoch']*117 for r in checkpoint['history']):
            raise ValueError('图像更新预算不一致。')
        if first_rng is None:first_rng=checkpoint['rng']
        elif checkpoint['rng']!=first_rng:raise ValueError('四条无放回采样RNG终态不同。')
        actual_updates=(60+a.epoch)*117 if annealed else a.epoch*117
        if any(int(item['step'])!=actual_updates for item in checkpoint['optimizer']['state'].values()):
            raise ValueError('实际Adam更新数与预算不同。')
        if annealed:
            if a.epoch!=30 or config['optimizer_schedule']!=ANNEAL or config['optimizer']!='Adam' or config['seed']!='继承v1@60的相同RNG状态':
                raise ValueError('退火预登记日程/seed不同。')
            if config['cumulative_epochs']!=90 or config['cumulative_updates']!=10530 or checkpoint['cumulative_epoch']!=90 or complete['cumulative_epochs']!=90:
                raise ValueError('退火累计预算不同。')
            if config['parent_status']!='OPTIMIZATION_BUDGET_INSUFFICIENT':raise ValueError('父失败状态被改写。')
            parent_path=Path(config['parent_checkpoint']);parent=torch.load(parent_path,map_location='cpu',weights_only=False)
            if v.sha(parent_path)!=config['parent_checkpoint_sha256'] or parent['epoch']!=60 or parent['constant']!=(name=='constant') or parent['initial_ratio']!=ratio:
                raise ValueError('退火父checkpoint身份/SHA不同。')
            if not torch.equal(parent['directions'],state['directions']):raise ValueError('退火改变了冻结方向。')
            if first_parent_rng is None:first_parent_rng=parent['rng']
            elif first_parent_rng!=parent['rng']:raise ValueError('四条父RNG不同。')
            zero_path=directory/'checkpoint_epoch_0.pt';zero=torch.load(zero_path,map_location='cpu',weights_only=False)
            if not torch.equal(parent['rho'],zero['rho']) or not torch.allclose(parent['environment'],zero['environment'],rtol=2e-7,atol=1e-7) or zero['rng']!=parent['rng']:
                raise ValueError('退火起点没有完整继承父参数/RNG。')
            if not same_tree(parent['optimizer'],zero['optimizer']):raise ValueError('退火重置了父Adam状态。')
            if 'log_e' not in checkpoint or not torch.allclose(checkpoint['log_e'].exp(),checkpoint['environment'],rtol=2e-7,atol=1e-7):
                raise ValueError('没有保存真实log EL或与环境不符。')
            restart=v.read(directory/'restart_parity.json')
            if restart['status']!='PASS_FIT_FORWARD_RESTART' or restart['max_per_frame_mse_difference']>1e-7:raise ValueError('重启前向不一致。')
            for row in checkpoint['history']:
                lr=.003 if row['epoch']<=10 else (.001 if row['epoch']<=20 else .0001)
                if row['lr']!=lr or any(value!=lr for value in row['param_group_lrs']) or row['cumulative_epoch']!=60+row['epoch']:
                    raise ValueError('实际退火学习率历史不符。')
            for path in (parent_path,zero_path,directory/'restart_parity.json'):hashes[str(path)]=v.sha(path)
        elif config['optimizer']!='Adam' or config['lr']!=.03 or config['seed']!=1019 or config['rho_initial']!=.5:
            raise ValueError('v1对照优化协议不同。')
        schedule=config['optimizer_schedule'] if 'optimizer_schedule' in config else dict(kind='constant',lr=config['lr'])
        if first_schedule is None:first_schedule=schedule
        elif schedule!=first_schedule:raise ValueError('四条预登记优化日程不同。')
        if first is not None:
            for key in ('input_sha256','backend_provenance','rho_linear_bounds','constant_color_bounds','environment_over_pi_bounds'):
                if config[key]!=first[key]:raise ValueError('四条来源/颜色域/库不同：'+key)
        else:first=config
        fixed_path=directory/'fixed_foreground.pt'
        if v.sha(fixed_path)!=config['foreground_sha256']:raise ValueError('固定mask SHA变化。')
        fixed=torch.load(fixed_path,map_location='cpu',weights_only=False)
        if fixed['frames']!=v.TRAIN or fixed['foreground'].dtype!=torch.bool or not bool(fixed['foreground'].flatten(1).any(1).all()):
            raise ValueError('固定mask无效。')
        if masks is None:masks=fixed['foreground']
        elif not torch.equal(masks,fixed['foreground']):raise ValueError('四条运行固定mask不同。')
        if not torch.isfinite(checkpoint['rho']).all() or not torch.isfinite(checkpoint['environment']).all() or bool((checkpoint['environment']<=0).any()):raise ValueError('参数非有限/EL非正。')
        lo,hi=(0.,30.) if name=='constant' else config['rho_linear_bounds']
        if bool(((checkpoint['rho']<lo-1e-7)|(checkpoint['rho']>hi+1e-7)).any()):raise ValueError('颜色超预登记域。')
        # 只有完整fit误差用于选择，held未计算。
        if len(fit['per_frame'])!=117 or not all(math.isfinite(x) for x in fit['per_frame']) or abs(sum(fit['per_frame'])/117-fit['frame_mean'])>1e-12:raise ValueError('完整fit摘要无效。')
        fit_counts=masks.flatten(1).sum(1)[torch.tensor([n in v.FIT for n in v.TRAIN])].double()
        expected_pixel_mean=float((torch.tensor(fit['per_frame'],dtype=torch.float64)*fit_counts).sum()/fit_counts.sum())
        if abs(fit['pixel_mean']-expected_pixel_mean)>1e-12:raise ValueError('完整fit像素加权分母不符。')
        scores[name]=fit['frame_mean'];checkpoints[name]=checkpoint;configs[name]=config
        if annealed:
            previous_path=directory/'fit_epoch_25.json';previous=v.read(previous_path)['frame_mean'];improvement=abs(previous-fit['frame_mean'])/max(previous,1e-12)
            hashes[str(previous_path)]=v.sha(previous_path)
        elif a.epoch==60:
            previous=v.read(directory/'fit_epoch_50.json')['frame_mean'];improvement=(previous-fit['frame_mean'])/max(previous,1e-12)
        else:improvement=0.
        budgets[name]=dict(updates=a.epoch*117,cumulative_updates=actual_updates,
            rule=RULES['v2_budget'] if annealed else RULES['v1_budget'],fit_last_interval_relative_change=improvement,sufficient=improvement<=.01)
        terminal=[]
        for state_path in (a.calibration/'runs').glob('*/state.json'):
            record=v.read(state_path)
            backend=config['backend_provenance']
            argv=record.get('actual_cmdline',[])
            same_output=('--output' in argv and Path(argv[argv.index('--output')+1]).resolve()==directory)
            same_epoch=('--epochs' not in argv or int(argv[argv.index('--epochs')+1])==a.epoch)
            if same_output and same_epoch and record.get('exit_code')==0 and record.get('runtime_modules',{}).get(backend['extension_path'])==backend['extension_sha256']:
                terminal.append(state_path)
        if not terminal:raise ValueError('没有该统一终点exit0收据：'+name)
        for path in [checkpoint_path,fixed_path,directory/'configuration.json',directory/f'fit_epoch_{a.epoch}.json',directory/f'train_epoch_{a.epoch}.json',*terminal]:
            hashes[str(path)]=v.sha(path)
        for path,expected in config['input_sha256'].items():
            if v.sha(path)!=expected:raise ValueError('校准来源变动：'+path)
            hashes[path]=expected
    selected=min(range(1,4),key=lambda i:(scores[f'init_{i}'],i))
    if selected!=a.selected_init:raise ValueError('主任务初值与仅完整fit选择不符，禁止运行held。')
    annealed='parent_checkpoint' in configs['constant']
    if any(('parent_checkpoint' in config)!=annealed for config in configs.values()):raise ValueError('四条来自不同轮次。')
    if a.epoch==30 and not annealed:
        need_extend=False
        for name in NAMES:
            previous=v.read(directory_for(a.calibration,name)/'fit_epoch_20.json')['frame_mean']
            need_extend |= (previous-scores[name])/max(previous,1e-12)>.01
        if need_extend:raise ValueError('原fit规则要求统一60，不接受30作为终点。')
    elif not annealed:
        decision=v.read(a.calibration/'budget_decision_30.json')
        if not decision['extend_all_to_60'] or decision['held_used']:raise ValueError('60预算来源不合法。')
        hashes[str(a.calibration/'budget_decision_30.json')]=v.sha(a.calibration/'budget_decision_30.json')
    return data,state,packed,checkpoints,configs,masks,scores,budgets,hashes


def plan(a):
    a.output.mkdir(parents=True,exist_ok=False)
    for folder in ('logs','visuals','reloaded_pc'):(a.output/folder).mkdir()
    paths={key:str(getattr(a,key)) for key in ('calibration','raster_cache','cache','fit','model','source','output')}
    source_files=[Path(__file__).resolve(),Path(forward.__file__),Path(v.__file__),ROOT/'scene/eldl_v3_photometric.py',DOC]
    sources={str(p):v.sha(p) for p in source_files}
    cmd=[str(v.PYTHON),str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),'scripts.validate_eldl_v3_raster','--action','run']
    for key,path in paths.items():cmd+=['--'+key.replace('_','-'),path]
    cmd+=['--epoch',str(a.epoch),'--selected-init',str(a.selected_init)]
    v.write(a.output/'plan.json',dict(rules=RULES,epoch=a.epoch,selected_init=a.selected_init,paths=paths,source_sha256=sources,
        status='PLANNED_NO_HELD_EXECUTION',command=shlex.join(cmd),internal_validation_passed=False))
    (a.output/DOC.name).write_bytes(DOC.read_bytes())
    (a.output/'README.md').write_text('# 全图冻结终点独立验收\n\n2026-10-19（Asia/Tokyo）。TRAIN-only，无GT/外部TEST，禁止重新拟合或按held选择。统一终点与fit初值由主任务给出，脚本独立核验。\n\n```bash\n'+shlex.join(cmd)+'\n```\n\nsource/model/iteration16000、校准终点、输入路径见plan.json。运行需协调GPU及原生二进制环境；日志logs/run.log。输出summary/per_frame、reloaded_pc、visuals。固定fit1/81/149与所有18held四类，每个candidate三bias、constant中心；RGB为观察/预测/绝对差，Alpha为观察/预测/差，Normal为世界法线，Separation为EL/DL/合计（constant为自由颜色/零/合计）。实际目检PENDING。internal_validation_passed=false，方向profile另行PENDING，不导出。\n')


def cpu_check(a):
    values=inputs(a);v.write(a.output/'cpu_receipt.json',dict(status='PASS_CPU_ENDPOINT_SOURCE_CHECK_NO_HELD_RENDER',
        selected_init=a.selected_init,fit_scores=values[6],budget=values[7],input_sha256=values[8],internal_validation_passed=False))
    return values


def png(path,x):
    array=x.detach().clamp(0,1).cpu().permute(1,2,0).numpy()
    if array.shape[-1]==1:array=np.repeat(array,3,-1)
    if not cv2.imwrite(str(path),(array[...,::-1]*255+.5).astype(np.uint8)):raise IOError('写图失败。')


def encode_material(rho):
    """只量化冻结rho的表示；固定±8 ULP，不接触观察或损失。"""
    from utils.sh_utils import RGB2SH,SH2RGB
    from scene.stage2_photometric import srgb_to_linear,linear_to_srgb
    initial=RGB2SH(linear_to_srgb(rho.double())).float()
    decode=lambda value:srgb_to_linear(SH2RGB(value).clamp(.03,.97))
    best=initial.clone();error=(decode(best)-rho).abs();initial_error=float(error.max())
    below=initial.clone();above=initial.clone()
    for unused in range(8):
        below=torch.nextafter(below,torch.full_like(below,-torch.inf))
        above=torch.nextafter(above,torch.full_like(above,torch.inf))
        for candidate in (below,above):
            candidate_error=(decode(candidate)-rho).abs();better=candidate_error<error
            best=torch.where(better,candidate,best);error=torch.minimum(error,candidate_error)
    return best.unsqueeze(1),dict(method='float64_initial_then_fixed_8_ULP_each_side_gpu_decode',
        neighbor_limit_each_side=8,initial_max_rho_error=initial_error,final_max_rho_error=float(error.max()),
        max_SH_encoding_change=float((best-initial).abs().max()),checkpoint_rho_modified=False,
        images_or_held_loss_used=False)


def pictures(out,frame,camera,package,environment):
    from scene.stage2_photometric import linear_to_srgb
    out.mkdir(parents=True,exist_ok=False)
    pred=linear_to_srgb(package['render_linear'].clamp_min(0));observed=camera.original_image_train_light.cuda()
    png(out/'RGB.png',torch.cat([observed,pred,(pred-observed).abs()*3],2))
    alpha=camera.gt_alpha_mask.cuda();png(out/'Alpha.png',torch.cat([alpha,package['rend_alpha'],(alpha-package['rend_alpha']).abs()],2))
    normal=torch.nn.functional.normalize(package['rend_normal'],dim=0);png(out/'Normal.png',(normal+1)/2*package['rend_alpha'])
    direct=package['render_linear']-environment
    png(out/'Separation.png',torch.cat([linear_to_srgb(environment.clamp_min(0)),linear_to_srgb(direct.clamp_min(0)),pred],2))


@torch.no_grad()
def run(a):
    record=v.read(a.output/'plan.json')
    if record['rules']!=RULES or record['epoch']!=a.epoch or record['selected_init']!=a.selected_init:raise ValueError('计划参数变动。')
    if record['paths']!={key:str(getattr(a,key)) for key in ('calibration','raster_cache','cache','fit','model','source','output')}:raise ValueError('计划路径变化。')
    v.check_hashes(record['source_sha256'])
    if (a.output/'RUN_STARTED.json').exists():raise FileExistsError('不能覆盖运行，失败需新目录。')
    v.write(a.output/'RUN_STARTED.json',dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),internal_validation_passed=False))
    data,state,packed,checkpoints,configs,masks,scores,budgets,hashes=cpu_check(a)
    if not all(item['sufficient'] for item in budgets.values()):
        v.write(a.output/'summary.json',dict(status='OPTIMIZATION_BUDGET_INSUFFICIENT_HELD_NOT_RUN',
            internal_validation_passed=False,training_asset_exported=False,held_rendered=False,
            epoch=a.epoch,fit_scores=scores,budget=budgets,input_sha256=hashes))
        with (a.output/'README.md').open('a') as f:f.write('\n最终FAILED（原fit预算门槛不足），held未运行，全部来源保留。\n')
        return
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from scripts.probe_dynamic_dl_selfcal import read_config
    from utils.camera_utils import loadCam
    from utils.general_utils import build_rotation
    from utils.sh_utils import RGB2SH
    from scene.stage2_photometric import srgb_to_linear,linear_to_srgb
    from scene.eldl_v3_photometric import render_photometric
    from scene.static_dl_photometric import StaticVisibility
    from scene.pbr_rasterizer import native_module
    backend=native_module().provenance
    if backend['source_sha256']!=configs['constant']['backend_provenance']['source_sha256'] or backend['extension_sha256']!=configs['constant']['backend_provenance']['extension_sha256']:
        raise ValueError('验收与校准实际原生库SHA不同，需相同TORCH_EXTENSIONS_DIR。')
    cfg=read_config(a.model/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
    def make_pc(path):
        pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(path))
        for value in vars(pc).values():
            if isinstance(value,torch.Tensor):value.requires_grad_(False)
        return pc
    base=make_pc(a.model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply')
    geometry={key:getattr(base,key).clone() for key in ('_xyz','_scaling','_rotation','_opacity')}
    original_albedo=base._albedo_dc.clone();pcs={};rhos={};environments={};material={}
    for name in NAMES:
        ck=checkpoints[name];rho=ck['rho'].cuda();environment=ck['environment'].cuda()
        if rho.shape!=base.get_xyz.shape or environment.shape!=(3,):raise ValueError('颜色/EL维数错误。')
        # 生产仍为原float32；只做与图像无关、确定性的冻结材质表示量化。
        if name=='constant':encoded=original_albedo;encoding=dict(method='constant_radiance_no_material_mapping')
        else:encoded,encoding=encode_material(rho)
        base._albedo_dc=torch.nn.Parameter(encoded,requires_grad=False)
        path=a.output/'reloaded_pc'/f'{name}.ply';base.save_ply(str(path));pc=make_pc(path)
        if not all(torch.equal(getattr(pc,key),value) for key,value in geometry.items()):raise ValueError('实际PLY重载改变几何。')
        if name=='constant':
            cp=a.output/'reloaded_pc/constant_radiance.pt';torch.save(rho.cpu(),cp);rho=torch.load(cp,map_location='cuda',weights_only=True)
            mapping_error=0.;meaning='独立线性辐亮度，native override_color；不是物理rho映射'
        else:
            mapping_error=float((srgb_to_linear(pc.get_albedo)-rho).abs().max())
            if mapping_error>RULES['material_roundtrip_tolerance']:raise ValueError('真实材质重载精度失败。')
            meaning='真实pc._albedo_dc材质映射，不用Proxy'
        pcs[name]=pc;rhos[name]=rho;environments[name]=environment
        material[name]=dict(semantics=meaning,roundtrip_max_error=mapping_error,ply_sha256=v.sha(path),
            representation_encoding=encoding,
            lower_bound_fraction=[float((rho[:,c]<=((0 if name=='constant' else configs[name]['rho_linear_bounds'][0])+1e-6)).float().mean()) for c in range(3)],
            upper_bound_fraction=[float((rho[:,c]>=((30 if name=='constant' else configs[name]['rho_linear_bounds'][1])-1e-6)).float().mean()) for c in range(3)],
            environment_over_pi=(environment/math.pi).tolist())
    cameras=[]
    for i,info in enumerate(readCamerasFromTransforms(str(a.source),'transforms_train.json','images',False)):
        camera=loadCam(cfg,i,info,1.);camera.load2device();frame=int(info.image_name_train_light.split('_')[-1])
        if frame!=v.TRAIN[i] or not torch.equal((camera.gt_alpha_mask>.5).cpu(),masks[i]):raise ValueError('原输入与冻结mask不符。')
        cameras.append(camera)
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    selected=f'init_{a.selected_init}';pc=pcs[selected];environment=environments[selected]
    normal=build_rotation(pc.get_rotation)[:,:,2]
    traces=[StaticVisibility(bias=float(data['shadow_bias'])*s,update_interval=1000000) for s in RULES['bias_scales']]
    rows={v.scale_key(s):[] for s in RULES['bias_scales']};constant_rows=[];parity=[];cache_checks=[];geometry_checks=[]
    other_center_held={name:[] for name in NAMES if name not in (selected,'constant')}
    for i,camera in enumerate(cameras):
        frame=v.TRAIN[i];light=state['directions'][i:i+1].cuda();target=srgb_to_linear(camera.original_image_train_light.cuda())
        saturated=camera.observed_saturation_mask.cuda().bool();fg=masks[i].cuda()
        constant=forward.render(camera,pcs['constant'],pipe,rhos['constant']);constant['render_linear']=constant['render']
        constant_rows.append(dict(frame=frame,split='fit' if frame in v.FIT else 'held',statistics=statistics(constant['render'],target,saturated,fg,constant['render'])))
        env=forward.render(camera,pc,pipe,srgb_to_linear(pc.get_albedo)*environment/math.pi)['render']
        if frame in v.HELD or frame in RULES['fixed_fit_representatives']:
            pictures(a.output/'visuals/constant/bias_1'/f'frame_{frame:04d}',frame,camera,constant,constant['render'])
        for si,(scale,trace) in enumerate(zip(RULES['bias_scales'],traces)):
            key=v.scale_key(scale);vis=trace(pc,light,0)
            facing=torch.where((normal*(camera.camera_center-pc.get_xyz)).sum(-1,keepdim=True)>=0,1.,-1.)
            q=(((normal*facing)@light.T).clamp_min(0)*vis)
            vis_error=float((vis.cpu()-packed['visibility'][si,i]).abs().max());q_error=float((q.cpu()-packed['q'][si,i]).abs().max())
            if max(vis_error,q_error)>RULES['exact_visibility_cache_tolerance']:raise ValueError('完整生产BVH与冻结缓存不符。')
            cache_checks.append(dict(frame=frame,bias_scale=scale,visibility_max_error=vis_error,q_max_error=q_error))
            lights=SimpleNamespace(ambient_rgb=environment,for_camera=lambda unused,L=light:(L,L.new_tensor([math.pi])))
            package=render_photometric(camera,pc,pipe,torch.zeros(3,device='cuda'),lights,lambda *unused,V=vis:V,0)
            stat=statistics(package['render_linear'],target,saturated,fg,env)
            if abs(stat['all']['mse']-float(forward.objective(package['render_linear'],camera)))>1e-7:raise ValueError('损失分母/删失语义不同。')
            rows[key].append(dict(frame=frame,split='fit' if frame in v.FIT else 'held',statistics=stat))
            if key=='1' and frame in v.HELD:
                for name in other_center_held:
                    alternate_lights=SimpleNamespace(ambient_rgb=environments[name],for_camera=lambda unused,L=light:(L,L.new_tensor([math.pi])))
                    alternate=render_photometric(camera,pcs[name],pipe,torch.zeros(3,device='cuda'),alternate_lights,lambda *unused,V=vis:V,0)
                    other_center_held[name].append(dict(frame=frame,split='held',statistics=statistics(alternate['render_linear'],target,saturated,fg)))
            geo={k:float((package[k]-constant[k]).abs().max()) for k in ('rend_alpha','rend_normal','surf_depth')}
            if any(value!=0 for value in geo.values()):raise ValueError('颜色改变了几何输出。')
            geometry_checks.append(dict(frame=frame,bias_scale=scale,**geo))
            if frame in RULES['fixed_fit_representatives']:
                for name in NAMES:
                    if name=='constant':
                        restored=forward.render(camera,pcs[name],pipe,rhos[name])['render'];reference=constant['render']
                    else:
                        li=SimpleNamespace(ambient_rgb=environments[name],for_camera=lambda unused,L=light:(L,L.new_tensor([math.pi])))
                        restored=render_photometric(camera,pcs[name],pipe,torch.zeros(3,device='cuda'),li,lambda *unused,V=vis:V,0)['render_linear']
                        reference=forward.render(camera,base,pipe,rhos[name]*(environments[name]/math.pi+q))['render']
                    error=float((restored-reference).abs().max())
                    if error>RULES['forward_absolute_tolerance']:raise ValueError('实际checkpoint/PLY生产parity失败。')
                    parity.append(dict(run=name,frame=frame,bias_scale=scale,max_error=error))
                direct=forward.render(camera,pc,pipe,srgb_to_linear(pc.get_albedo)*q)['render']
                if float((env+direct-package['render_linear']).abs().max())>RULES['forward_absolute_tolerance']:raise ValueError('黑背景EL+DL加和失败。')
            if frame in v.HELD or frame in RULES['fixed_fit_representatives']:
                pictures(a.output/'visuals/candidate'/('bias_'+key)/f'frame_{frame:04d}',frame,camera,package,env)
        print(dict(frame=frame,split='fit' if frame in v.FIT else 'held',status='生产完整三档完成'),flush=True)
    metrics={key:aggregate(value) for key,value in rows.items()};constant_metrics=aggregate(constant_rows)
    if abs(metrics['1']['fit']['groups']['all']['frame_mean']-scores[selected])>RULES['checkpoint_fit_mean_tolerance'] or abs(constant_metrics['fit']['groups']['all']['frame_mean']-scores['constant'])>RULES['checkpoint_fit_mean_tolerance']:
        raise ValueError('完整fit与checkpoint摘要不符。')
    checks,relative=decide(metrics,constant_metrics,environment,all(x['sufficient'] for x in budgets.values()))
    for name,pc in pcs.items():
        if not all(torch.equal(getattr(pc,key),value) for key,value in geometry.items()):raise ValueError('结束时几何改变。')
    hashes.update(record['source_sha256']);v.check_hashes(hashes)
    v.write(a.output/'per_frame.json',dict(candidate=rows,constant=constant_rows,unselected_center_held=other_center_held))
    status='PASS_NUMERIC_PENDING_DIRECTION_PROFILE_AND_REVIEW' if all(checks.values()) else 'FAILED_NUMERIC_RASTER_VALIDATION'
    v.write(a.output/'summary.json',dict(status=status,internal_validation_passed=False,training_asset_exported=False,
        gt_read=False,external_test_read=False,epoch=a.epoch,selected_init=a.selected_init,selected_by='完整终点fit等帧MSE',
        fit_selection_scores=scores,unselected_candidates_held_evaluated='CENTER_ONLY_AFTER_SELECTION_LOCKED',metrics=metrics,constant=constant_metrics,
        unselected_center_held_stability={name:aggregate(value)['held'] for name,value in other_center_held.items()},
        held_used_to_reselect=False,
        checks=checks,relative_held_degradation=relative,budget=budgets,material=material,
        checkpoint_reload_production_parity=parity,full_bvh_cache_checks=cache_checks,geometry_checks=geometry_checks,
        direction_profile='PENDING_SEPARATE_PROTOCOL_NOT_AUTOEXPORT',direction_semantics='三候选灯向冻结相同，不是三次独立估向',
        manual_review='PENDING',input_sha256=hashes))
    with (a.output/'README.md').open('a') as f:f.write('\n执行完成，最终'+status+'；完整指标summary.json、逐帧/分层per_frame.json；实际目检与方向profile尚PENDING，不导出。\n')


def self_check():
    pred=torch.tensor([[[.2,.8]],[[.3,.4]],[[.5,.9]]]);target=torch.ones_like(pred)*.5
    fg=torch.tensor([[[True,False]]]);sat=torch.zeros_like(pred,dtype=torch.bool);sat[0,0,0]=True
    s=statistics(pred,target,sat,fg)
    assert s['all']['channels']==3 and s['saturated']['channels']==1 and s['unsaturated']['channels']==2
    assert abs(s['all']['mse']-(.09+.04)/3)<1e-8
    pred[0,0,0]=.8
    assert abs(statistics(pred,target,sat,fg)['all']['mse']-.04/3)<1e-8
    def metric(value):return dict(fit=dict(frames=117),held=dict(frames=18,groups=dict(all=dict(frame_mean=value))))
    metrics={'0p5':metric(1.25),'1':metric(1.),'2':metric(1.2)}
    baseline=dict(held=dict(groups=dict(all=dict(frame_mean=2.))))
    checks,_=decide(metrics,baseline,torch.tensor([math.pi]*3),True)
    assert all(checks.values())
    metrics['0p5']=metric(1.25001)
    assert not decide(metrics,baseline,torch.tensor([math.pi]*3),True)[0]['0p5_degradation_within_25pct']
    metrics['0p5']=metric(1.25);baseline['held']['groups']['all']['frame_mean']=1.25
    assert not decide(metrics,baseline,torch.tensor([math.pi]*3),True)[0]['0p5_strictly_beats_constant']
    print('PASS：固定前景、逐通道删失与分层分母CPU检查；未读取模型或held。')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['plan','cpu-check','run','self-check'],required=True)
    for key in ('calibration','raster-cache','cache','fit','model','source','output'):p.add_argument('--'+key,type=Path)
    p.add_argument('--epoch',type=int,choices=[30,60]);p.add_argument('--selected-init',type=int,choices=[1,2,3]);p.add_argument('--iteration',type=int,default=16000)
    a=p.parse_args()
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():p.error('必须ushas环境。')
    if a.action=='self-check':self_check();return
    if any(getattr(a,key) is None for key in ('calibration','raster_cache','cache','fit','model','source','output','epoch','selected_init')):p.error('需主任务明确统一epoch/fit初值及全部来源。')
    for key,value in vars(a).items():
        if isinstance(value,Path):
            value=value.resolve();setattr(a,key,value)
            if 'eval_only' in value.parts or not value.is_relative_to(ROOT/'output'):p.error('仅output白名单TRAIN路径，禁eval_only。')
    if a.iteration!=16000:p.error('冻结I00@16000。')
    if a.action=='plan':plan(a);return
    if a.action=='cpu-check':cpu_check(a);return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('需与主任务协调并显式GPU。')
    try:run(a)
    except Exception as exc:
        v.write(a.output/'FAILED.json',dict(status='FAILED_ENGINEERING',error=str(exc),retained=True,internal_validation_passed=False))
        with (a.output/'README.md').open('a') as f:f.write('\n最终FAILED（工程异常），全部保留：'+str(exc)+'\n')
        raise


if __name__=='__main__':main()
