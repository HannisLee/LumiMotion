"""固定新旧方向成对开发验收；TRAIN-only、完整BVH/全图，不拟合或导出。"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import shlex
import socket
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts import validate_eldl_v3_raster as r
from scripts import calibrate_eldl_v3_raster as f

DOC=ROOT/'LIHAN/DOC/报告/1019-35-新旧方向成对完整开发验收协议.md'
CASES=('old','refined')
RULES=dict(r.RULES,checkpoint_fit_per_frame_tolerance=1e-7,
    refined_center_strictly_better_than_old=True,
    pair_status='PASS_EQUAL_BUDGET_FIT_PAIR',candidate_cumulative_epochs=120,
    constant_cumulative_epochs=90,constant_is_equal_budget_control=False,
    direction_identifiability='unresolved',internal_validation_passed=False)


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_receipt(receipt,pair):
    require(receipt['status']==RULES['pair_status'] and receipt['eligible_for_development_validation'] is True,'成对FIT门禁未通过，禁止开发验收。')
    require(receipt['pair']==str(pair) and receipt['gt_read'] is False and receipt['external_test_read'] is False,'成对路径或无GT/TEST声明不符。')
    require(receipt['held_scored'] is False and receipt['held_used_for_selection'] is False,'成对FIT阶段已使用开发评分。')
    require(receipt['same_initial_rho_loge_adam_rng'] and receipt['same_terminal_frame_rng'],'成对起点/终态RNG不一致。')
    require(receipt['epochs_each']==30 and receipt['updates_each']==3510 and receipt['cumulative_epochs']==120,'成对预算不同。')
    records=receipt['records']
    for case in CASES:
        values=records[case]['fit'];require(set(values)=={'0','10','20','25','30'},'成对完整FIT节点缺失。')
        require(all(math.isfinite(x) and x>=0 for x in values.values()),'成对FIT非有限。')
        change=abs(values['30']-values['25'])/max(values['25'],1e-12)
        require(change<=.01 and records[case]['budget_sufficient'] and abs(change-records[case]['terminal_absolute_relative_change'])<1e-12,'原1%预算门槛不符。')
    benefit=1-records['refined']['fit']['30']/max(records['old']['fit']['30'],1e-12)
    require(benefit>=.01 and abs(benefit-receipt['relative_refined_benefit'])<1e-12,'新方向FIT收益不足原1%。')


def inputs(a):
    """仅读FIT收据、参数与固定mask，不解码或评分开发相机。"""
    l00=a.pair.parent
    require(a.iteration==16000 and a.constant==l00/'raster_calibration_v2_anneal/constant','只接受I00@16k及原constant90参考。')
    receipt_path=a.pair/'fit_pair_receipt.json';receipt=v.read(receipt_path)
    validate_receipt(receipt,a.pair)
    original=l00/'raster_calibration_v2_anneal/init_3/checkpoint_epoch_30.pt'
    refined_path=l00/'raster_direction_refinement_v2/result/direction_state.pt'
    code_roots=[ROOT/p for p in ('scripts','utils','scene','gaussian_renderer','LIHAN/scripts')]
    snapshot_root=ROOT/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/snapshots'
    data_roots=[a.pair,a.cache.parent,a.fit,a.model,a.source,a.constant]
    exact_parents=[original,original.parent/'configuration.json',original.parent/'fit_epoch_30.json',original.parent/'fixed_foreground.pt']
    exact_refined=[refined_path,refined_path.parent/'configuration.json',refined_path.parent/'summary.json']
    # receipt可能列出源码，但不得借清单将外部数据登记为白名单。
    for text in receipt['input_sha256']:
        path=Path(text).resolve()
        allowed=any(path.is_relative_to(folder) for folder in data_roots) or path in exact_parents+exact_refined
        allowed|=path.suffix=='.py' and any(path.is_relative_to(folder) for folder in code_roots+[snapshot_root])
        require(allowed,'成对收据含白名单外来源：'+str(path))
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([*data_roots,a.output,*exact_parents,*exact_refined,
        *[Path(x) for x in receipt['input_sha256']]])
    v.check_hashes(receipt['input_sha256'])
    data,state,unused_summary,unused_result,hashes=v.load_inputs(a)
    hashes.update(receipt['input_sha256']);hashes[str(receipt_path)]=v.sha(receipt_path)
    parent=torch.load(original,map_location='cpu',weights_only=False)
    refined=torch.load(refined_path,map_location='cpu',weights_only=False)
    require(torch.equal(parent['directions'],state['directions']),'原冻结L与cache/fit不符。')
    require(refined['frames']==v.TRAIN and refined['fit_frames']==v.FIT and refined['held_rgb_used'] is False,'新方向序列/开发边界不符。')
    require(refined['source_checkpoint_sha256']==v.sha(original),'新方向父checkpoint SHA不符。')
    configs={};checkpoints={};fits={};masks=None;zeroes={};runtime=None
    for case in (*CASES,'constant'):
        directory=a.constant if case=='constant' else a.pair/case/'result'
        cfg=v.read(directory/'configuration.json');ck=torch.load(directory/'checkpoint_epoch_30.pt',map_location='cpu',weights_only=False)
        fit=v.read(directory/'fit_epoch_30.json');fixed_path=directory/'fixed_foreground.pt'
        require(cfg['gt_read'] is False and cfg['external_test_read'] is False and cfg['geometry_frozen'] and cfg['directions_frozen'],'非冻结无GT校准。')
        require(cfg['geometry_fit_frames']==v.FIT and cfg['held_frames']==v.HELD and cfg['loss']=='linear_censored_mse_original_alpha_gt_0p5_fixed_denominator','划分/损失/全图分母不同。')
        require(ck['epoch']==30 and ck['cumulative_epoch']==(90 if case=='constant' else 120) and len(ck['history'])==30,'checkpoint终点预算不符。')
        require(ck['constant']==(case=='constant') and ck['initial_ratio']==(1. if case=='constant' else 3.),'checkpoint身份不符。')
        require(cfg['optimizer_schedule']==r.ANNEAL and cfg['optimizer']=='Adam','优化日程被改变。')
        require(all(int(s['step'])==(10530 if case=='constant' else 14040) for s in ck['optimizer']['state'].values()),'实际Adam步数不同。')
        require(v.sha(fixed_path)==cfg['foreground_sha256'],'固定mask SHA变化。')
        fixed=torch.load(fixed_path,map_location='cpu',weights_only=False)
        require(fixed['frames']==v.TRAIN and fixed['foreground'].dtype==torch.bool and fixed['foreground'].shape==(135,1,400,400),'原mask格式不同。')
        if masks is None:masks=fixed['foreground']
        else:require(torch.equal(masks,fixed['foreground']),'新旧/constant固定mask不同。')
        require(torch.isfinite(ck['rho']).all() and torch.isfinite(ck['environment']).all() and (ck['environment']>0).all(),'rho/E无效。')
        require(ck['environment'].shape==(3,) and torch.allclose(ck['log_e'].exp(),ck['environment'],atol=1e-7,rtol=2e-7),'logE/EL不一致。')
        lo,hi=(0.,30.) if case=='constant' else cfg['rho_linear_bounds']
        require(((ck['rho']>=lo-1e-7)&(ck['rho']<=hi+1e-7)).all(),'rho越过原边界。')
        values=fit['per_frame'];require(len(values)==117 and all(math.isfinite(x) and x>=0 for x in values),'原逐帧FIT不完整。')
        require(abs(sum(values)/117-fit['frame_mean'])<1e-12,'FIT均值与117帧不一致。')
        count=masks.flatten(1).sum(1)[torch.tensor([n in v.FIT for n in v.TRAIN])].double()
        require(abs(float((torch.tensor(values,dtype=torch.float64)*count).sum()/count.sum())-fit['pixel_mean'])<1e-12,'FIT像素分母不同。')
        expected=refined['directions'] if case=='refined' else parent['directions']
        require(torch.equal(ck['directions'],expected),'方向不是冻结的旧/新候选。')
        require(torch.allclose(ck['directions'].norm(dim=-1),torch.ones(135),rtol=0,atol=1e-4),'方向不是单位向量。')
        fi=torch.tensor([n in v.FIT for n in v.TRAIN]);times=torch.tensor(v.TRAIN,dtype=torch.float64)
        expected_held=v.interpolate_frozen(times,times[fi],ck['directions'][fi].double()).float()
        require(torch.isfinite(ck['directions']).all() and torch.allclose(ck['directions'],expected_held,rtol=0,atol=2e-6),'开发方向不是FIT nlerp。')
        if case!='constant':
            terminal=v.read(directory/'train_epoch_30.json');supervisor=v.read(directory.parent/'state.json')
            require(supervisor['exit_code']==0 and terminal['geometry_bitexact_unchanged'] and terminal['directions_bitexact_unchanged'],'成对训练未正常完成或改变冻结参数。')
            require(cfg['case']==case and cfg['held_scored'] is False and cfg['held_used_for_selection'] is False,'FIT阶段使用开发评分。')
            require(fit['frame_mean']==receipt['records'][case]['fit']['30'],'FIT记录与成对receipt不同。')
            zero=torch.load(directory/'checkpoint_epoch_0.pt',map_location='cpu',weights_only=False);zeroes[case]=zero
            for key in ('rho','log_e','optimizer','rng'):require(r.same_tree(zero[key],parent[key]),'未完整继承同一父起点：'+key)
            modules=supervisor['runtime_modules'];require(modules.get(cfg['backend_provenance']['extension_path'])==cfg['backend_provenance']['extension_sha256'],'实际native库与metadata不符。')
            if runtime is None:runtime=modules
            else:require(runtime==modules,'成对实际native/tracer库不同。')
            for filename in (directory.parent/'state.json',directory/'train_epoch_30.json',directory/'checkpoint_epoch_0.pt'):hashes[str(filename)]=v.sha(filename)
        for filename in (directory/'configuration.json',directory/'checkpoint_epoch_30.pt',directory/'fit_epoch_30.json',fixed_path):hashes[str(filename)]=v.sha(filename)
        configs[case],checkpoints[case],fits[case]=cfg,ck,fit
    require(checkpoints['old']['rng']==checkpoints['refined']['rng'],'终态相机RNG不符。')
    for key in ('backend_provenance','rho_linear_bounds','environment_over_pi_bounds','loss','foreground_sha256'):
        require(configs['old'][key]==configs['refined'][key]==configs['constant'][key],'三者物理/来源不同：'+key)
    for path,expected in runtime.items():require(v.sha(path)==expected,'磁盘实际native库SHA改变。')
    sources=[Path(__file__).resolve(),Path(r.__file__),Path(f.__file__),Path(v.__file__),DOC,
        ROOT/'scene/eldl_v3_photometric.py',ROOT/'scene/stage2_photometric.py',
        ROOT/'gaussian_renderer/__init__.py',ROOT/'scene/pbr_rasterizer.py',ROOT/'utils/rgba16.py']
    hashes.update({str(x):v.sha(x) for x in sources});hashes.update(runtime)
    v.check_hashes(hashes)
    return data,checkpoints,configs,masks,fits,hashes,receipt,runtime


def plan(a):
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('logs','visuals','reloaded_pc'):(a.output/name).mkdir()
    paths={k:str(getattr(a,k)) for k in ('pair','cache','fit','model','source','constant','output')}
    command=[str(v.PYTHON),str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),'scripts.validate_eldl_v3_refined_pair','--action','run']
    for key,value in paths.items():command+=['--'+key,value]
    command+=['--iteration',str(a.iteration)]
    source_sha256={str(p):v.sha(p) for p in (Path(__file__).resolve(),DOC,Path(r.__file__),Path(f.__file__),Path(v.__file__),a.pair/'fit_pair_receipt.json')}
    v.write(a.output/'plan.json',dict(status='PLANNED_NO_GPU_OR_DEVELOPMENT_SCORES',paths=paths,iteration=a.iteration,rules=RULES,source_sha256=source_sha256,command=command,shell=shlex.join(command),internal_validation_passed=False))
    (a.output/DOC.name).write_bytes(DOC.read_bytes())
    (a.output/'README.md').write_text('# 新旧方向成对开发验收\n\n2026-10-19 Asia/Tokyo。source/model/I00 iteration16000/原constant90见plan.json。TRAIN-only，不读GT/TEST，不拟合，不导出。完整命令：\n\n```bash\n'+shlex.join(command)+'\n```\n\n日志logs/run.log；指标summary/per_frame.json；真实PLY reloaded_pc；old/refined三bias各18开发+FIT1/81/149四类visuals；constant中心同代表帧。RGB观察/预测/差，Alpha观察/预测/差，Normal世界法线，Separation为EL/DL/合计。所有实际目检和数字PENDING，不把plan/CPU核验当GPU验收。\n\n方向收益只能由同预算old/new120比较；constant90只是既有自由辐亮度参照。internal_validation_passed=false，不自动导出。\n')


def cpu_check(a):
    values=inputs(a)
    v.write(a.output/'cpu_receipt.json',dict(status='PASS_CPU_PAIR_BINDING_NO_DEVELOPMENT_RENDER',input_sha256=values[5],pair_receipt=values[6],constant_cumulative_epochs=90,candidate_cumulative_epochs=120,development_scored=False,gpu_forward=False,internal_validation_passed=False))
    return values


def numeric_decision(metrics,constant,environments):
    checks={};relative={}
    for case in CASES:
        local,drift=r.decide(metrics[case],constant,environments[case],True)
        checks[case]=local;relative[case]=drift
    center=lambda case:metrics[case]['1']['held']['groups']['all']['frame_mean']
    cross=center('refined')<center('old')
    passed=all(value for local in checks.values() for value in local.values()) and cross
    return passed,checks,relative,cross


@torch.no_grad()
def run(a):
    record=v.read(a.output/'plan.json')
    require(record['rules']==RULES and record['iteration']==a.iteration,'冻结规则不同。')
    require(record['paths']=={k:str(getattr(a,k)) for k in ('pair','cache','fit','model','source','constant','output')},'计划路径不同。')
    v.check_hashes(record['source_sha256']);require(not (a.output/'RUN_STARTED.json').exists(),'失败需新目录，禁止覆盖。')
    v.write(a.output/'RUN_STARTED.json',dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),internal_validation_passed=False))
    data,ck,cfg,masks,fits,hashes,receipt,runtime=cpu_check(a)
    # 不调用f.load，防止公共loader另装窄guard后误拒本次已登记checkpoint。
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from scripts.probe_dynamic_dl_selfcal import read_config
    from utils.camera_utils import loadCam
    from utils.general_utils import build_rotation
    from scene.stage2_photometric import srgb_to_linear
    from scene.eldl_v3_photometric import render_photometric
    from scene.static_dl_photometric import StaticVisibility
    from scene.pbr_rasterizer import native_module
    actual=native_module().provenance
    require(all(actual[k]==cfg['old']['backend_provenance'][k] for k in ('source_sha256','extension_sha256','extension_path')),'实际native光栅来源不同。')
    loaded=read_config(a.model/'cfg_args');loaded.resolution=2;loaded.load2gpu_on_the_fly=True
    def make_pc(path):
        pc=GaussianModel(loaded.sh_degree,no_binary_separation=loaded.no_binary_separation,fea_dim=loaded.hyper_dim);pc.load_ply(str(path))
        for value in vars(pc).values():
            if isinstance(value,torch.Tensor):value.requires_grad_(False)
        return pc
    weights=torch.load(a.model/f'deform/iteration_{a.iteration}/deform.pth',map_location='cpu',weights_only=True)
    require(all(torch.count_nonzero(weights[f'{h}.{s}'])==0 for h in ('gaussian_warp','gaussian_rotation','gaussian_scaling') for s in ('weight','bias')),'I00几何MLP并非静态零输出。')
    base=make_pc(a.model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply')
    geometry={k:getattr(base,k).clone() for k in ('_xyz','_scaling','_rotation','_opacity')}
    original_albedo=base._albedo_dc.clone();pcs={};rhos={};environments={};material={}
    for case in (*CASES,'constant'):
        rho=ck[case]['rho'].cuda();environment=ck[case]['environment'].cuda();require(rho.shape==base.get_xyz.shape,'rho维数不同。')
        encoded,encoding=(original_albedo,dict(method='constant_linear_radiance_override')) if case=='constant' else r.encode_material(rho)
        base._albedo_dc=torch.nn.Parameter(encoded,requires_grad=False);path=a.output/'reloaded_pc'/f'{case}.ply';base.save_ply(str(path));pc=make_pc(path)
        require(all(torch.equal(getattr(pc,k),value) for k,value in geometry.items()),'真实PLY重载改变几何。')
        if case=='constant':
            radiance=a.output/'reloaded_pc/constant_radiance.pt';torch.save(rho.cpu(),radiance);rho=torch.load(radiance,map_location='cuda',weights_only=True);error=0.
            require(torch.equal(rho.cpu(),ck[case]['rho']),'constant线性辐亮度重载不一致。')
        else:error=float((srgb_to_linear(pc.get_albedo)-rho).abs().max())
        require(error<=RULES['material_roundtrip_tolerance'],'真实PLY材质roundtrip失败。')
        pcs[case],rhos[case],environments[case]=pc,rho,environment
        lo,hi=(0.,30.) if case=='constant' else cfg[case]['rho_linear_bounds']
        material[case]=dict(ply_sha256=v.sha(path),roundtrip_max_error=error,encoding=encoding,
            lower_bound_fraction=[float((rho[:,c]<=lo+1e-6).float().mean()) for c in range(3)],
            upper_bound_fraction=[float((rho[:,c]>=hi-1e-6).float().mean()) for c in range(3)],environment_over_pi=(environment/math.pi).tolist())
    # torch JIT扩展对象不保证列入sys.modules；直接审计真实调用对象，
    # 仍严格比对两份实际.so身份与训练runtime，不以磁盘存在代替加载。
    from surfel_tracer.raytracer import _C as tracer_extension
    loaded_extensions=(native_module()._C,tracer_extension)
    loaded_native={str(Path(extension.__file__).resolve()):v.sha(extension.__file__) for extension in loaded_extensions}
    registered_native={str(Path(path).resolve()):expected for path,expected in runtime.items()}
    require(loaded_native==registered_native,'实际调用的native/tracer扩展路径或SHA与训练收据不同。')
    cameras=[]
    for i,info in enumerate(readCamerasFromTransforms(str(a.source),'transforms_train.json','images',False)):
        camera=loadCam(loaded,i,info,1.);camera.load2device()
        require(int(info.image_name_train_light.split('_')[-1])==v.TRAIN[i] and torch.equal((camera.gt_alpha_mask>.5).cpu(),masks[i]),'相机次序或原固定Alpha不同。')
        cameras.append(camera)
    require(len(cameras)==135,'需要完整135 TRAIN。')
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    rows={case:{v.scale_key(s):[] for s in RULES['bias_scales']} for case in CASES};constant_rows=[];parity=[];geometry_checks=[];additive=[]
    traces={case:[StaticVisibility(bias=float(data['shadow_bias'])*s,update_interval=1000000) for s in RULES['bias_scales']] for case in CASES}
    def restored_fit():
        result={}
        for case in (*CASES,'constant'):
            selected=constant_rows if case=='constant' else rows[case]['1']
            values=[row['statistics']['all']['mse'] for row in selected if row['split']=='fit']
            require(len(values)==117,'完整117 FIT尚未完成。')
            error=max(abs(x-y) for x,y in zip(values,fits[case]['per_frame']))
            require(error<RULES['checkpoint_fit_per_frame_tolerance'],'117逐帧FIT恢复失败：'+case)
            result[case]=dict(max_per_frame_mse_difference=error,frame_mean=sum(values)/117,source_frame_mean=fits[case]['frame_mean'])
        return result
    # 先完成全部FIT与生产工程门槛，确认后才评分任何18开发帧。
    order=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]+[i for i,n in enumerate(v.TRAIN) if n in v.HELD]
    fit_restore=None
    for ordinal,i in enumerate(order):
        camera=cameras[i]
        frame=v.TRAIN[i];target=srgb_to_linear(camera.original_image_train_light.cuda());saturated=camera.observed_saturation_mask.cuda().bool();fg=masks[i].cuda();split='fit' if frame in v.FIT else 'held'
        constant=f.render(camera,pcs['constant'],pipe,rhos['constant']);constant['render_linear']=constant['render']
        constant_rows.append(dict(frame=frame,split=split,statistics=r.statistics(constant['render_linear'],target,saturated,fg,constant['render_linear'])))
        if frame in RULES['fixed_fit_representatives']:
            native_reference=f.render(camera,base,pipe,ck['constant']['rho'].cuda())['render']
            error=float((constant['render_linear']-native_reference).abs().max())
            require(error<=RULES['forward_absolute_tolerance'],'constant真实辐亮度/native重载parity失败。')
            parity.append(dict(case='constant',frame=frame,bias='1',max_error=error))
        picture=frame in v.HELD or frame in RULES['fixed_fit_representatives']
        if picture:r.pictures(a.output/'visuals/constant/bias_1'/f'frame_{frame:04d}',frame,camera,constant,constant['render_linear'])
        for case in CASES:
            pc=pcs[case];environment=environments[case];decoded=srgb_to_linear(pc.get_albedo);normal=build_rotation(pc.get_rotation)[:,:,2]
            light=ck[case]['directions'][i:i+1].cuda();env=f.render(camera,pc,pipe,decoded*environment/math.pi)['render']
            for scale,trace in zip(RULES['bias_scales'],traces[case]):
                key=v.scale_key(scale);vis=trace(pc,light,0)
                facing=torch.where((normal*(camera.camera_center-pc.get_xyz)).sum(-1,keepdim=True)>=0,1.,-1.)
                q=((normal*facing)@light.T).clamp_min(0)*vis
                lights=SimpleNamespace(ambient_rgb=environment,for_camera=lambda unused,L=light:(L,L.new_tensor([math.pi])))
                package=render_photometric(camera,pc,pipe,torch.zeros(3,device='cuda'),lights,lambda *unused,V=vis:V,0)
                stat=r.statistics(package['render_linear'],target,saturated,fg,env)
                require(abs(stat['all']['mse']-float(f.objective(package['render_linear'],camera)))<1e-7,'删失损失/原Alpha分母不同。')
                rows[case][key].append(dict(frame=frame,split=split,statistics=stat))
                geo={k:float((package[k]-constant[k]).abs().max()) for k in ('rend_alpha','rend_normal','surf_depth')}
                require(all(value==0 for value in geo.values()),'颜色改变几何输出。');geometry_checks.append(dict(case=case,frame=frame,bias=key,**geo))
                if frame in RULES['fixed_fit_representatives']:
                    reference=f.render(camera,base,pipe,rhos[case]*(environment/math.pi+q))['render'];error=float((package['render_linear']-reference).abs().max())
                    require(error<=RULES['forward_absolute_tolerance'],'真实checkpoint→PLY→生产parity失败。')
                    parity.append(dict(case=case,frame=frame,bias=key,max_error=error))
                    direct=f.render(camera,pc,pipe,decoded*q)['render'];error=float((env+direct-package['render_linear']).abs().max())
                    require(error<=RULES['forward_absolute_tolerance'],'EL/DL加性失败。');additive.append(dict(case=case,frame=frame,bias=key,max_error=error))
                if picture:r.pictures(a.output/'visuals'/case/('bias_'+key)/f'frame_{frame:04d}',frame,camera,package,env)
        print(dict(frame=frame,split=split,status='old/refined真实全BVH三bias完成'),flush=True)
        if ordinal==116:
            fit_restore=restored_fit()
            v.write(a.output/'fit_restoration_before_development.json',dict(status='PASS_ALL_117_FIT_REPLAY_BEFORE_DEVELOPMENT_SCORES',fit_restore=fit_restore,production_parity=parity,additivity_checks=additive,development_scored=False))
    metrics={case:{key:r.aggregate(value) for key,value in by_bias.items()} for case,by_bias in rows.items()};constant_metrics=r.aggregate(constant_rows)
    require(fit_restore is not None,'缺少开发评分前FIT复现门槛。')
    passed,checks,relative,cross=numeric_decision(metrics,constant_metrics,environments)
    require(all(torch.equal(getattr(pc,k),value) for pc in pcs.values() for k,value in geometry.items()),'终末几何改变。')
    require(all(torch.equal(rhos[case].cpu(),ck[case]['rho']) and torch.equal(environments[case].cpu(),ck[case]['environment']) for case in (*CASES,'constant')),'终末rho/E改变。')
    v.check_hashes(hashes);v.check_hashes(record['source_sha256'])
    v.write(a.output/'per_frame.json',dict(candidates=rows,constant=constant_rows))
    status='PASS_NUMERIC_PENDING_MANUAL_REVIEW_NOT_EXPORT' if passed else 'FAILED_NUMERIC_REFINED_PAIR_VALIDATION'
    v.write(a.output/'summary.json',dict(status=status,gt_read=False,external_test_read=False,
        train_containers_loaded=135,development_containers_loaded=18,development_scored_only_after_fit_pair_lock=True,
        internal_validation_passed=False,training_asset_exported=False,direction_identifiability='unresolved',
        rules=RULES,metrics=metrics,constant=constant_metrics,checks=checks,relative_held_degradation=relative,
        refined_center_strictly_beats_old=cross,material=material,fit_restore=fit_restore,
        checkpoint_reload_production_parity=parity,additivity_checks=additive,geometry_checks=geometry_checks,
        runtime_modules=runtime,actual_extension_objects=loaded_native,
        pair_receipt_sha256=v.sha(a.pair/'fit_pair_receipt.json'),input_sha256=hashes,
        constant_interpretation='constant累计90是既有自由辐亮度参照，方向收益只由old/new累计120同预算对照',
        manual_review='PENDING_ACTUAL_VIEW',direction_profile='NOT_PERFORMED_NO_UNIQUENESS_CLAIM'))
    with (a.output/'README.md').open('a') as stream:stream.write('\n最终'+status+'。summary/per_frame与真实PLY/全部三bias四类保存；实际目检PENDING，不导出。\n')


def self_check():
    # 精确25%保留，超界失败；每档严格胜constant，refined中心严格胜old。
    def metric(value):return dict(fit=dict(frames=117),held=dict(frames=18,groups=dict(all=dict(frame_mean=value))))
    old={'0p5':metric(1.25),'1':metric(1.),'2':metric(1.25)};new={'0p5':metric(1.1),'1':metric(.9),'2':metric(1.1)}
    env={case:torch.full((3,),math.pi) for case in CASES};constant=metric(2.)
    require(numeric_decision(dict(old=old,refined=new),constant,env)[0],'25%边界或合法样例失败。')
    bad=dict(old);bad['2']=metric(1.25001)
    require(not numeric_decision(dict(old=bad,refined=new),constant,env)[0],'25%越界未拒绝。')
    require(not numeric_decision(dict(old=old,refined=old),constant,env)[0],'相同中心未拒绝。')
    require(not numeric_decision(dict(old=old,refined=new),metric(1.1),env)[0],'constant严格不等式未执行。')
    print('PASS：原25%边界、任一对照失败、严格constant/新旧中心规则；无输入/开发图/GPU读取。')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',choices=['plan','cpu-check','run'],default='plan');parser.add_argument('--self-check',action='store_true')
    for key in ('pair','cache','fit','model','source','constant','output'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--iteration',type=int,default=16000);a=parser.parse_args()
    if a.self_check:self_check();return
    require(all(getattr(a,k) is not None for k in ('pair','cache','fit','model','source','constant','output')),'需要完整路径参数。')
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    require(socket.gethostname()=='ushas' and Path(sys.executable).resolve()==v.PYTHON.resolve(),'需要ushas/lumimotion-ushas。')
    require(a.output.is_relative_to(ROOT/'output'),'输出必须位于output。')
    if a.action=='plan':plan(a);return
    if a.action=='cpu-check':cpu_check(a);return
    run(a)


if __name__=='__main__':main()
