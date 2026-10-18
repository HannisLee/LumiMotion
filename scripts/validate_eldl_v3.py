"""无GT精确BVH内部验证；固定rho/EL/方向/valid，不导出训练光照资产。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PYTHON=Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python')
PROTOCOL=ROOT/'LIHAN/DOC/报告/1019-05-ELDL-V3精确BVH内部验证预登记.md'
TRAIN=[n for n in range(1,151) if n%10]
HELD=[n for n in TRAIN if 61<=n<=79]
FIT=[n for n in TRAIN if n not in HELD]
SCALES=[.5,1.,2.]
REPRESENTATIVES=dict(fit=[1,81,149],held=[61,71,79])
RULES=dict(min_points_per_frame=100,min_held_frames=12,all_fit_frames_required=True,
    bias_scales=SCALES,max_held_relative_degradation=.25,degradation_floor=1e-8,
    center_strictly_better_than_constant=True,perturbations_strictly_better_than_constant=True,
    agreement_pair_median_strictly_below_deg=15.,representatives=REPRESENTATIVES,
    internal_validation_passed=False,refit=False,new_valid_filter=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def check_hashes(files):
    for path,expected in files.items():
        if sha(path)!=expected:raise ValueError('来源SHA不一致：'+path)


def interpolate_frozen(times,known_times,known):
    """固化生产fit的nlerp算法，不导入可能被并发修改的fit入口。"""
    right=torch.searchsorted(known_times,times).clamp(1,len(known_times)-1);left=right-1
    weight=((times-known_times[left])/(known_times[right]-known_times[left])).clamp(0,1)
    return torch.nn.functional.normalize(torch.lerp(known[left],known[right],weight[:,None]),dim=-1)


def receiver_cosines(data,directions):
    """保留旧plane余弦；surfel分支使用各真实receiver的统一朝向单位法线。"""
    if 'receiver_normals' in data:
        return (directions@data['receiver_normals'].T).clamp_min(0)
    return (directions@data['normal']).clamp_min(0)[:,None].expand(-1,len(data['points']))


def selected_files(args):
    summary=read(args.fit/'summary.json')
    selected=int(summary['selected_init'])
    if selected not in (1,2,3):raise ValueError('选定初值越界。')
    return summary,selected,args.fit/f'init_{selected}'


def install_guard(args):
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([args.cache.parent,args.fit,args.model,args.source,args.output])


def load_inputs(args):
    """只有CPU数据核验；调用者必须先安装白名单。"""
    meta=read(args.cache.parent/'manifest.json')
    summary,selected,init=selected_files(args)
    source_meta=read(args.source/'no_gt_manifest.json')
    marker=read(args.model/'eldl_v3_training.json')
    if meta.get('gt_read') is not False or meta['source_split']!='TRAIN' or summary.get('gt_read') is not False:
        raise ValueError('需要无GT的TRAIN缓存与fit结果。')
    if summary.get('internal_validation_passed') is not False or summary['selected_by']!='fit_censored_mse_only':
        raise ValueError('输入必须仍处于内部验证门禁，且只按fit选择初值。')
    if source_meta.get('gt_read') is not False or source_meta.get('init_weights_loaded') is not False:
        raise ValueError('TRAIN源不是白名单无GT输入。')
    if source_meta['train_frames']!=TRAIN or source_meta['internal_fit_frames']!=FIT or source_meta['internal_held_frames']!=HELD:
        raise ValueError('TRAIN/fit/held划分不同于预登记。')
    allowed_names={'images','points3d.ply','transforms_train.json','no_gt_manifest.json'}
    if set(p.name for p in args.source.iterdir())!=allowed_names:
        raise ValueError('源目录含白名单外项，拒绝外部test或GT混入。')
    if (marker.get('gt_supervision') is not False or marker.get('init_weights_loaded') is not False
            or marker['baseline']!='static_sh' or marker['train_frames']!=FIT):
        raise ValueError('必须使用无GT的117-fit静态SH模型。')
    if Path(meta['model']).resolve()!=args.model or Path(meta['source']).resolve()!=args.source or meta['iteration']!=args.iteration:
        raise ValueError('缓存几何/源/迭代与本次参数不一致。')
    if meta['fit_frames']!=FIT or meta['held_frames']!=HELD:raise ValueError('缓存划分不一致。')
    if sha(args.cache)!=meta['cache_sha256'] or sha(args.cache)!=summary['cache_sha256']:
        raise ValueError('cache.pt SHA与缓存/fit清单不一致。')
    # 只复核本白名单内缓存来源，源码另外冻结，不能追随任意清单路径打开外部数据。
    allowed_roots=[args.model,args.source,ROOT/'scripts',ROOT/'utils',ROOT/'scene']
    for path,expected in meta['input_sha256'].items():
        path=Path(path).resolve()
        registered_cache_code=(path.name=='cache_eldl_v3.py' and path.is_relative_to(
            ROOT/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/snapshots'))
        if not any(path.is_relative_to(p) for p in allowed_roots) and not registered_cache_code:
            raise ValueError('缓存清单包含未登记路径。')
        if path.name=='eldl_v3_training.json':
            # marker可能由正常训练终止更新；记录当前快照，checkpoint必须精确一致。
            continue
        if sha(path)!=expected:raise ValueError('缓存输入SHA变化：'+str(path))
    for relative,expected in source_meta['files'].items():
        path=(args.source/relative).resolve()
        if not path.is_relative_to(args.source):raise ValueError('TRAIN清单路径逃逸。')
        if sha(path)!=expected:raise ValueError('TRAIN输入SHA变化。')
    data=torch.load(args.cache,map_location='cpu',weights_only=False)
    state=torch.load(init/'fit.pt',map_location='cpu',weights_only=False)
    result=read(init/'result.json')
    all_results=[read(args.fit/f'init_{i}/result.json') for i in (1,2,3)]
    chosen=min(range(3),key=lambda i:all_results[i]['metrics']['fit'])+1
    if selected!=chosen:raise ValueError('选定初值不是仅按fit MSE最小值选择。')
    if result!=summary['selected']:raise ValueError('选定result与summary副本不一致。')
    npoints=len(data['points'])
    if npoints!=4096 or data['rgb'].shape!=(135,npoints,3) or data['valid'].shape!=(135,npoints):
        raise ValueError('缓存不是135帧×4096点的固定网格。')
    if [r['frame'] for r in data['frames']]!=TRAIN:
        raise ValueError('缓存序列编号不是完整135 TRAIN。')
    expected_fit=torch.tensor([n in FIT for n in TRAIN])
    if not torch.equal(data['fit'].bool(),expected_fit) or [bool(r['fit']) for r in data['frames']]!=expected_fit.tolist():
        raise ValueError('缓存fit标记不一致。')
    for name in ('rgb','points','normal','times','directions','visibility'):
        if not torch.isfinite(data[name]).all():raise ValueError('缓存有非有限值：'+name)
    if data['valid'].dtype!=torch.bool or data['saturated'].dtype!=torch.bool or data['saturated'].shape!=data['rgb'].shape:
        raise ValueError('支持/逐通道饱和标记类型不符。')
    if not torch.allclose(data['times'],torch.tensor([(n-1)/149 for n in TRAIN]),atol=1e-7,rtol=0):
        raise ValueError('缓存时间不对应完整原序列。')
    if not torch.equal(data['valid'],data['valid']&(data['valid'][expected_fit].sum(0)>=10)[None]):
        raise ValueError('原valid未绑定fit跨视角支持；禁止验证阶段重新筛选。')
    if 'receiver_normals' in data:
        normals=data['receiver_normals']
        if (normals.shape!=(npoints,3) or not torch.isfinite(normals).all()
                or not torch.allclose(normals.norm(dim=-1),torch.ones(npoints),atol=1e-4,rtol=0)):
            raise ValueError('真实receiver法线必须为4096个有限单位向量。')
        if meta.get('receiver_representation')!='surfel' or meta.get('fit_facing_disagreements')!=0:
            raise ValueError('缺少生产surfel/fit法线同号声明。')
        if ((normals@data['normal'])<math.cos(math.radians(10))-1e-5).any():
            raise ValueError('真实receiver法线未统一朝向或超出10°候选范围。')
        ids=data['receiver_ids'];candidate=data['receiver_candidate_support']
        if ids.shape!=(npoints,) or candidate.shape!=(npoints,) or candidate.dtype!=torch.bool:
            raise ValueError('surfel映射ID/支持形状不符。')
        if data['valid'][:,~candidate].any() or len(torch.unique(ids[candidate]))!=int(candidate.sum()):
            raise ValueError('真实receiver含重复ID支持或valid越过预登记匹配子集。')
        transforms=read(args.source/'transforms_train.json')['frames']
        centers={r['source_frame']:torch.tensor(r['transform_matrix'],dtype=torch.float32)[:3,3] for r in transforms}
        fit_centers=torch.stack([centers[n] for n in FIT])
        view_dot=((fit_centers[:,None]-data['points'][None])*normals[None]).sum(-1)
        if ((view_dot<0)&data['valid'][expected_fit]).any():
            raise ValueError('实际法线与fit可见相机朝向翻转不同号，拒绝生产语义偏差。')
    elif meta.get('receiver_representation','plane')!='plane':
        raise ValueError('缺少真实receiver法线，不能回退全局plane法线。')
    for name in ('albedo','environment','directions'):
        if not torch.isfinite(state[name]).all():raise ValueError('拟合产物非有限：'+name)
    rho,environment,directions=state['albedo'],state['environment'],state['directions']
    if rho.shape!=(npoints,3) or environment.shape!=(3,) or directions.shape!=(135,3):raise ValueError('拟合产物形状不符。')
    if (rho<0).any() or (rho>1).any() or (environment<0).any():raise ValueError('冻结rho/EL不合法。')
    if not torch.allclose(directions.norm(dim=-1),torch.ones(135),atol=1e-4,rtol=0):raise ValueError('方向不是单位向量。')
    if not torch.allclose(directions,torch.tensor(result['directions']),atol=1e-7,rtol=0):raise ValueError('方向张量与JSON不一致。')
    if not torch.allclose(environment,torch.tensor(result['ambient_rgb']),atol=1e-5,rtol=1e-6):raise ValueError('EL张量与JSON不一致。')
    if abs(result['direct_intensity']-math.pi)>1e-6:raise ValueError('固定DL尺度约定必须为π。')
    exact=interpolate_frozen(data['times'],data['times'][expected_fit],data['directions'][state['path']])
    if not torch.allclose(exact,directions,atol=2e-6,rtol=0):raise ValueError('held方向不等于fit时间nlerp。')
    bias=float(data['shadow_bias'])
    if not math.isfinite(bias) or bias<=0 or abs(bias-meta['shadow_bias'])>1e-9:raise ValueError('缓存bias不合法。')
    files=[args.cache,args.cache.parent/'manifest.json',args.fit/'summary.json',init/'fit.pt',
           args.model/'cfg_args',args.model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply',
           args.model/f'deform/iteration_{args.iteration}/deform.pth',args.source/'no_gt_manifest.json',
           args.source/'transforms_train.json',Path(__file__).resolve(),
           ROOT/'utils/no_gt_guard.py',ROOT/'scene/static_dl_photometric.py',
           ROOT/'scene/gaussian_model.py']+[args.fit/f'init_{i}/result.json' for i in (1,2,3)]
    files += [Path(path) for path in meta['input_sha256'] if Path(path).suffix=='.py']
    hashes={str(p):sha(p) for p in files}
    return data,state,summary,result,hashes


def censored_residual(prediction,observed,saturated):
    delta=prediction-observed
    return torch.where(saturated,delta.clamp_max(0),delta)


def score(prediction,data):
    valid=data['valid'];delta=censored_residual(prediction,data['rgb'],data['saturated'])
    count=valid.sum(1);squared=(delta.double().square()*valid[...,None]).sum((1,2))
    per=squared/(count.clamp_min(1)*3);enough=count>=RULES['min_points_per_frame'];fit=data['fit'].bool()
    result={}
    for name,split in [('fit',fit),('held',~fit)]:
        mask=split&enough;denom=int(count[split].sum())*3
        result[name]=dict(frame_mean=float(per[mask].mean()) if mask.any() else None,
            pixel_mean=float(squared[split].sum())/denom if denom else None,
            supported_frames=int(mask.sum()),all_frames=int(split.sum()),valid_points=int(count[split].sum()),
            valid_channels=denom,saturated_valid_channels=int((data['saturated']&valid[...,None])[split].sum()))
    result['per_frame']=[dict(frame=n,split='fit' if bool(fit[i]) else 'held',valid_points=int(count[i]),
        enough=bool(enough[i]),censored_mse=float(per[i]) if count[i] else None) for i,n in enumerate(TRAIN)]
    return result,delta,per


def numeric_checks(metrics,summary,result):
    center=metrics['1'];constant=float(result['metrics']['constant_held'])
    checks=dict(all_fit_supported=center['fit']['supported_frames']==117,
        enough_held=center['held']['supported_frames']>=12,
        original_three_start_agreement=all(r['median_deg']<15 for r in summary['three_start_agreement']),
        original_environment_not_bound=not result['bound_active'],
        original_preliminary_passed=bool(summary['preliminary_checks_passed']),
        selected_by_fit_only=summary['selected_by']=='fit_censored_mse_only')
    c=center['held']['frame_mean'];degradations={}
    checks['center_beats_constant']=c is not None and math.isfinite(constant) and c<constant
    for key in ('0p5','2'):
        held=metrics[key]['held']['frame_mean']
        degradation=max(held-c,0)/max(c,1e-8) if held is not None and c is not None else None
        degradations[key]=degradation
        checks[key+'_degradation_within_25pct']=degradation is not None and degradation<=.25
        checks[key+'_beats_constant']=held is not None and held<constant
    return checks,degradations


def scale_key(scale):return str(scale).removesuffix('.0').replace('.','p')


def true_visibility(pc,points,directions,bias,batch):
    results=[]
    for start in range(0,len(directions),batch):
        light=directions[start:start+batch]
        rays_d=light[None].expand(len(points),-1,-1).contiguous()
        rays_o=(points[:,None]+rays_d*bias).contiguous()
        traced=pc.trace(rays_o,rays_d,features=torch.zeros(len(pc.get_xyz),4,device='cuda'),
            xyz=pc.get_xyz,scales=pc.get_scaling,rotation=pc.get_rotation,opacity=pc.get_opacity,use_zeros=True)
        transmittance=(1-traced['alpha']).clamp(0,1).T
        if not torch.isfinite(transmittance).all():raise ValueError('精确BVH透射率有非有限值。')
        results.append(transmittance.cpu())
    return torch.cat(results)


def plane_figures(out,key,prediction,delta,data):
    import cv2
    valid=data['valid'].numpy();saturated=data['saturated'].numpy();observed=data['rgb'].numpy()
    prediction=prediction.numpy();raw=prediction-observed;censored=delta.numpy()
    def display(value):
        value=np.maximum(value,0)
        srgb=np.where(value<=.0031308,12.92*value,1.055*np.maximum(value,.0031308)**(1/2.4)-.055)
        return (np.clip(srgb,0,1)*255).round().astype(np.uint8)
    contacts=[]
    for split,frames in REPRESENTATIVES.items():
        for frame in frames:
            i=TRAIN.index(frame);mask=valid[i].reshape(64,64)
            panels=[display(observed[i].reshape(64,64,3)),display(prediction[i].reshape(64,64,3))]
            for residual in (raw[i],censored[i]):
                error=np.sqrt(np.mean(residual**2,axis=-1)).reshape(64,64)
                heat=cv2.applyColorMap((np.clip(error/.2,0,1)*255).round().astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1]
                panels.append(heat)
            panels.append(np.repeat(mask[...,None],3,axis=-1).astype(np.uint8)*255)
            panels.append(saturated[i].reshape(64,64,3).astype(np.uint8)*255)
            for j in (0,1,2,3,5):panels[j][~mask]=[80,80,80]
            tiles=[]
            for j,panel in enumerate(panels,1):
                tile=np.full((216,192,3),24,np.uint8)
                tile[24:]=cv2.resize(panel,(192,192),interpolation=cv2.INTER_NEAREST)
                cv2.putText(tile,str(j),(8,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                tiles.append(tile)
            image=np.concatenate(tiles,axis=1)
            cv2.putText(image,f'{split} {frame:04d} / bias {key}',(32,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            destination=out/f'{split}_{frame:04d}_bias_{key}.png'
            if not cv2.imwrite(str(destination),image[...,::-1]):raise IOError('平面图写入失败。')
            contacts.append(image)
    cv2.imwrite(str(out/f'contact_bias_{key}.png'),np.concatenate(contacts,axis=0)[...,::-1])


def plan(args):
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'logs').mkdir();(args.output/'figures').mkdir();(args.output/'arrays').mkdir()
    protocol=dict(version=1,rules=RULES,protocol_document_sha256=sha(PROTOCOL),
        paths={k:str(getattr(args,k)) for k in ('cache','fit','model','source','output')},iteration=args.iteration,
        evaluator_sha256=sha(Path(__file__)),status='PLANNED_NO_PRECISION_RESULT',internal_validation_passed=False)
    write(args.output/'plan.json',protocol)
    (args.output/PROTOCOL.name).write_bytes(PROTOCOL.read_bytes())
    command=[str(PYTHON),str(ROOT/'LIHAN/scripts/bootstrap_1011.py'),str(ROOT),
        'scripts.validate_eldl_v3','--action','run','--cache',str(args.cache),'--fit',str(args.fit),
        '--model',str(args.model),'--source',str(args.source),'--iteration',str(args.iteration),'--output',str(args.output),
        '--batch-directions',str(args.batch_directions)]
    write(args.output/'command.json',dict(argv=command,shell=shlex.join(command),gpu='执行前与主任务协调并显式CUDA_VISIBLE_DEVICES'))
    (args.output/'README.md').write_text('# V3精确BVH内部验证\n\n计划日期2026-10-19（Asia/Tokyo，主机日志UTC为10月18日）。仅读白名单TRAIN缓存/fit/I00/源；无GT，无外部test，不重新拟合。\n\n完整执行命令（需协调GPU并在环境指定CUDA_VISIBLE_DEVICES）：\n\n```bash\n'+shlex.join(command)+'\n```\n\nsource：`'+str(args.source)+'`；model：`'+str(args.model)+'`；iteration：'+str(args.iteration)+'。\n\n输出/渲染：本目录arrays与figures；日志logs；统计summary.json。平面图每行1观察、2预测、3原始残差、4删失残差、5原valid、6逐通道饱和；无效处灰色，残差色阶0～0.2线性RGB单位。fit固定1/81/149，held61/71/79，bias0.5/1/2均保存。\n\n本任务不渲染模型四类视频：RGB仅接收面观察/预测；Alpha仅原支持；Normal由缓存平面法向固定；分离不适用。所有实际目检PENDING，不能当四类几何验收。\n\n最终结论：PENDING。internal_validation_passed=false；不导出训练光照资产，失败保留全部结果。\n')
    print('计划已固定，尚未执行精确BVH。',flush=True)


def verify_plan(args):
    record=read(args.output/'plan.json')
    if record['rules']!=RULES or record['iteration']!=args.iteration or record['evaluator_sha256']!=sha(Path(__file__)):
        raise ValueError('计划门槛/迭代/源码已变化，需新目录预登记。')
    if record['paths']!={k:str(getattr(args,k)) for k in ('cache','fit','model','source','output')}:
        raise ValueError('计划输入路径不同。')
    if sha(args.output/PROTOCOL.name)!=record['protocol_document_sha256']:raise ValueError('协议快照变化。')


@torch.no_grad()
def run(args):
    verify_plan(args)
    if (args.output/'RUN_STARTED.json').exists():raise FileExistsError('禁止覆盖已有执行；失败重试新目录。')
    write(args.output/'RUN_STARTED.json',dict(host=socket.gethostname(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),internal_validation_passed=False))
    install_guard(args)
    try:
        data,state,summary,result,hashes=load_inputs(args)
        write(args.output/'input_receipt.json',dict(status='PASS_CPU_SOURCE_CHECK',input_sha256=hashes,
            selected_init=summary['selected_init'],selected_by=summary['selected_by'],gt_read=False,
            model_marker_snapshot=read(args.model/'eldl_v3_training.json')))
        import diff_surfel_rasterization as backend
        if getattr(backend,'provenance',{}).get('method')!='repository_lumimotion_cuda_rasterizer':
            raise RuntimeError('精确BVH需经bootstrap绑定仓库原生光栅器。')
        static=read(args.model/'eldl_v3_training.json')
        if static['rasterizer_provenance']['source_sha256']!=backend.provenance['source_sha256']:
            raise ValueError('I00训练和验证CUDA来源不同。')
        from scene.gaussian_model import GaussianModel
        from scene.static_dl_photometric import StaticVisibility
        from scripts.probe_dynamic_dl_selfcal import read_config
        cfg=read_config(args.model/'cfg_args')
        pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
        pc.load_ply(str(args.model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'))
        pc.active_sh_degree=min(args.iteration//1000,cfg.sh_degree)
        weights=torch.load(args.model/f'deform/iteration_{args.iteration}/deform.pth',map_location='cpu',weights_only=True)
        for head in ('gaussian_warp','gaussian_rotation','gaussian_scaling'):
            for suffix in ('weight','bias'):
                if torch.count_nonzero(weights[f'{head}.{suffix}']):raise ValueError('静态几何头非零。')
        points=data['points'].cuda();directions=state['directions'].cuda();bias=float(data['shadow_bias'])
        builder=StaticVisibility(bias=bias,update_interval=1)
        builder(pc,directions[:1],args.iteration)
        metrics={};files={}
        frozen={name:state[name].clone() for name in ('albedo','environment','directions')}
        for scale in SCALES:
            key=scale_key(scale)
            transmittance=true_visibility(pc,points,directions,bias*scale,args.batch_directions)
            cosine=receiver_cosines(data,state['directions'])
            prediction=state['albedo'][None]*(state['environment'][None,None]/math.pi+
                (result['direct_intensity']/math.pi)*transmittance[...,None]*cosine[...,None])
            if not torch.isfinite(prediction).all():raise ValueError('精确预测非有限。')
            score_value,delta,per=score(prediction,data);metrics[key]=score_value
            path=args.output/'arrays'/f'exact_bias_{key}.pt'
            torch.save(dict(visibility=transmittance,prediction=prediction,raw_residual=prediction-data['rgb'],
                censored_residual=delta,per_frame_mse=per,valid=data['valid'],saturated=data['saturated'],
                directions=state['directions'],bias=bias*scale),path)
            files[str(path)]=sha(path)
            plane_figures(args.output/'figures',key,prediction,delta,data)
            print(json.dumps(dict(bias_scale=scale,fit=score_value['fit'],held=score_value['held']),ensure_ascii=False),flush=True)
        for name,value in frozen.items():
            if not torch.equal(value,state[name]):raise AssertionError('冻结参数发生变化。')
        check_hashes(hashes)
        checks,degradations=numeric_checks(metrics,summary,result)
        passed=all(checks.values())
        report=dict(status='PASS_NUMERIC_PENDING_REVIEW' if passed else 'FAILED_NUMERIC_VALIDATION',
            internal_validation_passed=False,training_asset_exported=False,gt_read=False,external_test_read=False,
            selected_init=summary['selected_init'],selected_by='fit_censored_mse_only',rules=RULES,
            receiver_representation='surfel' if 'receiver_normals' in data else 'plane',
            receiver_normal_model='per_receiver_production_facing_consistent' if 'receiver_normals' in data else 'shared_plane_normal',
            fit_facing_disagreements=0 if 'receiver_normals' in data else None,
            checks=checks,relative_held_degradation=degradations,constant_held=result['metrics']['constant_held'],
            constant_fit=result['metrics']['constant_fit'],approximate_fit_metrics=result['metrics'],exact=metrics,
            parameters_refit=False,valid_changed=False,input_sha256=hashes,output_sha256=files,
            exact_bvh=True,cache_nearest_visibility_used_for_validation=False,effective_rasterizer=backend.provenance,
            manual_review='PENDING',scope='TRAIN内部受限混合照明可靠性；不是GT灯向或三维几何精度')
        write(args.output/'summary.json',report)
        with (args.output/'README.md').open('a') as f:
            f.write('\n## 实际执行\n\n数值状态：'+report['status']+'；三档完整指标与检查项见summary.json。原始固定valid与rho/EL/方向未变化；无GT/外部test读取。\n\n')
            f.write('最终结论：'+('PENDING（数值门槛通过，必须主任务实际目检）' if passed else '**FAILED（内部数值门槛未同时满足，完整现场保留）**')+'。internal_validation_passed=false，不导出资产。\n')
        write(args.output/'EXECUTION_COMPLETED.json',dict(status='PASS_EXECUTION',numeric_status=report['status'],manual_review='PENDING',internal_validation_passed=False))
    except Exception as exc:
        write(args.output/'FAILED.json',dict(status='FAILED',error=str(exc),retained=True,internal_validation_passed=False))
        with (args.output/'README.md').open('a') as f:f.write('\n最终结论：**FAILED（执行异常，全部结果保留）**。\n')
        raise


def self_check():
    observed=torch.ones((2,2,3));prediction=observed.clone();prediction[0,0]=2;prediction[1,0]=0
    saturated=torch.zeros_like(observed,dtype=torch.bool);saturated[:,0]=True
    data=dict(rgb=observed,saturated=saturated,valid=torch.tensor([[True,False],[True,False]]),fit=torch.tensor([True,False]))
    residual=censored_residual(prediction,observed,saturated)
    assert residual[0,0].eq(0).all() and residual[1,0].eq(-1).all()
    # score的固定135帧表单另验证完整合成输入；所有帧100点，不通过残差重筛。
    data=dict(rgb=torch.ones(135,100,3),saturated=torch.zeros(135,100,3,dtype=torch.bool),
        valid=torch.ones(135,100,dtype=torch.bool),fit=torch.tensor([n in FIT for n in TRAIN]))
    value,_,_=score(torch.zeros(135,100,3),data)
    assert value['fit']['frame_mean']==1 and value['held']['frame_mean']==1 and value['held']['supported_frames']==18
    metrics={k:dict(fit=dict(supported_frames=117),held=dict(supported_frames=18,frame_mean=v)) for k,v in [('0p5',1.25),('1',1.),('2',1.2)]}
    summary=dict(three_start_agreement=[dict(median_deg=1.)],preliminary_checks_passed=True,selected_by='fit_censored_mse_only')
    result=dict(bound_active=False,metrics=dict(constant_held=2.))
    checks,_=numeric_checks(metrics,summary,result);assert all(checks.values())
    metrics['2']['held']['frame_mean']=1.251
    checks,_=numeric_checks(metrics,summary,result);assert not checks['2_degradation_within_25pct']
    directions=torch.tensor([[1.,0,0],[0,0,1.]])
    cos=receiver_cosines(dict(points=torch.zeros(2,3),normal=torch.tensor([0.,0,1.])),directions)
    torch.testing.assert_close(cos,torch.tensor([[0.,0],[1.,1.]]))
    cos=receiver_cosines(dict(points=torch.zeros(2,3),receiver_normals=torch.tensor([[0.,0,1.],[1.,0,0.]])),directions)
    torch.testing.assert_close(cos,torch.tensor([[0.,1.],[1.,0.]]))
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([])
    for path in ('/tmp/validate_eldl_forbidden_gt.npz','/tmp/lights.json'):
        try:open(path,'rb')
        except PermissionError:pass
        else:raise AssertionError('no_gt_guard未拒绝未登记GT。')
    print('PASS：CPU删失/固定分母/支持/25%门槛边界与未登记GT拒绝检查；非精度结果。')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['plan','cpu-check','run','self-check'],required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--fit',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--model',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--iteration',type=int,default=16000);p.add_argument('--batch-directions',type=int,default=16)
    args=p.parse_args()
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=PYTHON.resolve():p.error('必须ushas/lumimotion-ushas。')
    if args.action=='self-check':self_check();return
    if any(getattr(args,k) is None for k in ('cache','fit','source','model','output')):p.error('需要cache/fit/source/model/output。')
    for key in ('cache','fit','source','model','output'):setattr(args,key,getattr(args,key).resolve())
    if not args.output.is_relative_to(ROOT/'output'):p.error('输出必须位于output。')
    if not 1<=args.batch_directions<=32:p.error('方向batch只允许1～32，不改科学协议。')
    if args.iteration!=16000 and 'smoke_test' not in args.output.parts:p.error('正式几何仅预登记I00@16000。')
    if args.action=='plan':plan(args);return
    if args.action=='cpu-check':
        verify_plan(args);install_guard(args)
        data,state,summary,result,hashes=load_inputs(args)
        write(args.output/'cpu_receipt.json',dict(status='PASS_CPU_SOURCE_CHECK',gt_read=False,selected_init=summary['selected_init'],input_sha256=hashes,
            actual_bvh_run=False,internal_validation_passed=False));print('PASS：CPU来源与冻结产物核验，尚未执行BVH。');return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('运行前必须协调并显式指定CUDA_VISIBLE_DEVICES。')
    run(args)


if __name__=='__main__':main()
