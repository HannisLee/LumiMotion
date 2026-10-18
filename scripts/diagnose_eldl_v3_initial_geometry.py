"""两个已登记几何起点的完整已见TRAIN光度兼容诊断；无优化、无择优。"""
import argparse,json,hashlib,math,sys,socket,os
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CONTROL=ROOT/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控'
FRAMES=[i for i in range(1,151) if i%10]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    require(socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve(),'需要ushas对应环境。')
    require(out.is_relative_to(CONTROL) and not out.exists(),'仅总控新目录，不覆盖。')
    source=CONTROL/'data_train_eldl_raster_v1';experiments=[next((ROOT/'output').glob(f'1019-{i}-*')) for i in (12,13)];models=[e/'model_mlp' for e in experiments]
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([source,out,*models])
    from LIHAN.scripts.eldl_v3_geometry_record import fixed_geometry_record_files
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from scene.static_dl_photometric import StaticVisibility
    from scene.eldl_v3_photometric import FixedELDLLights,render_photometric,censored_symmetric_loss
    from scene.stage2_photometric import srgb_to_linear
    import diff_surfel_rasterization as backend
    manifest=read(source/'no_gt_manifest.json');require(manifest['gt_read'] is False and manifest['train_frames']==FRAMES,'须原135TRAIN。')
    hashes={str(Path(__file__).resolve()):sha(__file__),str(source/'no_gt_manifest.json'):sha(source/'no_gt_manifest.json')}
    for name,value in manifest['files'].items():
        f=(source/name).resolve();require(f.is_relative_to(source) and sha(f)==value,'输入来源SHA变化。');hashes[str(f)]=value
    markers=[read(m/'eldl_v3_training.json') for m in models]
    for model,marker in zip(models,markers):
        extra=fixed_geometry_record_files(model,marker)
        static=read(model/'static_dl_training.json');provenance=static['rasterizer_provenance']
        code_root=Path(next(iter(marker['new_sources']))).parents[1]
        for relative in ('scene/eldl_v3_photometric.py','scene/static_dl_photometric.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','utils/rgba16.py','utils/camera_utils.py','scene/dataset_readers.py','scene/cameras.py'):
            actual,original=ROOT/relative,code_root/relative
            require(sha(actual)==sha(original),'初始诊断与训练核心源码不同。');hashes[str(actual)]=sha(actual);hashes[str(original)]=sha(original)
        require(backend.provenance['method']=='repository_lumimotion_cuda_rasterizer' and sha(backend._C.__file__)==provenance['extension_sha256'],'实际native二进制变化。')
        for f in [*extra,model/'eldl_v3_training.json',model/'static_dl_training.json',model/'point_cloud/iteration_0/point_cloud.ply',model/'static_dl_lighting/iteration_0/lighting.json',model/'optimizer_0.pt']:
            hashes[str(f)]=sha(f)
        opt=torch.load(model/'optimizer_0.pt',map_location='cpu',weights_only=False)
        require(opt['iteration']==0 and not opt['gaussian_optimizer']['state'] and not opt['light_optimizer']['state'],'只读零更新起点。')
    require([v['args']['geometry_step'] for v in markers]==[1000,16000],'必须双预登记起点。')
    require(markers[0]['geometry_initialization']['geometry_binding']['parent_manifest_sha256']==markers[1]['geometry_initialization']['geometry_binding']['parent_manifest_sha256'],'父轨迹不同。')
    import surfel_tracer.raytracer as tracer_module
    tracer_file=Path(tracer_module._C.__file__).resolve()
    require(sha(tracer_file)=='5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e','射线库不是已验收二进制。')
    hashes[str(tracer_file)]=sha(tracer_file);hashes[str(Path(backend._C.__file__).resolve())]=sha(backend._C.__file__)
    cfg=SimpleNamespace(**markers[0]['args']);cameras=[]
    for i,info in enumerate(readCamerasFromTransforms(str(source),'transforms_train.json','images',False,no_bg=True)):
        require(int(info.image_name_train_light.split('_')[-1])==FRAMES[i],'相机顺序不同。');cameras.append(loadCam(cfg,i,info,1.))
    lights=FixedELDLLights(source/'estimated_eldl_train.json');pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    out.mkdir();write(out/'environment.json',dict(hostname=socket.gethostname(),python=sys.executable,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),backend=backend.provenance))
    groups=[]
    for label,model,marker in zip(('M06','M07'),models,markers):
        pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(model/'point_cloud/iteration_0/point_cloud.ply'))
        require(torch.max((pc.get_albedo-.5).abs())<1e-6,'共同初始材质不是统一sRGB0.5。')
        env_color=srgb_to_linear(pc.get_albedo[0])*lights.ambient_rgb/math.pi
        trace=StaticVisibility(marker['args']['static_dl_shadow_bias'],1000000);rows=[];maximum_geometry_difference=0.
        for frame,camera in zip(FRAMES,cameras):
            camera.load2device();target=camera.original_image_train_light.cuda();alpha=camera.gt_alpha_mask.cuda();mask=alpha>=.5;denom=int(mask.sum())
            reference=None
            for visible in (False,True):
                package=render_photometric(camera,pc,pipe,torch.zeros(3,device='cuda'),lights,trace if visible else None,0)
                photo,_=censored_symmetric_loss(package['render'],target,alpha,camera.observed_saturation_mask)
                env=env_color[:,None,None]*package['rend_alpha'];direct=package['render_linear']-env
                require(float(direct.min())>-1e-5,'线性DL分解出现非舍入级负值。')
                # 仅统一rho零步才可由alpha精确恢复常量EL；不是对空间材质的一般分解。
                predicted=package['rend_alpha']>=.5;z=package['surf_depth'];valid=predicted & torch.isfinite(z) & (z>0)
                saturated=camera.observed_saturation_mask.cuda();unsat=mask.expand_as(target)&~saturated
                row=dict(frame=frame,mode='counterfactual_full_BVH' if visible else 'actual_initial_no_V',
                    censored_photo=float(photo),unsaturated_foreground_mae=float((package['render']-target).abs()[unsat].mean()),
                    alpha_iou=float((predicted&mask).sum()/(predicted|mask).sum().clamp_min(1)),
                    fixed_source_alpha_pixels=denom,valid_depth_pixels=int((valid&mask).sum()),
                    source_depth_coverage=float((valid&mask).sum()/max(1,denom)),
                    expected_Z_foreground_mean=float(z[valid&mask].mean()),
                    environment_linear_foreground_mean=env[:,mask[0]].mean(1).cpu().tolist(),
                    direct_linear_foreground_mean=direct[:,mask[0]].mean(1).cpu().tolist(),
                    visibility=float(package['mean_visibility']))
                current={k:package[k] for k in ('rend_alpha','rend_normal','surf_depth')}
                if reference is None:reference={k:v.clone() for k,v in current.items()}
                else:
                    d=max(float((current[k]-reference[k]).abs().max()) for k in current);require(math.isfinite(d) and d<1e-5,'可见性改变几何输出。');maximum_geometry_difference=max(maximum_geometry_difference,d)
                rows.append(row)
            camera.load2device('cpu')
        modes={}
        for mode in ('actual_initial_no_V','counterfactual_full_BVH'):
            selected=[r for r in rows if r['mode']==mode];require(len(selected)==135,'不完整TRAIN。')
            modes[mode]={key:sum(r[key] for r in selected)/135 for key in ('censored_photo','unsaturated_foreground_mae','alpha_iou','source_depth_coverage','expected_Z_foreground_mean','visibility')}
            for key in ('environment_linear_foreground_mean','direct_linear_foreground_mean'):modes[mode][key]=[sum(r[key][c] for r in selected)/135 for c in range(3)]
            modes[mode]['fixed_source_alpha_pixels']=sum(r['fixed_source_alpha_pixels'] for r in selected)
        result=dict(label=label,model=str(model),physical_iteration=0,parent_step=marker['args']['geometry_step'],points=len(pc.get_xyz),rows=rows,modes=modes,visibility_geometry_max_abs=maximum_geometry_difference)
        write(out/(label+'.json'),result);groups.append({k:v for k,v in result.items() if k!='rows'});print(label,modes,flush=True);del pc,trace
    for path,value in hashes.items():require(sha(path)==value,'运行期间来源改变。')
    write(out/'summary.json',dict(status='PASS_INITIAL_COMPATIBILITY_DIAGNOSTIC_EXECUTION',gt_read=False,external_test_read=False,no_parameter_updates=True,all_135_seen_TRAIN=True,no_checkpoint_selection=True,training_intervention=False,groups=groups,input_sha256=hashes,forward_count=540,
        scope='统一初始rho下的光度兼容，不是几何准确性；全BVH为反事实，真实初始训练无V。EL线性分解仅在全部surfel共同rho、黑背景时由alpha成立；sRGB不可直接相加。'))
if __name__=='__main__':main()
