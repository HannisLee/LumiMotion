"""固定旧混合光的纯几何预热初始化；不继承SH外观或优化器状态。"""
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from scene.eldl_v3_trainer import ELDLTrainer
from scripts.train_eldl_v3_late_anneal import LateAnnealTrainer
from utils.eldl_v3_late_lr import LateLearningRate,ACTIVE_GROUPS
from utils.sh_utils import RGB2SH

ENTRY='scripts/train_eldl_v3_geometry_warmstart.py'
KEYS={'_xyz':3,'_scaling':2,'_rotation':4,'_opacity':1}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)

def validate_parent(args,manifest):
    source=Path(args.source_path).resolve();control=source.parent;repo=control.parent.parent
    require(control.name=='1019-01-LH-static-jumpingjacks-ELDL-V3-总控' and source.name=='data_train_eldl_raster_v1','只接纳已登记旧混合光源。')
    base=repo/('output/smoke_test/1019-17-ELDL-V3-SH真实更新保存与纯几何包' if args.smoke_schedule else 'output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热')
    model=base/'model_mlp';path=model/'geometry_parent_manifest.json'
    require(Path(args.geometry_parent_manifest).resolve()==path,'父manifest不是本轮指定的SH实验。')
    review_path=base/'parent_geometry_engineering_review.json';review=read(review_path);parent=read(path)
    state=read(base/'state.json');record=read(model/'eldl_v3_training.json')
    require(state['exit_code']==0 and review['status']=='PASS_SH_PARENT_GEOMETRY_ENGINEERING','父训练/纯几何工程验收未完成。')
    require(review['smoke']==parent['smoke']==args.smoke_schedule,'禁止smoke权重进入正式训练。')
    allowed=[3,20] if args.smoke_schedule else [1000,16000]
    require(review['allowed_steps']==allowed and args.geometry_step in allowed,'未登记的父端点。')
    require(review['parent_manifest']==str(path) and review['parent_manifest_sha256']==sha(path),'父验收绑定失效。')
    require(parent['gt_supervision'] is False and parent['external_test_read'] is False and parent['pure_geometry_only'],'父模型来源越界。')
    require(parent['checkpoint_semantics']=='post_optimizer_post_native_topology' and parent['completed_optimizer_calls']==(20 if args.smoke_schedule else 16000),'拒绝旧更新前或N-1保存。')
    require(record['actual_entrypoint']=='scripts/train_eldl_v3_sh_warmup.py' and record['baseline']=='static_sh' and record['init_weights_loaded'] is False,'不是新的纯TRAIN随机SH父模型。')
    require(sha(model/'eldl_v3_training.json')==parent['model_record_sha256'],'父训练记录变更。')
    frames=[i for i in range(1,151) if i%10];require(parent['train_frames']==record['train_frames']==manifest['train_frames']==frames,'135TRAIN身份不一致。')
    require(len(parent['source_files'])==137 and {'points3d.ply','transforms_train.json'}.issubset(parent['source_files']),'父源缺必需文件。')
    for name,value in parent['source_files'].items():
        file=(source/name).resolve();require(file.is_relative_to(source) and manifest['files'].get(name)==value and sha(file)==value,'预热与物理阶段原始输入不同。')
    endpoint=parent['endpoints'][str(args.geometry_step)];folder=model/f'geometry_init/actual_{args.geometry_step}'
    require(endpoint['geometry']==str(folder/'geometry.pt') and endpoint['receipt']==str(folder/'receipt.json'),'父权重路径逃逸。')
    paths=[path,review_path,base/'state.json',model/'eldl_v3_training.json',folder/'geometry.pt',folder/'receipt.json']
    for key in ('geometry','receipt'):
        file=Path(endpoint[key]);require(file.resolve()==file and sha(file)==endpoint[key+'_sha256'],'纯几何来源SHA变化。')
        require(review['input_sha256'].get(str(file))==endpoint[key+'_sha256'],'工程验收未绑定当前权重。')
    for file in (base/'state.json',model/'eldl_v3_training.json'):
        require(review['input_sha256'].get(str(file))==sha(file),'父完成状态或模型记录在验收后变更。')
    require(review['gt_read'] is False and review['external_test_read'] is False,'父验收越界。')
    receipt=read(folder/'receipt.json')
    require(receipt['checkpoint_semantics']==parent['checkpoint_semantics'],'父端点保存语义不符。')
    require(receipt['requested_step']==receipt['completed_optimizer_calls']==args.geometry_step and receipt['geometry_heads_zero'],'实际保存步数或静态几何错误。')
    require(receipt['geometry_keys']==list(KEYS) and receipt['gt_supervision'] is False,'几何包声明非法。')
    return dict(parent_manifest=str(path),parent_manifest_sha256=sha(path),parent_review=str(review_path),parent_review_sha256=sha(review_path),
        geometry=str(folder/'geometry.pt'),geometry_sha256=endpoint['geometry_sha256'],geometry_step=args.geometry_step,
        checkpoint_semantics=parent['checkpoint_semantics'],parent_source=parent['source'],gt_supervision=False,
        pure_geometry_only=True,colors_loaded=False,mlp_loaded=False,optimizer_loaded=False,rng_loaded=False,
        approved_paths=[str(p) for p in paths],input_sha256={str(p):sha(p) for p in paths})


class GeometryWarmstartTrainer(LateAnnealTrainer):
    def __init__(self,*args,**kwargs):
        config=kwargs['args'];require(config.static_dl_physical_start==0 and config.late_lr_case=='anneal','必须纯物理相对步和同一退火。')
        if not config.smoke_schedule:require(config.iterations==24000,'物理阶段固定24000。')
        # 跳过LateAnnealTrainer的构造：先重建模型/Adam，再装相同lr策略。
        ELDLTrainer.__init__(self,*args,**kwargs)
        binding=config.geometry_initialization_binding
        require(sha(binding['geometry'])==binding['geometry_sha256'],'初始化权重变化。')
        tensors=torch.load(binding['geometry'],map_location='cpu',weights_only=True)
        require(set(tensors)==set(KEYS),'不得加载颜色/MLP/优化器。');count=len(tensors['_xyz']);require(count>0,'空父几何。')
        require(all(value.dtype==torch.float32 and value.shape==(count,KEYS[name]) and torch.isfinite(value).all() for name,value in tensors.items()),'纯几何schema错误。')
        scales=tensors['_scaling'].exp();require(torch.isfinite(scales).all() and (scales>0).all() and (tensors['_rotation'].norm(dim=1)>0).all(),'非法尺度/四元数。')
        pc=self.gaussians;require(self.scene.gaussians is pc and not hasattr(pc,'_albedo_dc_stage1'),'模型身份或stage1附加组错误。')
        for name,value in tensors.items():setattr(pc,name,torch.nn.Parameter(value.cuda().clone(),requires_grad=True))
        pc._albedo_dc=torch.nn.Parameter(torch.full((count,1,3),float(RGB2SH(.5)),device='cuda'))
        pc._albedo_rest=torch.nn.Parameter(torch.zeros((count,(pc.max_sh_degree+1)**2-1,3),device='cuda'))
        pc._roughness=torch.nn.Parameter(torch.full((count,1),math.log(3.),device='cuda'))
        pc.feature=torch.nn.Parameter(torch.full((count,pc.fea_dim),float(pc.binarization_init_value),device='cuda'))
        require(float(pc.binarization_init_value)==-.01,'固定feature初始化不同。')
        pc.active_sh_degree=0;pc.max_radii2D=torch.zeros(count,device='cuda');pc.training_setup(self.opt)
        require(len(pc.optimizer.state)==0,'新Adam必须为空。')
        self._begin_physical();require(len(pc.optimizer.state)==0,'开始物理时意外继承Adam。')
        require(all(not parameter.requires_grad for parameter in self.deform.deform.parameters()),'MLP未完全冻结。')
        require(all(torch.equal(getattr(pc,key).detach().cpu(),value) for key,value in tensors.items()),'原始四几何值被修改。')
        require(not pc.xyz_gradient_accum.any() and not pc.denom.any() and not pc.max_radii2D.any(),'增密统计未清零。')
        for group in pc.optimizer.param_groups:
            require(group['name']!='albedo_dc_stage1','禁用旧stage1参数组。')
            for parameter in group['params']:parameter.grad=None
        self.visibility.invalidate();self.viewpoint_stack=None;self._camera_logged_step=0
        self.late_start,self.late_end=(8,16) if config.smoke_schedule else (16000,20000)
        self.lr_policy=LateLearningRate(pc,self.late_start,self.late_end,enabled=True);pc.update_learning_rate=self.lr_policy
        root=Path(__file__).resolve().parents[1];model=Path(config.model_path)
        source_files=('scripts/train_eldl_v3_geometry_warmstart.py','scene/eldl_v3_geometry_init.py','scripts/train_eldl_v3_late_anneal.py','utils/eldl_v3_late_lr.py')
        sources={str(root/p):sha(root/p) for p in source_files}
        late=dict(actual_entrypoint=ENTRY,start=self.late_start,end=self.late_end,case='anneal',final_multiplier=.1,
            active_groups=ACTIVE_GROUPS,original_lrs=self.lr_policy.base,adam_moments_changed=False,
            lighting_unchanged=True,loss_unchanged=True,topology_policy_unchanged=True,gt_supervision=False,sources=sources,
            semantics='同相对物理步晚期退火；初始化前的SH年龄不影响优化日程')
        (model/'late_lr_training.json').write_text(json.dumps(late,ensure_ascii=False,indent=2)+'\n')
        torch.save(tensors,model/'effective_initial_geometry.pt')
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        # step0没有已完成的scheduler调用，显式经下层保存同一生产格式。
        current=self.iteration;self.iteration=0
        ELDLTrainer._save_physical(self)
        self.iteration=current
        zero_path=model/'static_dl_lighting/iteration_0/lighting.json';zero=read(zero_path)
        zero.update(actual_entrypoint=ENTRY,relative_physical_step=0,completed_optimizer_calls=0,
            geometry_parent_manifest_sha256=binding['parent_manifest_sha256'],geometry_initialization_step=config.geometry_step,
            next_optimizer_step=1,initial_lrs={g['name']:float(g['lr']) for g in pc.optimizer.param_groups})
        zero_path.write_text(json.dumps(zero,ensure_ascii=False,indent=2)+'\n')
        initial=dict(status='PASS_PURE_GEOMETRY_CONVERSION',actual_entrypoint=ENTRY,geometry_binding=binding,
            parent_weights_loaded=True,loaded_tensor_keys=list(KEYS),points=count,geometry_bitexact=True,
            empty_adam=True,densification_statistics_zero=True,visibility_invalidated=True,viewpoint_stack_empty=True,
            initial_srgb_albedo=.5,initial_linear_albedo=.21404114048223255,initial_feature=-.01,initial_roughness=.75,
            colors_loaded=False,mlp_loaded=False,optimizer_loaded=False,rng_loaded=False,
            reseeded_after_conversion=True,seed=0,relative_physical_step=0,gt_supervision=False,
            input_ply_interpretation='Scene原随机输入留档；本次有效起点见effective_initial_geometry.pt和iteration_0',
            effective_initial_geometry_sha256=sha(model/'effective_initial_geometry.pt'),source_sha256=sources)
        (model/'geometry_initialization.json').write_text(json.dumps(initial,ensure_ascii=False,indent=2)+'\n')
        for name in ('eldl_v3_training.json','unknown_light_training.json','static_dl_training.json'):
            path=model/name;record=read(path);record.update(actual_entrypoint=ENTRY,late_learning_rate=late,
                geometry_initialization=initial,weights_loaded_from_parent=True,
                initialization='本轮完整135TRAIN静态SH父模型仅四几何张量；新材质、Adam和RNG',relative_physical_steps=True)
            if 'initial_point_count' in record:record['initial_point_count']=count
            if 'new_sources' in record:record['new_sources'].update(sources)
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')

    def _render_photometric(self,camera,use_visibility):
        if self.iteration<=135 and self._camera_logged_step!=self.iteration:
            with (Path(self.args.model_path)/'first_cycle_cameras.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(physical_step=self.iteration,camera=camera.image_name_train_light))+'\n')
            self._camera_logged_step=self.iteration
        return super()._render_photometric(camera,use_visibility)

    def _save_physical(self):
        super()._save_physical()
        path=Path(self.args.model_path)/f'static_dl_lighting/iteration_{self.iteration}/lighting.json'
        meta=read(path);meta.update(actual_entrypoint=ENTRY,relative_physical_step=self.iteration,
            geometry_parent_manifest_sha256=self.args.geometry_initialization_binding['parent_manifest_sha256'],
            geometry_initialization_step=self.args.geometry_step)
        path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')

    @torch.no_grad()
    def audit_smoke_conversion(self):
        """真实训练对象与磁盘生产PLY重新加载的三TRAIN视角前向对照。"""
        from scene.gaussian_model import GaussianModel
        from scene.static_dl_photometric import StaticVisibility
        from scene.eldl_v3_photometric import render_photometric
        model=Path(self.args.model_path)
        before=(repr(random.getstate()),repr(np.random.get_state()),torch.get_rng_state(),torch.cuda.get_rng_state())
        clone=GaussianModel(self.gaussians.max_sh_degree,no_binary_separation=self.gaussians.no_binary_separation,fea_dim=self.gaussians.fea_dim)
        clone.load_ply(str(model/'point_cloud/iteration_0/point_cloud.ply'))
        require(all(torch.equal(getattr(clone,k),getattr(self.gaussians,k)) for k in KEYS),'step0生产PLY几何非逐位相等。')
        rows=[]
        for camera in self.scene.getTrainCameras():
            frame=int(camera.image_name_train_light.split('_')[-1])
            if frame not in (1,81,149):continue
            camera.load2device()
            for visible in (False,True):
                traces=[StaticVisibility(self.args.static_dl_shadow_bias,1000000) for _ in range(2)]
                packages=[render_photometric(camera,pc,self.pipe,self.background,self.physical_lights,traces[i] if visible else None,0) for i,pc in enumerate((self.gaussians,clone))]
                differences={name:float((packages[0][name]-packages[1][name]).abs().max()) for name in ('render','rend_alpha','rend_normal','surf_depth')}
                require(all(math.isfinite(v) and v<1e-5 for v in differences.values()),'step0生产重载前向不一致。')
                rows.append(dict(frame=frame,full_bvh=visible,max_abs=differences))
            camera.load2device('cpu')
        require(len(rows)==6,'缺少预登记三个TRAIN视角。')
        require(before[0]==repr(random.getstate()) and before[1]==repr(np.random.get_state()) and torch.equal(before[2],torch.get_rng_state()) and torch.equal(before[3],torch.cuda.get_rng_state()),'转换前向审计改变RNG。')
        require(not self.gaussians.optimizer.state and self.viewpoint_stack is None,'转换前向审计改变优化器或相机栈。')
        (model/'conversion_forward_audit.json').write_text(json.dumps(dict(status='PASS_LIVE_VS_PRODUCTION_RELOAD',rows=rows,gt_read=False,train_frames=[1,81,149],no_optimizer_updates=True,rng_unchanged=True),ensure_ascii=False,indent=2)+'\n')
