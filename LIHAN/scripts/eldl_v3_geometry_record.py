"""纯几何预热物理入口的来源核查；评测前向仍是同一固定EL+DL。"""
import hashlib
import json
from pathlib import Path
ENTRY='scripts/train_eldl_v3_geometry_warmstart.py'

def fixed_geometry_record_files(model,record):
    model=Path(model)
    def require(ok,message):
        if not ok:raise ValueError(message)
    def read(path):return json.loads(path.read_text())
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    initial_path=model/'geometry_initialization.json';late_path=model/'late_lr_training.json'
    initial=read(initial_path);late=read(late_path);args=record['args'];binding=initial['geometry_binding']
    require(record['actual_entrypoint']==ENTRY and initial['actual_entrypoint']==late['actual_entrypoint']==ENTRY,'纯几何入口身份不同。')
    require(initial==record['geometry_initialization'] and late==record['late_learning_rate'],'纯几何训练记录不一致。')
    require(args['static_dl_physical_start']==0 and record['relative_physical_steps'] and record['weights_loaded_from_parent'],'相对物理步或父权重声明错误。')
    require(binding==args['geometry_initialization_binding'] and binding['geometry_step']==args['geometry_step'],'父几何绑定不一致。')
    require(initial['status']=='PASS_PURE_GEOMETRY_CONVERSION' and initial['loaded_tensor_keys']==['_xyz','_scaling','_rotation','_opacity'],'转移集合错误。')
    require(all(initial[k] is False for k in ('colors_loaded','mlp_loaded','optimizer_loaded','rng_loaded','gt_supervision')),'纯几何加载越界。')
    require(all(initial[k] is True for k in ('geometry_bitexact','empty_adam','densification_statistics_zero','visibility_invalidated','viewpoint_stack_empty','reseeded_after_conversion')),'几何转换工程声明不全。')
    require(initial['points']==record['initial_point_count'] and initial['seed']==0 and initial['relative_physical_step']==0,'初始点数/随机种子/步数错误。')
    require(late['case']==args['late_lr_case']=='anneal' and late['final_multiplier']==.1 and
        (late['start'],late['end'])==((8,16) if args['smoke_schedule'] else (16000,20000)) and
        late['active_groups']==['xyz','albedo_dc','opacity','scaling','rotation'] and late['gt_supervision'] is False and late['adam_moments_changed'] is False and
        all(late[k] is True for k in ('lighting_unchanged','loss_unchanged','topology_policy_unchanged')),'共同退火语义不符。')
    require(initial['source_sha256']==late['sources'],'转换与退火来源不同。')
    names={Path(p).name for p in late['sources']}
    require(names=={'train_eldl_v3_geometry_warmstart.py','eldl_v3_geometry_init.py','train_eldl_v3_late_anneal.py','eldl_v3_late_lr.py'} and len(late['sources'])==4,'新入口源码集合不符。')
    files=[initial_path,late_path,Path(__file__).resolve(),model/'effective_initial_geometry.pt']
    require(sha(files[-1])==initial['effective_initial_geometry_sha256'],'有效初始几何变化。')
    for source,expected in late['sources'].items():
        path=Path(source)
        require(path.suffix=='.py' and record['new_sources'].get(source)==expected and sha(path)==expected,'新入口源码SHA变化。')
        files.append(path)
    # 不加载外部父权重，模型自身的完整训练来源已经绑定父manifest/收据及四张量SHA。
    return files
