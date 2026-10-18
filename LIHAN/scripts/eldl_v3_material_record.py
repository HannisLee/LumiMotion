"""登记已审计的TRAIN材质初始化；评测仍使用同一固定EL+DL前向。"""
import hashlib
import json
from pathlib import Path

ENTRY = 'scripts/train_eldl_v3_material_warmstart.py'


def fixed_material_record_files(model, record):
    from LIHAN.scripts.eldl_v3_geometry_record import fixed_geometry_record_files
    model = Path(model)

    def require(value, message):
        if not value:
            raise ValueError(message)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    require(record['actual_entrypoint'] == ENTRY and record['train_material_loaded'] is True, '不是登记的TRAIN材质入口。')
    # 旧记录明确属于构造中间阶段；只借用其四几何来源验证，不把最终DC称为灰值。
    geometry_record = dict(record, actual_entrypoint='scripts/train_eldl_v3_geometry_warmstart.py')
    files = fixed_geometry_record_files(model, geometry_record)
    path = model / 'material_initialization.json'
    material = json.loads(path.read_text())
    args = record['args']
    binding = material['material_binding']
    require(material == record['material_initialization'] and material['actual_entrypoint'] == ENTRY and material['status'] == 'PASS_TRAIN_MATERIAL_CONVERSION', '材质转换记录不一致。')
    require(binding == args['material_initialization_binding'] and binding['geometry_sha256'] == args['geometry_initialization_binding']['geometry_sha256'] and binding['parent_geometry_step'] == args['geometry_step'], '材质与几何来源不一致。')
    require(material['loaded_tensor_keys'] == ['_albedo_dc'] and material['points'] == binding['points'] == record['initial_point_count'], '不是仅DC加载或点数不同。')
    require(all(material[k] is True for k in ('geometry_bitexact', 'other_attributes_bitexact', 'train_material_loaded', 'empty_adam', 'densification_statistics_zero', 'viewpoint_stack_empty', 'reseeded_after_conversion')), '材质加载工程声明缺失。')
    require(all(material[k] is False for k in ('mlp_loaded', 'optimizer_loaded', 'rng_loaded', 'gt_supervision', 'external_test_read')), '材质加载越界。')
    require(binding['only_tensor'] == '_albedo_dc' and binding['optimizer_loaded'] is False and binding['rng_loaded'] is False and binding['gt_supervision'] is False and binding['external_test_read'] is False, '材质绑定越界。')
    require(binding['train_frames'] == [i for i in range(1, 151) if i % 10] and binding['preadapt_updates'] == (20 if args['smoke_schedule'] else 4050), '预适配数据或预算不符。')
    require(material['seed'] == material['relative_physical_step'] == 0 and binding['lighting_sha256'] == record['lighting_sha256'] and binding['source'] == args['source_path'], '随机种子/物理步/固定光不同。')
    require(set(Path(p).name for p in material['source_sha256']) == {'train_eldl_v3_material_warmstart.py', 'eldl_v3_material_init.py'} and len(material['source_sha256']) == 2, '新增入口源码不完整。')
    for source, expected in material['source_sha256'].items():
        file = Path(source)
        require(file.suffix == '.py' and record['new_sources'].get(source) == expected and sha(file) == expected, '新增材质源码变化。')
        files.append(file)
    effective = model / 'effective_initial_material.pt'
    require(sha(effective) == material['effective_initial_material_sha256'], '有效初始材质变化。')
    return [*files, path, effective, Path(__file__).resolve()]
