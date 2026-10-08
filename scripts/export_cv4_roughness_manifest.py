"""在 Blender 内只读核验 CV4 的常量 Principled 材质并导出 roughness 清单。

blender -b --factory-startup --disable-autoexec scene.blend --python SCRIPT -- --output FILE
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


MATERIALS = (
    ('cloth', '红布', 'FallingCloth_Burgundy'),
    ('seat', '椅面', 'DiningChair_Seat'),
    ('wood', '木框', 'DiningChair_Walnut'),
    ('platform', '平台', 'Circular_Platform_Material'),
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_material(material):
    if not material.use_nodes:
        raise ValueError('材质未使用节点：' + material.name)
    outputs = [n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
    if len(outputs) != 1 or len(outputs[0].inputs['Surface'].links) != 1:
        raise ValueError('材质输出不唯一：' + material.name)
    link = outputs[0].inputs['Surface'].links[0]
    node = link.from_node
    if node.type != 'BSDF_PRINCIPLED' or node.mute or link.from_socket.name != 'BSDF':
        raise ValueError('只支持直接连接的 Principled BSDF：' + material.name)
    for name in ('Roughness', 'Base Color', 'Alpha'):
        if node.inputs[name].is_linked:
            raise ValueError('不支持纹理/节点驱动的 ' + name + '：' + material.name)
    if node.inputs['Alpha'].default_value != 1:
        raise ValueError('首版仅支持不透明材质：' + material.name)
    roughness = float(node.inputs['Roughness'].default_value)
    rgb = list(node.inputs['Base Color'].default_value)[:3]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in [roughness] + rgb):
        raise ValueError('材质参数超出范围：' + material.name)
    srgb8 = [round(255 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)) for v in rgb]
    return {'blender_material': material.name, 'bsdf_node': node.name,
            'output_node': outputs[0].name, 'bsdf_type': node.type,
            'roughness': roughness, 'linear_rgb': rgb, 'srgb8': srgb8,
            'roughness_linked': False, 'base_color_linked': False,
            'direct_surface_connection': True}


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    blend = Path(bpy.data.filepath).resolve()
    before = sha256(blend)
    records = []
    for frame in range(8, 121, 8):
        bpy.context.scene.frame_set(frame)
        objects = [o for o in bpy.context.scene.objects if o.type == 'MESH' and not o.hide_render]
        used = {s.material.name for o in objects for s in o.material_slots if s.material}
        if used != {name for _, _, name in MATERIALS}:
            raise ValueError('可渲染物体材质集合与 CV4 四材质不一致。')
        current = []
        for name, label, blender_name in MATERIALS:
            record = inspect_material(bpy.data.materials[blender_name])
            record.update(name=name, label=label,
                          objects=sorted(o.name for o in objects if any(s.material and s.material.name == blender_name for s in o.material_slots)))
            current.append(record)
        if records and current != records:
            raise ValueError('测试帧之间材质发生变化，不支持常量清单。')
        records = current
    if sha256(blend) != before:
        raise AssertionError('源 Blender 文件发生变化。')
    manifest = {'schema_version': 1, 'dataset': 'only_clothV4',
                'parameter': 'principled_roughness_r', 'reference_blend': str(blend),
                'reference_blend_sha256': before, 'blender_version': bpy.app.version_string,
                'exporter_sha256': sha256(__file__), 'verified_frames': [f'{i:04d}' for i in range(8, 121, 8)],
                'source_unchanged': True, 'materials': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
