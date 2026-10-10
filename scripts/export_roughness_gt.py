"""从 Blender 导出逐帧原始 roughness RGBA EXR，无材质名称或颜色分类依赖。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--transforms', type=Path, required=True,
                        help='对应 split 的 transforms JSON，提供 source_frame、time 和相机。')
    parser.add_argument('--resolution', type=int, default=2, help='原始分辨率除数。')
    parser.add_argument('--alpha_composite', action='store_true',
                        help='允许单一Principled透明度；保持原Alpha，导出透射层alpha加权的有效roughness。')
    parser.add_argument('--engine', choices=['cycles', 'eevee', 'source'], default='cycles',
                        help='默认原有Cycles；透明数据推荐source以保留源引擎Alpha语义。')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.resolution < 1:
        parser.error('resolution 必须为正整数。')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    blend = Path(bpy.data.filepath).resolve()
    source_hash = digest(blend)
    metadata = json.loads(args.transforms.read_text())
    frames = metadata['frames']
    if not frames or any('source_frame' not in f or 'time' not in f for f in frames):
        raise ValueError('GT导出需要 transforms 中明确的 source_frame/time。')
    if len({f['source_frame'] for f in frames}) != len(frames):
        raise ValueError('source_frame 重复。')
    scene = bpy.context.scene
    if args.engine != 'source':
        scene.render.engine = 'CYCLES' if args.engine == 'cycles' else 'BLENDER_EEVEE'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.resolution_x = round(metadata['w'] / args.resolution)
    scene.render.resolution_y = round(metadata['h'] / args.resolution)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '32'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    # 临时替换着色输出，将真实 Roughness 输入接入自发光；不保存源 blend。
    replacements = []
    for material in bpy.data.materials:
        if not any(o.type == 'MESH' and not o.hide_render and any(s.material == material for s in o.material_slots)
                   for o in scene.objects):
            continue
        if not material.use_nodes:
            raise ValueError('不支持非节点材质：' + material.name)
        tree = material.node_tree
        outputs = [n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
        if len(outputs) != 1 or len(outputs[0].inputs['Surface'].links) != 1:
            raise ValueError('材质输出不唯一：' + material.name)
        output = outputs[0]
        link = output.inputs['Surface'].links[0]
        node = link.from_node
        if node.type != 'BSDF_PRINCIPLED' or node.mute:
            raise ValueError('首版仅支持直接连接的 Principled BSDF：' + material.name)
        alpha = node.inputs['Alpha']
        transparent = alpha.is_linked or alpha.default_value != 1
        if transparent and not args.alpha_composite:
            raise ValueError('首版不支持透明材质：' + material.name)
        emission = tree.nodes.new('ShaderNodeEmission')
        emission.inputs['Strength'].default_value = 1
        roughness = node.inputs['Roughness']
        if roughness.is_linked:
            # 与原 Roughness 浮点socket使用相同的颜色→标量转换，再广播为灰度RGB。
            scalar = tree.nodes.new('ShaderNodeMath')
            scalar.operation = 'MULTIPLY'
            scalar.inputs[1].default_value = 1
            scalar.use_clamp = True
            tree.links.new(roughness.links[0].from_socket, scalar.inputs[0])
            tree.links.new(scalar.outputs[0], emission.inputs['Color'])
        else:
            emission.inputs['Color'].default_value = [roughness.default_value] * 3 + [1]
            replacements.append((roughness, emission.inputs['Color'], True))
        shader = emission.outputs[0]
        if transparent:
            # 透明层的材质参数没有唯一第一命中值；保留原alpha，定义为
            # 各可见层按透射率×alpha加权的r，不将半透明头发强制变为不透明。
            mix = tree.nodes.new('ShaderNodeMixShader')
            clear = tree.nodes.new('ShaderNodeBsdfTransparent')
            tree.links.new(clear.outputs[0], mix.inputs[1])
            tree.links.new(shader, mix.inputs[2])
            if alpha.is_linked:
                scalar = tree.nodes.new('ShaderNodeMath')
                scalar.operation = 'MULTIPLY'
                scalar.inputs[1].default_value = 1
                scalar.use_clamp = True
                tree.links.new(alpha.links[0].from_socket, scalar.inputs[0])
                tree.links.new(scalar.outputs[0], mix.inputs[0])
            else:
                mix.inputs[0].default_value = alpha.default_value
                replacements.append((alpha, mix.inputs[0], False))
            shader = mix.outputs[0]
        tree.links.remove(link)
        tree.links.new(shader, output.inputs['Surface'])
    report = {'schema_version': 1, 'parameter': 'roughness_r', 'source_blend': str(blend),
              'source_blend_sha256': source_hash, 'transforms': str(args.transforms.resolve()),
              'transforms_sha256': digest(args.transforms), 'frames': {},
              'width': scene.render.resolution_x, 'height': scene.render.resolution_y,
              'method': 'Principled Roughness input -> emission -> raw float RGBA EXR',
              'alpha_association': 'premultiplied_RGB',
              'render_engine': scene.render.engine,
              'transparent_parameter_definition': ('transmittance-alpha weighted effective r'
                                                    if args.alpha_composite else 'opaque only'),
              'blender_version': bpy.app.version_string}
    for record in frames:
        frame = int(record['source_frame'])
        scene.frame_set(frame)
        for source, target, color in replacements:
            target.default_value = [source.default_value] * 3 + [1] if color else source.default_value
        matrix = scene.camera.matrix_world
        if max(abs(matrix[y][x] - record['transform_matrix'][y][x]) for y in range(4) for x in range(4)) > 1e-5:
            raise ValueError('Blender 相机与评测 transforms 不匹配：' + str(frame))
        # 导出必须沿用原始相机内参；禁止只凭相同帧号使用不同相机。
        width, height = metadata['w'], metadata['h']
        focal = scene.camera.data.lens / scene.camera.data.sensor_width * width
        fx = record.get('fl_x', metadata.get('fl_x'))
        fy = record.get('fl_y', metadata.get('fl_y'))
        if (scene.camera.data.type != 'PERSP' or scene.camera.data.sensor_fit == 'VERTICAL'
                or abs(scene.camera.data.shift_x) > 1e-6 or abs(scene.camera.data.shift_y) > 1e-6
                or fx is None or fy is None or abs(focal - fx) > 1e-3 or abs(fx - fy) > 1e-3
                or abs(record.get('cx', metadata.get('cx', width / 2)) - width / 2) > 1e-5
                or abs(record.get('cy', metadata.get('cy', height / 2)) - height / 2) > 1e-5):
            raise ValueError('首版导出要求与 transforms 一致的居中方形像素透视相机。')
        path = args.output_dir / f'roughness_{frame:04d}.exr'
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        report['frames'][f'{frame:04d}'] = {'file': path.name, 'sha256': digest(path), 'time': record['time'],
                                         'camera_to_world': record['transform_matrix']}
    if digest(blend) != source_hash:
        raise AssertionError('源 Blender 文件发生变化。')
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
