"""在内存中应用数据导出时的可见性配置，执行 GT 网格或粗糙度导出。"""
import json
import runpy
import sys
from pathlib import Path
import bpy

args = sys.argv[sys.argv.index('--') + 1:]
request, script = args[:2]
data = json.loads(Path(request).read_text())
for name in data.get('temporarily_hide_render_objects', []):
    if name in bpy.data.objects:
        bpy.data.objects[name].hide_render = True
        bpy.data.objects[name].hide_set(True)
bpy.context.view_layer.update()
sys.argv = [script, '--'] + args[2:]
runpy.run_path(script, run_name='__main__')
