import bpy,json,sys
from pathlib import Path
s=bpy.context.scene
r={'blender_version':bpy.app.version_string,'camera':s.camera.name,'materials':{},'visible_meshes':[],'packed_images':[]}
for o in s.objects:
 if o.type=='MESH' and not o.hide_render and not all(c.hide_render for c in o.users_collection):
  r['visible_meshes'].append({'name':o.name,'materials':[a.material.name if a.material else None for a in o.material_slots]})
for m in bpy.data.materials:
 if m.use_nodes:
  r['materials'][m.name]=[{'name':n.name,'type':n.type,'alpha_linked':n.inputs['Alpha'].is_linked,'alpha':n.inputs['Alpha'].default_value,'roughness_linked':n.inputs['Roughness'].is_linked,'roughness':n.inputs['Roughness'].default_value} for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
for i in bpy.data.images:r['packed_images'].append({'name':i.name,'packed':bool(i.packed_file),'size':list(i.size),'path':i.filepath})
Path(sys.argv[-1]).write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False))
