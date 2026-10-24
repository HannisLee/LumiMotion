"""拆分宽平面各门禁的TRAIN渲染贡献，不修改训练支持或模型。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull,Delaunay,cKDTree
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from utils.general_utils import build_rotation
from utils.no_gt_guard import install_no_gt_guard
from gaussian_renderer import render
from PIL import Image,ImageDraw,ImageFont


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main(args):
    for key in ['model','source','plane_audit','cache','output','parent','expected_support']:setattr(args,key,getattr(args,key).resolve())
    out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    plane=json.loads(args.plane_audit.read_text());assert plane['gt_read'] is False and plane['heldout_read'] is False
    assert sha(args.cache) in plane['source_sha256'].values()
    install_no_gt_guard([args.model,args.source,args.parent,args.plane_audit,args.cache,out,args.expected_support])
    torch.set_num_threads(4);model=DynamicSurfaceModel.load(args.model,0);cameras=load_train_cameras(args.source)
    manifest=json.loads((args.source/'no_gt_manifest.json').read_text());frames=[int(c.image_name_train_light.split('_')[-1]) for c in cameras];assert frames==manifest['train_frames'] and len(frames)==117
    normal=np.array(plane['plane_normal_unsigned']);origin=np.array(plane['plane_origin']);extent=float(plane['extent'])
    cache=torch.load(args.cache,map_location='cpu',weights_only=False);sample=cache['canonical_xyz'].numpy()
    sample_mask=(~cache['dynamic'].numpy())&(np.abs((sample-origin)@normal)<extent*.005)&(cache['valid'].sum(0).numpy()>=10)
    axis=np.cross(normal,[1.,0,0]);axis/=np.linalg.norm(axis);basis=np.stack([axis,np.cross(normal,axis)])
    uv=(sample[sample_mask]-origin)@basis.T;tri=Delaunay(uv[ConvexHull(uv).vertices])
    xyz=model.pc.get_xyz.cpu().numpy();gate=model.pc.get_binary_feature().cpu().numpy().reshape(-1)
    axes=build_rotation(model.pc.get_rotation)[:,:,2].cpu().numpy()
    masks=dict(high_confidence_static=gate<.1,near_plane=np.abs((xyz-origin)@normal)<extent*.03,
        within_train_hull=tri.find_simplex((xyz-origin)@basis.T)>=0,normal_not_side=np.abs(axes@normal)>.7)
    selected=np.flatnonzero(np.logical_and.reduce(list(masks.values())))
    low=np.full((len(selected),3),np.inf);high=-low.copy();nearest=np.full(len(selected),np.inf)
    for j,camera in enumerate(cameras):
        camera.load2device();current=model.effective(camera).get_xyz.cpu().numpy();points=current[selected]
        low=np.minimum(low,points);high=np.maximum(high,points)
        nearest=np.minimum(nearest,cKDTree(current[gate>.5]).query(points,workers=4)[0])
        camera.load2device('cpu')
        if j%20==0:print('TRAIN形变支持核验',j+1,'/117',flush=True)
    motion=np.linalg.norm(high-low,axis=-1);keep=(motion<extent*.001)&(nearest>extent*.02);indices=selected[keep]
    expected=np.load(args.expected_support)['indices'];assert np.array_equal(indices,expected),'只读复核必须逐索引一致'
    stage_masks=dict(raw_without_hull=(masks['high_confidence_static']&masks['near_plane']&masks['normal_not_side']),
        after_hull=np.isin(np.arange(len(xyz)),selected),
        after_motion=np.isin(np.arange(len(xyz)),selected[motion<extent*.001]),
        after_clearance=np.isin(np.arange(len(xyz)),indices))
    tensors={k:torch.tensor(v,device='cuda',dtype=torch.float32)[:,None].repeat(1,3) for k,v in stage_masks.items()}
    rows=[];sums={k:0. for k in tensors};total_alpha=0.
    for camera in cameras:
        frame=int(camera.image_name_train_light.split('_')[-1]);camera.load2device();geometry=model.effective(camera);packages={}
        foreground=camera.gt_alpha_mask[0].cuda()>.5
        for key,color in tensors.items():
            package=render(camera,geometry,model.pipe,model.pc.get_xyz.new_zeros(3),0.,0.,0.,override_color=color)
            packages[key]=package['render'][0];alpha=package['rend_alpha'][0]
        alpha_mass=float(alpha[foreground].sum());total_alpha+=alpha_mass
        values={key:float(value[foreground].sum()) for key,value in packages.items()}
        for key,value in values.items():sums[key]+=value
        rows.append(dict(frame=frame,fractions={key:value/alpha_mass for key,value in values.items()}))
        if frame in [1,81,149]:
            font=ImageFont.truetype('/tmp/lumimotion-ppt-1017/NotoSansCJKsc-Regular.otf',18)
            canvas=Image.new('RGB',(camera.image_width*4,camera.image_height+35),'white');draw=ImageDraw.Draw(canvas)
            for j,(key,label) in enumerate(zip(packages,['宽带原型','加TRAIN凸包','再加稳定门禁','再加动态净空'])):
                value=(packages[key]/alpha.clamp_min(1e-8)).clamp(0,1).cpu().numpy();color=np.stack([1-value,value,np.zeros_like(value)],-1);color[~foreground.cpu().numpy()]=0
                canvas.paste(Image.fromarray((color*255).astype('uint8')),(j*camera.image_width,35));draw.text((j*camera.image_width+5,5),label,fill='black',font=font)
            canvas.save(out/f'train_gates_{frame:04d}.png')
        camera.load2device('cpu')
    fractions={k:v/total_alpha for k,v in sums.items()};removed={}
    keys=list(stage_masks)
    for before,after in zip(keys[:-1],keys[1:]):
        subset=stage_masks[before]&~stage_masks[after];scale=model.pc.get_scaling.cpu().numpy()[subset];opacity=model.pc.get_opacity.cpu().numpy().reshape(-1)[subset]
        removed[after]=dict(count=int(subset.sum()),removed_alpha_mass_fraction=fractions[before]-fractions[after],
            scale_mean=float(np.linalg.norm(scale,axis=1).mean()),scale_p95=float(np.quantile(np.linalg.norm(scale,axis=1),.95)),opacity_mean=float(opacity.mean()))
    result=dict(gt_read=False,heldout_read=False,training_updates=0,frames=frames,source_support_sha256=sha(args.expected_support),
        reproduced_support_bitwise_equal=True,point_counts={k:int(v.sum()) for k,v in stage_masks.items()},
        alpha_mass_fraction=fractions,gate_removals=removed,per_frame=rows,
        interpretation='所有份额来自同117TRAIN前景Alpha合成，不使用GT对象或几何')
    np.savez_compressed(out/'stage_indices.npz',**{k:np.flatnonzero(v) for k,v in stage_masks.items()})
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_frame','frames']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','source','parent','plane-audit','cache','output','expected-support']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
