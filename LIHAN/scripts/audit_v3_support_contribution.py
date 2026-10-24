"""仅TRAIN前向审计父预测平面支持在实际光栅合成中的贡献；不训练、不读几何GT。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from gaussian_renderer import render
from utils.general_utils import build_rotation
from utils.no_gt_guard import install_no_gt_guard
from scene.stage2_photometric import linear_to_srgb


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main(args):
    args.source=args.source.resolve();args.model=args.model.resolve();args.support=args.support.resolve()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    install_no_gt_guard([args.source.resolve(),args.model.resolve(),out,args.support.resolve()])
    torch.set_num_threads(4);model=DynamicSurfaceModel.load(args.model,0);cameras=load_train_cameras(args.source)
    manifest=json.loads((args.source/'no_gt_manifest.json').read_text());assert len(cameras)==117
    assert [int(c.image_name_train_light.split('_')[-1]) for c in cameras]==manifest['train_frames']
    data=np.load(args.support);idx=torch.tensor(data['indices'],device='cuda');n=torch.tensor(data['normal'],dtype=torch.float32,device='cuda');origin=torch.tensor(data['origin'],dtype=torch.float32,device='cuda');extent=float(data['extent'])
    gate=model.pc.get_binary_feature().reshape(-1);dist=((model.pc.get_xyz-origin)*n).sum(-1).abs();axis=build_rotation(model.pc.get_rotation)[:,:,2];side=(axis*n).sum(-1).abs()>.7
    original=torch.zeros(len(gate),dtype=torch.bool,device='cuda');original[idx]=True
    masks=dict(original_support=original,high_static=gate<.1,
        near_plane_without_opacity=(gate<.1)&side&(dist<extent*.005),
        loose_plane_without_opacity=(gate<.1)&side&(dist<extent*.03),
        nonside_high_static=(gate<.1)&side)
    sums={k:0. for k in masks};all_alpha=0.;rows=[]
    for camera in cameras:
        frame=int(camera.image_name_train_light.split('_')[-1]);camera.load2device();geometry=model.effective(camera)
        contributions={};alpha=None
        for name,mask in masks.items():
            package=render(camera,geometry,model.pipe,model.pc.get_xyz.new_zeros(3),0.,0.,0.,override_color=mask.float()[:,None].repeat(1,3))
            contributions[name]=package['render'][0];alpha=package['rend_alpha'][0]
        foreground=camera.gt_alpha_mask[0].cuda()>.5
        static_dominated=foreground&(contributions['high_static']/alpha.clamp_min(1e-8)>.9)
        a=float(alpha[foreground].sum());all_alpha+=a
        row=dict(frame=frame,alpha_mass=a,static_dominated_pixels=int(static_dominated.sum()),groups={})
        for name,value in contributions.items():
            mass=float(value[foreground].sum());sums[name]+=mass
            row['groups'][name]=dict(alpha_mass_fraction=mass/a,mean_fraction_in_static_dominated=float((value[static_dominated]/alpha[static_dominated].clamp_min(1e-8)).mean()))
        rows.append(row)
        if frame in [1,81,149]:
            font=ImageFont.truetype('/tmp/lumimotion-ppt-1017/NotoSansCJKsc-Regular.otf',18)
            rgb=(linear_to_srgb(camera.observed_linear.cuda()).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')
            panels=[('TRAIN观测RGB',rgb)]
            for key,label in [('original_support','输入支持贡献/总Alpha'),('high_static','高置信静态贡献/总Alpha'),('near_plane_without_opacity','近面不裁opacity'),('loose_plane_without_opacity','宽面不裁opacity')]:
                value=(contributions[key]/alpha.clamp_min(1e-8)).clamp(0,1).cpu().numpy();color=np.stack([1-value,value,np.zeros_like(value)],-1);color[~foreground.cpu().numpy()]=0
                panels.append((label,(color*255).astype('uint8')))
            canvas=Image.new('RGB',(rgb.shape[1]*len(panels),rgb.shape[0]+35),'white');draw=ImageDraw.Draw(canvas)
            for i,(label,array) in enumerate(panels):canvas.paste(Image.fromarray(array),(i*rgb.shape[1],35));draw.text((i*rgb.shape[1]+5,5),label,fill='black',font=font)
            canvas.save(out/f'train_support_{frame:04d}.png')
        camera.load2device('cpu')
    result=dict(gt_read=False,heldout_read=False,training_updates=0,frames=manifest['train_frames'],
        source_sha256={str(args.support):sha(args.support),str(args.model/'point_cloud/iteration_0/point_cloud.ply'):sha(args.model/'point_cloud/iteration_0/point_cloud.ply')},
        point_counts={k:int(v.sum()) for k,v in masks.items()},
        alpha_mass_fraction={k:v/all_alpha for k,v in sums.items()},
        selected_fraction_of_high_static_mass=sums['original_support']/sums['high_static'],
        mean_selected_fraction_in_static_dominated=float(np.mean([r['groups']['original_support']['mean_fraction_in_static_dominated'] for r in rows])),
        per_frame=rows,interpretation='二值颜色渲染给出前向Alpha合成贡献，除总Alpha；前景仅TRAIN图像Alpha，不用GT几何或对象标签')
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_frame','frames','source_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','support','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
