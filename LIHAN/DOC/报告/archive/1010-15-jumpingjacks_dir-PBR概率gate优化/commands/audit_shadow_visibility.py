"""40k只读GGX/BVH遮挡、射线起点偏移与预测深度混合响应，0优化/无GT标签。"""
from pathlib import Path
import sys,json,hashlib,copy,datetime
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import torch
import torch.nn.functional as F
from scripts.render_stage2 import Stage2RenderRun,save_image
from gaussian_renderer.render_ir import render_ir
from PIL import Image,ImageDraw

@torch.no_grad()
def main():
    run=Stage2RenderRun(ROOT/'model_mlp',40000);ctx=run.context;pc=ctx.pc
    dest=ROOT/'inspection/shadow_visibility';dest.mkdir(exist_ok=False)
    rows=[];images=[]
    for key in ['0001','0075','0149']:
        cam=ctx.camera_by_key[key];ctx.load_camera(cam)
        d=ctx.deformation(cam)
        if run.bvh:pc.update_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']})
        else:pc.build_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']});run.bvh=True
        target=cam.original_image_train_light.cuda()
        native=None;records={};image_row=[('own RGB '+key,dest/(key+'_input.png'))]
        save_image(image_row[0][1],target)
        variants=[('native',False,float(run.args.light_t_min),float(run.args.depth_ratio)),
            ('unoccluded',True,float(run.args.light_t_min),float(run.args.depth_ratio)),
            ('bias_002',False,.02,float(run.args.depth_ratio)),
            ('bias_005',False,.05,float(run.args.depth_ratio)),
            ('depth_mix_half',False,float(run.args.light_t_min),.5)]
        for name,skip,bias,ratio in variants:
            pipe=copy.deepcopy(run.args);pipe.light_t_min=bias;pipe.depth_ratio=ratio
            package=render_ir(cam,pc,pipe,ctx.background,
                **{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                training=False,opt=pipe,skip_tracer=skip)
            assert torch.isfinite(package['render']).all() and torch.isfinite(package['visibility']).all()
            if native is None:native=package
            for field in ['rend_alpha','rend_normal']:
                assert torch.equal(package[field],native[field])
            own_alpha=cam.gt_alpha_mask.to(target)
            eligible=native['rend_alpha'][0]>.9
            bright=eligible&(own_alpha[0]>=.99)&(target.mean(0)>.1)
            visibility=(package['visibility']/package['rend_alpha'].clamp_min(1e-6))[0]
            per_pixel=(package['render']-target).abs().mean(0)
            records[name]=dict(shadow_ray_bias=bias,depth_ratio=ratio,skip_tracer=skip,
                native_photo_eligible_pixels=int(eligible.sum()),
                photo_L1=float(per_pixel[eligible].mean()),
                own_bright_pixels=int(bright.sum()),own_bright_photo_L1=float(per_pixel[bright].mean()),
                own_bright_mean_visibility=float(visibility[bright].mean()),
                own_bright_visibility_below_01_fraction=float((visibility[bright]<.1).float().mean()),
                mean_visibility=float(visibility[eligible].mean()),
                predicted_normal_alpha_buffers_bit_exact=True)
            image_path=dest/(key+'_'+name+'.png');save_image(image_path,package['render']);images.append(str(image_path.relative_to(ROOT)))
            if name in ['native','unoccluded','bias_005','depth_mix_half']:image_row.append((name+' '+key,image_path))
            if name=='native':save_image(dest/(key+'_visibility.png'),visibility.clamp(0,1)[None])
        rows.append(dict(frame=key,variants=records));ctx.unload_camera(cam)
        if key=='0001':canvas=Image.new('RGB',(400*len(image_row),425*3),'white');draw=ImageDraw.Draw(canvas)
        row_index=len(rows)-1
        for index,(label,path) in enumerate(image_row):
            im=Image.open(path).convert('RGB');assert im.size==(400,400)
            canvas.paste(im,(400*index,425*row_index+25));draw.text((400*index+6,425*row_index+6),label,fill='black')
    canvas.save(dest/'comparison.png')
    ctx.verify_frozen()
    manifest=json.loads((ROOT/'code_manifest.json').read_text());assert len(manifest)==474
    assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
    result=dict(status='PASS_READ_ONLY_SHADOW_VISIBILITY_DIAGNOSTIC',iteration=40000,rows=rows,
        optimizer_steps=0,GT_labels_read=False,original_TRAIN_RGB_and_own_alpha_only=True,
        production_pipeline_changed=False,training_prior_saved=False,images=images,
        source_files=ctx.source_files,created_at=datetime.datetime.now().astimezone().isoformat(),
        limitations='各variant只读同40k checkpoint原生GGX/BVH；阴影关闭或bias变化不是新Preset或训练结果，不能作为GT normal/7度结论。own bright group来自原RGB自身Alpha仅诊断，非训练/GT评分mask；所有点/评分/当前训练保持。depth_ratio半混合改变着色射线起点/预测surf normal，GS normal与alpha仍逐bit相同。')
    path=ROOT/'inspection/shadow_visibility_diagnostic.json';assert not path.exists()
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
