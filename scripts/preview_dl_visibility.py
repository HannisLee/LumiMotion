"""遮挡诊断的采样点可视化与高亮冲突统计，不生成完整重渲。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.dl_selfcal import SurfaceLightFit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    out=Path(args.directory)
    result=json.loads((out/'result.json').read_text())
    cache=Path(result['config']['cache'])
    data=torch.load(cache,weights_only=False,map_location='cpu')
    vis=torch.load(out/'visibility.pt',weights_only=False,map_location='cpu')
    valid=data['valid'];train=torch.arange(len(valid))%9!=4
    support=(valid[train].sum(0)>=3)&((valid&~data['saturated'].any(-1))[train].sum(0)>=1)
    valid=valid&support[None]
    stats={};predictions=[]
    for j,label in enumerate(vis['labels']):
        cosine=(data['normals']*vis['directions'][j,:,None]).sum(-1)
        bright=data['rgb'].mean(-1)>.8
        occluded=vis['visibility'][j]<.1
        stats[label]=dict(bright_valid_fraction=float((bright&valid).sum()/valid.sum()),
                          bright_occluded_fraction=float((bright&valid&occluded).sum()/(bright&valid).sum()),
                          zero_cosine_valid_fraction=float(((cosine<=0)&valid).sum()/valid.sum()),
                          exact_zero_visibility_valid_fraction=float(((vis['visibility'][j]==0)&valid).sum()/valid.sum()))
        state=torch.load(out/(label+'_visibility.pt'),weights_only=False,map_location='cpu')
        weights=state['model'];m=SurfaceLightFit(weights['times'],weights['raw_directions'],len(data['ids']))
        m.load_state_dict(weights)
        with torch.no_grad():
            _,energy=m.lighting(data['times'])
            prediction=torch.nn.functional.softplus(m.raw_albedo)[None]*cosine.clamp_min(0)[...,None]*energy[:,None,None]*vis['visibility'][j,...,None]
            predictions.append(prediction)
    (out/'visibility_conflicts.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    fig,axes=plt.subplots(3,5,figsize=(15,9))
    for row,index in enumerate([0,len(valid)//2,len(valid)-1]):
        name=data['frames'][index]['name'];w,h=Image.open(cache.parent/f'{name}_rgb.png').size
        mask=valid[index];uv=data['pixels'][index][mask]
        values=[data['rgb'][index][mask],vis['visibility'][0,index][mask],vis['visibility'][1,index][mask],predictions[0][index][mask],predictions[1][index][mask]]
        for col,value in enumerate(values):
            if value.ndim==1:
                color=plt.get_cmap('viridis')(value.numpy())[:,:3]
            else:
                color=torch.where(value<=.0031308,12.92*value,1.055*value.clamp_min(.0031308).pow(1/2.4)-.055).clamp(0,1).numpy()
            image=np.zeros((h,w,3),dtype=np.float32);image[uv[:,1],uv[:,0]]=color
            axes[row,col].imshow(image);axes[row,col].axis('off')
            axes[row,col].set_title(name+' '+['observed','V unknown','V reference','pred unknown+V','pred reference+V'][col])
    fig.tight_layout();fig.savefig(out/'visibility_samples.png',dpi=110);plt.close(fig)
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
