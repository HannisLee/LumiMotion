"""训练RGB颜色方向的跨时间一致性及任意标量光照残差下界；无GT。"""
import argparse,json,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--lights',required=True);p.add_argument('--material-run',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(4)
 d=torch.load(a.cache,map_location='cpu',weights_only=False);light=json.loads(Path(a.lights).read_text());assert not light['gt_read'];r=Path(a.material_run);ck=torch.load(r/'checkpoint.pt',map_location='cpu',weights_only=False);rt=json.loads((r/'runtime.json').read_text());assert not rt['gt_read'] and not ck['args']['gt_direction_reference'];assert light['cache_sha256']==hashlib.sha256(Path(a.cache).read_bytes()).hexdigest();assert rt['inputs'][str(Path(a.cache).resolve())]==light['cache_sha256']
 ref=torch.load(Path(a.reference)/'measurements.pt',map_location='cpu',weights_only=False);assert torch.equal(ref['ids'],d['ids']);dynamic=ref['dynamic'];y=d['rgb'].double();times=d['times'].double();train=light['train_indices'];held=light['held_indices'];assert train==[i for i in range(135) if i%9!=4] and held==[i for i in range(135) if i%9==4]
 base=d['valid'];lum=y.mean(-1);bright_ref=torch.nanquantile(torch.where(base[train],lum[train],torch.nan),.9,dim=0);bright=(lum/bright_ref.clamp_min(.05)[None]>=.65)&(bright_ref[None]>.05)&torch.isfinite(bright_ref)[None];raw_mask=base&bright&~d['saturated'].any(-1);supported=raw_mask[train].sum(0)>=10;mask=raw_mask&supported[None]
 def fit_axis(rgb):
  yy=rgb[train];mm=mask[train];scatter=torch.einsum('tnc,tnd,tn->ncd',yy,yy,mm.double());vals,vec=torch.linalg.eigh(scatter);axis=vec[:,:,-1];axis*=torch.where(axis.sum(-1,keepdim=True)>=0,1.,-1.);assert (axis[supported]>=-1e-8).all();return axis
 ray=fit_axis(y);alpha=(y*ray[None]).sum(-1).clamp_min(0);oracle=alpha[...,None]*ray[None]
 # 此处强度alpha读取被评估观测，仅为任意标量照明可达到的颜色残差下界，不是保留RGB预测。
 energy=F.softplus(ck['energy'].double());energy/=energy.mean();trtime=times[train];right=torch.searchsorted(trtime,times).clamp(1,len(train)-1);left=right-1;w=((times-trtime[left])/(trtime[right]-trtime[left])).clamp(0,1);E=torch.lerp(energy[left],energy[right],w)
 dirs=torch.tensor(light['directions'],dtype=torch.float64);cosine=(d['normals'].double()*dirs[:,None]).sum(-1).clamp_min(0);shading=E[:,None]*cosine+float(ck['ambient']);aa=(y[train]*shading[train,:,None]*mask[train,:,None]).sum(0)/(shading[train].square()*mask[train]).sum(0).clamp_min(1e-12)[:,None];pred=aa[None]*shading[...,None]
 # 固定未知灯向+既有训练能量下的点观测LS代理；非逐Gaussian光栅渲染损失。
 chroma=y/y.sum(-1,keepdim=True).clamp_min(1e-12);ray_chroma=ray/ray.sum(-1,keepdim=True).clamp_min(1e-12);normalized_angle=torch.rad2deg(torch.acos((F.normalize(y,dim=-1)*ray[None]).sum(-1).clamp(-1,1)))
 errors=dict(fixed_light=(pred-y).square().sum(-1),free_scalar_color_floor=(oracle-y).square().sum(-1),chroma_l1=(chroma-ray_chroma[None]).abs().sum(-1),color_axis_deg=normalized_angle)
 stats={}
 for region,reg in [('dynamic',dynamic),('static',~dynamic)]:
  for split,inds in [('train',train),('held',held)]:
   mm=mask[inds]&reg[None];count=int(mm.sum());den=3*count;floor=float(errors['free_scalar_color_floor'][inds][mm].sum());baseerr=float(errors['fixed_light'][inds][mm].sum());stats[region+'_'+split]=dict(observations=count,points=int(mm.any(0).sum()),raw_valid_observations=int((base[inds]&reg[None]).sum()),retained_fraction=count/max(int((base[inds]&reg[None]).sum()),1),fixed_light_mse=baseerr/den,color_floor_mse=floor/den,color_floor_fraction_of_fixed_error=floor/max(baseerr,1e-30),chroma_l1_mean=float(errors['chroma_l1'][inds][mm].mean()),color_axis_mean_deg=float(normalized_angle[inds][mm].mean()),color_axis_p90_deg=float(torch.quantile(normalized_angle[inds][mm],.9)))
 # 保留RGB隔离检查：更改保留RGB，不改变训练学到的颜色方向和albedo。
 altered=y.clone();altered[held]=.123456;isolation=float((fit_axis(altered)-ray).abs().max());assert isolation==0
 counts=mask[train].sum(0);perpoint=(errors['free_scalar_color_floor'][train]*mask[train]).sum(0)/counts.clamp_min(1);candidates=torch.where(supported&dynamic)[0];worst=candidates[torch.argsort(perpoint[candidates],descending=True)[:6]]
 torch.save(dict(ids=d['ids'],dynamic=dynamic,mask=mask,train=train,held=held,axis=ray,albedo=aa,energy=E,errors=errors,rgb=y,oracle_color=oracle,prediction=pred,worst_train_ids=worst),out/'diagnostics.pt')
 sources={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [a.cache,a.lights,str(r/'checkpoint.pt'),str(Path(a.reference)/'measurements.pt'),__file__]}
 result=dict(status='PASS',gt_read=False,gradient_updates=0,fit='训练观测闭式颜色主轴与固定照明albedo',source_hashes=sources,statistics=stats,held_rgb_isolation_max_difference=isolation,ambient=float(ck['ambient']),worst_dynamic_gaussian_ids=d['ids'][worst].tolist(),scope='颜色下界允许逐观测读取RGB求最优标量，非完整保留预测；默认Stage1与既有能量见过相关训练数据，非全管线泛化；误差不能唯一归因于对应')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
