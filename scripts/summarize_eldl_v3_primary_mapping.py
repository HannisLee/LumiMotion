"""汇总冻结主视线证据的空间覆盖，不拟合灯光或更改采样规则。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('primary','cache','plane-cache','selection','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():setattr(a,k,v.resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.primary,a.cache,a.plane_cache,a.selection,a.output])
    d=torch.load(a.primary,map_location='cpu',weights_only=False)
    old=torch.load(a.cache,map_location='cpu',weights_only=False);plane=torch.load(a.plane_cache,map_location='cpu',weights_only=False)
    assert 'center_front' in d,'本轮只接纳预登记前层实际贡献规则，不扫描median/neighbor'
    good=d['center_front'];qual=good.sum(0)>=10;pool=d['pool_ids'].numpy()
    assert len(d['frames'])==117 and not any(61<=f<=79 for f in d['frames'])
    lookup={int(v):i for i,v in enumerate(pool)}
    uv=((d['points']-old['origin'])@old['basis'].T).numpy();anchors=((plane['points']-old['origin'])@old['basis'].T).numpy()
    radius=float(max(np.linalg.norm(anchors[1]-anchors[0]),np.linalg.norm(anchors[64]-anchors[0])))
    eligible=np.flatnonzero(qual.numpy());assert len(eligible)>0,'无合格候选'
    # 与torch cdist/min相同：完全等距时按原pool ID升序，候选按原索引顺序。
    distance=np.linalg.norm(anchors[:,None]-uv[eligible][None],axis=-1)
    near=distance.argmin(1);dist=distance[np.arange(len(anchors)),near];idx=eligible[near]
    valid=dist<=radius;seen=set()
    for i,j in enumerate(idx):
        if not valid[i]:continue
        if int(pool[j]) in seen:valid[i]=False
        else:seen.add(int(pool[j]))
    old_support=old['valid'][old['fit']].any(0).numpy();oldids=old['receiver_ids'].numpy()
    rows=[]
    for rr in json.loads(a.selection.read_text())['rays']:
        k=rr['point'];oi=lookup[int(oldids[k])];ni=int(idx[k]);oldvalid=d['center_valid'][:,oi]
        rows.append(dict(frame=rr['frame'],kind=rr['kind'],grid_id=k,old_id=int(oldids[k]),old_front_views=int(good[:,oi].sum()),
          old_actual_views=int((d['actual_contribution'][:,oi]&oldvalid).sum()),new_id=int(pool[ni]),new_front_views=int(good[:,ni].sum()),
          new_distance=float(dist[k]),within_radius=bool(dist[k]<=radius),survives_dedup=bool(valid[k]),
          old_height=float(d['height'][oi]),new_height=float(d['height'][ni])))
    counts=good[:,torch.tensor(idx[valid])].sum(1)
    per_group={}
    for kind in ('sensitive','nearby_stable'):
        q=[r for r in rows if r['kind']==kind]
        per_group[kind]=dict(count=len(q),old_qualified=sum(r['old_front_views']>=10 for r in q),
          within_radius=sum(r['within_radius'] for r in q),survives_dedup=sum(r['survives_dedup'] for r in q),
          replacements=sum(r['survives_dedup'] and r['old_id']!=r['new_id'] for r in q))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    summary=dict(status='ZERO_OPTIMIZATION_MAPPING_AUDIT_NOT_ACCEPTANCE',gt_read=False,internal_validation_passed=False,
       qualified_pool=int(qual.sum()),original_pool=len(pool),matching_radius=radius,new_unique_grid_support=int(valid.sum()),
       old_support=int(old_support.sum()),old_grid_retained=int((old_support&valid).sum()),
       old_grid_same_id=int((old_support&valid&(oldids==pool[idx])).sum()),old_grid_replaced=int((old_support&valid&(oldids!=pool[idx])).sum()),
       old_grid_lost=int((old_support&~valid).sum()),new_fit_center_support=dict(min=int(counts.min()),median=float(counts.float().median()),max=int(counts.max())),
       selected48=rows,selected48_groups=per_group,input_sha256={str(p):sha(p) for p in (a.primary,a.cache,a.plane_cache,a.selection,Path(__file__).resolve())})
    torch.save(dict(pool_indices=torch.tensor(idx),distance=torch.tensor(dist),grid_support=torch.tensor(valid),qualified_pool=qual),a.output/'mapping.pt')
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('selected48','input_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
