#!/usr/bin/env python3
"""按冻结DEV选择规则汇总：python eldl_1024_dev_select.py --parent <A0 DEV dir> name=dir ..."""
import argparse, json, sys
import numpy as np
def L(d): return json.load(open(d+'/metrics/metrics.json'))
def g(m,r,k='normal'): return m['summary'][r]['normal']['gs_render']
def frame_P(m):
    out=[]
    for row in m['rows']:
        x=row['regions']['foreground']['normal']['gs_render']; out.append(x['missing180_mae'])
    return np.array(out)
def gates(m,p):
    f,pf=g(m,'foreground'),g(p,'foreground'); h,ph=g(m,'human'),g(p,'human'); w,pw=g(m,'footwear'),g(p,'footwear')
    s,ps=m['summary'],p['summary']
    dn=s['foreground']['normal']['depth_normal']['mae_pixel_mean_deg']; pdn=ps['foreground']['normal']['depth_normal']['mae_pixel_mean_deg']
    G={'fg_cov>=max(.9985,p-.0005)':f['coverage']>=max(.9985,pf['coverage']-.0005),
     'human_P<=p+.10':h['missing180_pixel_mean_deg']<=ph['missing180_pixel_mean_deg']+.10,
     'human_cov>=p-.002':h['coverage']>=ph['coverage']-.002,
     'foot_cov>=.99&>=p-.005':w['coverage']>=.99 and w['coverage']>=pw['coverage']-.005,
     'foot_P<=p+1.0':w['missing180_pixel_mean_deg']<=pw['missing180_pixel_mean_deg']+1.0,
     'foot_alpha>=.99&>=p-.005':s['footwear']['alpha']['coverage']>=.99 and s['footwear']['alpha']['coverage']>=ps['footwear']['alpha']['coverage']-.005,
     'human_alpha>=p-.002':s['human']['alpha']['coverage']>=ps['human']['alpha']['coverage']-.002,
     'depth<=p+.5':dn<=pdn+.5,
     'psnr>=p-2':m['rgb_psnr_frame_mean']>=p['rgb_psnr_frame_mean']-2,
     'aiou>=p-.001':m['alpha_iou_frame_mean']>=p['alpha_iou_frame_mean']-.001}
    return G
ap=argparse.ArgumentParser(); ap.add_argument('--parent'); ap.add_argument('--vs',action='append',default=[],help='A:B 比较(A相对B的dP与A更优帧数)'); ap.add_argument('items',nargs='+'); a=ap.parse_args()
p=L(a.parent); pf=frame_P(p); R={}
print('name | P cond cov frame | plat hum leg foot shoe footw (cond/P) | depth psnr aiou | dP | frames better | gates')
for it in a.items:
    n,d=it.split('=',1); m=L(d); R[n]=m
    f=g(m,'foreground'); fp=frame_P(m); G=gates(m,p)
    reg=' '.join('%s %.2f/%.2f'%(r[:4],g(m,r)['mae_pixel_mean_deg'],g(m,r)['missing180_pixel_mean_deg']) for r in ['platform','human','leg','foot','shoe','footwear'])
    print('%s | P %.4f cond %.4f cov %.5f fr %.4f | %s | D %.3f psnr %.2f aiou %.5f | dP %+.4f | better %d/18 | %s'%(n,f['missing180_pixel_mean_deg'],f['mae_pixel_mean_deg'],f['coverage'],f['mae_frame_mean_deg'],reg,m['summary']['foreground']['normal']['depth_normal']['mae_pixel_mean_deg'],m['rgb_psnr_frame_mean'],m['alpha_iou_frame_mean'],f['missing180_pixel_mean_deg']-g(p,'foreground')['missing180_pixel_mean_deg'],int((fp<pf).sum()),'ALL PASS' if all(G.values()) else 'FAIL:'+','.join(k for k,v in G.items() if not v)))
    # decomposition
    sh={r:p['summary'][r]['gt_pixels']/p['summary']['foreground']['gt_pixels'] for r in('platform','human')}
    print('   decomp dP: platform %+.4f human %+.4f (share-weighted)'%tuple(sh[r]*(g(m,r)['missing180_pixel_mean_deg']-g(p,r)['missing180_pixel_mean_deg']) for r in('platform','human')))
def frames(n): return frame_P(R[n])
def cmp(a,b,label):
    if a in R and b in R:
        da=frames(a)-frames(b); print('%s: dP=%+.4f, %d/18 frames lower(first better)'%(label,g(R[a],'foreground')['missing180_pixel_mean_deg']-g(R[b],'foreground')['missing180_pixel_mean_deg'],int((da<0).sum())))
for s in('4050','8100'):
    cmp('G1_'+s,'G0_'+s,'route detach-full LR1 @'+s); cmp('G3_'+s,'G2_'+s,'route detach-full LR10 @'+s)
    cmp('G2_'+s,'G0_'+s,'LR10-LR1 full @'+s); cmp('G3_'+s,'G1_'+s,'LR10-LR1 detach @'+s)

for pair in a.vs:
    x,y=pair.split(':'); cmp(x,y,'%s vs %s'%(x,y))
