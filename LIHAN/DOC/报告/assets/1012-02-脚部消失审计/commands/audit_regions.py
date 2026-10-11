"""只读复算已有几何数组；GT区域仅用于审计，不参与训练。"""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils.geometry_eval_utils import camera_rays, erode, normal_errors
OUT = Path(__file__).resolve().parents[1]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    directories = sorted(ROOT.glob('output/1011-*/evaluation_*/normal_depth'))
    directories += sorted(OUT.glob('checkpoint_renders/*'))
    rows, provenance = [], {}
    for folder in directories:
        if not (folder/'metrics.json').exists():
            continue
        meta = json.loads((folder/'metrics.json').read_text())
        if 'LH-static-jumpingjacks_' not in meta['command']['gt_root']:
            continue
        source = Path(meta['command']['source_path'])
        camera_file = source/'transforms_test.json'
        cams = json.loads(camera_file.read_text())
        records = {int(Path(r['file_path']).stem.split('_')[-1]): r for r in cams['frames']}
        tag = folder.parent.parent.name if folder.name=='normal_depth' else folder.name
        provenance[str(camera_file)] = digest(camera_file)
        provenance[str(folder/'metrics.json')] = digest(folder/'metrics.json')
        for file in sorted(folder.glob('prediction_*.npz')):
            n = int(file.stem.split('_')[-1]); record = records[n]
            data = dict(np.load(file)); h,w = data['alpha'].shape
            intrinsics = [record.get('fl_x',cams['fl_x'])*w/cams['w'], record.get('fl_y',cams['fl_y'])*h/cams['h'],w/2,h/2]
            rays = camera_rays(record['transform_matrix'],intrinsics,h,w)
            gz = rays[...,2]+rays[...,5]*data['gt_depth']
            pz = rays[...,2]+rays[...,5]*data['expected']
            mz = rays[...,2]+rays[...,5]*data['median']
            fg = data['gt_foreground'] & (data['gt_depth']>0)
            person = fg & (gz>.05)
            feet = person & (gz<.35)
            regions = dict(foreground=fg, platform=fg & (np.abs(gz)<.02),person=person,
                           feet=feet,feet_core=feet & erode(person,1))
            for cutoff in (.25,.30,.40,.45):
                regions[f'feet_upper_{cutoff:.2f}'] = person & (gz<cutoff)
            gt_rgb_file = source/'images'/record['file_path']
            with Image.open(gt_rgb_file) as im:
                rgb = np.array(im.convert('RGB').resize((w,h),Image.Resampling.BILINEAR))/255.
            provenance[str(file)] = digest(file)
            angular = {}
            gt_valid = np.linalg.norm(data['gt_normal'],axis=-1)>1e-6
            visible = np.isfinite(data['alpha']) & (data['alpha']>=.5)
            for method,field in [('gs_render','normal'),('depth_normal','depth_normal'),('mesh','mesh_normal')]:
                if field in data:
                    error,good=normal_errors(data[field],data['gt_normal'])
                    angular[method]=(error['angular_deg'],good)
            for name, mask in regions.items():
                valid=mask & (data['alpha']>=.5) & np.isfinite(data['expected']) & (data['expected']>0)
                row=dict(experiment=tag,iteration=meta['iteration'],frame=n,region=name,
                         pixels=int(mask.sum()),alpha_valid=int(valid.sum()),
                         expected_near_plane=int((valid & (abs(pz)<.04)).sum()),
                         median_near_plane=int((valid & (abs(mz)<.04)).sum()),
                         expected_depth_bad_5cm=int((valid & (abs(data['expected']-data['gt_depth'])>.05)).sum()),
                         expected_depth_bad_10cm=int((valid & (abs(data['expected']-data['gt_depth'])>.1)).sum()),
                         gt_dark_pixels=int((mask & (rgb.mean(-1)<.1)).sum()),
                         gt_saturated_pixels=int((mask & (rgb.max(-1)==1)).sum()),
                         expected_abs_sum=float(abs(data['expected']-data['gt_depth'])[valid].sum()),
                         rgb_mean_sum=float(rgb.mean(-1)[mask].sum()),
                         source=str(file))
                for method in ('gs_render','depth_normal','mesh'):
                    eligible=mask & gt_valid
                    support=erode(eligible & visible,1) if method=='depth_normal' else (eligible & (data.get('mesh_valid',np.ones_like(mask)) if method=='mesh' else visible))
                    if method in angular:
                        values,good=angular[method]; samples=values[support & good]
                        row[f'{method}_sum']=float(samples.sum());row[f'{method}_count']=len(samples)
                    else:
                        row[f'{method}_sum']=0.;row[f'{method}_count']=0
                rows.append(row)
    if not rows: raise RuntimeError('没有可审计数组')
    fields=list(rows[0])
    with (OUT/'regional_per_frame.csv').open('w') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    groups={}
    for row in rows:
        key=(row['experiment'],row['iteration'],row['region'])
        if key not in groups: groups[key]=dict(experiment=key[0],iteration=key[1],region=key[2],frames=0,**{k:0 for k in fields if k not in ('experiment','iteration','frame','region','source')})
        agg=groups[key];agg['frames']+=1
        for k in agg:
            if k not in ('experiment','iteration','region','frames'): agg[k]+=row[k]
    result=[]
    for agg in groups.values():
        for k in ('alpha_valid','expected_near_plane','median_near_plane','expected_depth_bad_5cm','expected_depth_bad_10cm','gt_dark_pixels','gt_saturated_pixels'):
            agg[k+'_fraction']=agg[k]/agg['pixels'] if agg['pixels'] else None
        agg['expected_depth_mae']=agg['expected_abs_sum']/agg['alpha_valid'] if agg['alpha_valid'] else None
        for key in ('gs_render','depth_normal','mesh'):
            agg[key+'_mae']=agg[key+'_sum']/agg[key+'_count'] if agg[key+'_count'] else None
        result.append(agg)
    (OUT/'regional_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'array_sources_sha256.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print('实验检查点数',len({(r['experiment'],r['iteration']) for r in rows}),'帧区域行数',len(rows))
    for a in result:
        if a['region']=='feet':
            print(a['experiment'],a['iteration'],a['pixels'],*[round(a[k],5) if a[k] is not None else None for k in ('gs_render_mae','expected_depth_mae','expected_near_plane_fraction','expected_depth_bad_5cm_fraction')])

if __name__=='__main__':main()
