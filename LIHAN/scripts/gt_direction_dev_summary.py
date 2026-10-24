"""只汇总DEV评测；不读取任何TEST统计，缺Mesh保留null和PENDING。"""
from pathlib import Path
import argparse,csv,json


def main(root):
    root=root.resolve();records=[]
    for folder in sorted(root.iterdir()):
        # 在打开指标前依目录白名单排除所有TEST路径。
        if not folder.is_dir() or '_dev' not in folder.name:continue
        path=folder/'metrics.json'
        if not path.exists():path=folder/'fast_metrics.json'
        if not path.exists():continue
        m=json.loads(path.read_text())
        if m.get('split')!='dev':raise ValueError('DEV目录出现非DEV指标：'+str(path))
        frozen=json.loads((folder/'freeze_snapshot.json').read_text());summary=m['summary'];fg=summary['foreground']['normal'];gs=fg['gs_render'];rgb=m['rgb'];albedo=m.get('albedo',{})
        row=dict(experiment=Path(frozen['model']).parent.name,iteration=frozen['iteration'],stage=frozen['stage'],
            split='DEV18',frames=m['frames'],model=frozen['model'],metric_file=str(path),
            gs_condition_pixel_deg=gs['mae_pixel_mean_deg'],gs_fixed_frame_deg=gs['missing180_frame_mean_deg'],
            gs_condition_frame_deg=gs['mae_frame_mean_deg'],gs_fixed_pixel_deg=gs['missing180_pixel_mean_deg'],
            gs_coverage_pixel=gs['coverage'],gs_coverage_frame=gs['coverage_frame_mean'],gt_pixels=gs['eligible_pixels'],
            depth_condition_pixel_deg=fg['depth_normal']['mae_pixel_mean_deg'],depth_fixed_frame_deg=fg['depth_normal']['missing180_frame_mean_deg'],
            depth_coverage=fg['depth_normal']['coverage'],mesh_condition_pixel_deg=fg.get('mesh',{}).get('mae_pixel_mean_deg'),
            mesh_fixed_frame_deg=fg.get('mesh',{}).get('missing180_frame_mean_deg'),mesh_coverage=fg.get('mesh',{}).get('coverage'),
            mesh_status='AVAILABLE' if fg.get('mesh',{}).get('mae_pixel_mean_deg') is not None else 'PENDING',
            psnr_foreground_frame=rgb['foreground']['psnr_frame_mean'],psnr_full_frame=rgb['all']['psnr_frame_mean'],
            ssim_full_linear_frame=m['ssim_linear_full_frame_mean'],
            albedo_condition_frame=albedo.get('conditional_frame_mean'),albedo_fixed_missing1_frame=albedo.get('fixed_missing1_frame_mean'),
            albedo_status=albedo.get('status','NOT_AVAILABLE'),protocol_file=str(folder/'protocol.json'),
            lighting='Stage1 SH无显式光照' if frozen['stage']=='stage1' else m['protocol']['lighting'])
        for region in ['human','platform','shoe','leg','foot']:
            x=summary[region]['normal']['gs_render']
            for key,field in [('condition_pixel_deg','mae_pixel_mean_deg'),('fixed_frame_deg','missing180_frame_mean_deg'),('coverage','coverage')]:row[f'{region}_gs_{key}']=x[field]
        records.append(row)
    records.sort(key=lambda r:(r['experiment'],r['iteration'],r['metric_file']))
    manifest=dict(scope='DEV_ONLY_NO_TEST_READ',normal_protocol='GS条件像素；固定GT缺失180度等帧；每区域同步报告coverage',
        mesh_missing_policy='null加PENDING，绝不以180占位当已评测结果',records=records)
    (root/'dev_summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if records:
        fields=[k for k in records[0] if k!='frames']
        with (root/'dev_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='frames'} for r in records)
    print('仅DEV汇总完成：'+str(len(records))+'条')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
