"""只读固定外部评测JSON/PNG产物；不打开模型、原始GT或光照文件。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
EVAL=ROOT/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/eval_only'
TEST=list(range(10,151,10))
TRAIN=[n for n in range(1,151) if n%10]
PROTOCOL_KEYS=('version','dataset','splits','evaluation_split','resolution','alpha_threshold','interior_radius',
    'source_manifest_sha256','normal_methods','depth_fields','aggregation','signed_normal',
    'missing_normal_penalty_deg','relative_distance_missing_penalty','distance_thresholds_world_units',
    'relative_distance_thresholds','distance_denominator','tsdf','gt_manifest_sha256','partition_definition_sha256')
REGIONS=('foreground','interior','human','leg','foot','shoe','footwear','platform')
METHODS=('gs_render','depth_normal','mesh')
DEPTHS=('expected','median','mesh_depth')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def fmt(value,places=5):return 'N/A' if value is None else f'{value:.{places}f}'


def same_protocol(left,right):
    for key in PROTOCOL_KEYS:
        if left.get(key)!=right.get(key):raise ValueError('评测协议不一致：'+key)


def load(label,path):
    if not path.is_relative_to(EVAL):raise ValueError('只允许eval_only中的已完成模型评测目录。')
    files={name:path/name for name in ('protocol.json','plan.json','metrics/metrics.json','four_classes/artifacts.json',
        'predicted_meshes/manifest.json','PIPELINE_COMPLETED.json')}
    if (path/'manual_review.json').is_file():files['manual_review.json']=path/'manual_review.json'
    values={name:read(file) for name,file in files.items()}
    protocol=values['protocol.json'];plan=values['plan.json'];metrics=values['metrics/metrics.json']
    if metrics['status']!='PASS_METRIC_EXECUTION' or values['PIPELINE_COMPLETED.json']['status']!='PASS_PIPELINE':
        raise ValueError('评测工程未完成：'+label)
    if protocol['evaluation_split']!='test' or plan['split']!='test' or [r['source_frame'] for r in metrics['rows']]!=TEST:
        raise ValueError('不是相同固定外部15帧：'+label)
    if sha(files['protocol.json'])!=plan['protocol_sha256'] or metrics['protocol_sha256']!=plan['protocol_sha256']:
        raise ValueError('协议SHA不一致：'+label)
    if metrics['gt_depth_manifest_sha256']!=protocol['gt_manifest_sha256']:
        raise ValueError('度量来源摘要与固定协议不同：'+label)
    artifacts=values['four_classes/artifacts.json'];mesh=values['predicted_meshes/manifest.json']
    if artifacts['frames']!=150 or any(item['pngs']!=150 or item['decoded_frames']!=150 for item in artifacts['videos'].values()):
        raise ValueError('四类150帧不完整：'+label)
    if mesh['gt_geometry_used'] or mesh['test_camera_used_for_reconstruction'] or mesh['fit_scale_or_pose'] or mesh['camera_frames']!=TRAIN:
        raise ValueError('预测TSDF不是135TRAIN-only固定协议：'+label)
    expected=dict(resolution=2,voxel_size=.02,sdf_trunc=.08,depth_trunc=20.,alpha_threshold=.5)
    if mesh['settings']!=expected or sorted(map(int,mesh['frames']))!=TEST:raise ValueError('TSDF参数/目标不一致：'+label)
    review=values.get('manual_review.json',artifacts.get('manual_review'))
    review_status=review.get('status','已留目检收据') if isinstance(review,dict) else review
    if not review_status or review_status=='PENDING':raise ValueError('实际目检未完成：'+label)
    hashes={str(file):sha(file) for file in files.values()}
    return dict(label=label,path=str(path),iteration=plan['iteration'],protocol=protocol,metrics=metrics,
        mesh_static=mesh['static_scene'],unique_predicted_meshes=len({x['mesh'] for x in mesh['frames'].values() if x['available']}),
        four_class_review=review_status,input_sha256=hashes)


def csv_file(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def contacts(out,datasets):
    import cv2
    import numpy as np
    destination=out/'contacts';destination.mkdir()
    source_hashes={};outputs=[]
    def tile(path,label,width,height):
        value=cv2.imread(str(path))
        if value is None:raise ValueError('固定图像缺失：'+str(path))
        source_hashes[str(path)]=sha(path)
        factor=min(width/value.shape[1],height/value.shape[0])
        value=cv2.resize(value,(round(value.shape[1]*factor),round(value.shape[0]*factor)),interpolation=cv2.INTER_AREA)
        panel=np.full((height+24,width,3),20,np.uint8);x=(width-value.shape[1])//2;panel[24:24+value.shape[0],x:x+value.shape[1]]=value
        cv2.putText(panel,label,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        return panel
    def save(name,array):
        path=destination/name
        if not cv2.imwrite(str(path),array):raise IOError('对照联系图写入失败。')
        outputs.append(path)
    for kind in ('rgb','alpha','normal','separation'):
        rows=[]
        for n in (150,130,10):
            rows.append(cv2.hconcat([tile(Path(item['path'])/'four_classes'/kind/f'{n:04d}.png',item['label']+f' frame{n}',400,400) for item in datasets]))
        save(f'{kind}_front150_side130_back10.png',cv2.vconcat(rows))
    for n in (10,80,130,150):
        rows=[tile(Path(item['path'])/'metrics/previews'/f'leg_foot_shoe_{n:04d}.png',item['label']+f' frame{n}',1200,240) for item in datasets]
        save(f'leg_foot_shoe_{n:04d}.png',cv2.vconcat(rows))
    write(destination/'manifest.json',dict(source_image_sha256=source_hashes,output_image_sha256={str(p):sha(p) for p in outputs},
        fixed_by='原协议10/80/150，预先由输入朝向确定补130；没有按误差选优',manual_review='PENDING_ACTUAL_VIEW'))
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',action='append',default=[],help='标签=已完成固定评测目录')
    p.add_argument('--control-audit',type=Path,default=EVAL.parent/'M00_M01_preintervention_control.json')
    p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.self_check:
        same_protocol({},{});failed=False
        try:same_protocol(dict(resolution=2),dict(resolution=1))
        except ValueError:failed=True
        assert failed
        print('PASS：协议不一致被拒绝，未打开模型/GT或评测结果。');return
    if len(a.evaluation)<2 or (not a.output and not a.check_only):p.error('需要至少两个evaluation及output。')
    out=a.output.resolve() if a.output else None
    if out is not None and not out.is_relative_to(EVAL):p.error('汇总只允许eval_only新目录。')
    datasets=[]
    for text in a.evaluation:
        label,path=text.split('=',1)
        if any(d['label']==label for d in datasets):p.error('标签重复。')
        datasets.append(load(label,Path(path).resolve()))
    for item in datasets[1:]:
        same_protocol(datasets[0]['protocol'],item['protocol'])
        for region in datasets[0]['metrics']['summary']:
            for method in METHODS:
                left=datasets[0]['metrics']['summary'][region]['normal'][method]
                right=item['metrics']['summary'][region]['normal'][method]
                if left['eligible_pixels']!=right['eligible_pixels'] or left['eligible_frames']!=right['eligible_frames']:
                    raise ValueError('模型固定GT分母不一致：'+region+'/'+method)
    if a.check_only:
        print('PASS：既有评测协议/分母/四类/来源收据一致；只读核验，无新汇总或训练选择。');return
    control=None
    if a.control_audit.is_file():
        if a.control_audit.resolve().parent!=EVAL.parent:p.error('控制审计只允许总控JSON。')
        control=dict(path=str(a.control_audit.resolve()),sha256=sha(a.control_audit),record=read(a.control_audit))
    out.mkdir(parents=True,exist_ok=False)
    normal_rows=[];depth_rows=[];overview=[]
    for item in datasets:
        metric=item['metrics'];overview.append(dict(model=item['label'],iteration=item['iteration'],
            rgb_psnr_frame_mean=metric['rgb_psnr_frame_mean'],alpha_iou_frame_mean=metric['alpha_iou_frame_mean'],
            mesh_static=item['mesh_static'],unique_predicted_meshes=item['unique_predicted_meshes'],
            mesh_complete=metric['all_predicted_meshes_available'],four_class_review=item['four_class_review'],output=item['path']))
        for region,data in metric['summary'].items():
            for method,value in data['normal'].items():normal_rows.append(dict(model=item['label'],iteration=item['iteration'],region=region,method=method,**value))
            for method,value in data['depth'].items():
                depth_rows.append(dict(model=item['label'],iteration=item['iteration'],region=region,method=method,
                    valid_mae_m=value['absolute']['pixel_mean'],valid_mae_frame_m=value['absolute']['frame_mean'],
                    coverage=value['coverage'],eligible_pixels=value['eligible_pixels'],valid_pixels=value['valid_pixels'],
                    close_1cm=value['close_fraction_pixel'][0],close_2cm=value['close_fraction_pixel'][1],
                    close_5cm=value['close_fraction_pixel'][2],close_10cm=value['close_fraction_pixel'][3],
                    relative_close_1pct=value['relative_close_fraction_pixel'][0],relative_close_2pct=value['relative_close_fraction_pixel'][1],
                    relative_close_5pct=value['relative_close_fraction_pixel'][2],relative_close_10pct=value['relative_close_fraction_pixel'][3],
                    clipped_relative_missing_mean=value['clipped_relative_missing_pixel_mean']))
    csv_file(out/'overview.csv',overview);csv_file(out/'normal_comparison.csv',normal_rows);csv_file(out/'depth_comparison.csv',depth_rows)
    contact_paths=contacts(out,datasets)
    write(out/'comparison.json',dict(status='PASS_FIXED_ARTIFACT_COMPARISON_EXECUTION',
        raw_gt_or_light_files_opened=False,training_or_lighting_selection=False,
        preintervention_control=control,
        scientific_precision_pass='NOT_DECIDED_NO_PREREGISTERED_V3_TARGET',models=datasets,
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),script_sha256=sha(Path(__file__))))
    labels=[x['label'] for x in datasets]
    lines=['# ELDL-V3 固定外部模型对照','','2026-10-19（Asia/Tokyo）。本工具只读取已完成评测的JSON/manifest、PNG及目检收据，不打开原始GT、光照、材质文件或模型；几何GT派生指标只用于外部报告，不接入训练与估光。','',
        '所有模型必须相同15TEST、400分辨率、评测源/GT深度/分区摘要、固定分母和缺失惩罚及TRAIN-only TSDF协议。B00@16k与M00/M01@16k按名义迭代预算并列，B00@35k为额外收敛参照；训练器/照明模型不同，不把跨方法差异直接解释为唯一因果。','',
        'M00/M01虽同输入/source/snapshot/相机序列且配置仅末次reset不同，15000前已存在原生CUDA跨GPU非确定性及增密轨迹小差异。单变量配置不等于逐位配对轨迹；终点小幅差异不能全部归因reset，大趋势与需要重复实验的现象分别解读。控制审计原记录与SHA已绑定comparison.json；不据外部指标更改训练。','',
        '|模型|迭代|黑背景RGB PSNR|Alpha IoU|独立TSDF份数|静态复用|','|---|---:|---:|---:|---:|---|']
    for row in overview:lines.append(f'|{row["model"]}|{row["iteration"]}|{fmt(row["rgb_psnr_frame_mean"])}|{fmt(row["alpha_iou_frame_mean"],7)}|{row["unique_predicted_meshes"]}|{row["mesh_static"]}|')
    lines+=['','法线表单元格为像素MAE° / 等帧MAE° / 覆盖率% / 缺失180°计罚像素MAE。完整逐区域细项见CSV/JSON，缺失不被删掉。','','|区域/方法|'+'|'.join(labels)+'|','|---|'+'---:|'*len(labels)]
    for region in REGIONS:
        for method in METHODS:
            cells=[]
            for item in datasets:
                n=item['metrics']['summary'][region]['normal'][method]
                cells.append(f'{fmt(n["mae_pixel_mean_deg"],3)} / {fmt(n["mae_frame_mean_deg"],3)} / {fmt(n["coverage"]*100,2)} / {fmt(n["missing180_pixel_mean_deg"],3)}')
            lines.append('|'+region+'/'+method+'|'+'|'.join(cells)+'|')
    lines+=['','深度表单元格为有效MAE米 / 覆盖率% / ≤2cm固定分母比例%。完整1/2/5/10cm及相对1/2/5/10%与缺失计罚均在depth_comparison.csv，不能以高覆盖率替代距离准确。','','|区域/深度|'+'|'.join(labels)+'|','|---|'+'---:|'*len(labels)]
    for region in ('foreground','human','leg','foot','shoe','platform'):
        for method in DEPTHS:
            cells=[]
            for item in datasets:
                d=item['metrics']['summary'][region]['depth'][method]
                cells.append(f'{fmt(d["absolute"]["pixel_mean"],5)} / {fmt(d["coverage"]*100,2)} / {fmt(d["close_fraction_pixel"][1]*100,2)}')
            lines.append('|'+region+'/'+method+'|'+'|'.join(cells)+'|')
    lines+=['','各模型完整命令、checkpoint/asset、日志、四类代表图片与实际目检记录在原输出README/manual_review.json；本次输入文件SHA与完整source指标摘要在comparison.json。','']
    lines += ['- '+item['label']+'：`'+item['path']+'`，目检收据状态 '+str(item['four_class_review']) for item in datasets]
    lines+=['','固定正150/侧130/背10四类及10/80/130/150腿足对照联系图如下，输入/输出图像SHA见contacts/manifest.json；新对照联系图实际目检待执行后另填，不能把生成视为已查看。','']
    for path in contact_paths:lines+=['!['+path.stem+']('+str(path)+')','']
    lines+=['','最终PASS（固定产物只读汇总）；无新的训练、光照选择或精度阈值声明。','']
    (out/'README.md').write_text('\n'.join(lines))
    print('PASS：固定协议只读汇总',out)


if __name__=='__main__':main()
