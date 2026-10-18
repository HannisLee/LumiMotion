"""仅在数值验收与真实目检完成后导出冻结TRAIN混合灯光；不接受失败资产。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','cache','fit','validation','manual-review','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    if a.output.exists():raise ValueError('禁止覆盖任何已有训练源或失败产物')
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.source,a.cache,a.cache.parent/'manifest.json',a.fit,a.validation,a.manual_review,a.output])
    source=read(a.source/'no_gt_manifest.json');meta=read(a.cache.parent/'manifest.json')
    fit=read(a.fit/'summary.json');validation=read(a.validation/'summary.json');review=read(a.manual_review)
    if source['gt_read'] or source['init_weights_loaded'] or source['dataset']!='LH-static-jumpingjacks-ELDL-V3':
        raise ValueError('需要V3原始135TRAIN无GT源')
    if validation['status']!='PASS_NUMERIC_PENDING_REVIEW' or not all(validation['checks'].values()):
        raise ValueError('数值验收未通过，禁止导出')
    if not fit['preliminary_checks_passed'] or fit['selected_by']!='fit_censored_mse_only':
        raise ValueError('原fit门槛未通过')
    if validation['gt_read'] or validation['external_test_read'] or validation['parameters_refit'] or validation['valid_changed']:
        raise ValueError('验证越过冻结参数/固定支持边界')
    if review.get('status')!='PASS_TRAIN_INTERNAL_LIGHTING' or review.get('gt_read') is not False:
        raise ValueError('主任务尚未完成真实TRAIN内部目检')
    bindings={str(a.cache):sha(a.cache),str(a.fit/'summary.json'):sha(a.fit/'summary.json'),
              str(a.validation/'summary.json'):sha(a.validation/'summary.json')}
    if review.get('input_sha256')!=bindings:raise ValueError('目检收据与当前结果未绑定')
    if not review.get('observations') or len(review.get('images_sha256',{}))<6:
        raise ValueError('必须记录至少六张固定代表图的实际观察和SHA')
    for path,expected in review['images_sha256'].items():
        image=Path(path).resolve()
        if not image.is_relative_to(a.validation/'figures') or sha(image)!=expected:
            raise ValueError('代表图未绑定当前内部验证')
    if review.get('production_forward_check')!='PASS_FULL_BVH_REAL_RECEIVER_SEMANTICS':
        raise ValueError('必须核验生产全BVH真实接收点语义')
    if meta.get('receiver_representation')!='surfel' or meta.get('fit_facing_disagreements')!=0:
        raise ValueError('当前导出仅接纳与生产起射一致的真实surfel验证')
    if sha(a.cache)!=meta['cache_sha256'] or sha(a.cache)!=fit['cache_sha256']:
        raise ValueError('缓存绑定失败')
    for path,expected in bindings.items():
        if path==str(a.validation/'summary.json'):continue
        if validation['input_sha256'].get(path)!=expected:raise ValueError('精确验证未绑定cache/fit')
    selected=fit['selected'];frames=source['train_frames']
    if frames!=[i for i in range(1,151) if i%10] or len(selected['directions'])!=135:
        raise ValueError('必须恰好原135 TRAIN')
    for relative,expected in source['files'].items():
        path=(a.source/relative).resolve()
        if not path.is_relative_to(a.source) or sha(path)!=expected:raise ValueError('源文件SHA变化')
    provenance=dict(input_sha256=bindings,manual_review_sha256=sha(a.manual_review),
                    cache_manifest_sha256=sha(a.cache.parent/'manifest.json'),
                    exporter_sha256=sha(__file__),geometry_model=meta['model'],geometry_iteration=meta['iteration'],
                    absolute_scale='DL强度π仅在恒定白DL子模型内为尺度规范，时间恒定/共享EL仍是已验证范围内的假设')
    records={f'{frame:04d}':dict(light_dir_world=direction) for frame,direction in zip(frames,selected['directions'])}
    asset=dict(source_split='TRAIN',gt_read=False,engineering_only=False,internal_validation_passed=True,
               environment_mode='shared_rgb_constant',ambient_rgb=selected['ambient_rgb'],
               direct_intensity=math.pi,frames=records,shadow_bias=meta['shadow_bias'],
               held_direction_rule='fit时间nlerp，末端hold；内部18在正式几何训练中属于已见TRAIN',
               provenance=provenance)
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.source/'images',a.output/'images')
    for name in ('points3d.ply','transforms_train.json'):shutil.copy2(a.source/name,a.output/name)
    for name,value in [('estimated_eldl_train.json',asset),('estimated_directions_train.json',
                       dict(frames=records,gt_read=False,source_split='TRAIN',provenance=provenance))]:
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    source.update(lighting_provenance=provenance,lighting_frozen=True)
    source['files']={str(path.relative_to(a.output)):sha(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
    (a.output/'no_gt_manifest.json').write_text(json.dumps(source,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='PASS_EXPORT',source=str(a.output),lighting_sha256=sha(a.output/'estimated_eldl_train.json')),ensure_ascii=False))


if __name__=='__main__':main()
