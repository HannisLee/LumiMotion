#!/usr/bin/env python3
"""冻结T06精确法线，工程误差和严格不退化分别报告；视觉独立验收。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal_eval_protocol import read,write,FRAMES,digest
BASELINE={'gs_render':2.6683778383745302,'depth_normal':5.404577799172339,'mesh':4.20848728697561}
TOLERANCE=1e-5
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--visual_review',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('不得覆盖科学验收。')
    mae=read(a.evaluation/'goal_mae_summary.json');rgb=read(a.evaluation/'rgb/rgb_summary.json');reference=read(a.reference)
    if mae['test_frames']!=15 or mae['completeness']!='PASS' or rgb['frames']!=FRAMES:raise AssertionError('原15测试完整性失败。')
    geometry={k:dict(baseline_deg=v,current_deg=mae['values'][k],delta_deg=mae['values'][k]-v,
        strict_non_regression=mae['values'][k]<=v,within_repeat_tolerance=abs(mae['values'][k]-v)<=TOLERANCE) for k,v in BASELINE.items()}
    comparison=rgb['reference_comparison']
    if Path(comparison['reference']).resolve()!=a.reference.resolve() or not reference['baseline']:raise AssertionError('RGB参考不符。')
    visual=None if a.visual_review is None else read(a.visual_review)
    visual_pass=False if visual is None else visual.get('status')=='PASS' and visual.get('foot_geometry')=='PASS' and visual.get('black_edge')=='PASS'
    numeric=all(x['within_repeat_tolerance'] for x in geometry.values()) and comparison['psnr_gain_pass'] and comparison['foreground_guardrail_pass']
    status=('PENDING_VISUAL' if numeric else 'FAILED_NUMERIC') if visual is None else 'PASS' if numeric and visual_pass else 'FAILED'
    report=dict(status=status,numeric_pass=numeric,strict_normal_non_regression=all(x['strict_non_regression'] for x in geometry.values()),
        geometry_repeat_tolerance_deg=TOLERANCE,normal_comparison=geometry,rgb_comparison=comparison,visual_review=visual,
        geometry_scope='tol仅原GPU重复测量的工程一致性；正delta不能称改善，严格不退化单列。',
        goal_scope='冻结几何无法补齐原缺鞋；全图PSNR与前景守卫不代替完整脚形/黑边目检。',
        sources={str(p):digest(p) for p in (a.evaluation/'goal_mae_summary.json',a.evaluation/'rgb/rgb_summary.json',a.reference,Path(__file__))})
    if a.visual_review:report['sources'][str(a.visual_review)]=digest(a.visual_review)
    a.output.mkdir(parents=True);write(a.output/'acceptance.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
