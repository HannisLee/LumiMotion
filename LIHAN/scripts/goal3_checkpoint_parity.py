#!/usr/bin/env python3
"""中间检查点独立fresh包装：保留原parity所有真实训练来源/前向门禁。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal3_checkpoint_audit import audit,destination
from goal_eval_protocol import digest
import goal3_densify_parity

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model_path',required=True);p.add_argument('--source_path',required=True);p.add_argument('--load_iter',type=int,required=True);p.add_argument('--frame',type=int,default=96);p.add_argument('--output_dir',type=Path,required=True);a=p.parse_args();out=destination(a.output_dir)
 before=audit(a.model_path,a.load_iter)
 if a.source_path!=before['source']:raise AssertionError('必须原评测source。')
 goal3_densify_parity.main() # 参数兼容；finished仍真实检查18000，optimizer/light检查load_iter。
 after=audit(a.model_path,a.load_iter)
 if before['sources']!=after['sources']:raise AssertionError('fresh前后中间资产或原证据变化。')
 report=json.loads((out/'parity.json').read_text())
 if report['iteration']!=a.load_iter or report['point_count']!=after['point_count']:raise AssertionError('真实加载点数/迭代不符。')
 report.update(checkpoint_retrospective=True,actual_training_terminal_iteration=18000,checkpoint_cpu_evidence=after,checkpoint_four_asset_sha256=after['checkpoint_four_asset_sha256'])
 report['sources'].update(after['sources']);report['sources'][str(Path(__file__).resolve())]=digest(Path(__file__));report['sources'][str(Path(audit.__code__.co_filename).resolve())]=digest(audit.__code__.co_filename)
 with (out/'parity.json').open('w') as f:json.dump(report,f,ensure_ascii=False,indent=2)
 with (out/'checkpoint_sha256.json').open('x') as f:json.dump(after['checkpoint_four_asset_sha256'],f,ensure_ascii=False,indent=2)
 with (out/'checkpoint_after_audit.json').open('x') as f:json.dump(dict(status='PASS',unchanged=True,before=before['checkpoint_four_asset_sha256'],after=after['checkpoint_four_asset_sha256']),f,ensure_ascii=False,indent=2)
 with (out/'README.md').open('a') as f:f.write('\n本次为已有中间保存点回溯，原真实训练终点仍18000/2000步；原finished未改写。四资产匹配旧保存清单与归档SHA，元数据/Adam step/相机余栈/RNG/XYZ LR独立CPU核对并前后不变。只验前向接入，不提前判科学指标或完整足形PASS。\n')
 print('中间检查点真实fresh PASS',a.load_iter,flush=True)
if __name__=='__main__':main()
