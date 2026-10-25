"""实际目检完成后归档六实验README、中文报告、交付校验清单与素材包。"""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LIHAN.scripts.stage3_ppt_export import ROOT, EXPERIMENTS


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--export', type=Path, default=ROOT/'Exporter/1025-01-Stage3静动态实验展示')
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    assert review['status'] == 'REVIEW_COMPLETED' and set(review['experiments']) == set(EXPERIMENTS)
    assert args.export.is_dir()
    control = ROOT/'output/1025-02-Stage3-PPT配对总控'
    assert not (control/'archive_summary.json').exists()
    records = {}
    for arm, name in EXPERIMENTS.items():
        experiment = ROOT/'output'/name
        state = json.loads((experiment/'state.json').read_text())
        assert state['status'] == 'READY_FOR_REVIEW'
        audit = json.loads((experiment/'independent_audit.json').read_text())
        metric_audit = json.loads((experiment/'independent_metric_validation.json').read_text())
        assert audit['status'] == metric_audit['status'] == 'PASS'
        assert len(metric_audit['frames']) == 15 and all(x == 15 for x in metric_audit['full_video_decoded_frames'].values())
        item = review['experiments'][arm]
        assert item['inspection_status'] in ['PASS', 'FAILED']
        assert all(key in item for key in ['rgb', 'alpha', 'normal', 'separation', 'limitations', 'scope'])
        metrics = json.loads((experiment/'evaluation/metrics.json').read_text())
        write_new(experiment/'manual_review.json', item)
        environment = dict(state['environment'], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='4', PYTHONUNBUFFERED='1')
        commands = {label: shlex.join(['env']+[key+'='+str(value) for key, value in environment.items()]+state[key])
                    for label, key in [('完整训练', 'command'), ('完整15TEST渲染', 'evaluation_command')]}
        text = '\n## 最终指标、命令与实际目检归档\n\n'
        text += f'服务器ushas，环境lumimotion-ushas；source={state["source"]}；父model={state["parent"]}@{state["source_iteration"]}；输出model={experiment/"model_mlp"}@{state["iteration"]}。\n\n'
        text += '实际训练20000循环、4999轮原生同步更新（Gaussian优化器每轮step一次）；最后循环先保存且不更新。XYZ/rotation/scale/fea门控与全部MLP逐元素同父，opacity和材质允许学习，因此不将父法线MAE作为终点法线MAE。\n\n'
        for label, command in commands.items():
            text += label+'命令：\n```bash\n'+command+'\n```\n\n'
        text += f'15TEST：PSNR={metrics["metrics"]["psnr"]:.10f}dB，SSIM={metrics["metrics"]["ssim"]:.10f}，前景PSNR={metrics["metrics"]["psnr_foreground"]:.10f}dB。全图float sRGB逐帧后均值；SSIM11×11/sigma1.5；NumPy/SciPy独立复算PASS，四类及材质视频完整程序解码15帧。\n\n'
        text += '渲染目录evaluation/{rgb,gt,alpha,normal,separation,albedo,roughness,error,arrays}；代表图各目录0010.png/0080.png/0150.png；代表视频evaluation/rgb.mp4、alpha.mp4、normal.mp4、separation.mp4（15采集视角，非固定相机）。日志train.log/evaluation.log/independent_audit.log/independent_metric_validation.log。\n\n'
        if arm in ['动态原生', '动态改进']:
            fixed = json.loads((experiment/'fixed_view_0149/receipt.json').read_text())
            assert len(fixed['frames']) == 150 and fixed['target_buffers_zeroed']
            text += '固定TRAIN相机0149：fixed_view_0149/{rgb,alpha,normal,separation,albedo,roughness}.mp4，各150帧/25fps；日志fixed_view_0149.log。目标RGB/Alpha置零，未观测固定视角不计算GT指标；实际命令另见上方后处理命令和总控finish_queue.json。\n\n'
        text += '实际目检范围：'+item['scope']+'\n\n'
        for key, label in [('rgb', 'RGB'), ('alpha', 'Alpha'), ('normal', 'Normal'), ('separation', '分离特征')]:
            text += '- '+label+'：'+item[key]+'\n'
        text += '\n限制：'+item['limitations']+'\n\n最终结论：'+item['inspection_status']+'（'+item['conclusion_scope']+'）；训练、指标计量与证据交付PASS。失败/负对照的checkpoint、渲染、统计和日志全部保留，不覆盖先前目检记录。\n'
        with (experiment/'README.md').open('a') as stream:
            stream.write(text)
        records[arm] = dict(experiment=str(experiment), metrics=metrics['metrics'], source_iteration=state['source_iteration'],
                            iteration=state['iteration'], inspection=item, engineering='PASS', metric_validation='PASS',
                            checkpoint_sha256=metrics['checkpoint_sha256'])
    section = '\n## 最终实际验收与交付结论\n\n'
    section += '全部六组固定预算训练、15TEST评价、独立冻结/指标复算、视频全解码及中文展示整理完成，交付验收 **PASS**。四类目检与完整几何/材质恢复分开判断；父模型足部缺陷没有被Stage3位置更新修复。\n\n'
    section += '| 实验 | 实际目检结论 | 适用范围 |\n|---|---|---|\n'
    section += '\n'.join('| '+arm+' | '+records[arm]['inspection']['inspection_status']+' | '+records[arm]['inspection']['conclusion_scope']+' |' for arm in EXPERIMENTS)+'\n\n'
    section += review['overall_conclusion']+'\n\n'
    section += '实际查看范围与逐实验结论保存在各实验manual_review.json，联系表在总控final_review。平台真实彩色棋盘格不单独作为伪影；按同视角GT的亮度、投影与结构判断。所有程序全解码均不冒充人工逐帧播放。\n'
    document = args.export/'1025-01-Stage3实验展示说明.md'
    with document.open('a') as stream:
        stream.write(section)
    report = ROOT/'LIHAN/DOC/报告/1025-01-Stage3静动态PBR基线与方向光完整方法最终报告.md'
    with report.open('x') as stream:
        stream.write(document.read_text()+'\n交付目录：'+str(args.export)+'。主展示：1025-01-Stage3静动态展示合集.pdf；原图与视频及CSV均在该目录。\n')
    summary = dict(status='PASS_DELIVERY', completed=time.time(), experiments=records, manual_review_source=str(args.review),
                   export=str(args.export), report=str(report), overall_conclusion=review['overall_conclusion'],
                   git_before_report_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_new(control/'archive_summary.json', summary)
    write_new(args.export/'1025-03-最终验收.json', summary)
    slides = list(args.export.glob('1025-*.png'))
    assert len(slides) == 9
    for slide in slides:
        assert Image.open(slide).size == (1920, 1080)
    hashes = {str(path.relative_to(args.export)): sha(path) for path in sorted(args.export.rglob('*')) if path.is_file()}
    write_new(args.export/'1025-04-交付文件校验.json', dict(sha256=hashes, scope='校验清单自身不包含在自身哈希中'))
    archive = args.export.parent/'1025-01-Stage3静动态实验展示.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(args.export.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(args.export.parent))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    queue = json.loads((control/'finish_queue.json').read_text())
    queue.update(status='COMPLETED_WITH_MANUAL_REVIEW', archive=str(archive), archive_sha256=sha(archive), manual_review=str(args.review))
    (control/'finish_queue.json').write_text(json.dumps(queue, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(report=str(report), export=str(args.export), archive=str(archive), status='PASS_DELIVERY'), ensure_ascii=False))


if __name__ == '__main__':
    main()
