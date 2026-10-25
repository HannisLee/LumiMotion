"""将同视角 Stage3 配对结果导出为中文报告、PNG、PDF 和离线 HTML。"""
import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT/'exports/1024-01-静态ELDL-研究展示素材/NotoSansCJKsc-Regular.otf'
EXPERIMENTS = {
    '静态原生': '1025-03-staticDL-Stage3原生基线',
    '静态改进': '1025-08-staticDL-Stage3已知方向光',
    '动态原生': '1025-05-dynamicDL-Stage3原生基线',
    '动态改进': '1025-07-dynamicDL-Stage3估计方向光',
    '静态仅几何': '1025-04-staticDL-Stage3改进几何',
    '动态仅几何': '1025-06-dynamicDL-Stage3改进几何',
}
INK = '#15283e'
MUTED = '#607187'
ACCENT = '#087e8b'


def font(size):
    return ImageFont.truetype(str(FONT), size)


def canvas(title, subtitle, index):
    image = Image.new('RGB', (1920, 1080), '#f7f9fc')
    d = ImageDraw.Draw(image)
    d.rectangle((0, 0, 1920, 12), fill=ACCENT)
    d.text((70, 42), 'LUMIMOTION / STAGE 3', fill=ACCENT, font=font(25))
    d.text((70, 90), title, fill=INK, font=font(52))
    d.text((72, 170), subtitle, fill=MUTED, font=font(25))
    d.line((70, 1006, 1850, 1006), fill='#d6dfe9', width=2)
    d.text((70, 1025), '2026.10.25  ·  同一数据内配对  ·  15 TEST / 400×400  ·  固定终点', fill=MUTED, font=font(21))
    d.text((1780, 1025), f'{index:02d}', fill=ACCENT, font=font(24))
    return image


def put(image, path, box):
    source = Image.open(path).convert('RGB')
    scale = min((box[2]-box[0])/source.width, (box[3]-box[1])/source.height)
    source = source.resize((round(source.width*scale), round(source.height*scale)), Image.Resampling.LANCZOS)
    image.paste(source, (box[0]+(box[2]-box[0]-source.width)//2, box[1]+(box[3]-box[1]-source.height)//2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'Exporter/1025-01-Stage3静动态实验展示')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output/'素材').mkdir()
    (output/'视频').mkdir()
    reports = {}
    states = {}
    source_manifest = []
    for name, dirname in EXPERIMENTS.items():
        experiment = ROOT/'output'/dirname
        states[name] = json.loads((experiment/'state.json').read_text())
        assert states[name]['status'] == 'READY_FOR_REVIEW', (name, states[name]['status'])
        reports[name] = json.loads((experiment/'evaluation/metrics.json').read_text())
        assert len(reports[name]['frames']) == 15 and reports[name]['checkpoint_unchanged']
        for category in ['rgb', 'gt', 'alpha', 'normal', 'separation', 'albedo', 'roughness', 'error']:
            for key in ['0010', '0080', '0150']:
                original = experiment/'evaluation'/category/(key+'.png')
                destination = output/'素材'/f'{name}-{category}-{key}.png'
                shutil.copy2(original, destination)
                source_manifest.append(dict(file=str(destination.relative_to(output)), source=str(original),
                                            sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
            original = experiment/'evaluation'/(category+'.mp4')
            shutil.copy2(original, output/'视频'/f'{name}-{category}-15测试视角.mp4')
    rows = []
    for kind in ['静态', '动态']:
        a, b = reports[kind+'原生'], reports[kind+'改进']
        for frame in a['frames']:
            key = frame['frame']
            arrays = [np.load(ROOT/'output'/EXPERIMENTS[kind+arm]/'evaluation/arrays'/(key+'.npz')) for arm in ['原生', '改进']]
            assert np.array_equal(arrays[0]['target'], arrays[1]['target'])
            assert np.array_equal(arrays[0]['mask'], arrays[1]['mask'])
            for array in arrays:
                array.close()
        for arm in ['原生', '改进']:
            row = {'数据': kind+'DL-V2', '方法': '原生LumiMotion环境PBR' if arm == '原生' else '改进几何＋'+('已知GT方向PBR' if kind == '静态' else '估计方向PBR'),
                   **reports[kind+arm]['metrics'], 'model': reports[kind+arm]['model'], 'iteration': reports[kind+arm]['iteration']}
            rows.append(row)
    controls = [{'数据': kind+'DL-V2', '方法': '改进几何＋环境PBR', **reports[kind+'仅几何']['metrics'],
                 'model': reports[kind+'仅几何']['model'], 'iteration': reports[kind+'仅几何']['iteration']} for kind in ['静态', '动态']]
    with (output/'1025-01-Stage3同协议指标.csv').open('w', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows+controls)
    slides = []
    image = canvas('几何与方向照明承接后的 Stage3 外观', '同数据、相同新增预算；静态使用已知GT方向，动态只用TRAIN估计方向；固定20000循环终点', 1)
    d = ImageDraw.Draw(image)
    columns = [(90, '数据与方法'), (950, 'PSNR ↑'), (1230, 'SSIM ↑'), (1510, '前景PSNR ↑')]
    for x, label in columns:
        d.text((x, 285), label, font=font(31), fill=MUTED)
    for i, row in enumerate(rows):
        y = 365+i*112
        d.rounded_rectangle((70, y-12, 1850, y+81), radius=14, fill='white' if i%2 == 0 else '#e9f4f5')
        for x, value in [(90, row['数据']+' / '+row['方法']), (950, f'{row["psnr"]:.3f} dB'),
                         (1230, f'{row["ssim"]:.5f}'), (1510, f'{row["psnr_foreground"]:.3f} dB')]:
            d.text((x, y), value, font=font(31), fill=INK)
    for i, kind in enumerate(['静态', '动态']):
        a, b = reports[kind+'原生']['metrics'], reports[kind+'改进']['metrics']
        d.text((90, 846+i*55), f'{kind}改进−原生：PSNR {b["psnr"]-a["psnr"]:+.3f} dB；SSIM {b["ssim"]-a["ssim"]:+.5f}', font=font(29), fill=ACCENT)
    slides.append(('1025-01-Stage3定量结果', image))
    image = canvas('仅改几何与完整方法分开比较', '三个设置：原生环境PBR／改进几何＋环境PBR／改进几何＋方向PBR；不将全部收益归因于法线', 2)
    d = ImageDraw.Draw(image)
    for i, row in enumerate([rows[0], controls[0], rows[1], rows[2], controls[1], rows[3]]):
        y = 290+i*103
        d.rounded_rectangle((70, y-8, 1850, y+78), radius=10, fill='white' if i%3 == 0 else '#e9f4f5')
        d.text((95, y), row['数据']+' / '+row['方法'], font=font(30), fill=INK)
        d.text((1170, y), f'{row["psnr"]:.3f} dB', font=font(31), fill=ACCENT)
        d.text((1510, y), f'{row["ssim"]:.5f}', font=font(31), fill=ACCENT)
    d.text((95, 948), '同一数据内相同RGB/Alpha；照明模型属于方法变量。静态GT方向条件不等同于动态未知光估计。', font=font(23), fill=MUTED)
    slides.append(('1025-02-Stage3几何承接与照明对照', image))
    for kind in ['静态', '动态']:
        for key in ['0080', '0150']:
            light_label = '已知GT方向' if kind == '静态' else 'TRAIN估计方向（无GT灯向）'
            image = canvas(kind+'场景：同视角外观对比', f'固定 TEST {key}；改进组使用{light_label}；保留全图与统一人体窗口', len(slides)+1)
            d = ImageDraw.Draw(image)
            sources = [output/'素材'/f'{kind}原生-gt-{key}.png', output/'素材'/f'{kind}原生-rgb-{key}.png', output/'素材'/f'{kind}改进-rgb-{key}.png']
            for col, (label, source) in enumerate(zip(['真实观测 GT', '原生 LumiMotion', '改进几何＋方向 PBR'], sources)):
                x = 95+col*610
                d.text((x, 235), label, fill=INK, font=font(32))
                put(image, source, (x, 290, x+520, 810))
                # 固定中心人体窗口；同一场景、三种方法完全同坐标。
                crop = (150, 120, 250, 205)
                d.rectangle((x+crop[0]*1.3, 290+crop[1]*1.3, x+crop[2]*1.3, 290+crop[3]*1.3), outline='#f6a23a', width=4)
                zoom = Image.open(source).convert('RGB').crop(crop)
                zoom_scale = min(190/zoom.width, 160/zoom.height)
                zoom = zoom.resize((round(zoom.width*zoom_scale), round(zoom.height*zoom_scale)), Image.Resampling.LANCZOS)
                image.paste(zoom, (x+380, 815))
                if col:
                    metrics = next(r for r in reports[kind+('原生' if col == 1 else '改进')]['frames'] if r['frame'] == key)
                    d.text((x, 840), f'PSNR {metrics["psnr"]:.2f} dB', fill=ACCENT, font=font(27))
                    d.text((x, 885), f'SSIM {metrics["ssim"]:.4f}', fill=ACCENT, font=font(27))
            slides.append((f'1025-{len(slides)+1:02d}-{kind}RGB对比-{key}', image))
        image = canvas(kind+'场景：几何、覆盖与材质诊断', '固定 TEST 0150；法线为当前 Stage3 checkpoint 渲染；材质图不能当作 GT 反照率', len(slides)+1)
        d = ImageDraw.Draw(image)
        categories = [('normal', 'GS法线'), ('alpha', 'Alpha覆盖'), ('albedo', '反照率'), ('separation', '原生红绿特征')]
        for col, (category, label) in enumerate(categories):
            x = 78+col*460
            d.text((x+60, 238), label, font=font(29), fill=INK)
            for row, arm in enumerate(['原生', '改进']):
                y = 294+row*345
                put(image, output/'素材'/f'{kind}{arm}-{category}-0150.png', (x, y, x+340, y+330))
                d.text((x+345, y+135), arm, font=font(25), fill=ACCENT)
        d.text((95, 972), '红/绿特征只反映模型门控；动态候选全绿源于全点形变设置。鞋部缺陷和材质残留须单独说明。', font=font(22), fill=MUTED)
        slides.append((f'1025-{len(slides)+1:02d}-{kind}四类与材质诊断', image))
    for name, image in slides:
        image.save(output/(name+'.png'))
    slides[0][1].save(output/'1025-01-Stage3静动态展示合集.pdf', save_all=True, append_images=[im for _, im in slides[1:]], resolution=150)
    table = '| 数据 | 方法 | PSNR(dB) | SSIM | 前景PSNR(dB) |\n|---|---|---:|---:|---:|\n'
    table += '\n'.join(f'| {r["数据"]} | {r["方法"]} | {r["psnr"]:.6f} | {r["ssim"]:.6f} | {r["psnr_foreground"]:.6f} |' for r in rows)
    report = '# Stage3 静动态实验展示与复现说明\n\n日期：2026-10-25。服务器 ushas，Conda lumimotion-ushas。逻辑 Stage3 对应原生代码 Stage2 的环境 PBR 外观训练。\n\n'+table+'\n\n'
    report += '同一数据内 Stage3 RGB/Alpha输入、512半球采样、400×400、20000新增循环/4999真实更新、原生PBR损失和混合追踪深度1一致；环境光与方向光属于方法变量，相应光照参数/正则按各入口语义处理。静态完整方法使用已知GT方向；动态使用父TRAIN估计方向及插值，不读GT灯向或GT强度。历史API参数名称gt_directional不表示动态数据读取了GT。动态原始RGBA净化与原生视图132张输入RGB/Alpha逐字节相同，证据在总控directional_input_comparison.json。15固定测试视角全图浮点sRGB逐帧PSNR/SSIM后均值；SSIM为项目11×11、sigma1.5；前景分数另列。NumPy/SciPy独立复算指标，保存float数组，配对GT和Mask逐元素一致。\n\n'
    report += '纯环境PBR仅换几何的控制结果：\n\n| 数据 | PSNR(dB) | SSIM | 前景PSNR(dB) |\n|---|---:|---:|---:|\n'+'\n'.join(f'| {r["数据"]} | {r["psnr"]:.6f} | {r["ssim"]:.6f} | {r["psnr_foreground"]:.6f} |' for r in controls)+'\n\n这两臂与方向PBR完整方法一起保留，不能将方向光承接收益全部归因于法线。\n\n'
    report += '对齐的是新增外观预算，不是整条训练链的总计算成本；原生父35000循环、改进父16500更新且包含不同阶段。单场景/单种子实验不声明微小差异的统计显著性。\n\n'
    report += '静态父为原生S1@35000和历史GS最优F-T04@16500（2.663858°）；动态为原生S1@35000和完整路线预登记主检查点@16500（7.329092°，原生14.545605°）。这些法线来自父模型，不冒充当前Stage3的同checkpoint法线MAE。原生动态父见过135训练时刻、改进父117，且动态门控和颜色头设置不同，属于完整方法对照，不能证明法线单因素因果。\n\n'
    report += '训练source移除了TEST；所有TEST指标均在对应模型完成固定预算后计算，TEST不参与优化或选参。早期冒烟只评分和查看TRAIN，但Scene初始化曾加载含TEST图像的评价视图；输入逐字节一致性审计也读取TEST输入，均未用于梯度、TEST评分或按图选参。后续TRAIN诊断改用无TEST视图。历史TEST已被多轮实验查看，不称新盲测。现有1024 EL44.685147dB/0.997835与DL29.316748dB/0.974770为不同照明/已知光设置，不能混入本表。\n\n'
    report += '展示顺序：先定量页，再静态RGB和四类诊断，再动态RGB和诊断。局部框使用固定相同坐标，并保留完整图。视频目录为按采集顺序的15个TEST视角，静态也会移动相机，动态同时变化相机和时间；不是固定相机动态视频。PNG与PDF可直接插入PPT，HTML离线浏览；CSV可编辑。\n\n'
    report += '## 来源与完整命令\n\n'
    for name, state in states.items():
        report += f'- {name}：{ROOT/"output"/EXPERIMENTS[name]}；source={state["source"]}；parent={state["parent"]}@{state["source_iteration"]}；model={reports[name]["model"]}@{reports[name]["iteration"]}。完整训练/渲染命令及环境在该目录README.md、state.json；train.log和evaluation.log，evaluation/arrays与metrics.json保留。\n'
    report += '\n## 验收\n\n工程审计和四类实际目检见实验目录 independent_audit.json / manual_review.json。工程完成和数值提升不能替代完整分解质量。最终科学结论见本次报告；没有通过的检查明确写FAILED，全部checkpoint/图片/统计/日志保留。\n'
    (output/'1025-01-Stage3实验展示说明.md').write_text(report)
    (output/'1025-02-来源清单.json').write_text(json.dumps(dict(artifacts=source_manifest, metrics=reports, states=states), ensure_ascii=False, indent=2)+'\n')
    body = '<!doctype html><meta charset="utf-8"><title>Stage3静动态实验展示</title><style>body{background:#172338;color:white;font-family:sans-serif;margin:0}main{max-width:1400px;margin:auto}img{width:100%;display:block;margin:30px 0}a{color:#83ddd8}p{padding:20px}</style><main><p><a href="1025-01-Stage3实验展示说明.md">实验说明</a> · <a href="1025-01-Stage3静动态展示合集.pdf">PDF合集</a> · <a href="1025-01-Stage3同协议指标.csv">指标CSV</a></p>'
    body += ''.join('<img src="'+html.escape(name+'.png')+'" alt="'+html.escape(name)+'">' for name, _ in slides)
    body += '<p>视频为15个TEST视角的采样序列，非固定相机。'+''.join('<br><a href="'+html.escape(str(p.relative_to(output)))+'">'+html.escape(p.name)+'</a>' for p in (output/'视频').glob('*rgb*'))+'</p></main>'
    (output/'1025-01-Stage3展示.html').write_text(body)
    print(output)


if __name__ == '__main__':
    main()
