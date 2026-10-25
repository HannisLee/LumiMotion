"""从已冻结的四组 RGB 导出同位置框选放大，保留不支持提升的核对结论。"""
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PdfParser

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'LIHAN/exports/1025-02-静动态EL与GT方向光ELDL-Stage3外观对比'
DEST = SOURCE / '1026-01-几何细节放大版'
FONT = ROOT / 'output/1025-33-静动态EL与GT方向光ELDL-Stage3总控/assets/NotoSansCJKsc-Regular.otf'
INK = '#192b42'
MUTED = '#536278'
BOX = '#ef6248'


def font(size):
    return ImageFont.truetype(str(FONT), size)


def text(draw, xy, message, size=26, fill=INK):
    draw.text(xy, message, font=font(size), fill=fill)


def columns(kind):
    return [(kind + '_EL', 'gt', 'EL 数据 GT'),
            (kind + '_EL', 'rgb', '原生 EL · Stage3'),
            (kind + '_ELDL', 'gt', 'ELDL 数据 GT'),
            (kind + '_ELDL', 'rgb', '改进 GT方向光 ELDL · Stage3')]


def read(arm, sub, frame):
    image = Image.open(SOURCE / arm / sub / f'{frame:04d}.png').convert('RGB')
    assert image.size == (400, 400)
    return image


CASES = [
    dict(kind='dynamic', name='动态', frame=50, roi=[210, 345, 250, 377],
         title='平台前缘：局部凸出伪影减少', status='局部轮廓改善可见',
         lines=['基线：平台圆弧下方多出黄灰色凸出物，两侧 GT 对应位置均无此结构。',
                '改进：该凸出物消失，局部外轮廓更接近 GT；结论仅针对此处可见轮廓。']),
    dict(kind='static', name='静态', frame=150, roi=[157, 230, 221, 294],
         title='小腿与鞋部：几何改善未获确认', status='核对页 · 不作为提升例子',
         lines=['两侧均可辨认小腿与鞋部形状；当前 RGB 未显示改进结果具有明确的轮廓优势。',
                '改进侧仍有边缘不规则和地面碎点，不能将阴影更强或亮度变化视为几何改善。']),
]


def detail(case):
    canvas = Image.new('RGB', (1920, 1160), '#f5f6f8')
    draw = ImageDraw.Draw(canvas)
    text(draw, (55, 28), f"{case['name']} Stage3｜{case['title']}", 40)
    text(draw, (55, 88), f"视角 {case['frame']:04d} · {case['status']} · 四列使用相同像素框", 25)
    roi = case['roi']
    crop_scale = 360 / (roi[2] - roi[0])
    for i, (arm, sub, label) in enumerate(columns(case['kind'])):
        x = 40 + i * 470
        text(draw, (x, 146), label, 24)
        original = read(arm, sub, case['frame'])
        # 整图仅标框；放大图直接读取原始像素，不携带标框，不做增强或修补。
        canvas.paste(original, (x, 194))
        draw.rectangle((x + roi[0], 194 + roi[1], x + roi[2] - 1,
                        194 + roi[3] - 1), outline=BOX, width=2)
        crop = original.crop(roi)
        crop.save(DEST / '原始裁剪' / f"{case['kind']}-{arm}-{sub}.png")
        crop = crop.resize((360, round(crop.height * crop_scale)), Image.Resampling.NEAREST)
        y = 662
        canvas.paste(crop, (x + 20, y))
        draw.rectangle((x + 19, y - 1, x + 380, y + crop.height), outline=BOX, width=2)
        draw.line((x + (roi[0] + roi[2]) / 2, 194 + roi[3], x + 200, 600), fill=BOX, width=2)
        text(draw, (x, 608), f"原图裁剪 · 等比例 {crop_scale:g} 倍放大", 23)
    for j, line in enumerate(case['lines']):
        text(draw, (55, 1040 + j * 36), line, 25)
    text(draw, (55, 1124), 'EL 与 ELDL 照明不同，分别对照各自 GT；原始像素等倍率放大，不做锐化、补图或亮度调整。', 22, MUTED)
    return canvas


def overview(kind, name, frame):
    canvas = Image.new('RGB', (1920, 720), '#f5f6f8')
    draw = ImageDraw.Draw(canvas)
    text(draw, (55, 35), f'{name}同视角 RGB｜{frame:04d}', 42)
    for i, (arm, sub, label) in enumerate(columns(kind)):
        x = 40 + i * 470
        text(draw, (x, 135), label, 24)
        canvas.paste(read(arm, sub, frame), (x, 195))
    text(draw, (55, 654), '相同采集视角；EL 与 ELDL 照明不同，分别对照自身 GT。', 25, MUTED)
    return canvas


def legacy_main():
    DEST.mkdir(exist_ok=False)
    (DEST / '原始裁剪').mkdir()
    pages = []
    for case in CASES:
        page = detail(case)
        case['image'] = f"1026-01-{case['name']}几何细节.png"
        page.save(DEST / case['image'])
        pages.append(page)
    for kind, name in [('static', '静态'), ('dynamic', '动态')]:
        folder = DEST / f'{name}同视角'
        folder.mkdir()
        for frame in range(10, 151, 10):
            page = overview(kind, name, frame)
            page.save(folder / f'{frame:04d}.png')
            if frame in (10, 80, 150):
                pages.append(page)
    pdf = DEST / '1026-01-Stage3静动态几何细节与同视角RGB.pdf'
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
    parser = PdfParser.PdfParser(filename=str(pdf))
    assert len(parser.pages) == 8
    parser.close()
    provenance = json.loads((SOURCE / '1025-02-来源清单.json').read_text())
    manifest = dict(cases=CASES, sources=provenance['sources'],
                    checked_frames=list(range(10, 151, 10)),
                    resize='四列统一最近邻等倍率；无超分、锐化、修补或曝光修改',
                    coordinate='原图400×400，左上为零，裁剪框为左上包含、右下不包含',
                    verdict='PARTIAL：动态局部轮廓改善可见，静态未确认；未完成两组均有明确改善的目标')
    # 将每一份原始裁剪与源图做像素级核对。
    for case in CASES:
        for arm, sub, _ in columns(case['kind']):
            actual = Image.open(DEST / '原始裁剪' / f"{case['kind']}-{arm}-{sub}.png").convert('RGB')
            expected = read(arm, sub, case['frame']).crop(case['roi'])
            assert actual.tobytes() == expected.tobytes()
    (DEST / '1026-01-框选来源与验收.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    description = '''# Stage3 静动态几何细节放大导出

新版展示不含 PSNR、SSIM。PDF 共 8 页：动态、静态各一页框选放大核对，后附静态与动态各 3 个同视角 RGB（0010、0080、0150）；静态与动态全部 15 个视角另存 PNG。

动态视角 0050，原图框 [210,345,250,377)，9 倍最近邻放大：原生 EL 基线的平台前缘下方存在黄灰色凸出物，两侧 GT 中均无此结构，改进结果此处凸出物消失。支持局部可见轮廓改善，不据此推断完整三维几何或人体运动质量均提升。

静态视角 0150，原图框 [157,230,221,294)，5.625 倍最近邻放大：作为核对页保留。检查全部 15 个视角后，尚未发现足以确认改进优于基线的静态几何细节；小腿、鞋部仍有边缘不规则。该页不称为提升例子。

四列为 EL 数据 GT、原生 EL Stage3、ELDL 数据 GT、改进 GT方向光 ELDL Stage3。两种数据照明不同，分别对照自身 GT。几何来源仍为已确认的静态 G9@8100、动态 23@1500，沿用上一版最终 Stage3 保存点；本次仅重新排版裁剪，没有训练或调参。源模型、iteration、评估路径见框选来源 JSON 和上级目录原始来源清单；完整渲染命令、日志和历史验收保留于对应实验 README。

只读取原始导出 RGB；同一页四列裁剪坐标、倍率、插值一致，无增强或修补。原始裁剪单独保留并逐像素校验；JSON 记录坐标与来源。RGB 中的轮廓改善不能单独证明三维几何误差下降。原有数值版和失败实验记录保留在上级目录，新版未将其改判为通过。

## 验收

导出完整性 PASS：8 页 PDF、2 张细节页、30 张无数值同视角图、8 份未放大原始裁剪，像素校验通过。科学结论 PARTIAL：动态局部改善已找到，静态改善未确认，未达到“两组均显示明确几何改善”的目标。新版只展示 RGB，不添加法线、覆盖、材质诊断或固定相机内容。
'''
    (DEST / '1026-01-使用与核对说明.md').write_text(description)
    body = '<h1>Stage3 静动态几何细节</h1><p>同视角 RGB · 相同位置框选放大</p>'
    body += '<p>动态平台前缘可见局部轮廓改善；静态页为核对，尚未确认几何提升。</p>'
    body += f'<p><a href="{html.escape(pdf.name)}">下载 8 页 PDF</a> · <a href="1026-01-使用与核对说明.md">来源与说明</a></p>'
    for case in CASES:
        body += f'<img src="{case["image"]}" alt="{case["title"]}">'
    for name in ['静态', '动态']:
        body += f'<details><summary>{name}全部 15 个同视角</summary>'
        for frame in range(10, 151, 10):
            body += f'<img loading="lazy" src="{name}同视角/{frame:04d}.png">'
        body += '</details>'
    (DEST / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage3几何细节放大</title><style>body{font:18px sans-serif;max-width:1500px;margin:35px auto;color:#192b42;background:#f5f6f8}img{width:100%;margin:18px 0}summary{cursor:pointer;padding:20px}a{color:#215b8d}</style>' + body + '</html>')
    hashes = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DEST.rglob('*') if p.is_file()}
    (DEST / '1026-01-文件校验.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(destination=str(DEST), pdf_pages=8, detail_pages=2, overview_images=30,
                          raw_crops=8, crop_pixel_validation='PASS', conclusion='PARTIAL'), ensure_ascii=False))


def main():
    """当前交付：四张框选全图、四张独立放大图，无总览附页。"""
    global DEST
    DEST = SOURCE / '1026-02-静态两组动态两组局部对比'
    DEST.mkdir(exist_ok=False)
    cases = [
        dict(kind='static', name='静态', frame=30, roi=[170, 138, 230, 198], region='背肩与衣褶',
             observation='可见衣褶明暗与表面颗粒差异；未确认几何优于基线。'),
        dict(kind='static', name='静态', frame=110, roi=[165, 115, 225, 175], region='肘部与腰侧轮廓',
             observation='肘部与腰侧外轮廓接近；亮度差异不能作为几何改善证据。'),
        dict(kind='dynamic', name='动态', frame=140, roi=[120, 25, 180, 85], region='手腕与袖口',
             observation='两侧手腕和袖口均有不规则边缘；没有明确优于基线的结构证据。'),
        dict(kind='dynamic', name='动态', frame=150, roi=[145, 65, 205, 125], region='抬臂肘部与腋下',
             observation='可见袖部颗粒与明暗差异；两侧存在伪影，未确认几何提升。'),
    ]
    source_info = json.loads((SOURCE / '1025-02-来源清单.json').read_text())['sources']
    pages = []
    crop_hashes = []
    for number, case in enumerate(cases, 1):
        prefix = f"1026-{number:02d}-{case['name']}-{case['frame']:04d}-{case['region']}"
        full = Image.new('RGB', (1920, 680), '#f5f6f8')
        zoom = Image.new('RGB', (1920, 680), '#f5f6f8')
        fd, zd = ImageDraw.Draw(full), ImageDraw.Draw(zoom)
        title = f"{case['name']} {number if number <= 2 else number - 2}/2｜视角 {case['frame']:04d}｜{case['region']}"
        text(fd, (45, 28), title + ' · 框选全图', 36)
        text(zd, (45, 28), title + ' · 局部放大', 36)
        for i, (arm, sub, label) in enumerate(columns(case['kind'])):
            x = 40 + 470 * i
            original = read(arm, sub, case['frame'])
            text(fd, (x, 117), label, 24)
            text(zd, (x, 117), label, 24)
            full.paste(original, (x, 172))
            l, t, r, b = case['roi']
            fd.rectangle((x + l, 172 + t, x + r - 1, 172 + b - 1), outline=BOX, width=2)
            crop = original.crop(case['roi'])
            zoom.paste(crop.resize((400, 400), Image.Resampling.NEAREST), (x, 172))
            zd.rectangle((x, 172, x + 399, 571), outline=BOX, width=2)
            crop_hashes.append(dict(case=number, arm=arm, sub=sub,
                                    source=str(SOURCE / arm / sub / f"{case['frame']:04d}.png"),
                                    source_sha256=hashlib.sha256((SOURCE / arm / sub / f"{case['frame']:04d}.png").read_bytes()).hexdigest(),
                                    crop_pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest()))
            # 导出放大图的每块像素须与同倍率源裁剪完全一致。
            assert zoom.crop((x, 172, x + 400, 572)).crop((2, 2, 398, 398)).tobytes() == crop.resize((400, 400), Image.Resampling.NEAREST).crop((2, 2, 398, 398)).tobytes()
        text(fd, (45, 608), '四列相同原图坐标框选；照明不同，分别对照各自 GT。', 24, MUTED)
        text(zd, (45, 608), '60×60 原始像素统一放大至 400×400（约 6.667 倍最近邻）；仅作局部 RGB 对照。', 24, MUTED)
        case['full_image'], case['zoom_image'] = prefix + '-对比.png', prefix + '-放大.png'
        full.save(DEST / case['full_image'])
        zoom.save(DEST / case['zoom_image'])
        pages.extend([full, zoom])
    pdf = DEST / '1026-02-静态两组动态两组RGB对比与放大.pdf'
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
    parser = PdfParser.PdfParser(filename=str(pdf))
    assert len(parser.pages) == 8
    parser.close()
    assert len(list(DEST.glob('*.png'))) == 8
    manifest = dict(cases=cases, sources=source_info, pixel_checks=crop_hashes,
                    reviewed=dict(test_frames=list(range(10, 151, 10)), dev_frames=[f for f in range(61, 80) if f != 70], views_per_condition=33),
                    conclusion='未找到满足要求的四组明确几何提升；交付为四组局部RGB对照，不标为成功案例。',
                    validation='8张PNG，8页PDF；放大区域与原始裁剪最近邻缩放逐像素一致。')
    (DEST / '1026-02-来源与核对.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    description = '''# 静态两组、动态两组局部 RGB 对照

本版严格导出四张框选对比图与四张独立放大图，共八张 PNG；另附八页 PDF，按每组全图、放大图交替排列，无指标、总览或其他诊断附页。上一版 `1026-01-几何细节放大版` 已按用户要求从导出目录撤下；原始训练、评估与失败记录保留。

静态选择 0030 背肩衣褶、0110 肘部腰侧；动态选择 0140 手腕袖口、0150 抬臂肘部腋下。均不采用鞋面或平台边缘小凸点。每组四列顺序为 EL 数据 GT、原生 EL Stage3、ELDL 数据 GT、改进 GT方向光 ELDL Stage3。两种照明不同，不能把明暗差异解释为几何提升。

重新检查了每种条件 15 个 TEST 视角及已保存的 18 个 DEV 视角，共每种 33 个视角。没有找到四组明显、可靠的几何提升，故导出的四组仅为人体局部形态对照，不作为“改进均优于原生”的证据。静态轮廓总体接近；动态手腕、手臂存在不规则边缘，改进侧并非稳定更好。各组逐项观察见来源 JSON。不得将本版图表中中性的区域标题解读为提升结论。

原图 400×400；四列使用完全相同的 60×60 裁剪框，最近邻统一放大到 400×400，无锐化、曝光调整、超分、修补或合成。框坐标为左上包含、右下不包含，记录于 JSON。全部取自此前冻结的最终 TEST 导出，未改变模型、iteration 或重新训练。来源模型与完整评估路径见 JSON，完整渲染命令与历史验收仍见实验 README。

导出完整性 PASS：8 张 PNG、8 页 PDF，16 个放大区域像素校验通过。科学结论 FAILED：未找到四组具有说服力的几何提升，当前图像不足以支持所需结论。
'''
    (DEST / '1026-02-使用与核对说明.md').write_text(description)
    body = '<h1>静态两组 · 动态两组</h1><p>四张框选对比 + 四张局部放大。当前为 RGB 形态核对，未证实四组几何提升。</p>'
    body += f'<p><a href="{pdf.name}">下载八页 PDF</a> · <a href="1026-02-使用与核对说明.md">核对说明</a></p>'
    for case in cases:
        for key in ['full_image', 'zoom_image']:
            body += f'<img src="{html.escape(case[key])}">'
    (DEST / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>静动态四组局部对照</title><style>body{font:18px sans-serif;margin:35px auto;max-width:1500px;background:#f5f6f8;color:#192b42}img{width:100%;margin:18px 0}</style>' + body + '</html>')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file()}
    (DEST / '1026-02-文件校验.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(destination=str(DEST), png_images=8, pdf_pages=8, enlarged_regions=16, validation='PASS', geometric_improvement='FAILED'), ensure_ascii=False))


if __name__ == '__main__':
    main()
