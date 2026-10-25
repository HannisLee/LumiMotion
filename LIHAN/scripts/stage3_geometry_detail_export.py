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


def main():
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


if __name__ == '__main__':
    main()
