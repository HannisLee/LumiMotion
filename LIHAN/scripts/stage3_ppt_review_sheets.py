"""最终只读目检联系表：全15TEST RGB、三视角四类和固定相机时序采样。"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LIHAN.scripts.stage3_ppt_export import EXPERIMENTS, ROOT, FONT


def main():
    output = ROOT/'output/1025-02-Stage3-PPT配对总控/final_review'
    output.mkdir(exist_ok=False)
    font = ImageFont.truetype(str(FONT), 20)
    manifest = []

    def save(image, name, details):
        path = output/name
        image.save(path, quality=94)
        manifest.append(dict(file=str(path), **details))

    for kind in ['静态', '动态']:
        for page in range(3):
            image = Image.new('RGB', (1600, 2150), 'white')
            draw = ImageDraw.Draw(image)
            for row, index in enumerate(range(page*5+1, page*5+6)):
                key = f'{index*10:04d}'
                sources = [('GT', kind+'原生', 'gt'), ('原生环境', kind+'原生', 'rgb'),
                           ('改进父环境', kind+'环境承接', 'rgb'), ('方向PBR', kind+'改进', 'rgb')]
                for col, (label, arm, category) in enumerate(sources):
                    source = ROOT/'output'/EXPERIMENTS[arm]/'evaluation'/category/(key+'.png')
                    image.paste(Image.open(source).convert('RGB'), (col*400, row*430+30))
                    draw.text((col*400+8, row*430+2), label+' '+key, font=font, fill='black')
            save(image, f'{kind}-全15TEST-RGB-第{page+1}页.jpg', dict(dataset=kind, scope='每数据全部15TEST，三页完整RGB，含环境承接控制'))
    for arm, directory in EXPERIMENTS.items():
        image = Image.new('RGB', (1600, 1290), 'white')
        draw = ImageDraw.Draw(image)
        for row, key in enumerate(['0010', '0080', '0150']):
            for col, category in enumerate(['rgb', 'alpha', 'normal', 'separation']):
                source = ROOT/'output'/directory/'evaluation'/category/(key+'.png')
                image.paste(Image.open(source).convert('RGB'), (col*400, row*430+30))
                draw.text((col*400+8, row*430+2), category+' '+key, font=font, fill='black')
        save(image, arm+'-四类.jpg', dict(experiment=arm, frames=[10, 80, 150], categories=['rgb', 'alpha', 'normal', 'separation']))
    for kind in ['静态', '动态']:
        image = Image.new('RGB', (1600, 1290), 'white')
        draw = ImageDraw.Draw(image)
        for row, key in enumerate(['0010', '0080', '0150']):
            for col, (arm, category) in enumerate([(kind+'原生', 'albedo'), (kind+'改进', 'albedo'),
                                                   (kind+'原生', 'roughness'), (kind+'改进', 'roughness')]):
                source = ROOT/'output'/EXPERIMENTS[arm]/'evaluation'/category/(key+'.png')
                image.paste(Image.open(source).convert('RGB'), (col*400, row*430+30))
                draw.text((col*400+8, row*430+2), arm+' '+category+' '+key, font=font, fill='black')
        save(image, kind+'-材质.jpg', dict(dataset=kind, frames=[10, 80, 150], scope='原生与方向PBR材质；不等同GT材质定量'))
    for arm in ['动态原生', '动态改进']:
        source = ROOT/'output'/EXPERIMENTS[arm]/'fixed_view_0149'
        image = Image.new('RGB', (1600, 1290), 'white')
        draw = ImageDraw.Draw(image)
        for row, key in enumerate(['0001', '0075', '0150']):
            for col, category in enumerate(['rgb', 'alpha', 'normal', 'separation']):
                image.paste(Image.open(source/category/(key+'.png')).convert('RGB'), (col*400, row*430+30))
                draw.text((col*400+8, row*430+2), category+' '+key, font=font, fill='black')
        save(image, arm+'-固定相机四类.jpg', dict(experiment=arm, view=149, frames=[1, 75, 150], categories=['rgb', 'alpha', 'normal', 'separation']))
        image = Image.new('RGB', (960, 1080), 'white')
        draw = ImageDraw.Draw(image)
        reader = imageio.get_reader(str(source/'rgb.mp4'))
        selected = list(range(0, 150, 10))+[149]
        try:
            for i, index in enumerate(selected):
                frame = Image.fromarray(reader.get_data(index)).resize((240, 240), Image.Resampling.LANCZOS)
                x, y = (i%4)*240, (i//4)*270
                image.paste(frame, (x, y+30))
                draw.text((x+6, y+2), '固定149，时间'+str(index+1), font=font, fill='black')
        finally:
            reader.close()
        save(image, arm+'-固定相机16时刻RGB.jpg', dict(experiment=arm, view=149, frames=[x+1 for x in selected], scope='16时刻实际目检样本；不声称逐帧人工查看全部150帧'))
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
