"""将项目阴影估光方案绘成中文演示配图；只读实验，不运行训练或选择灯向。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Arc

ROOT = Path(__file__).resolve().parents[2]
C = dict(ink='#152D46', muted='#65788A', line='#DEE6EC', teal='#008C85', blue='#4C80B0',
         orange='#DF9C35', red='#D56C59', pale='#F4F7FA', shadow='#364A5B')


def txt(ax, x, y, s, size=17, color=None, ha='left', weight='normal', **kw):
    return ax.text(x, y, s, fontsize=size, color=color or C['ink'], va='center', ha=ha,
                   fontweight=weight, **kw)


def box(ax, x, y, w, h, fill='white', edge=None, lw=1, radius=.14):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                      facecolor=fill, edgecolor=edge or C['line'], linewidth=lw)
    ax.add_patch(p)
    return p


def line(ax, p, q, color=None, lw=2, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color or C['line'], lw=lw,
            solid_capstyle='round', **kw)


def arrow(ax, p, q, color=None, lw=2, ms=15, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms,
                                color=color or C['teal'], linewidth=lw, **kw))


def base(title, subtitle, n):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 16), ylim=(0, 9), aspect='equal')
    ax.axis('off')
    box(ax, .58, 7.85, .07, .70, C['teal'], C['teal'], radius=.02)
    txt(ax, .83, 8.28, title, 29, weight='bold')
    txt(ax, .84, 7.73, subtitle, 14, C['muted'])
    line(ax, (.75, .63), (15.25, .63), lw=.8)
    txt(ax, .8, .36, 'LumiMotion  /  阴影驱动的方向光估计  /  2026.10.15—10.17', 9.8, C['muted'])
    txt(ax, 15.2, .36, f'{n:02d} / 03', 11, C['muted'], ha='right')
    return fig, ax


# 同一三维骨架既画作遮挡物，也沿候选方向精确投到平面上。
JOINT = np.array([[-.18, 0, 0], [-.16, 0, .85], [0, 0, 1.3], [.18, 0, .85],
                  [.3, 0, 0], [0, 0, 1.85], [-.53, 0, 1.08], [.53, 0, 1.38]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (5, 7)]


def scene(ax, cx, cy, scale=1, light=(-1.3, -.7, 1.4), ghost=None, rays=False):
    def project(points):
        v = np.array(points)
        return np.stack([cx+scale*(v[..., 0]+.45*v[..., 1]),
                         cy+scale*(-.20*v[..., 0]+.45*v[..., 1]+v[..., 2])], -1)
    floor = project([[-1.2, -1.0, 0], [2.5, -1.0, 0], [2.5, 1.4, 0], [-1.2, 1.4, 0]])
    ax.add_patch(Polygon(floor, facecolor='#EBF0F4', edgecolor='#CAD7E0', lw=1))
    for x in np.arange(-.8, 2.5, .55):
        a, b = project([[x, -1., 0], [x, 1.4, 0]])
        line(ax, a, b, '#D9E2E9', .6)
    for y in np.arange(-.7, 1.4, .55):
        a, b = project([[-1.2, y, 0], [2.5, y, 0]])
        line(ax, a, b, '#D9E2E9', .6)
    def shadow(direction, color, width, alpha):
        l = np.array(direction)
        sh = JOINT-JOINT[:, 2, None]/l[2]*l
        p = project(sh)
        for a, b in EDGES:
            line(ax, p[a], p[b], color, width*scale, alpha=alpha)
        ax.add_patch(Ellipse(p[5], .42*scale, .17*scale, facecolor=color, alpha=alpha))
    if ghost is not None:
        shadow(ghost, C['red'], 10, .42)
    shadow(light, C['shadow'], 12, .55)
    p = project(JOINT)
    for a, b in EDGES:
        line(ax, p[a], p[b], '#6095B3', 15*scale)
    head = project([[0, 0, 2.15]])[0]
    ax.add_patch(Circle(head, .22*scale, color='#83AEC4', zorder=5))
    if rays:
        l = np.array(light)
        for j in [5, 7]:
            v = JOINT[j]
            q = v-v[2]/l[2]*l
            a, b = project([v+.8*l, q])
            arrow(ax, a, b, C['orange'], 1.7, 12, zorder=2)
        v = JOINT[5]
        a, b = project([v, v+.9*l])
        arrow(ax, a, b, C['teal'], 2.8, 17, zorder=7)
    return project


def principle():
    fig, ax = base('从阴影的位置，反推照亮场景的方向光',
                   '已知粗几何与接收平面后：光向决定投影位置，观测阴影反过来约束光向。', 1)
    box(ax, .75, 2.0, 7.0, 5.18, C['pale'])
    txt(ax, 1.03, 6.80, '① 看懂几何关系', 21, weight='bold')
    pr = scene(ax, 2.75, 3.12, 1.12, rays=True)
    txt(ax, 3.4, 6.13, '光向 l（指向光源）', 15, C['teal'])
    txt(ax, 3.18, 5.37, '粗遮挡物', 13, C['blue'])
    txt(ax, 5.05, 2.95, '投影阴影', 14, C['shadow'])
    txt(ax, 1.08, 2.34, '光线沿 −l 传播；平面上的投影随 l 改变', 13.5, C['muted'])
    # 右侧将同一骨架的三种投影与固定观测轮廓直接叠加。
    box(ax, 8.05, 2.0, 7.2, 5.18)
    txt(ax, 8.35, 6.80, '② 试光向，让预测阴影对上观测', 21, weight='bold')
    observed = np.array([[0., 0.], [1.35, -.33], [2.10, -.12], [1.42, .24], [1.75, .65]])
    # 示意轮廓只用于解释选择逻辑，不填入任何真实分数。
    for i, (dy, angle, lab, color) in enumerate([(5.74, -.75, '方向 A：偏转', C['red']),
                                                (4.48, .40, '方向 B：影子过短', C['orange']),
                                                (3.22, 0, '方向 C：位置吻合', C['teal'])]):
        x = 9.8
        box(ax, 8.35, dy-.48, 6.55, 1.07, '#F8FAFC', edge='#E6ECF1', radius=.10)
        ax.add_patch(Ellipse((x+1.0, dy), 2.6, .78, facecolor='#EAF0F4', edgecolor='none'))
        o = observed.copy()
        # 将同一形状旋转或缩短，表示候选灯向的预测投影。
        if i == 0:
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            pred = o@rot.T
        elif i == 1:
            pred = o*.52
        else:
            pred = o.copy()
        for branch in [(0, 1, 2), (0, 3, 4)]:
            ax.plot(x+o[list(branch), 0], dy+o[list(branch), 1]*.30, color='#71818F', lw=9,
                    alpha=.4, solid_capstyle='round')
            ax.plot(x+pred[list(branch), 0], dy+pred[list(branch), 1]*.30, color=color, lw=2.4,
                    solid_capstyle='round')
        ax.add_patch(Circle((x, dy), .09, color=C['ink']))
        txt(ax, 12.45, dy+.06, lab, 15, color)
    line(ax, (8.53, 2.4), (8.94, 2.4), '#71818F', 8, alpha=.4)
    txt(ax, 9.08, 2.4, '观测暗区', 11, C['muted'])
    line(ax, (10.65, 2.4), (11.06, 2.4), C['teal'], 2.4)
    txt(ax, 11.20, 2.4, '候选方向的预测阴影', 11, C['muted'])
    box(ax, .75, .91, 14.50, .74, '#E9F5F3', '#D7EBE7')
    labels = ['训练图像＋粗几何', '模拟多个方向的阴影', '比较位置与覆盖', '输出世界坐标光向 l(t)']
    for x, label in zip([1.02, 4.75, 8.52, 11.92], labels):
        txt(ax, x, 1.28, label, 14.8, C['teal'], weight='bold')
    for x in [4.08, 7.92, 11.37]:
        arrow(ax, (x, 1.28), (x+.4, 1.28), ms=13)
    return fig


def methods():
    fig, ax = base('实际怎样求解：先搜索方向，再细化时间轨迹',
                   '主线采用投影代理＋候选搜索；估光阶段固定几何，不依赖 GT 灯向或 GT 阴影。', 2)
    cards = [(.75, '01', '找到接收阴影的平面'), (5.68, '02', '从图像建立明暗约束'),
             (10.61, '03', '候选光向 → 预测阴影')]
    for x, n, title in cards:
        box(ax, x, 4.03, 4.64, 3.10, C['pale'])
        txt(ax, x+.22, 6.76, n, 19, C['teal'], weight='bold')
        txt(ax, x+.79, 6.76, title, 17, weight='bold')
    # 由平面内点和离面遮挡点可视地表示RANSAC分组。
    rng = np.random.default_rng(1017)
    x0 = .75
    for u, v in rng.uniform([0, 0], [2.6, .55], (60, 2)):
        ax.add_patch(Circle((x0+.7+u, 5.36+v), .028, color=C['teal'], alpha=.7))
    for u, v in rng.uniform([0, 0], [.65, 1], (30, 2)):
        ax.add_patch(Circle((x0+1.68+u, 5.73+v*.6), .025, color=C['blue'], alpha=.75))
    line(ax, (x0+.54, 5.35), (x0+3.83, 5.35), C['teal'], 1.5)
    txt(ax, x0+.25, 4.86, 'Stage1 表面轨迹 → RANSAC＋SVD', 13)
    txt(ax, x0+.25, 4.39, '稳定近地面点：接收；离面点：遮挡', 12.2, C['muted'])
    x0 = 5.68
    vals = [.82, .55, .14, .11, .63, .9, .36, .78]
    for j, v in enumerate(vals):
        box(ax, x0+.42+j*.46, 5.50, .37, .53, str(v), str(v), radius=.035)
    txt(ax, x0+.35, 6.23, '同一表面点的跨时间亮度', 12, C['muted'])
    txt(ax, x0+.25, 5.03, '亮度 ÷ 训练期 90% 分位参考', 14)
    txt(ax, x0+.25, 4.55, '暗 < 0.20    亮 > 0.65    中间忽略', 12.5, C['muted'])
    x0 = 10.61
    center = np.array([x0+2.24, 5.55])
    for ang in np.linspace(.08, np.pi-.08, 13):
        p = center + np.array([1.35*np.cos(ang), .67*np.sin(ang)])
        ax.add_patch(Circle(p, .04, color=C['blue']))
    for ang in [.38, .92, 1.57, 2.10, 2.75]:
        p = center + np.array([1.2*np.cos(ang), .59*np.sin(ang)])
        arrow(ax, center, p, C['blue'], 1, 8)
    txt(ax, x0+.25, 5.02, '512 个上半球候选方向', 14)
    txt(ax, x0+.25, 4.55, '投到 128² 网格，3×3 膨胀补点隙', 12.3, C['muted'])
    for x in [5.43, 10.36]:
        arrow(ax, (x, 5.52), (x+.21, 5.52), C['teal'], 1.8, 12)
    # 评分公式与显式投影公式。
    box(ax, .75, 1.74, 7.0, 1.96)
    txt(ax, 1.0, 3.31, '04  比较观测与预测：两类错误等权', 18, weight='bold')
    txt(ax, 1.02, 2.73, 'L = ½ × 暗点漏检率 ＋ ½ × 亮点误遮挡率', 16, C['teal'])
    txt(ax, 1.02, 2.20, '既解释实际暗区，也避免把本来亮的区域挡住。', 12.8, C['muted'])
    box(ax, 8.05, 1.74, 7.20, 1.96)
    txt(ax, 8.3, 3.31, '05  动态规划选路径，再做局部细化', 18, weight='bold')
    for j, deg in enumerate([113, 118, 125, 121, 128]):
        x, y = 8.80+j*.68, 2.41
        theta = np.deg2rad(deg)
        arrow(ax, (x, y), (x+.48*np.cos(theta), y+.48*np.sin(theta)), C['teal'], 1.8, 11)
        if j < 4:
            line(ax, (x+.07, y-.09), (x+.60, y-.09), '#C8D9DF', 1)
    txt(ax, 12.34, 2.68, '半径 6° / 步长 1°', 13, C['teal'])
    txt(ax, 12.34, 2.23, '保留时刻仅插值', 12.5, C['muted'])
    box(ax, .75, .90, 14.50, .57, '#E9F5F3', '#D7EBE7', radius=.1)
    txt(ax, 1.0, 1.19, r'投影关系：  $q=p-\frac{n\cdot(p-o)}{n\cdot l}\,l$'
        '     ｜     p：遮挡点    q：投影点    (o, n)：接收平面    l：指向光源', 13.5)
    return fig


def evidence(data, state, image_path):
    fig, ax = base('真实数据示例：候选灯向能否解释观测暗区？',
                   '静态 DL · T05 · seed 1015 · 固定中间保留帧 frame_0075；展示同一组接收点的预测。', 3)
    idx = next(i for i, f in enumerate(data['frames']) if f['name'] == 'frame_0075')
    assert not state['train'][idx]
    rgba = Image.open(image_path).convert('RGBA')
    bg = Image.new('RGBA', rgba.size, 'white')
    bg.alpha_composite(rgba)
    # 跟原始缓存相同的分辨率；所有点图采用同一个显示窗口和支持。
    w, h = Image.open(ROOT/'output/1015-06-LH-static-DL-T02表面轨迹初始化/cache/frame_0075_rgb.png').size
    image = bg.convert('RGB')
    mask = state['positive'][idx] | state['negative'][idx]
    uv = data['pixels'][idx][mask].numpy()
    target = state['positive'][idx][mask].bool().numpy()
    values = [None, target, state['predicted']['rgb_baseline'][idx][mask].numpy()>.5,
              state['predicted']['smooth'][idx][mask].numpy()>.5]
    titles = ['原始训练 RGB', '由 RGB 得到的明暗标签', '仅 RGB 拟合光向的投影', '阴影约束估光的投影']
    colors = [C['ink'], C['ink'], C['red'], C['teal']]
    for j, x in enumerate([.75, 4.43, 8.11, 11.79]):
        box(ax, x, 3.50, 3.46, 3.61, C['pale'])
        txt(ax, x+1.73, 6.76, titles[j], 14, colors[j], ha='center', weight='bold')
        sub = fig.add_axes([(x+.13)/16, 3.89/9, 3.20/16, 2.55/9])
        sub.set_facecolor('#E5EBF0')
        if j == 0:
            sub.imshow(image, extent=(0, w, h, 0), interpolation='lanczos')
            sub.set_xlim(0, w); sub.set_ylim(h, 0)
        else:
            # 适度放大地面，并保留真实采样空洞，不做阴影轮廓补画。
            v = values[j]
            cols = np.where(v, C['shadow'], '#FFFFFF')
            sub.scatter(uv[:, 0], uv[:, 1], c=cols, s=10, linewidths=0)
            lo = uv.min(0)-8; hi = uv.max(0)+8
            sub.set_xlim(lo[0], hi[0]); sub.set_ylim(hi[1], lo[1])
        sub.set_aspect('equal'); sub.set_xticks([]); sub.set_yticks([])
        for spine in sub.spines.values():
            spine.set_visible(False)
        if j >= 2:
            value = values[j]
            balanced = .5*(~value[target]).mean()+.5*value[~target].mean()
            txt(ax, x+1.73, 3.70, f'本帧明暗均衡错误率：{balanced:.1%}', 11.3, colors[j], ha='center')
    txt(ax, .87, 3.12, '点图：深色＝暗／预测阴影；白色＝亮／预测无阴影；灰底＝无观测。后三图放大同一地面范围。',
        12.2, C['muted'])
    box(ax, .75, 1.43, 9.21, 1.34, '#E9F5F3', '#D7EBE7')
    txt(ax, 1.03, 2.37, '早期 T05 的可复核结果', 15, C['teal'], weight='bold')
    txt(ax, 1.03, 1.88, '静态：40.68° → 4.81°', 23, C['teal'], weight='bold')
    txt(ax, 5.91, 1.90, '15 个 Stage2 保留时间的平均光向误差', 11.2, C['muted'])
    box(ax, 10.22, 1.43, 5.03, 1.34, C['pale'])
    txt(ax, 10.48, 2.36, '10 月 17 日的进一步改进', 14, weight='bold')
    txt(ax, 10.48, 1.90, '完整遮挡几何＋局部细化＋时间约束', 12.7, C['teal'])
    txt(ax, .84, 1.00, '边界：这些是投影代理；GT 只做离线角度评测。Stage1 曾见过上述保留时间，不能视为全管线未见；几何目标仍未完成。',
        10.5, C['muted'])
    errors = {name: float(.5*(~value[target]).mean()+.5*value[~target].mean())
              for name, value in [('RGB基线', values[2]), ('阴影估光', values[3])]}
    return fig, dict(frame='frame_0075', index=idx, samples=int(mask.sum()),
                     positive=int(target.sum()), negative=int((~target).sum()), 均衡错误率=errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--font', type=Path, required=True, help='中文字体，例如 NotoSansCJKsc-Regular.otf')
    p.add_argument('--output', type=Path, default=ROOT/'LIHAN/export')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fm.fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family': fm.FontProperties(fname=str(args.font)).get_name(),
                         'svg.fonttype': 'path', 'pdf.fonttype': 42, 'axes.unicode_minus': False})
    cache = ROOT/'output/1015-06-LH-static-DL-T02表面轨迹初始化/cache/tracks.pt'
    ckpt = ROOT/'output/1015-14-LH-static-DL-T05复核/seed_1015/checkpoint.pt'
    data = torch.load(cache, map_location='cpu', weights_only=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == state['cache_sha256']
    original = Path(data['config']['source'])/'images/frame_0075.png'
    figs = [principle(), methods()]
    third, sample = evidence(data, state, original)
    figs.append(third)
    names = ['1017-01-阴影反推方向光原理', '1017-02-阴影估光方法流程', '1017-03-真实观测与阴影投影对比']
    files = []
    pdf = args.output/'1017-04-阴影反推方向光演示合集.pdf'
    with PdfPages(pdf) as pages:
        for fig, name in zip(figs, names):
            for ext in ['png', 'svg', 'pdf']:
                path = args.output/f'{name}.{ext}'
                fig.savefig(path, dpi=240, facecolor='white')
                files.append(str(path.relative_to(ROOT)))
            pages.savefig(fig)
            plt.close(fig)
    files.append(str(pdf.relative_to(ROOT)))
    manifest = dict(服务器='ushas', 环境='lumimotion-ushas', 说明='只读实验的演示配图，无训练、无GT资产读取',
                    真实示例=sample, 输入文件=[dict(路径=str(x), SHA256=hashlib.sha256(x.read_bytes()).hexdigest())
                                             for x in [cache, ckpt, original, args.font]],
                    输出文件=files, PNG尺寸=[3840, 2160], PDF页数=3)
    (args.output/'1017-05-导出来源与文件清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
