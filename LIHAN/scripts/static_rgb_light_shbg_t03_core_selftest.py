"""T03 核心合成数据自检；不读取真实数据或 GT。"""
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t03_core import template, backlight, calibrate, R


def main():
    rng = np.random.default_rng(3)
    h = rng.uniform(.1, .4, (150, 40)); H = template(h)['H']
    assert H.min() >= 0 and H.max() <= .1 + 1e-15
    assert np.max(template(np.ones((150, 40)))['H']) == 0
    anchor = np.tile([1., 0., 0.], (150, 1))
    n = np.tile([-1., 0., 0.], (150, 40, 1)); valid = np.ones((150, 40), bool)
    nn, A = backlight(n, anchor, valid)
    assert A.all() and np.cos(R) > np.sin(R)
    y = .2 + .4*H; w = np.ones_like(H); ids = np.arange(40)
    rec = calibrate(H, y, w, A, ids, 1e-4)
    assert rec['identified']
    assert abs(rec['records']['all']['gamma']-calibrate(H, y+3, w, A, ids, 1e-4)['records']['all']['gamma']) < 1e-12
    assert not calibrate(np.zeros_like(H), y, w, A, ids, 1e-4)['identified']
    assert not calibrate(H, -y, w, A, ids, 1e-4)['identified']
    n2 = np.tile([1., 0., 0.], (150, 40, 1)); n2[:75] = nn[:75]
    _, A2 = backlight(n2, anchor, valid)
    injection = np.maximum(np.einsum('tgc,tc->tg', n2, anchor), 0.)
    a = calibrate(H, y, w, A2, ids, 1e-4)['records']['all']['gamma']
    b = calibrate(H, y+injection, w, A2, ids, 1e-4)['records']['all']['gamma']
    assert abs(a-b) <= 1e-12
    print('T03 核心合成数据自检通过；未读取真实数据或 GT。')


if __name__ == '__main__': main()
