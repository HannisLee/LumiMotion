"""T01(1107-03) 合成数据自检：不读任何真实 GT/EL 数据，不需要 GPU。

  python LIHAN/scripts/pipeline1107_selftest.py --work <空目录>
覆盖：E4->方向 JSON(含失败用例)、相机一致性、snapshot 补丁锚点/门禁、E1 与 E2 命令仅差 --wo_indirect、
metrics 钩子、评测探针合成、泄漏监测(旗标可触发/不触发)、资源汇总、曲线与并排图、预注册冻结。
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = []


def check(name, condition):
    RESULTS.append((name, bool(condition)))
    print(('PASS ' if condition else 'FAIL ') + name, flush=True)


def raises(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as error:
        return type(error).__name__
    return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_e4(root, mutate=None):
    root.mkdir(parents=True)
    (root / 'fitting').mkdir()
    rng = np.random.RandomState(0)
    v = rng.normal(size=(150, 3))
    v[:, 2] = np.abs(v[:, 2]) + .5
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    frames = np.arange(1, 151)
    if mutate == 'nonunit':
        v = v * 1.01
    if mutate == 'frames':
        frames = frames[::-1].copy()
    np.savez(root / 'fitting/predictions.npz', frames=frames, E4=v)
    cfg = dict(selected='a1e+00_e1e+01', direction_convention='surface_to_light', coordinate_system='world', gt_read=False)
    (root / 'fitting/method_config.json').write_text(json.dumps(cfg))
    digest = sha(root / 'fitting/predictions.npz')
    if mutate == 'sha':
        digest = '0' * 64
    (root / 'freeze_manifest.json').write_text(json.dumps(dict(
        prediction_sha256=digest, gt_read=False, direction_convention='surface_to_light', coordinate_system='world',
        selected='a1e+00_e1e+01')))
    return v


def test_e4(work):
    from LIHAN.scripts import pipeline1107_e4_directions as e4
    v = fake_e4(work / 'e4ok')
    record = e4.convert(work / 'dirs', work / 'e4ok')
    check('E4 转换成功且三份 JSON 帧数为 117/135/132', record['splits'] == dict(train=117, dev=135, test=132))
    check('E4 JSON gt_read=false 且登记 predictions SHA', record['gt_read'] is False and
          json.loads((work / 'dirs/e4_test_directions.json').read_text())['predictions_sha256'] == record['predictions_sha256'])
    for name, mutate in [('非单位范数', 'nonunit'), ('帧序错误', 'frames'), ('SHA 与冻结清单不符', 'sha')]:
        fake_e4(work / f'e4_{mutate}', mutate)
        check(f'E4 断言拦截：{name}', raises(e4.convert, work / f'dirs_{mutate}', work / f'e4_{mutate}') is not None)
    # 往返：原生 GTDirectionalLights 读取
    from scene.pbr_directional import GTDirectionalLights
    table = GTDirectionalLights(work / 'dirs/e4_test_directions.json')
    import torch
    check('pbr_directional 读取全部 TEST 方向帧', len(table.frames) == 132)
    d, irr = table.frames['0010']
    check('往返方向与 E4 一致(<=1e-6)', float((d[0].double() - torch.tensor(v[9])).abs().max()) < 1e-6 and float((irr - 1).abs().max()) == 0)
    cam = types.SimpleNamespace(image_name_train_light='image_0070')
    check('缺帧必须报错(DEV 帧不在 train JSON)', raises(GTDirectionalLights(work / 'dirs/e4_train_directions.json').for_camera, cam, 1., 'cpu') is not None)
    ids = {int(k) for k in json.loads((work / 'dirs/e4_train_directions.json').read_text())['frames']}
    check('train JSON = FIT117，不含 10 的倍数与 61-79', ids == set(e4.fit_frames()) and not any(i % 10 == 0 or 61 <= i <= 79 for i in ids))
    # 相机一致性
    def transforms(path, frames, shift=0.):
        path.write_text(json.dumps(dict(frames=[dict(source_frame=i, transform_matrix=(np.eye(4) * 1 + shift * i).tolist()) for i in frames])))
    transforms(work / 'ref.json', range(1, 151))
    transforms(work / 'ok.json', e4.test_frames())
    transforms(work / 'bad.json', e4.test_frames(), 1e-3)
    check('A10 相机一致通过', e4.check_camera_consistency(work / 'ref.json', {'ok': work / 'ok.json'})['ok']['max_abs_diff'] == 0)
    check('A10 相机偏差拦截', raises(e4.check_camera_consistency, work / 'ref.json', {'bad': work / 'bad.json'}) is not None)


def load_text_module(name, text):
    module = types.ModuleType(name)
    exec(compile(text, name, 'exec'), module.__dict__)
    return module


def test_patches(work):
    from LIHAN.scripts import pipeline1107_prepare as prep
    train = (ROOT / 'scripts/train_stage2.py').read_text()
    patched = prep.patch_train_stage2(train)
    compile(patched, 'train_stage2_patched', 'exec')
    check('train_stage2 补丁后可编译，五处锚点各生效一次', all(patched.count(s) == 1 for s in [
        'attach(self)', 'self.testing_iterations = []', 'self._pipeline_geometry_terms = geometry_terms', 'log_step(self, loss, loss_result, Ll1)']))
    check('原仓库 train_stage2.py 未被修改(补丁只作用于文本)', 'attach(self)' not in train)
    bad = raises(prep.patch_train_stage2, train.replace('self.iters_only_diffuse = 2000', 'self.x = 1'))
    check('锚点缺失时补丁报错而非静默', bad == 'AssertionError')
    geometry = prep.patch_pbr_geometry((ROOT / 'scene/pbr_geometry.py').read_text())
    module = load_text_module('pbr_geometry_patched', geometry)
    base = dict(pbr_optimize_geometry=True, render_mode='original_ir', pbr_geometry_xyz_lr=2e-5, pbr_geometry_rotation_lr=1e-3,
                pbr_geometry_max_displacement=.08, pbr_plane_band=.08, lambda_pbr_normal_depth=.05, lambda_pbr_plane_normal=.5,
                lambda_pbr_plane_position=.1, lambda_pbr_xyz_anchor=1e-3, lambda_pbr_rotation_anchor=1e-3, lambda_pbr_silhouette=.1,
                pbr_geometry_warmup=2000, pbr_geometry_max_angle=60., iterations=55000, load_iter=35000)
    ns = lambda **kw: types.SimpleNamespace(**{**base, **kw})
    check('门禁：envmap+混合方向光 放行', raises(module.validate_pbr_geometry, ns(pbr_light_mode='envmap', pbr_mixed_lights_path='x.json')) is None)
    check('门禁：envmap 无混合光 仍拒绝', raises(module.validate_pbr_geometry, ns(pbr_light_mode='envmap', pbr_mixed_lights_path='')) == 'ValueError')
    check('门禁：gt_directional 行为不变', raises(module.validate_pbr_geometry, ns(pbr_light_mode='gt_directional')) is None)
    check('门禁：render_mode 非 original_ir 仍拒绝', raises(module.validate_pbr_geometry, ns(pbr_light_mode='envmap', pbr_mixed_lights_path='x', render_mode='photometric_lambertian')) == 'ValueError')
    check('冒烟A 24 循环默认 warmup 2000 被原校验拒绝(故冒烟 A 默认 8)', raises(module.validate_pbr_geometry, ns(pbr_light_mode='envmap', pbr_mixed_lights_path='x', iterations=35024)) == 'ValueError')
    check('冒烟A warmup=8 通过', raises(module.validate_pbr_geometry, ns(pbr_light_mode='envmap', pbr_mixed_lights_path='x', iterations=35024, pbr_geometry_warmup=8)) is None)


def test_commands(work):
    from LIHAN.scripts import pipeline1107_prepare as prep
    ctl = work / 'ctl'
    cmds = {a: prep.training_command(ctl, a, work / 'out' / a, 20000, None, True, None)[0] for a in prep.ARMS}
    strip = lambda c: [x for x in c if not x.startswith(str(work))]
    e1, e2 = strip(cmds['E1']), strip(cmds['E2'])
    check('E2 命令 = E1 命令 + 仅 --wo_indirect', e2 == e1 + ['--wo_indirect'])
    check('E0 无混合光、无几何开关', '--pbr_mixed_lights_path' not in cmds['E0'] and '--pbr_optimize_geometry' not in cmds['E0'])
    check('E1/E2 含 N1 两开关与 E4 train JSON', all(x in cmds['E1'] for x in ['--pbr_optimize_geometry', '--pbr_freeze_opacity']) and
          str(ctl / 'e4_train_directions.json') in cmds['E1'])
    check('保存点为 39000/47000/55000', cmds['E1'][cmds['E1'].index('--save_iterations') + 1: cmds['E1'].index('--save_iterations') + 4] == ['39000', '47000', '55000'])
    n0 = strip(prep.training_command(ctl, 'E2', work / 'o', 20000, None, False, None)[0])
    check('N0 回退去掉两个几何开关且保留 --wo_indirect', '--pbr_optimize_geometry' not in n0 and '--wo_indirect' in n0)
    a, _, end = prep.training_command(ctl, 'E1', work / 'o', 20000, 'A', True, 8)
    check('冒烟A：24 循环、warmup 8', end == 35024 and a[a.index('--pbr_geometry_warmup') + 1] == '8')
    b = prep.training_command(ctl, 'E1', work / 'o', 20000, 'B', True, None)[0]
    check('冒烟B：2040 循环，保存点含 37000 与 37040', b[b.index('--iterations') + 1] == '37040' and '37000' in b)
    # 预注册
    ctl.mkdir()
    (ctl / 'manifest.json').write_text(json.dumps(dict(snapshot={'a.py': 'x'}, e4=dict(files={'e4_train_directions.json': 'y'}))))
    ns = types.SimpleNamespace(control=ctl, no_geometry=False)
    prep.prereg(ns)
    check('预注册文件生成且再次冻结被拒绝', (ctl / 'preregistration.json').exists() and raises(prep.prereg, ns) == 'FileExistsError')


def test_metrics_and_probe(work):
    import torch
    from LIHAN.scripts import pipeline1107_metrics as metrics
    from LIHAN.scripts import pipeline1107_evaluate as ev
    model = work / 'model'
    model.mkdir()

    class Env:
        base = torch.zeros(4, 8, 3)
        activation = staticmethod(torch.exp)
        mixed_log_intensity = torch.tensor(0.)
    class Time:
        def elapsed_time(self, other): return 12.5
    trainer = types.SimpleNamespace(iteration=100, args=types.SimpleNamespace(model_path=str(model)), env_light=Env(),
                                    iter_start=Time(), iter_end=Time(), _pipeline_geometry_terms={'pbr_silhouette': torch.tensor(.1)},
                                    gaussians=types.SimpleNamespace(get_xyz=torch.zeros(7, 3)))
    result = types.SimpleNamespace(terms={'rgb_l1': torch.tensor(.2)})
    metrics.log_step(trainer, torch.tensor(1.5), result, torch.tensor(.2))
    trainer.iteration = 150
    metrics.log_step(trainer, torch.tensor(1.5), result, torch.tensor(.2))
    rows = [json.loads(x) for x in (model / 'metrics.jsonl').read_text().splitlines()]
    check('metrics 钩子只在 100 的倍数写一行，字段齐全', len(rows) == 1 and rows[0]['iteration'] == 100 and rows[0]['env_mean'] == 1.0 and
          rows[0]['sun_intensity'] == 1.0 and rows[0]['geo_pbr_silhouette'] == torch.tensor(.1).item() and rows[0]['gaussians'] == 7)
    trainer.env_light = None
    trainer.iteration = 200
    trainer.args.model_path = str(work / 'missing/dir')
    metrics.log_step(trainer, torch.tensor(1.), result, torch.tensor(.2))
    check('metrics 钩子异常不抛出(不中断训练)', True)
    check('独立 PSNR 复算与 torch 口径一致', abs(ev.independent_psnr(np.full((3, 4, 4), .5), np.full((3, 4, 4), .4)) - ev.psnr_from_mse(.01)) < 1e-9)
    # 探针
    n = 6
    gen = torch.Generator().manual_seed(1)
    def original(*args, **kwargs):
        has_sun = kwargs.get('directional_lights') is not None
        base = .1 if not has_sun else .3
        out = dict(diffuse=torch.full((n, 3), base), specular=torch.full((n, 3), .01), light_direct=torch.full((n, 3), base),
                   light=torch.full((n, 3), base), light_blocked=torch.zeros(n, 3),
                   light_indirect=torch.zeros(n, 3) if (has_sun or args[6].wo_indirect) else torch.full((n, 3), .05),
                   visibility=torch.full((n, 1), .4 if not has_sun else .9))
        return out
    probe = ev.ComponentProbe()
    holder = types.SimpleNamespace(rendering_equation=original)
    ev.install_probe(holder, original, probe)
    normals = torch.tensor([[0., 0., 1.]]).repeat(n, 1)
    pipe = types.SimpleNamespace(wo_indirect=False)
    lights = (torch.tensor([[0., 0., 1.]]), torch.ones(1, 3))
    args = (torch.zeros(n, 3), torch.zeros(n, 1), normals, torch.zeros(n, 3), torch.zeros(n, 3), None, pipe)
    out = holder.rendering_equation(*args, envlight=object(), directional_lights=lights)
    parts = probe.collect()
    check('探针合成 = 环境 + SUN(diffuse 0.1+0.3)', torch.allclose(out['diffuse'], torch.full((n, 3), .4)))
    check('探针记录环境/SUN/间接/可见度/s', torch.allclose(parts['env'], torch.full((n, 3), .11)) and torch.allclose(parts['sun'], torch.full((n, 3), .31)) and
          torch.allclose(parts['ind'], torch.full((n, 3), .05)) and torch.allclose(parts['s'], torch.full((n,), .9)) and torch.allclose(parts['env_vis'], torch.full((n,), .4)))
    pipe2 = types.SimpleNamespace(wo_indirect=True)
    probe.reset()
    holder.rendering_equation(*(args[:6] + (pipe2,)), envlight=object(), directional_lights=lights)
    check('E2(wo_indirect)：间接分量记录为 0', float(probe.collect()['ind'].abs().max()) == 0.)
    probe.reset()
    holder.rendering_equation(*args, envlight=object(), directional_lights=None)
    check('E0 路径(无方向光)只记录环境分支', 'sun' not in probe.collect())
    image = ev.scatter(torch.ones(2, 3), torch.tensor([[True, False], [False, True]]), (2, 2))
    check('scatter 还原前景像素', image.shape == (3, 2, 2) and float(image[:, 0, 1].sum()) == 0 and float(image[:, 1, 1].sum()) == 3)


def make_dump(root, frames, rng, albedo_low_v_scale=1., resid_shadow=0., n=2000):
    (root / 'leak').mkdir(parents=True)
    for f in range(frames):
        v = rng.uniform(0, 1, n).astype(np.float32)
        s = (rng.uniform(0, 1, n) * (v >= .5) * (rng.uniform(size=n) > .2)).astype(np.float32)  # 阴影像素多为低V
        lum = rng.uniform(.2, .5, n)
        lum = np.where(v < .5, lum * albedo_low_v_scale, lum)
        albedo = np.stack([lum] * 3, 1).astype(np.float32)
        resid = rng.normal(0, .02, (n, 3)).astype(np.float32)
        resid[s < .05] += resid_shadow
        np.savez(root / 'leak' / f'{f + 1:04d}.npz', fg=np.ones((n // 50, 50), bool), env_vis=v, sun_vis=np.ones(n, np.float32), s=s,
                 env=np.ones((n, 3), np.float32), sun=np.ones((n, 3), np.float32), ind=np.ones((n, 3), np.float32) * .1,
                 albedo_lin=albedo, resid_lin=resid, normal=np.tile(np.array([[0, 0, 1.]], np.float32), (n, 1)))


def write_ply(path, dc, rot, xyz):
    from plyfile import PlyData, PlyElement
    n = len(dc)
    dtype = [(k, 'f4') for k in ['x', 'y', 'z'] + [f'albedo_dc_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]]
    data = np.empty(n, dtype=dtype)
    for i, k in enumerate('xyz'):
        data[k] = xyz[:, i]
    for i in range(3):
        data[f'albedo_dc_{i}'] = dc[:, i]
    for i in range(4):
        data[f'rot_{i}'] = rot[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, 'vertex')]).write(str(path))


def write_model(root, iteration, dc, rot, xyz, env_value, intensity, test_frames, ind):
    import torch
    write_ply(root / f'point_cloud/iteration_{iteration}/point_cloud.ply', dc, rot, xyz)
    folder = root / f'envmap/iteration_{iteration}'
    folder.mkdir(parents=True)
    torch.save({'state_dict': {'base': torch.log(torch.full((16, 32, 3), env_value))}, 'activation': 'exp'}, folder / 'envmap.pth')
    (folder / 'mixed.json').write_text(json.dumps({'effective_intensity': intensity}))
    return root


def test_leakage(work):
    from LIHAN.scripts import pipeline1107_leakage as lk
    rng = np.random.RandomState(3)
    n = 500
    dc = rng.normal(0, .5, (n, 3))
    rot = np.tile(np.array([[1., 0, 0, 0]]), (n, 1))
    xyz = rng.normal(size=(n, 3))
    q = rot + rng.normal(0, .15, rot.shape)
    for name, kw, expect in [('trigger', dict(scale=1.5, shadow=-.03, envv=1.5), True), ('quiet', dict(scale=1., shadow=0., envv=1.0), False)]:
        base = work / f'leak_{name}'
        write_model(base / 'b00', 35000, dc, rot, xyz, 1., None, None, None)
        e1 = write_model(base / 'e1', 55000, dc, q, xyz, 1., 1., None, None)
        e2 = write_model(base / 'e2', 55000, dc * (1.2 if kw['scale'] > 1 else 1.), q, xyz, kw['envv'], 1., None, None)
        for arm, scale, shadow in (('e1', 1., 0.), ('e2', kw['scale'], kw['shadow'])):
            for split, frames in (('fit', 4), ('dev', 2)):
                make_dump(base / f'{arm}_{split}', frames, np.random.RandomState(7), scale, shadow)  # 同种子：quiet 时两模型像素完全一致
            (base / f'{arm}_test').mkdir()
            ind_max = 0.1 if arm == 'e1' else 0.
            (base / f'{arm}_test/metrics.json').write_text(json.dumps(dict(frames=[
                dict(frame='0010', energy_env=.1, energy_sun=.2, energy_indirect=ind_max / 2, energy_indirect_max=ind_max)])))
        args = types.SimpleNamespace(output=base / 'out', iteration=55000, b00_iteration=35000, b00=base / 'b00', e1=e1, e2=e2,
                                     e1_fit=base / 'e1_fit', e1_dev=base / 'e1_dev', e2_fit=base / 'e2_fit', e2_dev=base / 'e2_dev',
                                     e1_test=base / 'e1_test', e2_test=base / 'e2_test')
        res = lk.run(args)
        check(f'泄漏监测[{name}] M6 正确性通过(E2间接=0,E1>0)', res['m6']['engineering_error'] is False)
        check(f'泄漏监测[{name}] 输出 json 与两张图', all((base / 'out' / f).exists() for f in ['leakage.json', 'm3_residual_hist.png', 'm1_albedo_by_visibility.png']))
        if expect:
            check('泄漏监测[trigger] M1/M3/M4 旗标触发', res['flags']['m1'] is True and res['flags']['m3'] is True and res['flags']['m4'] is True)
        else:
            check('泄漏监测[quiet] M1-M4 旗标均不触发', not any(res['flags'][k] for k in ('m1', 'm2', 'm3', 'm4')))
        check(f'泄漏监测[{name}] M5 逐Gaussian法线变化被计算', res['m5']['E1'] is not None and res['m5']['E1']['p90'] > 0)
    broken = work / 'leak_trigger'
    (broken / 'e2_test/metrics.json').write_text(json.dumps(dict(frames=[dict(frame='0010', energy_indirect_max=.2)])))
    check('M6：E2 间接非零判工程错误', lk.m6(broken / 'e1_test/metrics.json', broken / 'e2_test/metrics.json')['engineering_error'] is True)
    check('Gaussian clamp 贴边比例随阈值', lk.gaussian_albedo(np.array([[(0.02 - .5) / lk.C0, (.99 - .5) / lk.C0, 0.]]))[1] == dict(low=1 / 3, high=1 / 3))


def test_resources_figures(work):
    from LIHAN.scripts import pipeline1107_resources as res
    from LIHAN.scripts import pipeline1107_figures as fig
    for arm in ('E1', 'E2'):
        run = work / f'run_{arm}'
        (run / 'model_mlp').mkdir(parents=True)
        (run / 'state.json').write_text(json.dumps(dict(arm=arm, status='TRAIN_COMPLETED', started=0., finished=3600., steps=20000, gpu=0)))
        (run / 'model_mlp/metrics.jsonl').write_text('\n'.join(json.dumps(dict(
            iteration=35000 + 100 * i, total_loss=1 / (i + 1), rgb_l1=.1 / (i + 1), step_ms=100. + i, cuda_max_allocated_mb=5000. + i,
            env_mean=1., sun_intensity=1.)) for i in range(5)) + '\n')
    out = res.summarize_run(work / 'run_E1')
    check('资源汇总：墙钟 1h、峰值显存、单步毫秒', out['wall_seconds'] == 3600 and out['torch_max_allocated_mb'] == 5004. and out['step_ms']['median'] == 102.)
    csvp = work / 'smi.csv'
    csvp.write_text('time,pid_memory_mb,processes_on_node,gpu_memory_used_mb_json,gpu_names_json\n1,100,2,"{""0"": 300}","{""0"": ""A40""}"\n2,,3,"{""0"": 500}","{""0"": ""A40""}"\n')
    check('nvidia-smi 采样汇总(PID 缺失行不崩溃)', res.read_samples(csvp)['node_peak_mb'] == 500 and res.read_samples(csvp)['pid_peak_mb'] == 100)
    fig.curves(types.SimpleNamespace(run=[f'E1={work / "run_E1"}', f'E2={work / "run_E2"}'], output=work / 'curves.png'))
    check('曲线叠加图生成', (work / 'curves.png').stat().st_size > 1000)
    rng = np.random.RandomState(0)
    for arm in ('E0', 'E1', 'E2'):
        (work / f'ev_{arm}/arrays').mkdir(parents=True)
        for frame in ('0010', '0020'):
            gt = rng.uniform(size=(3, 16, 16)).astype(np.float32) if arm == 'E0' else np.load(work / 'ev_E0/arrays' / f'{frame}.npz')['target']
            if arm == 'E0':
                np.random.RandomState(int(frame)).uniform(size=1)
            np.savez(work / f'ev_{arm}/arrays/{frame}.npz', rgb=np.clip(gt + rng.normal(0, .05, gt.shape), 0, 1).astype(np.float32), target=gt, mask=np.ones((1, 16, 16), np.float32))
    fig.sidebyside(types.SimpleNamespace(eval=[f'{a}={work / ("ev_" + a)}' for a in ('E0', 'E1', 'E2')], output=work / 'sbs', error_max=None))
    check('并排图：逐帧 + 总览生成', (work / 'sbs/frame_0010.png').exists() and (work / 'sbs/overview.png').exists())
    np.savez(work / 'ev_E2/arrays/0010.npz', rgb=np.zeros((3, 16, 16), np.float32), target=np.zeros((3, 16, 16), np.float32), mask=np.ones((1, 16, 16), np.float32))
    check('并排图：GT 不一致被断言拦截', raises(fig.assert_same_targets, {a: work / f'ev_{a}' for a in ('E0', 'E1', 'E2')}) == 'AssertionError')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    for test in (test_e4, test_patches, test_commands, test_metrics_and_probe, test_leakage, test_resources_figures):
        try:
            test(a.work)
        except Exception as error:
            import traceback
            traceback.print_exc()
            check(f'{test.__name__} 无异常', False)
    failed = [n for n, ok in RESULTS if not ok]
    print(f'\n{len(RESULTS) - len(failed)}/{len(RESULTS)} PASS')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
