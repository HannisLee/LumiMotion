"""固定预测表面轨迹的主光／共享材质拟合；独立于 GT 评测。"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from utils.dl_selfcal import SurfaceLightFit, observation_loss, initialize_directions


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--initialization', choices=['random', 'rgb', 'sh_aux'], default='rgb')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--seed', type=int, default=1015)
    p.add_argument('--smoothness', type=float, default=.001)
    return p


def fit(args, oracle_directions=None):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists() or (out/'checkpoint.pt').exists():
        raise FileExistsError('禁止覆盖已有结果。')
    torch.manual_seed(args.seed)
    data = torch.load(args.cache, map_location='cpu', weights_only=False)
    times, normals, rgb, saturated, valid = [data[k].cuda() for k in ['times','normals','rgb','saturated','valid']]
    train = torch.arange(len(times), device='cuda') % 9 != 4
    # 仅训练时间至少三次有效且有一次未饱和支持的表面进入拟合／验证。
    supported = (valid[train].sum(0) >= 3) & ((valid & ~saturated.any(-1))[train].sum(0) >= 1)
    valid = valid & supported[None]
    if (valid[train].sum(1) < 16).any() or valid[~train].sum() == 0:
        raise ValueError('有效表面支持不足。')
    init_target = rgb if args.initialization != 'sh_aux' else .75*rgb+.25*data['sh_rgb'].cuda()
    started = time.monotonic()
    initial = (initialize_directions(normals[train], init_target[train], valid[train], saturated[train], args.initialization, args.seed)
               if oracle_directions is None else oracle_directions.to(normals)[train])
    initialization_seconds = time.monotonic()-started
    model = SurfaceLightFit(times[train], initial, normals.shape[1]).cuda()
    if oracle_directions is not None:
        model.raw_directions.requires_grad_(False)
    optimizer = torch.optim.Adam([
        {'params': [model.raw_directions], 'lr': .015},
        {'params': [model.log_energy], 'lr': .01},
        {'params': [model.raw_albedo], 'lr': .02}])

    @torch.no_grad()
    def metrics():
        prediction = model(times, normals)
        result = {}
        for name, mask in [('fit', train), ('held_time', ~train)]:
            result[name+'_robust'] = float(observation_loss(prediction[mask], rgb[mask], saturated[mask], valid[mask]))
            unsat = valid[mask][..., None] & ~saturated[mask]
            result[name+'_unsaturated_mse'] = float(((prediction[mask]-rgb[mask]).square()*unsat).sum()/unsat.sum().clamp_min(1))
            result[name+'_observations'] = int(valid[mask].sum())
        return result

    before = metrics()
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = observation_loss(model(times[train], normals[train]), rgb[train], saturated[train], valid[train])
        objective = loss + args.smoothness*model.regularizer()
        if not torch.isfinite(objective):
            raise RuntimeError('非有限目标。')
        objective.backward()
        optimizer.step()
        if (step+1) % 100 == 0 or step == 0:
            record = dict(update=step+1, loss=float(loss), **metrics())
            history.append(record)
            print(json.dumps(record), flush=True)
    after = metrics()
    directions, energy = model.lighting(times)
    final_directions, final_energy = model.lighting(model.times)
    # 保存原始参数和优化器、数据身份；验证重新加载后的全部时间预测。
    checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), args=vars(args),
                      train_mask=train.cpu(), supported=supported.cpu(), updates=args.steps,
                      cache_sha256=hashlib.sha256(Path(args.cache).read_bytes()).hexdigest())
    torch.save(checkpoint, out/'checkpoint.pt')
    restored = SurfaceLightFit(times[train], initial, normals.shape[1]).cuda()
    restored.load_state_dict(torch.load(out/'checkpoint.pt', weights_only=False)['model'])
    with torch.no_grad():
        restore_error = float((restored(times, normals)-model(times, normals)).abs().max())
    result = dict(config=vars(args), before=before, after=after, actual_updates=args.steps,
                  initialization_seconds=initialization_seconds, total_seconds=time.monotonic()-started,
                  supported_patches=int(supported.sum()), total_patches=len(supported),
                  train_indices=torch.where(train)[0].tolist(), held_indices=torch.where(~train)[0].tolist(),
                  times=times.tolist(), frames=[f['name'] for f in data['frames']],
                  directions=directions.detach().tolist(), relative_energy=energy.detach().tolist(),
                  train_mean_relative_energy=float(final_energy.mean()), restore_max_error=restore_error,
                  history=history, cache_sha256=checkpoint['cache_sha256'],
                  engineering_status='PASS' if restore_error == 0 else 'FAILED',
                  gt_direction_reference=oracle_directions is not None,
                  scientific_status='未验收：无GT光向评测，未更新几何',
                  heldout_scope='仅Stage2保留时间，默认Stage1已经看过这些RGB')
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(after), flush=True)


if __name__ == '__main__':
    fit(argument_parser().parse_args())
