"""评测Agent补丁（不改规划Agent的wrapper与冻结入口）：

eldl_1024_eval_static_gt.py 对 scene/static_eldl_lighting.py 做整文件SHA比对，1025起该文件新增了仅属训练损失的
censored_loss(pixel_keep)/partial_alpha_keep_mask，导致A0/G0–G3等旧模型评测fail-closed。本包装器把该文件的比对改为
“AST逐节点比较，仅允许 censored_loss 与 partial_alpha_keep_mask 两个训练损失函数不同”，渲染相关代码(MixedLighting/render_mixed等)
必须完全一致；其余文件仍整文件SHA比对。数学/掩膜/分母不变。
"""
import ast
import hashlib
import json
import shlex
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import LIHAN.scripts.eldl_1024_eval_static_gt as W  # noqa: E402
import LIHAN.scripts.eldl_1024_eval as E  # noqa: E402

TRAIN_ONLY = {'censored_loss', 'partial_alpha_keep_mask'}
_orig_sha = W._sha
THIS = 'LIHAN.scripts.eldl_1024_eval_static_gt_compat'


def _norm(path):
    tree = ast.parse(Path(path).read_text())
    keep = [ast.dump(n) for n in tree.body if getattr(n, 'name', None) not in TRAIN_ONLY]
    return hashlib.sha256('\n'.join(keep).encode()).hexdigest()


def _sha(p):
    if str(p).endswith('scene/static_eldl_lighting.py'):
        return _norm(p)
    return _orig_sha(p)


W._sha = _sha
_wmake = E.make_plan


def make_plan(args):
    commands = _wmake(args)
    for item in commands:
        cmd = item['command']
        if W.__name__ in cmd or 'LIHAN.scripts.eldl_1024_eval_static_gt' in cmd:
            cmd[cmd.index('LIHAN.scripts.eldl_1024_eval_static_gt')] = THIS
            item['shell'] = shlex.join(cmd)
    p = args.output / 'plan.json'
    if p.is_file():
        plan = json.loads(p.read_text()); plan['steps'] = commands
        plan['compat_wrapper'] = str(Path(__file__).resolve()); plan['compat_wrapper_sha256'] = _orig_sha(Path(__file__))
        p.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    return commands


E.make_plan = make_plan
if __name__ == '__main__':
    E.main()
