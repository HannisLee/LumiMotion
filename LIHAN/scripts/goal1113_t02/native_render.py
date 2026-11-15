"""T02-native 渲染+四类图：进程内替换 16×32 网格后复用冻结 render.run，再生成 native_figures。
cwd=仓库根：python -u -m LIHAN.scripts.goal1113_t02.native_render --output ROOT"""
import argparse, json, os, shlex, socket, sys
from datetime import datetime, timezone
from pathlib import Path
from LIHAN.scripts.goal1113 import physics as P
from LIHAN.scripts.goal1113_t02 import native_run as NR

ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); ap.add_argument('--device', default='cuda')
a = ap.parse_args(); out = Path(a.output)
P.grid = NR.grid
from LIHAN.scripts.goal1113 import render
from LIHAN.scripts.goal1113_t02 import native_figures as NF
started = datetime.now(timezone.utc).isoformat(); code = 1
try:
    render.run(out, a.device); NF.run(out, a.device); code = 0
finally:
    r = dict(schema_version=1, mode='render', command=shlex.join(sys.orig_argv), cwd=os.getcwd(), hostname=socket.gethostname(),
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(), started_utc=started,
             finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=code, log_path='render.log')
    (out/'render_execution.json').write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')
