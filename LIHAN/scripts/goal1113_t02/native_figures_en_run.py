"""Redraw T02-native figures with English text into <ROOT>/figures-en (figures/ untouched). cwd=repo root."""
import argparse
from pathlib import Path
from LIHAN.scripts.goal1113 import physics as P
from LIHAN.scripts.goal1113_t02 import native_run as NR
ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); ap.add_argument('--device', default='cuda')
a = ap.parse_args(); P.grid = NR.grid
from LIHAN.scripts.goal1113_t02 import native_figures_en as NF
NF.run(Path(a.output), a.device)
