#!/bin/bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-02-jumpingjacks_dir-PBR方向光基线/commands/audit_artifacts.py'
