#!/bin/bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/lihan/miniconda3/envs/lumimotion/bin/python -u '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/commands/write_results.py'
