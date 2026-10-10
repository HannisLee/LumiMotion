#!/bin/bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.verify_roughness_eval '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复/roughness'
