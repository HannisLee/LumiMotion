#!/bin/bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1010-15-jumpingjacks_dir-PBR概率gate优化/predicted_meshes'
