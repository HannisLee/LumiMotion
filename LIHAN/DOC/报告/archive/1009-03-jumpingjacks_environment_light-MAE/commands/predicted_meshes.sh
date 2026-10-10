#!/usr/bin/env bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.extract_geometry_mesh_dynamic --model_path '/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-01-jumpingjacks_environment_light-原版LumiMotion两阶段训练/model_mlp' --load_iter 55000 --split test --resolution 2 --voxel_size 0.02 --sdf_trunc 0.08 --alpha_threshold 0.5 --output_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-03-jumpingjacks_environment_light-MAE/predicted_meshes
