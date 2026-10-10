#!/usr/bin/env bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.render_stage2 --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp --load_iter 55000 --task all --hdr_filepath /home/lihan/reproduce/LumiMotion-stage2-perlight/example_envmaps/golden_bay_4k_32x16_rot330.hdr --hdr_samples 2048
