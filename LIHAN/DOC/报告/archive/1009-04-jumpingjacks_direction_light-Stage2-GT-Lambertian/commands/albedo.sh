#!/usr/bin/env bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
/home/lihan/miniconda3/envs/lumimotion/bin/python -u -m scripts.eval_albedo_gt --model_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/model_mlp --iterations 55000 --gt_albedo_dir /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/albedo --split test --output_path /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/albedo --save_float
