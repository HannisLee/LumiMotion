#!/usr/bin/env bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
/home/lihan/miniconda3/envs/lumimotion/bin/python -u /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/commands/verify_all_mae.py
