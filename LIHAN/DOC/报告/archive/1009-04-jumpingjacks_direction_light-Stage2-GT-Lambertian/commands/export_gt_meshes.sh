#!/usr/bin/env bash
set -euo pipefail
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
/home/lihan/reproduce/LumiMotion-stage2-perlight/output/Archive/1003-06-onlyclothV4-geometry_gt_eval/tools/blender-5.2.2-linux-x64/blender --factory-startup --disable-autoexec -b /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/scene_snapshot.blend --python /home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/export_blender_geometry.py -- --camera_json /home/lihan/data/LH-data/danamic/jumpingjacks_direction_light/camera.json --frames 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 --output_dir /home/lihan/reproduce/LumiMotion-stage2-perlight/output/1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/gt_meshes --export_ply
