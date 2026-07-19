#!/usr/bin/env bash
# Baseline Stage1 reproduction for d-nerf-relight-spec32.
# Scenes  : spheres_v5_spec32 / hook150_v5_spec32 / mouse150_v5_spec32
# Combo   : chapelday_goldenbay  (train_light = chapel_day_4k_32x16_rot0)
# Env     : lumimotion-minakshi  (server: minakshi)
# Derived from bash_scripts/synthetic_results_from_paper.sh — Stage1 + render_stage1_insights only.
# NOTE: --model_path is passed as ..._r2; the code auto-appends the deform suffix (_mlp).
#       --depth_ratio MUST be 1.0 for synth Stage1 (0.0 only at render time).
set -uo pipefail

cd /home/han.li/reproduce/LumiMotion

CONDA=/home/han.li/miniconda3/bin/conda
ENV=lumimotion-minakshi

RESOLUTION=2
ITERATIONS=35000
TRAIN_LIGHT="chapel_day_4k_32x16_rot0"
OUT_BASE="output/Baseline/0712-d-nerf-relight-spec32-baseline"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

# run_scene <scene_basename> <wbin(lambda_separation)> <wxyz(d_xyz_loss_weight)> <gpu>
run_scene() {
  local SCENE="$1" WBIN="$2" WXYZ="$3" GPU="$4"
  local SRC="data/d-nerf-relight-spec32/${SCENE}"
  local MODEL="$OUT_BASE/${SCENE}_r${RESOLUTION}"   # code auto-appends _mlp
  local S1_LOG="$LOG_DIR/${SCENE}_stage1.log"
  local RI_LOG="$LOG_DIR/${SCENE}_render_stage1.log"
  echo ">>> [$SCENE] GPU=$GPU  wbin=$WBIN wxyz=$WXYZ  ->  $MODEL"
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run --no-capture-output -n "$ENV" \
      python -m scripts.train_stage1 \
        --source_path="$SRC" --model_path="$MODEL" \
        --is_blender --eval --gt_alpha_mask_as_scene_mask \
        --resolution="$RESOLUTION" --iterations "$ITERATIONS" \
        --train_light_folder "$TRAIN_LIGHT" \
        --densify_until_iter 20000 \
        --lambda_separation "$WBIN" --d_xyz_loss_weight "$WXYZ" \
        --binarization_warm_up 1000 \
        --depth_ratio 1.0 --d_color_reg_loss_weight 0.01 \
        2>&1 | tee "$S1_LOG" \
    && \
    CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run --no-capture-output -n "$ENV" \
      python -m scripts.render_stage1_insights \
        --source_path="$SRC" --model_path="$MODEL" \
        --is_blender --eval --resolution="$RESOLUTION" --load_iter "$ITERATIONS" \
        --train_light_folder "$TRAIN_LIGHT" --depth_ratio 0.0 \
        2>&1 | tee "$RI_LOG"
    echo ">>> [$SCENE] FINISHED (pipeline exit=$?)"
  ) &
}

# chapelday_goldenbay CONFIGS (wbin / lossxyz) per scene, from synthetic_results_from_paper.sh
run_scene "spheres_v5_spec32"  0.005 0.001 4
run_scene "hook150_v5_spec32"  0.001 0.001 0
run_scene "mouse150_v5_spec32" 0.005 0.001 1

wait
echo ">>> ALL DONE"
