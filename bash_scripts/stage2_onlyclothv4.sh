#!/usr/bin/env bash
# 只读共享SH来源，分别运行原始IR/GT/learned并完整评估；禁止覆盖已有实验。
set -euo pipefail
stage2_mode=${1:?用法: stage2_onlyclothv4.sh original|GT|learned output/日期-序号-数据集-特征}
stage2_output=${2:?缺少输出目录}
case "$stage2_output" in output/*) ;; *) echo '输出必须位于output/'; exit 2;; esac
case "$stage2_mode" in original|GT|learned) ;; *) echo '未知模式';exit 2;; esac
stage2_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$stage2_root"
stage2_server=$(hostname)
case "$stage2_server" in
  venus) stage2_env=lumimotion;;
  mahadevi|minakshi|parvati|ushas|garuda) stage2_env=lumimotion-$stage2_server;;
  *) echo "没有环境映射: $stage2_server";exit 2;;
esac
if [[ -e "$stage2_output/STARTED" || -e "$stage2_output/FAILED" || -e "$stage2_output/COMPLETED" ]]; then
  echo '已有实验禁止覆盖';exit 2
fi
mkdir -p "$stage2_output/logs" "$stage2_output/commands"
exec > >(tee -a "$stage2_output/logs/pipeline.log") 2>&1
touch "$stage2_output/STARTED"
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "FAILED exit=$status"; touch "$stage2_output/FAILED"; fi' EXIT
stage2_source=${LM_STAGE1_MODEL:-/home/lihan/reproduce/LumiMotion-perlight/output/1001-01-onlyclothV4-FOVxy修复重训/model_mlp}
stage2_data=${LM_STAGE2_DATA:-/home/lihan/data/LH-data/transfer-static/only_clothV4}
stage2_lights=${LM_STAGE2_LIGHTS:-/home/lihan/data/LH-data/static/only_clothV4/lights.json}
stage2_hdr=example_envmaps/golden_bay_4k_32x16_rot330.hdr
git rev-parse HEAD > "$stage2_output/code_commit.txt"
git diff > "$stage2_output/code_diff.patch"
git status --short > "$stage2_output/code_status.txt"
date --iso-8601=seconds > "$stage2_output/start_time.txt"
hostname > "$stage2_output/server.txt"
run_step() {
  local step_name=$1;shift
  printf '%q ' "$@" > "$stage2_output/commands/$step_name.sh"
  printf '\n' >> "$stage2_output/commands/$step_name.sh"
  "$@" > "$stage2_output/logs/$step_name.log" 2>&1
}
stage2_common=(conda run --no-capture-output -n "$stage2_env" python -m scripts.train_stage2
  --source_path "$stage2_data" --stage1_model_path "$stage2_source" --model_path "$stage2_output/model"
  --load_iter 35000 --iterations 55000 --resolution 2 --is_blender --eval --load2gpu_on_the_fly
  --train_light_folder images --test_light_folder images --depth_ratio 0 --diffuse_sample_num 512
  --save_iterations 40000 50000 55000 --test_iterations 55000)
if [[ "$stage2_mode" == original ]]; then
  run_step train "${stage2_common[@]}" --render_mode original_ir --loss_preset irgs_baseline
  run_step evaluate conda run --no-capture-output -n "$stage2_env" python -m scripts.render_stage2 \
    --model_path "$stage2_output/model_mlp" --load_iter 55000 --task eval
  run_step insights conda run --no-capture-output -n "$stage2_env" python -m scripts.render_stage2 \
    --model_path "$stage2_output/model_mlp" --load_iter 55000 --task insights \
    --output_path "$stage2_output/model_mlp/stage2_insights/ours_55000"
  run_step hdr conda run --no-capture-output -n "$stage2_env" python -m scripts.render_relight_with_hdr \
    --model_path "$stage2_output/model_mlp" --load_iter 55000 --hdr "$stage2_hdr" \
    --depth_ratio 0 --diffuse_sample_num 2048 --load2gpu_on_the_fly
else
  stage2_light_mode=learned_directional
  stage2_preset=lambertian_default
  if [[ "$stage2_mode" == GT ]]; then stage2_light_mode=gt_directional;stage2_preset=lambertian_gt;fi
  run_step train "${stage2_common[@]}" --render_mode photometric_lambertian --loss_preset "$stage2_preset" \
    --photometric_light_mode "$stage2_light_mode" --photometric_gt_lights_path "$stage2_lights" \
    --photometric_light_intensity 6.032453522706779 --photometric_log_interval 500
  run_step render conda run --no-capture-output -n "$stage2_env" python -m scripts.render_stage2 \
    --model_path "$stage2_output/model_mlp" --load_iter 55000 --task all --hdr_filepath "$stage2_hdr"
fi
date --iso-8601=seconds > "$stage2_output/finish_time.txt"
touch "$stage2_output/PIPELINE_COMPLETED"
echo "管线执行完成，等待四类目检与README验收: $stage2_output"
