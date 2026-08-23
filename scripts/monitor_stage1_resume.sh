#!/usr/bin/env bash
set -u

# 仅记录状态，不重启、不终止训练。由 tmux 独立运行，直至两条训练会话均结束。
root_dir='/home/han.li/reproduce/LumiMotion-perlight'
declare -A session_by_dataset=(
  [CV3]='lumimotion_cv3_0818_resume'
  [CV3L]='lumimotion_cv3l_0818_resume'
)
declare -A log_by_dataset=(
  [CV3]='output/0817-04-CV3-GTlight_i5p5_A500_Nonly_lr1e4/train_stage1_resume_from_5000.log'
  [CV3L]='output/0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4/train_stage1_resume_from_5000.log'
)
declare -A model_by_dataset=(
  [CV3]='output/0817-04-CV3-GTlight_i5p5_A500_Nonly_lr1e4/CV3_A500_Nonly_lr1e4_mlp'
  [CV3L]='output/0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4/CV3L_A500_Nonly_lr1e4_mlp'
)
monitor_log='output/0817-04-CV3-GTlight_i5p5_A500_Nonly_lr1e4/process_monitor_0818.log'
monitor_log_cv3l='output/0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4/process_monitor_0818.log'

cd "$root_dir"
while true; do
  now=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  any_running=0
  for dataset in CV3 CV3L; do
    session=${session_by_dataset[$dataset]}
    log=${log_by_dataset[$dataset]}
    model=${model_by_dataset[$dataset]}
    if tmux has-session -t "$session" 2>/dev/null; then
      state='RUNNING'
      any_running=1
    else
      state='EXITED'
    fi
    remaining=$(tr '\r' '\n' < "$log" 2>/dev/null | rg -o '[0-9]+/30000' | tail -n 1 || true)
    if [[ -n "$remaining" ]]; then
      step=$((5000 + ${remaining%%/*}))
    else
      step='unknown'
    fi
    checkpoint=$(find "$model/point_cloud" -mindepth 1 -maxdepth 1 -type d -name 'iteration_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)
    checkpoint=${checkpoint:-none}
    error_count=$(tail -n 120 "$log" 2>/dev/null | rg -i -c 'traceback|cuda out of memory|out of memory|segmentation fault|killed' || true)
    error_count=${error_count:-0}
    line="$now dataset=$dataset state=$state global_step=$step checkpoint=$checkpoint recent_error_markers=$error_count"
    printf '%s\n' "$line" | tee -a "$monitor_log" "$monitor_log_cv3l" >/dev/null
  done
  if [[ "$any_running" -eq 0 ]]; then
    exit 0
  fi
  sleep 60
done
