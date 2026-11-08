#!/usr/bin/env bash
# T02 固定执行入口；正式拟合后须人工目检并填写文档，再调用 freeze。
set -euo pipefail
project=/home/han.li/reproduce/LumiMotion-stage2-perlight
root=$(git rev-parse --show-toplevel)
source_path="$project/data/LH-data/static/LH-static-jumpingjacks-ELDL-V3"
model="$project/output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp"
observations="$project/output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz"
baseline="$project/output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角"
plan="$project/LIHAN/DOC/规划/1103-02-T02低频轨迹切空间保护SH背景代理规划.md"
if [[ $(hostname -s) != ushas ]]; then echo '服务器不符' >&2; exit 1; fi
cd "$root"
mode=${1:?需要 self-test、smoke、formal、figures 或 freeze}
if [[ $mode == worker ]]; then
    phase=${2:?}; out=${3:?}; self_dir=${4:-}; smoke_dir=${5:-}
    args=(--output "$out" --baseline "$baseline" --observations "$observations" --source "$source_path" --model "$model" --iteration 35000 --plan "$plan")
    [[ -n $self_dir ]] && args+=(--self-test-dir "$self_dir")
    [[ -n $smoke_dir ]] && args+=(--smoke-dir "$smoke_dir")
    run() {
        local command=(conda run --no-capture-output -n lumimotion-ushas python -m LIHAN.scripts.static_rgb_light_shbg_t02 "$1" "${args[@]}")
        shift; command+=("$@")
        printf '%q ' "${command[@]}" >> "$out/command.txt"; printf '\n' >> "$out/command.txt"
        "${command[@]}"
    }
    code=0
    case "$phase" in
        self-test) run self-test || code=$? ;;
        smoke) run lock && run extract && run fit --smoke || code=$? ;;
        formal) run lock && run extract && run fit && run choose || code=$? ;;
        figures) run figures || code=$? ;;
        freeze) run freeze || code=$? ;;
        *) echo '未知阶段' >&2; code=2 ;;
    esac
    if [[ $phase != freeze || $code != 0 ]]; then printf '%s\n' "$code" > "$out/${phase}.exit_code"; fi
    exit "$code"
fi
case "$mode" in
    self-test|smoke|formal)
        self_dir=${2:-}; smoke_dir=${3:-}
        [[ $mode == self-test || -n $self_dir ]] || { echo '需要合成自检目录' >&2; exit 2; }
        [[ $mode != formal || -n $smoke_dir ]] || { echo '需要冒烟目录' >&2; exit 2; }
        mkdir -p "$project/output/smoke_test"
        exec 9>"$project/output/.t02-allocation.lock"; flock 9
        day=$(date -u +%m%d); max=0
        for p in "$project/output/$day"-* "$project/output/smoke_test/$day"-*; do
            base=${p##*/}; if [[ $base =~ ^[0-9]{4}-([0-9]+)- ]]; then num=$((10#${BASH_REMATCH[1]})); ((num>max)) && max=$num; fi
        done
        printf -v seq '%02d' "$((max+1))"
        case "$mode" in
            self-test) out="$project/output/smoke_test/$day-$seq-LH-static-ELDL-V3-T02-合成自检" ;;
            smoke) out="$project/output/smoke_test/$day-$seq-LH-static-ELDL-V3-T02-C01冒烟" ;;
            formal) out="$project/output/$day-$seq-LH-static-ELDL-V3-T02-低频轨迹SH" ;;
        esac
        mkdir "$out"; flock -u 9
        printf '{"mode":"%s","utc":"%s"}\n' "$mode" "$(date -u +%FT%TZ)" > "$out/allocation.json"
        ;;
    figures|freeze)
        out=${2:?需要正式输出目录}; self_dir=${3:?需要自检目录}; smoke_dir=${4:?需要冒烟目录}
        [[ -d $out && ! -e $out/$mode.exit_code ]] || { echo '目录不存在或阶段已执行' >&2; exit 2; }
        ;;
    *) echo '未知阶段' >&2; exit 2 ;;
esac
printf '{"hostname":"%s","conda_env":"lumimotion-ushas","start_utc":"%s"}\n' "$(hostname -s)" "$(date -u +%FT%TZ)" > "$out/environment.json"
nohup bash "$root/LIHAN/scripts/static_rgb_light_shbg_t02_launch.sh" worker "$mode" "$out" "$self_dir" "$smoke_dir" >> "$out/run.log" 2>&1 </dev/null &
pid=$!; printf '%s\n' "$pid" > "$out/launcher.pid"
printf '输出目录: %s\nPID: %s\n' "$out" "$pid"
