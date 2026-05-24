#!/usr/bin/env bash
set -u

# Lightweight checkpoint watcher for WAN-FT debugging on Snellius.
# This is intentionally not a Slurm job script: it only submits generation/eval
# jobs when new LoRA checkpoints become stable on disk.

ROOT="${DL2RUNTIME_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TASK="${WANFT_TASK:-pusht}"
LABEL="${WANFT_WATCH_LABEL:-staggered_81f_linear}"
MODEL_DIR="${WANFT_MODEL_DIR:-$ROOT/models/wanft_${TASK}_t25_${LABEL}_832x480}"
CHECKPOINTS="${WANFT_CHECKPOINTS:-step-125 step-250 step-375 step-500 epoch-0 epoch-1 epoch-2 epoch-3 epoch-4 epoch-5 epoch-6 epoch-7 epoch-8 epoch-9}"
SPECS="${WANFT_EPISODE_SPECS:-12:97,6:38,5:29,7:17,2:64}"
EPISODE_IDS="${WANFT_EPISODE_IDS:-0}"
WIDTH="${WANFT_GENERATE_WIDTH:-832}"
HEIGHT="${WANFT_GENERATE_HEIGHT:-480}"
FRAMES="${WANFT_NUM_FRAMES:-81}"
STEPS="${WANFT_NUM_INFERENCE_STEPS:-50}"
SLEEP_SECONDS="${WANFT_WATCH_SLEEP_SECONDS:-180}"
STABILITY_SECONDS="${WANFT_STABILITY_SECONDS:-20}"

mkdir -p "$ROOT/reports" "$ROOT/logs"
cd "$ROOT" || exit 1

echo "[wanft-watch] started $(date -Is)"
echo "[wanft-watch] root=$ROOT"
echo "[wanft-watch] task=$TASK label=$LABEL"
echo "[wanft-watch] model_dir=$MODEL_DIR"
echo "[wanft-watch] checkpoints=$CHECKPOINTS"

stable_checkpoint() {
  local ckpt="$1"
  [ -s "$ckpt" ] || return 1
  local size1 size2
  size1="$(stat -c %s "$ckpt" 2>/dev/null || echo 0)"
  sleep "$STABILITY_SECONDS"
  size2="$(stat -c %s "$ckpt" 2>/dev/null || echo 1)"
  [ "$size1" = "$size2" ] && [ "$size2" -gt 0 ]
}

submit_checkpoint() {
  local stem="$1"
  local ckpt="$MODEL_DIR/$stem.safetensors"
  local marker="$ROOT/reports/submitted_${LABEL}_${stem}_first5.txt"

  [ ! -e "$marker" ] || return 0
  stable_checkpoint "$ckpt" || return 1

  local run_label video_root result_json gen_out gen_job eval_out
  run_label="wanft_${TASK}_seed99_chunk0_${LABEL}_${stem}_first5_${STEPS}steps_${WIDTH}x${HEIGHT}"
  video_root="$ROOT/videos/$run_label"
  result_json="$ROOT/reports/${run_label}.json"

  gen_out="$(env \
    DL2RUNTIME_ROOT="$ROOT" \
    WANFT_TASK="$TASK" \
    WANFT_EPISODE_IDS="$EPISODE_IDS" \
    WANFT_EPISODE_SPECS="$SPECS" \
    WANFT_VIDEO_ROOT="$video_root" \
    WANFT_MODEL_DIR="$MODEL_DIR" \
    WANFT_LORA_CHECKPOINT="$ckpt" \
    WANFT_GENERATE_WIDTH="$WIDTH" \
    WANFT_GENERATE_HEIGHT="$HEIGHT" \
    WANFT_NUM_FRAMES="$FRAMES" \
    WANFT_NUM_INFERENCE_STEPS="$STEPS" \
    WANFT_SKIP_EVAL=1 \
    WANFT_FORCE=1 \
    sbatch --export=ALL jobs/26_wanft_pusht_selected_t25.slurm)"
  gen_job="$(printf "%s\n" "$gen_out" | awk '/Submitted batch job/{print $4}' | tail -1)"

  if [ -z "$gen_job" ]; then
    echo "[wanft-watch] failed to parse generation job for $stem: $gen_out"
    return 1
  fi

  eval_out="$(env \
    DL2RUNTIME_ROOT="$ROOT" \
    WANFT_TASK="$TASK" \
    WANFT_EPISODE_IDS="$EPISODE_IDS" \
    WANFT_EPISODE_SPECS="$SPECS" \
    WANFT_VIDEO_ROOT="$video_root" \
    WANFT_RESULT_JSON="$result_json" \
    sbatch --dependency="afterok:${gen_job}" --export=ALL jobs/28_wanft_eval_existing_t25.slurm)"

  {
    printf "label=%s\n" "$LABEL"
    printf "checkpoint=%s\n" "$ckpt"
    printf "num_frames=%s\n" "$FRAMES"
    printf "video_root=%s\n" "$video_root"
    printf "result_json=%s\n" "$result_json"
    printf "gen_out=%s\n" "$gen_out"
    printf "eval_out=%s\n" "$eval_out"
  } > "$marker"

  echo "[wanft-watch] submitted $stem gen=$gen_job result=$result_json"
}

while true; do
  for stem in $CHECKPOINTS; do
    submit_checkpoint "$stem" || echo "[wanft-watch] $stem not ready"
  done
  sleep "$SLEEP_SECONDS"
done
