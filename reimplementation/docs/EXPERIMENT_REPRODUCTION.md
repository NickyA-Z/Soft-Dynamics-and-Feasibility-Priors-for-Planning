# Experiment Reproduction Guide

This document describes how to reproduce the experiments represented by the
`dinowm` branch. It focuses on reproducible entry points, fixed evaluation
specs, Slurm jobs, and result aggregation. It does not duplicate large external
assets in Git.

## Scope

The maintained experiment surface is:

- Local smoke tests for the GVP-WM reimplementation.
- Oracle, WAN-0S, and WAN-FT Table 2-style evaluations on PushT and Wall.
- Fixed sampled-50 `episode:offset` specs using seed `99`.
- Rescue jobs for known failed PushT `T=25` and `T=50` cases.
- JSON report aggregation against the paper thresholds.

The original paper targets used by `scripts/summarize_table2_goal.py` are:

| method | task | T | paper success |
| --- | --- | ---: | ---: |
| oracle | pusht | 25 | 0.98 |
| oracle | pusht | 50 | 0.72 |
| oracle | pusht | 80 | 0.36 |
| oracle | wall | 25 | 1.00 |
| oracle | wall | 50 | 1.00 |
| wan0s | pusht | 25 | 0.56 |
| wan0s | pusht | 50 | 0.12 |
| wan0s | pusht | 80 | 0.04 |
| wan0s | wall | 25 | 0.86 |
| wan0s | wall | 50 | 0.76 |
| wanft | pusht | 25 | 0.80 |
| wanft | pusht | 50 | 0.30 |
| wanft | pusht | 80 | 0.06 |
| wanft | wall | 25 | 0.94 |
| wanft | wall | 50 | 0.90 |

`summarize_table2_goal.py` marks a setting as passing when at least 50 specs are
present and the reproduced success rate is at least `min(0.9 * paper, paper -
0.05)`.

## External Assets

These assets must be provided outside Git:

- DINO-WM code dependencies and pretrained checkpoints.
- PushT data under a `pusht_noise` directory with `train`/`val` splits.
- Wall data under a `wall_single` directory.
- Wan2.1 image-to-video checkpoints and runtime dependencies.
- WAN-FT LoRA checkpoints.
- Generated WAN videos, JSON reports, Slurm logs, and model training outputs.

Keep these assets in a runtime directory such as:

```bash
export DL2RUNTIME_ROOT=/gpfs/home2/$USER/dl2runtime
mkdir -p "$DL2RUNTIME_ROOT"/{jobs,specs,reports,logs,videos,models}
```

## Local Environment and Smoke Tests

From a fresh checkout:

```bash
cd reimplementation
conda env create -n dl2 -f environment.yml
conda run -n dl2 python -m pytest
conda run -n dl2 python scripts/run_toy_demo.py
```

The local tests cover the optimizer, MPC loop, video preprocessing helpers, and
argument parsing. They do not require DINO-WM checkpoints or Wan2.1.

## Runtime Environment on Snellius

The Slurm scripts expect a runtime directory containing `env.sh`, a `jobs/`
copy of the Slurm scripts, and generated `specs/`.

Example setup:

```bash
export REPO_ROOT=/path/to/checkout
export DL2RUNTIME_ROOT=/gpfs/home2/$USER/dl2runtime
mkdir -p "$DL2RUNTIME_ROOT"/{jobs,specs,reports,logs,videos,models}
rsync -a "$REPO_ROOT/reimplementation/scripts/snellius/" "$DL2RUNTIME_ROOT/jobs/"
```

Create `$DL2RUNTIME_ROOT/env.sh` with paths for your account:

```bash
export DL2RUNTIME_ROOT=/gpfs/home2/$USER/dl2runtime
export REIMPLEMENTATION_ROOT=/path/to/checkout/reimplementation
export DINO_WM_ROOT=/path/to/checkout/dino_wm
export DINO_WM_PYTHON=/path/to/python
export DINO_DATA_ROOT=/path/to/dino_wm/data
export PUSHT_DATA_ROOT=$DINO_DATA_ROOT/pusht_noise
export WALL_DATA_ROOT=$DINO_DATA_ROOT/wall_single
export DL2_REPORT_DIR=$DL2RUNTIME_ROOT/reports
export DL2_LOG_DIR=$DL2RUNTIME_ROOT/logs
```

The scripts also use cache variables such as `HF_HOME`, `TORCH_HOME`,
`MPLCONFIGDIR`, and `TMPDIR`. By default, the newer Table 2 scripts place these
under `/gpfs/scratch1/nodespecific/int5/$USER_ID/dl2_cache`; override
`TABLE2_SCRATCH_CACHE` if your account uses a different scratch location.

## Fixed Sampled-50 Specs

Generate deterministic sampled-50 specs for PushT `T=25,50,80` and Wall
`T=25,50`:

```bash
cd "$REPO_ROOT/reimplementation"
python scripts/make_table2_specs.py \
  --runtime-dir "$DL2RUNTIME_ROOT" \
  --dino-data-root "$DINO_DATA_ROOT" \
  --seed 99 \
  --num-specs 50 \
  --chunk-size 5 \
  --frame-skip 5 \
  --force
```

This writes files such as:

- `$DL2RUNTIME_ROOT/specs/t25_pusht_seed99_50.specs`
- `$DL2RUNTIME_ROOT/specs/t25_pusht_seed99_50_chunks5.txt`
- `$DL2RUNTIME_ROOT/specs/t50_wall_seed99_50.specs`
- `$DL2RUNTIME_ROOT/specs/t50_wall_seed99_50_chunks5.txt`

The checked-in files under `reimplementation/specs/` are targeted rescue specs,
not the complete sampled-50 Table 2 specs.

## Table 2 Evaluation Jobs

Submit jobs from the runtime directory so that `SLURM_SUBMIT_DIR` resolves to
`$DL2RUNTIME_ROOT`:

```bash
cd "$DL2RUNTIME_ROOT"
sbatch jobs/34_table2_oracle_full_array.slurm
sbatch jobs/35_table2_wan0s_full_array.slurm
sbatch jobs/36_table2_wanft_existing_full_array.slurm
```

The array mappings are:

- `34_table2_oracle_full_array.slurm`: tasks `0..4` are PushT `T=25,50,80`,
  Wall `T=25,50`.
- `35_table2_wan0s_full_array.slurm`: tasks `0..49`, ten chunks per setting.
- `36_table2_wanft_existing_full_array.slurm`: tasks `0..49`, ten chunks per
  setting using the configured existing LoRA checkpoints.

For direct evaluation of existing videos without regenerating them, use:

```bash
cd "$DL2RUNTIME_ROOT"
sbatch jobs/41_table2_direct_eval_existing_array.slurm
```

`41_table2_direct_eval_existing_array.slurm` evaluates all three methods over
the five Table 2 settings. It expects the generated video directories named by
the script, or compatible overrides through environment variables.

## WAN-FT Training and Probes

The main WAN-FT helpers are:

- `37_wanft_train_pusht_720p_staggered_resume.slurm`: resume PushT WAN-FT
  training at 1280x720.
- `38_wanft_pusht720_checkpoint_t25_array.slurm`: evaluate a specific PushT
  720p checkpoint. Set `WANFT_LORA_CHECKPOINT=/path/to/checkpoint.safetensors`.
- `40_wanft_eval_precomputed_videos.slurm`: evaluate already generated WAN-FT
  videos. Use `WANFT_EPISODE_SPECS_FILE` when passing comma-separated specs
  through Slurm would be fragile.

The 832x480 existing-checkpoint baseline uses:

```bash
cd "$DL2RUNTIME_ROOT"
sbatch jobs/36_table2_wanft_existing_full_array.slurm
```

## Rescue Jobs

The targeted rescue jobs use checked-in spec files under `reimplementation/specs/`:

- `oracle_pusht_t50_union_fail_chunks3.txt`
- `wanft_pusht_t25_fail_root22785961_chunks4.txt`
- `wanft_pusht_t25_fail_root22786013_chunks4.txt`
- `wanft_pusht_t50_fail_chunks4.txt`

Submit the oracle PushT rescue:

```bash
cd "$DL2RUNTIME_ROOT"
RESCUE_SPECS_FILE="$REPO_ROOT/reimplementation/specs/oracle_pusht_t50_union_fail_chunks3.txt" \
RESCUE_RAW_HORIZON=50 \
RESCUE_MODE=refine2000 \
sbatch jobs/44_oracle_pusht_targeted_rescue.slurm
```

Submit the WAN-FT PushT rescue:

```bash
cd "$DL2RUNTIME_ROOT"
RESCUE_SPECS_FILE="$REPO_ROOT/reimplementation/specs/wanft_pusht_t50_fail_chunks4.txt" \
RESCUE_RAW_HORIZON=50 \
RESCUE_MODE=nopad_refine2000 \
RESCUE_VIDEO_ROOT="$DL2RUNTIME_ROOT/videos/wanft_existing_pusht_s50_t50_22786013" \
sbatch jobs/45_wanft_pusht_targeted_rescue.slurm
```

For a chunked oracle retry over the standard sampled-50 chunk files, use:

```bash
cd "$DL2RUNTIME_ROOT"
RESCUE_RAW_HORIZON=50 RESCUE_MODE=nopad \
sbatch jobs/42_oracle_pusht_direct_rescue_array.slurm
```

## Result Summaries

Aggregate Table 2-style reports:

```bash
cd "$REPO_ROOT/reimplementation"
python scripts/summarize_table2_reports.py \
  --report-dir "$DL2RUNTIME_ROOT/reports" \
  --details
```

Evaluate against paper-relative thresholds:

```bash
cd "$REPO_ROOT/reimplementation"
python scripts/summarize_table2_goal.py \
  --report-dir "$DL2RUNTIME_ROOT/reports" \
  --pattern "direct_*.json" \
  --union-by-spec \
  --show-fail-specs
```

Use `--job-id` on `summarize_table2_goal.py`, or `--require-token` /
`--any-token` on `summarize_table2_reports.py`, to isolate specific Slurm
submissions.

## Reproducibility Notes

- The evaluation specs are deterministic when `seed=99`, `num-specs=50`,
  `chunk-size=5`, and `frame-skip=5` are kept fixed.
- Report filenames include Slurm job ids. Preserve the JSON files when sharing
  results; logs alone are not enough to recompute aggregate success rates.
- WAN generation and WAN-FT training are stochastic. Record checkpoint paths,
  `WANFT_SEED_BASE`, `WANFT_SEED_OFFSET`, video roots, and Slurm job ids with
  every reported number.
- Wall uses `env-replay` targets for the Table 2 protocol in these scripts.
- Generated artifacts should stay outside Git. Commit scripts, specs, and
  summary documents; keep videos, checkpoints, logs, caches, and report archives
  in the runtime directory.
