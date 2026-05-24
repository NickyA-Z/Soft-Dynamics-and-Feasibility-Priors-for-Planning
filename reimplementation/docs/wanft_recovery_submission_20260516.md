# WAN-FT recovery status, 2026-05-16

## Goal

Keep the full Table 2 WAN-0S and Oracle jobs running, and use the same window to get WAN-FT into a reproducible state for the Table 2 settings:

- PushT: T=25, T=50, T=80
- Wall: T=25, T=50

Paper WAN-FT targets from Table 2 are PushT 0.80/0.30/0.06 and Wall 0.94/0.90.

## Code and scripts added

- `reimplementation/scripts/snellius/36_table2_wanft_existing_full_array.slurm`
  - Full 5-setting x 10-chunk WAN-FT evaluation array over the same sampled-50 specs used by the WAN-0S/Oracle Table 2 jobs.
  - Uses the existing best available checkpoints:
    - PushT: `$DL2RUNTIME_ROOT/models/wanft_pusht_t25_uniform4_100d_81f_linear_832x480/step-100.safetensors`
    - Wall: `$DL2RUNTIME_ROOT/models/wanft_wall_t25_paper_100d_81f_832x480/epoch-0.safetensors`
  - Generates at 832x480 with 81 frames and 50 inference steps, then runs the existing trajectory optimizer/evaluator through `jobs/25_wanft_t25.slurm`.
  - This is a pragmatic pipeline/baseline run, not yet the final paper-faithful PushT 720p checkpoint.

- `reimplementation/scripts/snellius/37_wanft_train_pusht_720p_staggered_resume.slurm`
  - 4xH100 48h PushT WAN-FT training job.
  - Prepares a 100-demo, 81-frame, staggered 1s-offset, linear-resampled, 1280x720 dataset if missing.
  - Resumes from `$DL2RUNTIME_ROOT/models/wanft_pusht_t25_paper_100d_81f_1280x720/step-50.safetensors`.
  - Writes to `$DL2RUNTIME_ROOT/models/wanft_pusht_t25_staggered_100d_81f_linear_1280x720_resume_from_prefix50`.

- `reimplementation/scripts/prepare_wanft_dataset.py`
  - Wall candidate filtering now uses the visual-frame count from `obses/episode_000.pth` instead of only the state tensor length.
  - Reason: Wall T=50 has enough visual frames, while the previous state-length check could incorrectly exclude all candidates.

- `reimplementation/scripts/summarize_table2_reports.py`
  - Aggregates `oracle_*`, `wan0s_*`, and `wanft_*` JSON reports by method/task/horizon.
  - Supports content-token filters so array jobs can be summarized even when Slurm child job ids, rather than array ids, appear in the report filenames.

- `reimplementation/scripts/snellius/38_wanft_pusht720_checkpoint_t25_array.slurm`
  - Evaluates an arbitrary PushT 720p WAN-FT checkpoint on the sampled-50 T=25 specs.
  - Requires `WANFT_LORA_CHECKPOINT=/path/to/checkpoint.safetensors`.
  - Intended use: submit small probes, for example `--array=0-1%2`, as soon as training writes `step-25.safetensors`, then expand to all 10 chunks if the probe is reasonable.

- `reimplementation/scripts/snellius/40_wanft_eval_precomputed_videos.slurm`
  - Eval-only helper for already generated `wanft.mp4` files.
  - Used to get early sanity metrics from the first `step-25` 720p videos before the full 10-case probe finishes.
  - Supports `WANFT_EPISODE_SPECS_FILE` so comma-separated specs do not get broken by Slurm `--export` parsing.

## Submitted jobs

- Existing-checkpoint WAN-FT eval:
  - Initial job: `22785961`
  - Running tasks preserved: `22785961_0` to `22785961_4`
  - These are PushT T=25 chunks 0 to 4.
  - Pending tasks `22785961_[5-49]` were cancelled because the first wrapper requested `220G`, which increased billing for a 1-GPU eval.
  - Resubmitted remaining tasks after lowering wrapper memory to `180G`: `22786013_[5-49%5]`.

- PushT 720p staggered resume training:
  - Job: `22785978`
  - Script: `$DL2RUNTIME_ROOT/jobs/37_wanft_train_pusht_720p_staggered_resume.slurm`
  - Training entered the dataloader at roughly 100 seconds per step; with `WANFT_SAVE_STEPS=25`, the first probe checkpoint is expected around 40 minutes after training proper starts.
  - `step-25.safetensors` was written at 2026-05-16 13:22.
  - Submitted a 720p T=25 probe on chunks 0-1 with this checkpoint: job `22786701_[0-1%2]`.
  - `step-50.safetensors` was written at 2026-05-16 14:02; not probed yet because the `step-25` 720p probe is still running.
  - Submitted eval-only sanity for the first two generated `step-25` probe videos: job `22787049`.
  - `22787049` only evaluated `12:97` because comma-separated `--export` truncated the specs; `12:97` succeeded.
  - Resubmitted the first-two eval through a specs file: job `22787147`.
  - `22787147` evaluated `12:97,12:135`: 1/2 success.
  - Submitted a `step-50` first-two 720p generate+eval probe on the same specs: job `22787215`.
  - `step-75.safetensors` was written at 2026-05-16 14:43; not probed yet.
  - Existing 832x480 WAN-FT PushT T=25 completed at 26/50 = 0.52, below the paper target 0.80.
  - Submitted eval-only for the first generated `step-50` video (`12:97`): job `22787738`.
  - `22787738` evaluated `step-50` on `12:97`: 0/1 success, worse than `step-25` on the same case.
  - Submitted a single-case `step-75` probe on `12:97`: job `22788006`.

## Immediate next action

Monitor `22785961`, `22786013`, and `22785978`. The full existing-checkpoint WAN-FT eval is split across the preserved initial PushT T=25 chunks and the resubmitted remaining chunks.

## Current interpretation

WAN-FT is being handled in two tracks:

1. Existing-checkpoint full evaluation to get complete, comparable numbers and catch evaluation/runtime bugs across all Table 2 settings.
2. PushT 720p staggered resume training to address the known low PushT WAN-FT baseline caused by undertrained and prefix-distribution-mismatched checkpoints.

The first track should produce quick diagnostic numbers. The second track is the likely path to a paper-faithful PushT WAN-FT result.
