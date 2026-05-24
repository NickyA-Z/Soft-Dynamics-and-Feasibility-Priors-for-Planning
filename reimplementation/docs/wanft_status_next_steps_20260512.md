# WAN-FT Reproduction Status and Next Steps

Date: 2026-05-12

## Paper Targets

WAN-FT is a main Table 2 condition, not an auxiliary experiment.

Relevant Table 2 targets:

- PushT, GVP-WM (WAN-FT), T=25: `0.80`
- Wall, GVP-WM (WAN-FT), T=25: `0.94`

The PushT ablation table reports `0.82` for GVP-WM with WAN-FT at T=25, but the main Table 2 number to compare against is `0.80`.

## What Is Implemented

The WAN-FT pipeline is runnable end to end:

- Dataset preparation: `reimplementation/scripts/prepare_wanft_dataset.py`
- LoRA training: `reimplementation/scripts/snellius/24_wanft_train_t25.slurm`
- Generated-video evaluation: `reimplementation/scripts/snellius/25_wanft_t25.slurm`
- Existing-video evaluation: `reimplementation/scripts/snellius/28_wanft_eval_existing_t25.slurm`

The current implementation supports:

- PushT and Wall task datasets
- `prefix`, `full`, `uniform`, and `staggered` clip extraction
- 1 or multiple segments per demo
- `25f` and `81f` training clips
- `832x480` and `1280x720` training/generation sizes
- LoRA rank `32`, learning rate `1e-5`, dataset repeat `10`
- FLF2V conditioning via `input_image,end_image`
- GVP-WM evaluation through the same `wan0s_eval.py` path, using generated `wanft.mp4`

## Current Results

### Wall

Best current Wall result:

- Result: `10/10 = 1.00`
- Report: `/gpfs/home2/scur0196/dl2runtime/reports/wanft_wall_t25_22610626.json`
- LoRA: `/gpfs/home2/scur0196/dl2runtime/models/wanft_wall_t25_paper_100d_81f_832x480/epoch-0.safetensors`
- Eval set: selected 10 episodes, not full sampled-50

Interpretation: Wall WAN-FT looks very promising and likely should be evaluated on the full 50-pair set next. We cannot yet claim the Table 2 `0.94` result because the current success is selected-batch only.

### PushT

PushT is still the bottleneck.

Main full sampled-50 result for the `full_epoch0_832x480` model:

- Chunk results: `3/10`, `4/10`, `1/10`, `5/10`, `4/10`
- Aggregate: `17/50 = 0.34`
- Reports:
  - `/gpfs/home2/scur0196/dl2runtime/reports/wanft_eval_existing_pusht_seed99_chunk0_full_epoch0_832x480.json`
  - `/gpfs/home2/scur0196/dl2runtime/reports/wanft_eval_existing_pusht_seed99_chunk1_full_epoch0_832x480.json`
  - `/gpfs/home2/scur0196/dl2runtime/reports/wanft_eval_existing_pusht_seed99_chunk2_full_epoch0_832x480.json`
  - `/gpfs/home2/scur0196/dl2runtime/reports/wanft_eval_existing_pusht_seed99_chunk3_full_epoch0_832x480.json`
  - `/gpfs/home2/scur0196/dl2runtime/reports/wanft_eval_existing_pusht_seed99_chunk4_full_epoch0_832x480.json`

Other PushT probes:

- Selected 9 episodes with `paper_100d_81f_832x480/epoch-0`: `3/9 = 0.33`
- `prefix_epoch1_832x480`, partial chunks: `14/30 = 0.47`
- `uniform4_25f_epoch0_832x480`, chunk0: `5/10 = 0.50`
- `uniform4_81f_linear_step100_832x480`, first5: `3/5 = 0.60`
- 720p probes are weaker so far: best observed first5 was `2/5 = 0.40`

Interpretation: PushT WAN-FT is not close to the paper target yet. The best small probe is promising but too small to trust; the only full sampled-50 result is far below target.

## Training Status

We have not completed a fully faithful paper-scale WAN-FT training run.

Paper-scale expectation:

- Wan2.1-FLF2V-14B 720p
- 100 task-specific demonstrations
- LoRA rank `32`
- 10 epochs
- 4 x A100 80GB
- Paper reports roughly 5h40m per epoch

Current reality:

- The working `832x480` models produced multiple checkpoints, but these are lower-resolution than the paper's 720p setup.
- The `1280x720` PushT runs were short probes. They produced only early step checkpoints such as `step-25` and `step-50`, then hit short wall-time limits.
- Several training jobs were intentionally stopped or timed out; this means our current PushT 720p checkpoints are undertrained and should not be treated as paper-faithful.
- Wall succeeds even with the lower-resolution selected setup, but still needs full 50-pair validation.

## Likely Failure Modes

PushT failure is probably not one single bug.

Most likely issues:

- The PushT LoRA training distribution is not aligned with the evaluation distribution. We used several variants, but many early runs trained on prefix/offset-0 clips while evaluation uses sampled `episode:offset` windows.
- 832x480 is convenient but not faithful to the paper's 720p Wan setting.
- The 720p models are undertrained; the existing probes are too short to conclude that 720p does not help.
- PushT GVP-WM oracle is still below the paper upper bound in our implementation. If the planner ceiling is low on the same sampled-50 set, WAN-FT cannot reliably reach `0.80`.
- Small probes are noisy. A `3/5` result is useful for prioritization, but it is not a metric.

## Recommended Next Steps

1. Wait for the current full T=25 ORACLE/WAN-0S jobs to finish. The new oracle numbers tell us the current planner ceiling on the same sampled-50 specs.

2. Submit Wall WAN-FT sampled-50 using the existing Wall LoRA. This is the highest-probability quick win because selected Wall is already `10/10`.

3. For PushT, run a controlled first20/full50 evaluation of the most promising existing checkpoint before training more. The strongest candidate is currently the `uniform4_81f_linear_step100_832x480` path because it reached `3/5` on the first probe.

4. If PushT still fails, do not blindly continue training the same setup. First inspect generated videos on failures and classify whether failure is video-quality, GVP-WM action recovery, or oracle/planner ceiling.

5. If video quality is the bottleneck, run a faithful 720p WAN-FT training job in resumable chunks. Use long enough wall time and checkpoints, because the previous 720p runs only reached early steps.

6. Use a staged gate for expensive training: first evaluate first10 or first20; continue/resume only if the probe reaches roughly `0.6+` and videos look physically plausible.

## Immediate Proposed Order

After current jobs free resources:

1. Wall WAN-FT sampled-50 with `/gpfs/home2/scur0196/dl2runtime/models/wanft_wall_t25_paper_100d_81f_832x480/epoch-0.safetensors`.
2. PushT WAN-FT first20 or full50 with the `uniform4_81f_linear_step100_832x480` checkpoint.
3. Analyze current PushT oracle sampled-50 result.
4. Decide whether to invest in a long 720p PushT WAN-FT resume run.

