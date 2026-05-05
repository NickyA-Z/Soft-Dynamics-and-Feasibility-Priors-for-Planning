# WAN-0S Execution Report

Date: 2026-05-05

## Scope

This round focused on making WAN-0S runnable end to end on Snellius, not on matching Table 2 yet. I first saved and pushed the existing oracle sanity work, then added a zero-shot Wan2.1 FLF2V pipeline that prepares start/goal frames, generates a video, feeds the generated frames into GVP-WM, and records planner metrics.

The earlier checkpoint was committed and pushed as `9bd592c` (`Save oracle sanity diagnostics`). The WAN-0S implementation in this report is the follow-up work.

## Implementation

Added repo-side files:

- `reimplementation/docs/wan0s_memo.md`: working memo and success criteria for WAN-0S.
- `reimplementation/scripts/prepare_wan0s_inputs.py`: extracts PushT/Wall start and goal frames for Wan FLF2V.
- `reimplementation/src/gvpwm/examples/wan0s_utils.py`: prompt, letterbox, video loading, and frame conversion utilities.
- `reimplementation/src/gvpwm/examples/wan0s_eval.py`: evaluates generated WAN-0S videos as visual-only GVP-WM video plans.
- `reimplementation/scripts/snellius/05_wan0s_t25.slurm`: Snellius job script for T=25 WAN generation plus evaluation.

The same WAN-0S job script is staged under `~/dl2runtime/jobs/05_wan0s_t25.slurm`. Runtime configuration stays in `~/dl2runtime/env.sh`, so the job script uses environment variables such as `DL2RUNTIME_ROOT`, `REIMPLEMENTATION_ROOT`, `WAN21_REPO_DIR`, `WAN21_MODEL_DIR`, and `SLURM_SUBMIT_DIR` instead of hard-coded project paths.

Existing runtime resources verified:

- Wan repo: `~/dl2runtime/wan2.1/Wan2.1`
- Wan FLF2V checkpoint: `~/dl2runtime/models/Wan2.1-FLF2V-14B-720P`
- Wan Python runtime: `~/dl2runtime/venvs/wan21-runtime-py311`
- Output root: `~/dl2runtime/videos/wan0s`
- Reports/logs: `~/dl2runtime/reports`, `~/dl2runtime/logs`

## Important Fixes

The first WAN-0S attempt used `WAN_FRAME_NUM=21`, which failed inside the official Wan2.1 FLF2V code because this checkout builds an 81-frame conditioning mask. I changed the default to `WAN_FRAME_NUM=81`; shorter frame counts can still be passed manually, but they are expected to fail unless the Wan FLF2V mask logic is patched.

I also corrected the Wall default prompt from "blue dot" to "red dot" after inspecting the generated start/goal images. The Wall environment renders a red dot, so the prompt should not fight the conditioning images.

## Jobs Run

`22472792`: Wall ep0, 21 frames, failed as expected after discovering the Wan FLF2V 81-frame mask assumption.

`22472799`: Wall ep0, 81 frames, quick eval, completed. Result: `state_dist=6.73`, `success=false`. This proved the pipeline could generate a non-empty mp4 and run evaluation.

`22472830`: Wall ep0, 81 frames, 8 Wan steps, full planner eval (`25/25/500`), completed in 9 minutes. Result JSON: `~/dl2runtime/reports/wan0s_wall_t25_22472830.json`. Metric: `state_dist=5.55`, `success=false`, `dynamics_residual=3.62`. The generated video visibly moves the red dot toward the goal and is close to the Wall success threshold of `4.5`, but does not pass it.

`22472831`: PushT ep0, 81 frames, quick eval, completed in 8 minutes. Result JSON: `~/dl2runtime/reports/wan0s_pusht_t25_22472831.json`. It reported success, but this episode has zero block-pose change over T=25, so it is only a trivial smoke test.

`22472875`: PushT ep11, 81 frames, quick eval, completed in 7 minutes. This episode has meaningful block movement over T=25 (`block_diff` about `70px` from start to goal). Result: `block_diff=29.35`, `angle_diff=1.32`, `success=false`.

`22472894`: PushT ep11, reusing the generated video, full planner eval (`25/25/500`), completed in 2 minutes. Result JSON: `~/dl2runtime/reports/wan0s_pusht_t25_22472894.json`. Metric: `block_diff=16.87`, `angle_diff=1.76`, `success=false`. The full eval crosses the position threshold (`<20px`) but misses the angle threshold (`<pi/9`).

Contact sheets for visual inspection:

- `~/dl2runtime/reports/wan0s_wall_ep000_steps8_contact.jpg`
- `~/dl2runtime/reports/wan0s_pusht_ep011_steps8_contact.jpg`

## Current Assessment

WAN-0S is now runnable on Snellius for both Wall and PushT. The pipeline produces mp4 videos, consumes them in GVP-WM, executes five macro actions for raw `T=25` with `frame_skip=5`, and writes metrics to JSON.

The results are reasonable as implementation smoke tests, but they are not Table 2 reproduction numbers. Wall ep0 gets close but not under the success threshold. PushT ep11 substantially improves block position but misses orientation. The large PushT dynamics residuals suggest there is still a mismatch between generated visual plans, DINO-WM dynamics, and action optimization; this is separate from simply making Wan generate and run.

## Next Steps For Table 2

To move from runnable WAN-0S to paper-level WAN-0S reproduction:

- Run a multi-episode batch for Wall and PushT instead of single-case smoke tests.
- Keep the official 81-frame Wan FLF2V setting unless we patch Wan's mask construction.
- Tune prompts per task, especially PushT orientation language and whether to describe the green target T explicitly.
- Compare generated-video planner metrics against oracle-video planner metrics on the same episode ids.
- Revisit action scaling/proprio handling only if oracle-video and generated-video failures point to the same planner-side issue.

## Rerun Commands

From `~/dl2runtime`:

```bash
WAN0S_TASK=wall WAN0S_EPISODE_IDS=0 WAN_FORCE=1 WAN_FRAME_NUM=81 WAN_SAMPLE_STEPS=8 \
  EVAL_INNER_STEPS=25 EVAL_OUTER_STEPS=25 EVAL_REFINEMENT_SAMPLES=500 \
  sbatch --export=ALL jobs/05_wan0s_t25.slurm

WAN0S_TASK=pusht WAN0S_EPISODE_IDS=11 WAN_FORCE=1 WAN_FRAME_NUM=81 WAN_SAMPLE_STEPS=8 \
  EVAL_INNER_STEPS=25 EVAL_OUTER_STEPS=25 EVAL_REFINEMENT_SAMPLES=500 \
  sbatch --export=ALL jobs/05_wan0s_t25.slurm
```

