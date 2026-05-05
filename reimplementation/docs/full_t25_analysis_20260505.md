# Full T=25 PushT/Wall Oracle And WAN-0S Analysis

Date: 2026-05-05

Repository commit: `31481fc Add group meeting T25 update script`

Submission manifest: `/home/scur0196/dl2runtime/reports/full_t25_submission_20260505_094113.txt`

Fresh WAN-0S video root: `/home/scur0196/dl2runtime/videos/wan0s_full_t25_20260505`

## Scope

This run submitted the requested T=25 experiments for PushT and Wall under both oracle-video GVP-WM and WAN-0S GVP-WM.

Common settings:

- `raw_horizon=25`
- `frame_skip=5`
- world-model macro horizon `5`
- ALM inner/outer steps `25/25`
- local refinement `500` samples with variance `0.3`
- WAN-0S generation used Wan2.1 FLF2V with `WAN_FRAME_NUM=81` and `WAN_SAMPLE_STEPS=8`

Evaluation set caveat:

- PushT local `val` only has 21 eligible T=25 episodes, so PushT results are over all 21 available local episodes, not the paper's 50 initial-goal pairs.
- Wall used the first 50 eligible local `wall_single/all` episodes.

Paper Table 2 targets, verified from `original_paper/arXiv-2602.01960v3.tar.gz:example_paper.tex`:

- PushT, GVP-WM (WAN-0S), T=25: `0.56`
- PushT, GVP-WM (ORACLE), T=25: `0.98`
- Wall, GVP-WM (WAN-0S), T=25: `0.86`
- Wall, GVP-WM (ORACLE), T=25: `1.00`

## Slurm Status

All submitted jobs completed with Slurm state `COMPLETED` and exit code `0:0`.

Oracle jobs:

- PushT oracle: `22475441`, elapsed `00:50:06`
- Wall oracle: `22475445`, elapsed `01:40:20`

WAN-0S jobs:

- PushT WAN-0S chunks: `22475459`, `22475460`, `22475462`, `22475463`, `22475464`
- Wall WAN-0S chunks: `22475465` through `22475474`
- Completed WAN chunk elapsed times were roughly `13-48` minutes.

No traceback, OOM, cancelled, or failed markers were found in the relevant logs.

## Results

| Task | Video Source | Episodes | Success | Success Rate | Paper Target | Main Error Metric |
|---|---:|---:|---:|---:|---:|---|
| PushT | ORACLE | 21 | 15 | `0.714` | `0.98` | mean block diff `8.77`, mean angle diff `0.223` |
| PushT | WAN-0S | 21 | 10 | `0.476` | `0.56` | mean block diff `21.69`, mean angle diff `0.366` |
| Wall | ORACLE | 50 | 23 | `0.460` | `1.00` | mean state dist `6.03` |
| Wall | WAN-0S | 50 | 22 | `0.440` | `0.86` | mean state dist `6.26` |

Dynamics residuals:

- PushT oracle: mean `0.187`, median `0.184`
- PushT WAN-0S: mean `21756.787`, median `22306.775`
- Wall oracle: mean `1.002`, median `0.361`
- Wall WAN-0S: mean `1.192`, median `0.469`

Failed episodes:

- PushT oracle: `5,6,8,9,11,12`
- PushT WAN-0S: `1,4,5,6,7,8,9,11,12,13,19`
- Wall oracle: `1,2,3,4,6,7,8,11,12,13,14,15,19,22,23,24,29,30,31,35,37,40,41,43,44,47,49`
- Wall WAN-0S: `3,4,5,6,7,9,11,12,13,14,15,17,18,19,20,23,26,29,30,31,35,37,40,41,43,44,47,49`

## Interpretation

The experiments are technically runnable end to end on Snellius, but the results do not reproduce the paper.

The most important signal is that oracle-video GVP-WM is already far below the paper's oracle upper bound. PushT oracle is `0.714` instead of `0.98`, and Wall oracle is `0.460` instead of `1.00`. Therefore the current bottleneck is not only WAN-0S video generation quality. The GVP-WM oracle planning/evaluation pipeline is still not paper-faithful or is being evaluated on a different episode distribution.

Wall is the clearest failure case. Wall WAN-0S (`0.440`) is almost the same as Wall oracle (`0.460`), while the paper reports Wall oracle `1.00` and Wall WAN-0S `0.86`. This suggests the main Wall issue is likely in action recovery, action/proprio normalization, objective scaling, evaluation-pair selection, or another planner/protocol mismatch, rather than zero-shot Wan videos alone.

PushT is mixed. WAN-0S reaches `0.476`, which is closer to the paper's WAN-0S target `0.56`, but this is over only 21 local episodes and the oracle ceiling is only `0.714`. The large PushT WAN-0S dynamics residuals also indicate that generated-video latents are much less dynamically compatible with the world model than oracle-video latents.

## Likely Protocol Mismatches To Audit

- The current oracle scripts default to `residual_reduction=mean`, while `config.py` labels `sum` as the paper-style dynamics residual reduction. This can materially change the ALM tradeoff.
- PushT is not using the paper's 50 initial-goal pairs because the local `val` split exposes only 21 eligible T=25 episodes.
- Wall uses first 50 local `wall_single/all` eligible episodes; this may not match the paper's sampled 50 expert initial-goal pairs.
- The planner code uses task-specific implementation choices such as action reparameterization and local refinement; these need a direct audit against the original implementation.
- Wall logs repeatedly show planner macro actions deviating from expert/oracle displacement even with oracle videos, so Wall action normalization/proprio handling remains a high-priority suspect.

## Recommended Next Step

Do not claim Table 2 reproduction from this run.

Before spending more compute on WAN-0S, fix the oracle sanity path:

1. Re-run small and full oracle sweeps with paper-style `residual_reduction=sum` and verify whether oracle approaches PushT `0.98` and Wall `1.00`.
2. Establish exact paper evaluation pairs, or create a documented local substitute if the original 50 pairs are unavailable.
3. For Wall, compare three rollouts on the same episodes: dataset expert replay, state-track upper bound, and GVP-WM oracle. This will isolate whether the remaining failure is environment/action scale or GVP-WM optimization.
4. For PushT, separate failures caused by block position from failures caused by wrapped angle error, since several oracle failures are orientation-driven.
5. Only after oracle-video GVP-WM is near the paper upper bound should WAN-0S be rerun and compared to Table 2.
