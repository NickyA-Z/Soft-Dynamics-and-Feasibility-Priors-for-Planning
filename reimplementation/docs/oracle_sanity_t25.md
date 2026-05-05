# Oracle Sanity T=25 Status

Date: 2026-05-04

## Goal

Reproduce the Table 2 oracle upper bounds before WAN-0S:

- PushT, GVP-WM (ORACLE), T=25: paper target 0.98.
- Wall, GVP-WM (ORACLE), T=25: paper target 1.00.

## Code/Runtime Changes Made

- Added `--raw-horizon` and `--frame-skip` handling so paper T=25 with frame skip 5 runs as macro horizon 5.
- Switched PushT reporting to a block-pose success metric (`block_diff < 20`, `angle_diff < pi/9`) while keeping legacy env metrics in logs.
- Patched Wall oracle loss to include full latent/proprio instead of visual-only adapter loss for oracle diagnostics.
- Added diagnostic knobs to PushT/Wall oracle scripts:
  - `--lambda-action`
  - `--lambda-action-prior`
  - `--lambda-goal`
  - `--lambda-video`
  - `--residual-reduction`
  - `--fix-states-to-video`
  - `--disable-refinement`
  - `--disable-action-reparameterization`
  - PushT only: `--expert-action-warmstart`
- Updated external Slurm scripts in `~/dl2runtime/jobs` to accept extra args through `PUSHT_EXTRA_ARGS`, `WALL_EXTRA_ARGS`, `PUSHT_EPISODE_IDS`, and `WALL_EPISODE_IDS`.

Validation:

- `py_compile` passed for modified planner and oracle scripts.
- `pytest reimplementation/tests/test_argparse.py reimplementation/tests/test_mpc_rollout.py reimplementation/tests/test_dino_wm_rollout.py -q` passed.

## Slurm Jobs

- `22468083`: PushT raw T=25 default oracle; canceled after ep0-6 once failures made 0.98 impossible locally.
- `22468259`: Wall raw T=25 default oracle with full latent/proprio loss; canceled after ep1 failed.
- `22468514`, `22468515`, `22468516`: Wall ep1 diagnostics for lambda/fixed-video/no-refinement.
- `22468630`, `22468631`, `22468632`: Wall ep0/ep1 matrix diagnostics.
- `22468845`: Wall 50-episode candidate with `--fix-states-to-video --lambda-action 0.2`; canceled after ep3 failed.
- `22468633`, `22468634`, `22468635`: PushT ep5/ep6 fixed-video/lambda diagnostics.
- `22468902`, `22468903`, `22468904`: PushT ep5/ep6 direct-action diagnostics.
- `22469134`, `22469136`: PushT ep5/ep6 expert-action-warmstart diagnostics.

## Results

PushT:

- Default raw T=25: ep0-4 succeeded under block-pose metric, ep5 and ep6 failed; local rate at that point was 5/7 and could not reach 0.98 on the 21-episode local val set.
- Expert replay check on ep5/ep6 succeeds nearly exactly, so env action scale and frame skip are correct.
- `fix_states_to_video`, `lambda_action=0`, direct action optimization, action-prior diagnostics, and expert-action warm-start all still failed on ep5/ep6.
- Failure mode: planner actions do not contact/move the block on episodes where the oracle block actually moves. ALM washes even expert warm-start actions back toward actions that have low latent objective but poor env rollout.

Wall:

- Default raw T=25 with full latent/proprio loss: ep0 succeeded, ep1 failed.
- `--fix-states-to-video --lambda-action 0.2` with refinement succeeded on ep0/ep1.
- The same candidate failed on ep3 in the full selected set, so it is not a stable 1.00 reproduction.
- Expert replay filtering/scale check confirms selected episode IDs are replayable by environment actions; the issue is planner inverse-action recovery from oracle video.

## Current Conclusion

The project cannot currently reproduce Table 2 oracle upper bounds for PushT 0.98 / Wall 1.00 at T=25 using the existing GVP-WM oracle planner.

The core issue is no longer basic frame skip or env action scaling. Those are now corrected and checked. The remaining issue is planner/objective alignment: the latent collocation objective can prefer low-residual latent trajectories whose decoded/executed actions do not accomplish the physical push/navigation in the real env. This is clearest on PushT ep5/ep6, where expert actions replay successfully but ALM-planned actions leave the block essentially unmoved.

## Next Plan Before WAN-0S

The user asked to proceed with WAN-0S regardless of oracle accuracy. These oracle fixes are therefore paused, but the next oracle-specific plan is:

1. Add an oracle rollout evaluator that always reports expert replay, planner replay, and teacher-forced WM residual side by side for the same episode IDs.
2. Add action-init strategies derived from oracle proprio/video rather than dataset expert actions.
3. Add a proprio/state-weighted objective, not just full-latent mean MSE.
4. Add solver safeguards for warm starts, including optional proximal action prior around initialized actions.
5. Re-run oracle sanity on PushT local 21 val episodes and Wall first 50 expert-replay-successful IDs.
