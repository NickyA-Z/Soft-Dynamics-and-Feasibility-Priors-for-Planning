# Feasibility-only planner

This folder is an isolated planning implementation. It does not modify or copy
training code. Existing DINO-WM loading, encoding, oracle-data, and environment
helpers are imported from `local/`; checkpoint scoring is wrapped behind one
normalization boundary.

## Structure

- `models/`: thin wrappers around the frozen DINO-WM and feasibility checkpoint.
- `objectives/`: goal, DSM, contrastive, and regularization terms.
- `optimization/`: bounded actions, alternating/joint optimization, best iterate.
- `planners/fixed_transition.py`: one-step action recovery with states fixed.
- `planners/open_loop.py`: horizon action recovery with every latent fixed.
- `planners/free_energy.py`: free latent/action feasibility-only planning.
- `planners/mpc.py`: replans from actual observations; references must be regenerated.
- `evaluation/`: action and environment recovery diagnostics.
- `run.py`: staged oracle diagnostic entry point.

## Required validation order

First run fixed-transition recovery:

```bash
$PYTHON -u -m codex.run \
  --mode fixed --horizon 5 --episode-idx 2 \
  --feasibility-checkpoint "$HOME/checkpoints/feasibility_pusht/better_varcon_final.pt"
```

It prints `E_fixed(expert)`, `E_fixed(zero)`, `E_fixed(planner)`, action MSE, and
cosine similarity. The fixed test must discriminate and recover actions before
moving on:

```bash
$PYTHON -u -m codex.run --mode open_loop --horizon 5 \
  --feasibility-checkpoint "$HOME/checkpoints/feasibility_pusht/better_varcon_final.pt"
```

Only after that succeeds, test free latent planning with `--mode free`. MPC is
available as `codex.planners.MPCPlanner`; its executor callback owns environment
stepping. It always re-encodes the actual observation, and any target-video
reference must come from a callback that is invoked anew at every MPC step.

## Important conventions

Actions passed to `FeasibilityScorer` are already normalized exactly as during
DINO-WM/feasibility training. Latents are raw encoder latents and are normalized
exactly once inside the scorer. DSM defaults to the paper's clean-query energy
`mean(e_theta(h,a,z_next,sigma)^2)`; stochastic probing is optional and uses
fixed noise during an optimization solve.

Action optimization uses the dataset-derived lower and upper bound of each
individual macro-action coordinate. It does not collapse those bounds into one
global scalar range.

## Planner-aligned training

The new training entry point is `codex.training.run`. It reuses the existing
`extension.feasibility2` tensor dataset, transformer model, DSM loss, sigma
scheduler, AMP helper, and checkpoint layout. No training implementation under
`extension/` or `local/` is changed.

The new contrastive portion is deliberately action-only: history and next
latent remain fixed while actions are perturbed or optimized. Training has
three stages within one run:

1. DSM warmup.
2. Multi-scale local action ranking.
3. Online bounded adversarial action mining using the current energy head.

Every epoch is validated by minimizing energy over actions on a separate
episode-disjoint dataset. The best checkpoint is selected using recovery MSE,
cosine similarity, and whether the expert still beats the optimized action.

Build two dataset artifacts with `extension.feasibility2.build`, for example
episodes 0--499 for training and 500--549 for validation. The validation build
must use the training artifact through `--norm-stats`, so both contain identical
latent normalization statistics. The trainer rejects overlapping episode
ranges or mismatched statistics.

After setting `TRAIN_DATASET` and `VAL_DATASET` in the job file, submit:

```bash
sbatch codex/jobs/train_planner_aligned.sbatch
```

The selected checkpoint is `planner_aligned.pt`. The accompanying
`planner_aligned_final.pt` is only the last epoch and should not be preferred
automatically.

This setup is more expensive because mining runs inner action-optimization
steps. With the supplied defaults (three starts, 12 steps every fourth batch),
expect roughly 10--20 times the cost of DSM-only training. The existing mixed
contrastive implementation already evaluates many negatives, so relative to
that full DSM+contrastive setup the likely increase is closer to 1.5--3 times.
Actual cost depends strongly on attention memory, batch size, and GPU
utilization; the first epoch should be timed before requesting a long run.
