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
