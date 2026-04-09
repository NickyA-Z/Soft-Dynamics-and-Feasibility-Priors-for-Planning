# Maintainer Guide

This document explains what has been implemented in `reimplementation/`, how it
maps to the GVP-WM paper, what was deliberately left out, and what the most
important maintenance tasks are going forward.

It is written for future contributors who need to understand the current state
without re-deriving the design from the paper or reverse-engineering the code.

## 1. Scope of What Was Done

The work completed so far is a reimplementation of the **planning method** from
the paper:

- video-guided latent collocation
- augmented Lagrangian optimization
- smooth bounded action parameterization
- receding-horizon MPC execution
- optional local action refinement
- scale-invariant latent alignment to video guidance

The work **does not** reproduce the paper's exact experimental pipeline end to
end, because the required external artifacts are not present in this workspace:

- pretrained DINO-WM checkpoints
- Push-T and Wall datasets / evaluation harness
- Wan2.1 video generation pipeline
- LoRA fine-tuning code and weights for Wan
- the paper's exact benchmark scripts and metrics aggregation

The implementation was designed so those external assets can be plugged in
later without rewriting the core algorithm.

## 2. Current High-Level Architecture

The package is intentionally small and organized around two abstractions:

1. `WorldModelAdapter`
2. `GVPWMPlanner`

`WorldModelAdapter` provides a minimal interface around any latent
action-conditioned world model:

- `encode_observation`
- `encode_sequence`
- `predict_next_latent`
- inherited `rollout`

`GVPWMPlanner` implements the method from the paper on top of that interface.
It delegates the constrained optimization to `LatentCollocationSolver`.

The code is in:

- `src/gvpwm/interfaces.py`
- `src/gvpwm/planner.py`
- `src/gvpwm/solver.py`

## 3. File-by-File Overview

### Core package

- `src/gvpwm/config.py`
  - Dataclasses for solver, MPC, and refinement hyperparameters.
  - Keeps algorithm settings explicit and serializable.

- `src/gvpwm/interfaces.py`
  - Defines the adapter interface and result dataclasses.
  - Contains the generic `WorldModelAdapter.rollout()` method used by both
    solver and refinement.

- `src/gvpwm/losses.py`
  - Implements the scale-invariant alignment loss and terminal goal loss.
  - The alignment loss is the code version of the paper's normalized latent
    distance / cosine-equivalent objective.

- `src/gvpwm/utils.py`
  - Helper functions for history padding, warm starts, and simple image
    alignment utilities.

- `src/gvpwm/video.py`
  - Minimal video plan helpers.
  - Includes temporal resampling and lightweight motion blur utilities.
  - Video generation itself is not implemented here; this module only handles
    plan transport and alignment.

- `src/gvpwm/solver.py`
  - Implements the latent collocation problem and the augmented Lagrangian
    solver.
  - This is the main implementation of the paper's optimization core.

- `src/gvpwm/planner.py`
  - Wraps the solver in an MPC loop.
  - Handles video-plan encoding, optional local action refinement, warm starts,
    and receding-horizon execution.

### Adapters

- `src/gvpwm/adapters/dino_wm.py`
  - Adapter for a DINO-WM-style world model.
  - This is the bridge to the type of pretrained model the paper used.
  - It assumes a DINO-WM model object similar to the public reference code and
    maps its encode/predict APIs into `WorldModelAdapter`.

### Examples and tests

- `src/gvpwm/examples/toy_world.py`
  - A tiny deterministic linear world model and point-mass environment.
  - Used to validate solver and planner behavior without requiring external
    checkpoints.

- `src/gvpwm/examples/toy_demo.py`
  - End-to-end demo showing that infeasible video guidance can still be turned
    into executable actions.

- `tests/test_losses.py`
  - Confirms scale invariance of the latent alignment loss.

- `tests/test_solver.py`
  - Confirms the solver can hit the goal in the toy system.
  - Confirms the MPC loop executes the full horizon and reaches the target.

### Setup

- `environment.yml`
  - Conda environment definition for `dl2`.

- `pyproject.toml`
  - Package metadata and editable install support.

- `scripts/run_toy_demo.py`
  - Convenience launcher for the demo.

## 4. Mapping from Paper to Code

This section is the most important one for maintainers.

### 4.1 Video guidance

Paper concept:

- Generate or obtain a video plan.
- Encode it into the world model latent space.
- Use it both to initialize the trajectory and to guide optimization.

Code mapping:

- `VideoPlan`, `VideoPlanSource` in `interfaces.py`
- `_encode_video()` in `planner.py`
- `temporal_resample_sequence()` in `video.py`
- `_initialize_latents()` in `solver.py`
- `_objective()` in `solver.py`

Current behavior:

- The planner accepts either:
  - a `VideoPlanSource`, or
  - a precomputed `VideoPlan`
- The video may already be encoded or may be raw observations.
- The encoded latent sequence is resampled to `horizon + 1`.

### 4.2 Scale-invariant alignment loss

Paper concept:

- Latent magnitudes from generated video may drift relative to latent states
  from the world model.
- Normalize embeddings and compare directions instead of raw magnitudes.

Code mapping:

- `safe_l2_normalize()` in `losses.py`
- `scale_invariant_alignment()` in `losses.py`

Current behavior:

- Each latent tensor is flattened, normalized, and compared with squared error.
- This is equivalent to a cosine-style alignment penalty up to constants.

### 4.3 Direct collocation in latent space

Paper concept:

- Optimize both latent states and actions.
- Do not treat the video latents as executable waypoints.

Code mapping:

- `LatentCollocationSolver.solve()` in `solver.py`

Current behavior:

- The optimization variables are:
  - latent states `z_1 ... z_T`
  - actions `a_0 ... a_{T-1}`
- `z_0` is clamped to the current latent state.
- There is an ablation switch `fix_states_to_video` that reproduces the
  "no collocation" behavior by not optimizing latent states.

### 4.4 Dynamics constraints

Paper concept:

- World model dynamics are enforced as hard constraints.
- The paper solves this with an augmented Lagrangian.

Code mapping:

- `_dynamics_residuals()` in `solver.py`
- augmented objective inside `solve()` in `solver.py`

Current behavior:

- For each timestep, the solver computes:
  - predicted next latent from the adapter
  - residual = optimized next latent - predicted next latent
- The residuals are accumulated into the augmented Lagrangian with:
  - dual term
  - quadratic penalty term

### 4.5 Augmented Lagrangian method

Paper concept:

- Alternate between primal optimization and dual updates.
- Gradually increase the penalty weight `rho`.

Code mapping:

- outer and inner loops in `solve()` in `solver.py`
- `ALMConfig` in `config.py`

Current behavior:

- Inner loop:
  - Adam updates on latent and action parameters
- Outer loop:
  - multiplier update
  - geometric increase of `rho`

Important note:

- The implementation is intentionally straightforward and readable.
- It is not yet optimized for throughput or for large-batch evaluation.

### 4.6 Smooth bounded action parameterization

Paper concept:

- Actions are constrained to valid bounds.
- The paper notes a smooth parameterization is better than projected SGD.

Code mapping:

- `_actions_from_parameter()` in `solver.py`
- `_raw_parameter_from_actions()` in `solver.py`

Current behavior:

- If enabled, the solver optimizes unconstrained action parameters and maps them
  to bounded actions with `tanh`.
- This mirrors the paper's "smooth action reparameterization" choice.

### 4.7 MPC execution

Paper concept:

- Replan from the current state after executing only a short prefix.

Code mapping:

- `run_mpc()` in `planner.py`
- `MPCConfig` in `config.py`

Current behavior:

- The planner:
  - solves the collocation problem over the remaining horizon
  - executes `execution_stride` actions
  - re-encodes the actual next observation from `step_fn`
  - warm-starts the next solve

### 4.8 Local refinement

Paper concept:

- After optimization, optionally sample around the solution and choose a better
  action sequence by world-model rollout.

Code mapping:

- `_refine_actions()` in `planner.py`
- `RefinementConfig` in `config.py`

Current behavior:

- Samples noisy candidates around the optimized action sequence.
- Rolls them out with the adapter.
- Selects the candidate with lowest terminal goal loss.

## 5. Runtime Data Model

Maintainers should keep these shape assumptions in mind.

### Generic planner assumptions

- Latent trajectory:
  - shape `(T + 1, ...)`
- Action trajectory:
  - shape `(T, action_dim)`
- Latent history passed to the adapter:
  - shape `(history_length, ...)`
- Action history passed to the adapter:
  - shape `(history_length - 1, action_dim)` for history-based models

The planner does not impose a specific latent layout. The latent can be:

- a vector
- patch tokens
- patch tokens plus proprio token
- any other structured tensor

All losses flatten the latent event dimensions when needed.

### DINO-WM adapter assumptions

The adapter in `adapters/dino_wm.py` assumes a DINO-WM-like model with:

- `num_hist`
- `concat_dim`
- `num_action_repeat`
- `num_proprio_repeat`
- `action_dim`
- `encode_obs()`
- `encode_act()`
- `predict()`

It expects dict-like observations with at least:

- `visual`
- optional `proprio`

The adapter tries to be conservative, but this part is the most likely place
for future integration work if the external DINO-WM checkpoint format differs.

## 6. What Was Intentionally Left Minimal

Several pieces are deliberately lightweight rather than "fully productized."

### 6.1 Video generation

There is no built-in Wan integration yet.

Reason:

- the workspace does not contain the model, weights, pipeline, or prompts
- the main goal here was to reimplement the paper's planner rather than a
  specific external video model stack

Consequence:

- maintainers must provide a `VideoPlanSource` or precomputed `VideoPlan`

### 6.2 Evaluation harness

There is no reproduction script for Push-T / Wall.

Reason:

- the datasets, checkpoints, and benchmark runners are absent

Consequence:

- current validation is algorithmic rather than benchmark-level

### 6.3 Batch planning and acceleration

The implementation favors clarity over raw speed.

Current limitations:

- solver is single-trajectory oriented
- loops are explicit
- no vectorized multi-episode planner
- no mixed precision, no JIT, no custom kernels

This is acceptable for maintainability right now, but if the next task is full
benchmark reproduction, this is an obvious optimization target.

## 7. Validation Performed So Far

The following was completed and verified:

- created the `dl2` conda environment from `environment.yml`
- installed the package editable into that environment
- ran the test suite in `dl2`
- ran the toy end-to-end demo in `dl2`

Verified commands:

```bash
conda run -n dl2 python -m pytest tests
conda run -n dl2 python scripts/run_toy_demo.py
```

Observed result:

- all tests passed
- the toy demo reached the goal with very small final error

This validates:

- the alignment loss
- the collocation solver
- the MPC loop
- the refinement path
- the package/environment setup

It does **not** validate:

- DINO-WM checkpoint compatibility against a real external checkpoint
- Wan video generation integration
- Push-T / Wall success rates from the paper

## 8. Known Caveats and Integration Risks

These are the main things a maintainer should scrutinize before using this on
real checkpoints.

### 8.1 DINO-WM latent packing details

The DINO-WM adapter was written against the public reference structure, but
checkpoint-specific packaging may still differ.

Watch for:

- proprio shape mismatches
- action embedding shape mismatches
- differences in how action-conditioning is inserted into tokens
- differences between training-time and inference-time preprocessor logic

### 8.2 Observation preprocessing

The paper mentions:

- symmetric padding for aspect-ratio mismatch
- cropping before world-model encoding
- temporal interpolation or subsampling

Only the generic utilities are present right now. No end-to-end preprocessing
pipeline has been wired to a real video generator or DINO-WM checkpoint yet.

### 8.3 Action scaling

The planner assumes action bounds are already known and expressed in the same
scale expected by the world model.

If the external model expects normalized actions and the environment uses raw
actions, maintainers must insert the correct transform in the adapter or call
site.

### 8.4 No environment-specific metrics

The package does not currently define:

- Push-T success computation
- Wall success computation
- benchmark logging / aggregation

Those belong in a future evaluation layer, not inside the core planner.

## 9. Most Likely Next Tasks

If maintenance continues, the next tasks should probably happen in this order.

### Priority 1: real DINO-WM integration

Add a script that:

- loads a public DINO-WM checkpoint
- instantiates `DinoWorldModelAdapter`
- runs one planner call on a stored example trajectory

Goal:

- verify the adapter against real checkpoint tensors

### Priority 2: real video-plan ingestion

Add one of:

- a loader for saved video frames from disk
- a loader for saved latent video plans
- an adapter around an actual video-generation pipeline

Goal:

- replace the toy `PrecomputedVideoPlanSource` usage with realistic inputs

### Priority 3: evaluation harness

Build scripts for:

- Push-T evaluation
- Wall evaluation
- ablation toggles

The current config objects already expose the important knobs:

- video init on/off
- video loss on/off
- fixed states vs collocation
- smooth action parameterization on/off
- MPC and refinement controls

### Priority 4: performance work

Only after correctness is established on real models:

- batch multiple planning instances
- profile bottlenecks
- vectorize residual computation
- reduce repeated encoding work

## 10. Practical Tips for Future Maintainers

- Treat `solver.py` as the ground truth for the optimization logic.
- Treat `planner.py` as orchestration around that solver.
- Keep adapters thin. Do not put planner logic inside the world-model adapter.
- Preserve the toy example. It is the fastest way to catch algorithmic
  regressions without needing external infrastructure.
- If you change latent packing or adapter behavior, add an integration test for
  the exact tensor shapes involved.
- If you introduce environment-specific code, keep it outside `src/gvpwm/`
  unless it is genuinely planner-generic.

## 11. Short Summary

Current state:

- the paper's core planner is implemented
- the package is installable
- the `dl2` conda env exists and works
- toy validation is passing

Current gap:

- benchmark reproduction still needs real checkpoints, data, and video
  generation assets

The implementation is therefore best understood as a **maintainable reference
implementation of GVP-WM's algorithmic core**, not yet as a full experimental
reproduction package.
