# Soft Dynamics and Feasibility Priors for Grounding Generated Video Plans

Code for the experiments in **Soft Dynamics and Feasibility Priors for Grounding
Generated Video Plans**.

The project investigates how visual trajectories can be converted into executable
actions using DINO-WM world-model dynamics and learned feasibility objectives.

## Overview

The experiments compare four planning methods:

1. **GVP-WM** — latent collocation with augmented-Lagrangian dynamics constraints.
2. **Soft dynamics** — latent collocation with a fixed DINO-WM dynamics penalty.
3. **Recursive rollout** — action-only optimization through recursive DINO-WM rollouts,
   using multi-chain Langevin–Adam search.
4. **Learned feasibility prior** — a model combining denoising score matching,
   latent-transition prediction, and contrastive feasibility scoring.

All methods use video guidance and receding-horizon model-predictive control.
The main experiments use oracle video guidance from expert trajectories to isolate
physical grounding and action recovery.

## Environments

Experiments are evaluated in:

- **Push-T** — contact-rich manipulation requiring the agent to move a T-shaped
  object to a target pose.
- **Wall** — navigation with obstacle avoidance.

Planning horizons are:

- Push-T: `T ∈ {25, 50, 80}`
- Wall: `T ∈ {25, 50}`

The repository also includes feasibility-model diagnostics, soft-dynamics
ablations, recursive-rollout search sweeps, and auxiliary feasibility experiments.

## Main findings

The reported experiments show that:

- Action-conditioned DINO-WM dynamics provide effective grounding for Push-T.
- Soft dynamics is substantially faster than GVP-WM and more robust at longer
  Push-T horizons.
- Recursive rollout performs competitively at short horizons but becomes harder
  to optimize as the horizon increases.
- Offline feasibility ranking does not necessarily translate into successful
  action recovery.
- The learned feasibility prior performs poorly on contact-rich Push-T but is
  substantially more effective on Wall.
- Adding the learned feasibility objective to soft dynamics reduces performance
  under the evaluated equal-weight configuration.

## Repository structure

```text
.
├── .env.example
├── dino_wm/                 # DINO-WM code and environment wrappers
├── reimplementation/        # GVP-WM reproduction and planner experiments
├── extension/               # Feasibility-prior and residual experiments
├── original_paper/          # Paper sources and reference material
└── final_report_draft/      # LaTeX report draft
```

Large assets are not included in Git. These may include datasets, model
checkpoints, generated videos, experiment logs, JSON results, and Slurm
artifacts.

## Setup

Create the environment used by the reimplementation:

```bash
cd reimplementation
conda env create -n dl2 -f environment.yml
conda activate dl2
```

Install or configure the imported DINO-WM environment using the setup files
under `dino_wm/`.

Create the local environment configuration from the example:

```bash
cp .env.example .env
```

The available variables are:

```text
DINO_WM_ROOT=./dino_wm
WALL_VERBOSE_DIAGNOSTICS=0
PUSHT_ORACLE_PROPRIO_GUIDANCE=0
```

`DINO_WM_ROOT` should point to the DINO-WM repository or checkout used by the
experiments. The diagnostic variables enable optional Wall and Push-T
experiment behavior.

## Testing

Run the reimplementation test suite with:

```bash
cd reimplementation
conda run -n dl2 python -m pytest
```

For a quick local validation, use the smoke-test instructions in
`reimplementation/README.md`.

## Running experiments

The maintained experiment entry points and configuration files are located in:

- `reimplementation/` — GVP-WM, soft-dynamics, and rollout planners.
- `extension/` — feasibility-prior datasets, training, and evaluation.
- `reimplementation/docs/EXPERIMENT_REPRODUCTION.md` — cluster reproduction
  procedures, Slurm jobs, result collection, and table-generation commands.

Start with the relevant README or reproduction document before launching a
full experiment. Most experiments require pretrained DINO-WM checkpoints,
prepared datasets, and task-specific configuration files.

## Feasibility-prior data format

The feasibility extension expects tensors with the following shapes:

```text
histories:              [N, 3, 196, 394]
actions:                [N, 10]
next_latents:           [N, 196, 394]
past_action_histories:  [N, 3, 10]
latent_mean:            [196, 394]
latent_std:             [196, 394]
action_mean:            [2]
action_std:             [2]
```

The latent representation is the packed DINO-WM latent. Each macro-action
contains five two-dimensional actions, giving an action dimension of `10`.

## Reproduction notes

The main reported configurations use:

- Oracle expert video guidance.
- One macro-action executed before replanning.
- DINO-WM latent trajectories subsampled from the raw video trajectory.
- Adam optimization for collocation-based planners.
- Multi-chain Langevin–Adam optimization for recursive rollout.
- A combined DSM, transition, and contrastive objective for the learned
  feasibility prior.

Exact learning rates, horizon mappings, action bounds, checkpoint locations,
Slurm commands, and evaluation settings are documented in
`reimplementation/docs/EXPERIMENT_REPRODUCTION.md` and the configuration files.

## Limitations

The reported study uses a limited number of simulated evaluation episodes and
primarily oracle video guidance. Results should therefore be interpreted as an
evaluation of physical grounding and action recovery rather than a complete
assessment of generated-video planning.

Important limitations include:

- Limited evaluation-set sizes.
- Single optimization seeds for the main configurations.
- Dependence on pretrained DINO-WM checkpoints.
- Sensitivity to planner hyperparameters.
- Predefined corruption types for offline feasibility evaluation.
- No claim that offline feasibility ranking guarantees executable plans.

## Reference

This repository accompanies:

> **Soft Dynamics and Feasibility Priors for Grounding Generated Video Plans**

It also contains reproduction code related to:

> **Grounding Generated Videos in Feasible Plans via World Models**
> (GVP-WM).