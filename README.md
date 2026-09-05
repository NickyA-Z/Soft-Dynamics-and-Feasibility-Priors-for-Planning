# Soft-Dynamics-and-Feasibility-Priors-for-Grounding-Generated-Video-Plans
World-model planning with DINO-WM and learned feasibility priors, including dataset building, model training, and PushT and Wall experiments.
# DL2 - Grounding Generated Videos

This repository contains our DL2 reproduction and extension work for
"Grounding Generated Videos in Feasible Plans via World Models" (GVP-WM,
arXiv:2602.01960). The project focuses on planning in DINO-WM latent space and
testing whether generated video plans can be grounded into executable action
sequences.

## Project Overview

The main workflow uses generated video plans as high-level guidance, then
optimizes action sequences that make the generated latent trajectory feasible
under a learned world model or feasibility model.

The current project includes:

- DINO-WM latent-space planning.
- ALM-style latent collocation planning.
- Soft DINO-WM dynamics planning.
- Rollout-mode action-only planning.
- DSM and transformer feasibility models.
- Transition / delta feasibility heads.
- Paired DINO-WM residual experiments.
- PushT and Wall evaluation scripts.

## Repository Layout

- `original_paper/`: paper sources and PDF snapshots used for reference.
- `dino_wm/`: imported DINO-WM code and environment/task wrappers.
- `reimplementation/`: the maintained GVP-WM reimplementation, tests, Snellius
  jobs, experiment utilities, and reproduction notes.
- `extension/`: feasibility-model and residual-extension experiments.
- `final_report_draft/`: LaTeX source for the course report draft.

## Reproduction Entry Points

Start with `reimplementation/README.md` for the package overview and local
smoke tests.

For the cluster experiments, fixed sampled-50 Table 2 protocol, WAN-0S/WAN-FT
video generation/evaluation jobs, rescue jobs, and result summarization, use:

- `reimplementation/docs/EXPERIMENT_REPRODUCTION.md`

For the feasibility-model extension, including dataset construction, training, evaluation, and checkpoint paths, see `extension/README.md`.

Large external assets are intentionally not stored in Git. This includes
datasets, DINO-WM checkpoints, Wan2.1 checkpoints, LoRA checkpoints, generated
videos, JSON reports, Slurm logs, and report ZIP exports.


## Local Setup

For the reimplementation package:

```bash
cd reimplementation
conda env create -n dl2 -f environment.yml
conda run -n dl2 python -m pytest
```

For the imported DINO-WM environment, use the environment files and install
scripts under `dino_wm/`.

## Feasibility Extension Data Format

The feasibility extension expects tensors with these shapes:

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
