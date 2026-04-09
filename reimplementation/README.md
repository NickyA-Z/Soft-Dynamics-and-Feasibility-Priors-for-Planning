# GVP-WM Reimplementation

This directory contains a clean reimplementation of the core algorithm from
"Grounding Generated Videos in Feasible Plans via World Models" (GVP-WM,
arXiv:2602.01960).

The paper's exact Push-T and Wall results depend on external assets that are
not in this workspace:

- Pretrained DINO-WM checkpoints
- Push-T and Wall datasets / evaluation setups
- Wan2.1 image-to-video generation and LoRA fine-tuning

What is implemented here is the paper's planning method itself:

- Video-guided latent collocation
- Scale-invariant latent video alignment
- Augmented Lagrangian optimization with primal-dual updates
- Smooth bounded action parameterization via `tanh`
- Receding-horizon MPC execution
- Optional local action refinement
- A DINO-WM adapter for plugging into a pretrained DINO-WM model
- A fully runnable toy example and tests

## Layout

- `src/gvpwm/`: core package
- `src/gvpwm/adapters/dino_wm.py`: adapter for DINO-WM-style latent world models
- `src/gvpwm/examples/toy_world.py`: lightweight linear world model and environment
- `src/gvpwm/examples/toy_demo.py`: runnable end-to-end demo
- `docs/MAINTAINERS.md`: detailed implementation and maintenance notes
- `tests/`: unit tests for the optimizer and MPC loop
- `scripts/run_toy_demo.py`: convenience launcher

For a detailed explanation of what has been implemented, what is still missing,
and where to extend the code safely, see
[`docs/MAINTAINERS.md`](docs/MAINTAINERS.md).

## Environment

Create the conda environment from inside this directory:

```bash
conda env create -n dl2 -f environment.yml
```

Then run:

```bash
conda run -n dl2 python -m pytest
conda run -n dl2 python scripts/run_toy_demo.py
```

## Quick Start

The package centers around two abstractions:

1. `WorldModelAdapter`: exposes `encode_observation`, `encode_sequence`, and
   `predict_next_latent`.
2. `GVPWMPlanner`: runs collocation once or in MPC mode.

For a pretrained DINO-WM model, wrap it with `DinoWorldModelAdapter`. For a toy
or custom model, subclass `WorldModelAdapter`.

## Notes

- Actions are assumed to already be in the world model's expected scale.
- Temporal alignment between the video plan and the world model horizon is done
  through linear latent interpolation / subsampling.
- Spatial alignment helpers for aspect-ratio padding and center cropping are
  included, matching the paper's preprocessing description.
