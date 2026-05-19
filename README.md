# DL2 — Grounding Generated Videos

This repository contains our project code for reproducing and extending the idea of grounding generated videos with learned world models. The project focuses on planning in DINO-WM latent space and testing whether additional feasibility models can make generated latent trajectories executable in environments such as Push-T.

## Project overview

The main goal is to use generated video plans as high-level guidance, then optimize action sequences that make the generated latent trajectory feasible under a learned world model or feasibility model.

The current project includes:

- DINO-WM latent-space planning
- ALM-style latent collocation planning
- Soft DINO-WM dynamics planning
- Rollout-mode action-only planning
- DSM feasibility models
- Transformer feasibility models
- Transition / delta feasibility heads
- Paired DINO-WM residual experiments
- Push-T evaluation scripts

## Repository structure EXSTENSION

The most relevant code is in:

```text
nicky_dl/
├── dino_wm/
│   ├── build.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── feasibility2/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── planner.py
│
└── ...
```

## Important files
nicky_dl/feasibility2/train.py       Train DSM / transformer feasibility models
nicky_dl/feasibility2/evaluate.py    Evaluate feasibility model diagnostics
nicky_dl/feasibility2/model.py       MLP and transformer feasibility models
nicky_dl/feasibility2/planner.py     Feasibility-based planning utilities

## Setup 
conda activate dino_wm

## Notes
Execpted Data: 
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