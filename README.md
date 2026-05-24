# Grounding Generated Videos Reproduction

This repository contains our DL2 reproduction and extension work for
"Grounding Generated Videos in Feasible Plans via World Models" (GVP-WM,
arXiv:2602.01960).

## Repository Layout

- `original_paper/`: paper sources and PDF snapshots used for reference.
- `dino_wm/`: imported DINO-WM code and environment/task wrappers.
- `reimplementation/`: the maintained GVP-WM reimplementation, tests, Snellius
  jobs, experiment utilities, and reproduction notes.
- `final_report_draft/`: LaTeX source for the course report draft.

## Reproduction Entry Points

Start with `reimplementation/README.md` for the package overview and local
smoke tests.

For the cluster experiments, fixed sampled-50 Table 2 protocol, WAN-0S/WAN-FT
video generation/evaluation jobs, rescue jobs, and result summarization, use:

- `reimplementation/docs/EXPERIMENT_REPRODUCTION.md`

Large external assets are intentionally not stored in Git. This includes
datasets, DINO-WM checkpoints, Wan2.1 checkpoints, LoRA checkpoints, generated
videos, JSON reports, Slurm logs, and report ZIP exports.
