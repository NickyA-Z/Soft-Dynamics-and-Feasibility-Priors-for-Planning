# Running the workflows

Run from the repository root in your DINO-WM Python environment:

```bash
python -m gvpwm.task.pusht.plan_pusht --help
python -m gvpwm.task.wall.plan_wall --help
python -m scripts.pusht.train_pusht --help
python -m scripts.pusht.build_pusht --help
python -m scripts.wall.build_wall --help
python -m scripts.wall.train_wall --help
```

Append your existing arguments in place of `--help`. Use module commands (`python -m`) so Python can find the project packages.

DINO-WM defaults to the repository's `dino_wm/` directory. Set `DINO_WM_ROOT` to use another checkout. Datasets, checkpoints and the dependencies in `dino_wm/environment.yaml` must be available separately.

PushT's `pusht_utils.py` is the dataset-building loader. `oracle_utils.py` preserves the original planner loader and its normalization; these are intentionally separate.

## Local environment settings

Install the project additions in your existing Python environment:

```bash
python -m pip install -r requirements.txt
```

The root `.env` loads automatically when `gvpwm` is imported. Edit
`DINO_WM_ROOT` to point to your checkout; relative paths resolve from the
repository root. Existing shell environment variables take precedence.
`.env` is ignored by Git; `.env.example` provides the shared defaults.
