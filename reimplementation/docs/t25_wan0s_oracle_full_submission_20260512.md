# T=25 WAN-0S and Oracle Full Submission Memo

Date: 2026-05-12

## Scope

Submitted the full T=25 reproduction jobs for the Table 2 GVP-WM settings:

- PushT, GVP-WM (ORACLE), T=25
- Wall, GVP-WM (ORACLE), T=25
- PushT, GVP-WM (WAN-0S), T=25
- Wall, GVP-WM (WAN-0S), T=25

This submission does not include WAN-FT. WAN-FT is a separate Table 2 condition and should be submitted/analyzed separately.

## Evaluation Specs

To avoid selected-episode sanity bias, I generated fixed 50-pair `episode:offset` specs with `sample_seed=99`, `raw_horizon=25`, and `frame_skip=5`. The same specs are shared by ORACLE and WAN-0S for each task.

Spec files:

- PushT all 50: `/gpfs/home2/scur0196/dl2runtime/specs/t25_pusht_seed99_50.specs`
- PushT chunks of 5: `/gpfs/home2/scur0196/dl2runtime/specs/t25_pusht_seed99_50_chunks5.txt`
- Wall all 50: `/gpfs/home2/scur0196/dl2runtime/specs/t25_wall_seed99_50.specs`
- Wall chunks of 5: `/gpfs/home2/scur0196/dl2runtime/specs/t25_wall_seed99_50_chunks5.txt`

Both generated spec sets have 50 entries and no duplicate `episode:offset` pairs.

## Submitted Jobs

| Setting | Job ID | Script |
| --- | --- | --- |
| PushT ORACLE T=25 sampled-50 | `22693196` | `/gpfs/home2/scur0196/dl2runtime/jobs/30_pusht_oracle_sampled50_t25.slurm` |
| Wall ORACLE T=25 sampled-50 | `22693200` | `/gpfs/home2/scur0196/dl2runtime/jobs/31_wall_oracle_sampled50_t25.slurm` |
| PushT WAN-0S T=25 sampled-50 | `22693202_[0-9]` | `/gpfs/home2/scur0196/dl2runtime/jobs/32_wan0s_pusht_sampled50_t25_array.slurm` |
| Wall WAN-0S T=25 sampled-50 | `22693205_[0-9]` | `/gpfs/home2/scur0196/dl2runtime/jobs/33_wan0s_wall_sampled50_t25_array.slurm` |

Repo copies:

- `reimplementation/scripts/snellius/30_pusht_oracle_sampled50_t25.slurm`
- `reimplementation/scripts/snellius/31_wall_oracle_sampled50_t25.slurm`
- `reimplementation/scripts/snellius/32_wan0s_pusht_sampled50_t25_array.slurm`
- `reimplementation/scripts/snellius/33_wan0s_wall_sampled50_t25_array.slurm`

## Key Settings

- Raw horizon: `T=25`
- World-model frame skip: `5`
- Macro horizon: `5`
- ALM: `inner=25`, `outer=25`, `learning_rate=0.05`
- Local refinement: `500` samples, variance `0.3`
- WAN-0S model: local `Wan2.1-FLF2V-14B-720P`
- WAN-0S generation: `81` frames, `1280*720`, `50` sampling steps, shift `16`, guidance scale `5`
- WAN-0S chunks: 10 array tasks per environment, 5 pairs per task
- Wall target source: `env-replay`

Note: the WAN-0S array jobs request `220G` memory, so Slurm warned that they are charged at the 2-GPU accounting level even though each job uses one GPU process. I kept the submitted jobs because many tasks immediately obtained H100 resources and started running.

## Expected Outputs

Oracle JSONs:

- `/gpfs/home2/scur0196/dl2runtime/reports/oracle_pusht_t25_seed99_50_22693196.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/oracle_wall_t25_seed99_50_22693200.json`

WAN-0S videos:

- `/gpfs/home2/scur0196/dl2runtime/videos/wan0s_pusht_s50_t25_22693202`
- `/gpfs/home2/scur0196/dl2runtime/videos/wan0s_wall_s50_t25_22693205`

WAN-0S JSONs:

- `/gpfs/home2/scur0196/dl2runtime/reports/wan0s_pusht_t25_*.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/wan0s_wall_t25_*.json`

The WAN-0S result JSON names are generated inside the array tasks using Slurm job IDs, so aggregate by matching job/log timestamps and generated-video roots.

## Monitoring Commands

```bash
squeue -u scur0196 -o '%.18i %.9P %.45j %.8T %.10M %.9l %.6D %R' | rg '22693196|22693200|22693202|22693205|dl2_'
```

```bash
sacct -j 22693196,22693200,22693202,22693205 --format=JobID,JobName%40,Partition,State,Elapsed,ExitCode,Start,End -P
```

```bash
tail -f /gpfs/home2/scur0196/dl2runtime/logs/dl2_wan0s_pusht_s50_t25_22693202_0.out
tail -f /gpfs/home2/scur0196/dl2runtime/logs/dl2_wan0s_wall_s50_t25_22693205_0.out
```

## Important Caveats

- This is the first full sampled-50 WAN-0S run with 50-step Wan generation, so runtime may be much longer than the earlier 8-step sanity jobs.
- PushT ORACLE uses the paper-style action reparameterization path, not the direct-action diagnostic variant.
- Wall ORACLE now writes a compact JSON summary; this required adding `--output-json` support to `dino_wall_oracle_demo.py`.
- These jobs are intended to produce comparable full sampled-50 numbers, but the sampled seed may still differ from the paper authors' exact hidden 50-pair set.

