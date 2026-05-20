# Table 2 WAN-0S and Oracle Full Submission

Date: 2026-05-16

## Scope

Submitted complete sampled-50 jobs for the GVP-WM rows in the original paper's Table 2:

- GVP-WM (WAN-0S): PushT `T=25,50,80`; Wall `T=25,50`
- GVP-WM (ORACLE): PushT `T=25,50,80`; Wall `T=25,50`

This submission does not include WAN-FT.

## Paper Targets

| Method | PushT T=25 | PushT T=50 | PushT T=80 | Wall T=25 | Wall T=50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GVP-WM (WAN-0S) | 0.56 | 0.12 | 0.04 | 0.86 | 0.76 |
| GVP-WM (ORACLE) | 0.98 | 0.72 | 0.36 | 1.00 | 1.00 |

## Submitted Jobs

| Job | Slurm ID | Script | Mapping |
| --- | --- | --- | --- |
| WAN-0S full array | `22785801_[0-49%10]` | `/gpfs/home2/scur0196/dl2runtime/jobs/35_table2_wan0s_full_array.slurm` | 10 chunks per setting |
| ORACLE full array | `22785802_[0-4]` | `/gpfs/home2/scur0196/dl2runtime/jobs/34_table2_oracle_full_array.slurm` | 1 full sampled-50 job per setting |

WAN-0S array mapping:

- `0-9`: PushT `T=25`
- `10-19`: PushT `T=50`
- `20-29`: PushT `T=80`
- `30-39`: Wall `T=25`
- `40-49`: Wall `T=50`

ORACLE array mapping:

- `0`: PushT `T=25`
- `1`: PushT `T=50`
- `2`: PushT `T=80`
- `3`: Wall `T=25`
- `4`: Wall `T=50`

## Specs

Generated fixed 50-pair `episode:offset` specs with seed `99`, frame skip `5`, and chunk size `5`:

- `/gpfs/home2/scur0196/dl2runtime/specs/t25_pusht_seed99_50.specs`
- `/gpfs/home2/scur0196/dl2runtime/specs/t50_pusht_seed99_50.specs`
- `/gpfs/home2/scur0196/dl2runtime/specs/t80_pusht_seed99_50.specs`
- `/gpfs/home2/scur0196/dl2runtime/specs/t25_wall_seed99_50.specs`
- `/gpfs/home2/scur0196/dl2runtime/specs/t50_wall_seed99_50.specs`

All five spec files have 50 entries, 50 unique entries, and 10 chunks of 5.

## Key Settings

- World-model frame skip: `5`
- WAN-0S: Wan2.1-FLF2V-14B-720P, `1280*720`, `81` frames, `50` sampling steps, shift `16`, guide scale `5`
- ALM: `inner=25`, `outer=25`, learning rate `0.05`
- Local refinement: `500` samples, variance `0.3`
- PushT action regularization: `lambda_action=0.05` for `T=25`, `0.1` for `T=50,80`
- Wall action regularization: `lambda_action=0.05` for `T=25`, `0.1` for `T=50`
- Wall rho growth: `1.5` for `T=25`, `1.9` for `T=50`
- Wall target source: `env-replay`

## Implementation Notes

Added reusable scripts:

- `reimplementation/scripts/make_table2_specs.py`
- `reimplementation/scripts/snellius/34_table2_oracle_full_array.slurm`
- `reimplementation/scripts/snellius/35_table2_wan0s_full_array.slurm`

Fixed Wall `T=50` handling before submission. The Wall dataset stores `states.pth` with 50 states but stores 50 actions and `obses/episode_*.pth` with 51 visual frames. For `env-replay`, the correct `T=50` protocol is to start from the initial state, execute 50 primitive actions, and use the replayed 51st frame/state as the target. The loader now preserves all 50 actions and pads terminal proprio/layout metadata where the tensor dataset only has 50 state/layout entries.

Smoke check before submission:

- Wall `T=50` first spec `827:0` loaded with `length=51`, `actions=(50,2)`, and replayed to `video_plan=11`, `states=(51,2)`.

## Monitoring

```bash
squeue -u scur0196 -o '%.18i %.12P %.35j %.10T %.10M %.12L %.20R' | rg '22785801|22785802|dl2_table2'
```

```bash
sacct -j 22785801,22785802 --format=JobID,JobName%35,Partition,State,Elapsed,ExitCode,Start,End -P
```

Expected report patterns:

- `/gpfs/home2/scur0196/dl2runtime/reports/oracle_pusht_t25_seed99_50_22785802_0.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/oracle_pusht_t50_seed99_50_22785802_1.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/oracle_pusht_t80_seed99_50_22785802_2.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/oracle_wall_t25_seed99_50_22785802_3.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/oracle_wall_t50_seed99_50_22785802_4.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/wan0s_pusht_t25_*.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/wan0s_pusht_t50_*.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/wan0s_pusht_t80_*.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/wan0s_wall_t25_*.json`
- `/gpfs/home2/scur0196/dl2runtime/reports/wan0s_wall_t50_*.json`
