# Old Successful Run vs Latest Failure Run: Config Comparison

Based on the old successful code/run and the latest failing run config/log.

| Setting / behavior | Old successful run | Latest failing run | Same? | Why it matters |
|---|---:|---:|---:|---|
| Episode | `2` | `2` | Yes | Same dataset episode. |
| Split | `val` | `val` | Yes | Same split. |
| Macro horizon | `25` | `25` | Yes | Both use 25 macro actions = 125 primitive steps. |
| Frame skip | `5` | `5` | Yes | Each macro action contains 5 primitive actions. |
| Video plan length | `26` | `26` | Yes | Correct for macro horizon 25. |
| States length | `126` | `126` | Yes | Correct for raw horizon 125. |
| Rel actions length | `125` | `125` | Yes | Correct for raw horizon 125. |
| `inner_steps` | `25` | `25` | Yes | Same number of optimizer inner steps. |
| `outer_steps` | `1` | `1` | Yes | Same outer loop count. |
| `learning_rate` | `0.01` | `0.01` | Yes | Same Adam learning rate. |
| `rho_init` | `1.0` | `1.0` | Yes | Same. |
| `rho_growth` | `1.9` | `1.9` | Yes | Same. |
| `rho_max` | `1000.0` | `1000.0` | Yes | Same. |
| `lambda_video` | `1.0` | `1.0` | Yes | Same nominal video weight. |
| `lambda_goal` | `10.0` | `10.0` | Yes | Same nominal goal weight. |
| `lambda_action` | `0.001` | `0.0` | **No** | Old run had a small action penalty. Latest run has none. Not likely the sole cause, but configs differ. |
| `lambda_action_prior` | `0.0` | `0.0` | Yes | Neither was anchored to expert/warm-start actions through an action prior. |
| `clip_grad_norm` | `None` | `None` | Yes | Same. |
| `use_video_init` | `True` | `True` | Yes | Both initialize latents from video. |
| `use_video_loss` | `False` | `True` | **No** | Old run used video only for initialization, not as a loss. Latest run actively optimized visual video loss, which can change the optimization basin. |
| `fix_states_to_video` | `False` | `False` | Yes | Latents are optimized in both. |
| `use_action_reparameterization` | `True` | `False` | **No** | Big difference. Old run optimized bounded action parameters through `tanh`; latest run directly optimized/clamped actions. This can strongly affect whether large useful actions are found. |
| `adam_eps` | `1e-8` | `1e-8` | Yes | Same. |
| `residual_reduction` | `mean` | `mean` | Yes | Same residual scaling. |
| `history_action_pad` | `zeros` | `zeros` | Yes | Same. |
| `pad_initial_history` | `True` | `True` | Yes | Same. |
| `dynamics_mode` | `soft` | `soft` | Yes | Same nominal mode. |
| `lambda_dynamics` | `10.0` | `10.0` | Yes | Same dynamics weight. |
| `use_dynamics_constraints` | `True` | `False` | **No** | Big difference in config, although in the pasted solver code this flag may not control the `soft` branch directly. |
| `lambda_anti_stillness` | not present / effectively `0.0` | `0.0` | Probably same | Not relevant here. |
| `min_action_norm` | not present | `0.0` | Probably same | Not relevant since anti-stillness is off. |
| MPC `execution_stride` | `1` | `1` | Yes | Replans every macro step. |
| MPC `warm_start` | `True` | `False` | **No** | Very important. Old run shifted the previous planned trajectory into the next solve. Latest run starts each solve from scratch. |
| Refinement `enabled` | `False` | `False` | Yes | No refinement in either. |
| Refinement `num_samples` | `0` | `0` | Yes | Same. |
| Refinement `noise_variance` | `0.3` | `0.3` | Yes | Same but unused. |
| Feasibility `enabled` | `False` | `False` | Yes | Feasibility model disabled in both. |
| Feasibility `lambda_feasibility` | `1.0` but disabled | `0.0` and disabled | Effectively same | Since feasibility is disabled, this should not matter. |
| Feasibility `lambda_transition` | `10.0` but disabled | `0.0` and disabled | Effectively same | Since feasibility is disabled, this should not matter. |
| Initial expert action warm-start | Not passed | Not passed | Yes | Neither directly passed `initial_warm_start_actions`. |
| First useful large actions | Yes | No | **No** | Old step 1: planner around `(104.5, 67.9)`. Latest step 1: planner around `(20.1, 25.3)`. |
| Block movement | Yes, from around step 10 onward | No / insufficient | **No** | Latest run never reaches useful contact with the block. |
| Final result | Success | Failure | **No** | Old reaches the block goal; latest leaves the episode far from success. |

## Main differences likely explaining the failure

The latest failing run is not the same configuration as the old successful run.

Old successful run:

```text
warm_start=True
use_action_reparameterization=True
use_dynamics_constraints=True
use_video_loss=False
lambda_action=0.001
```

Latest failing run:

```text
warm_start=False
use_action_reparameterization=False
use_dynamics_constraints=False
use_video_loss=True
lambda_action=0.0
```

The biggest likely cause is the combination of:

```text
warm_start=False
use_action_reparameterization=False
```

The old planner found a useful long-horizon plan and carried it forward through MPC. The latest one replans from scratch every step and collapses to small actions.

The action traces support this:

```text
Old successful step 1: planner=(104.5, 67.9), expert=(118.5, 72.7)
Latest failing step 1: planner=(20.1, 25.3), expert=(118.5, 72.7)
```

That is not a minor numerical variation; it is a different optimization basin.

## Suggested reproduction test

Start from the latest code and set exactly these old values:

```python
lambda_action = 0.001
use_video_loss = False
use_action_reparameterization = True
use_dynamics_constraints = True
warm_start = True
```

Then change only one of them at a time to isolate the cause.
