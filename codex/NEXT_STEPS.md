# Next Steps: Feasibility-Model Planning Study

- Use the episode-disjoint Push-T test set (episodes 550–599) only for final results.
- Evaluate the same transitions, episode offsets, and random seeds for every model.

## Models

- `better_varcon_final.pt`: original DSM + contrastive checkpoint.
- `planner_aligned.pt`: planner-aligned v1 checkpoint.
- `planner_aligned_v2.pt`: planner-aligned v2 checkpoint.
- Include expert, zero-action, and random-action baselines.

## Energy ablations

- DSM only: `lambda_dsm=1.0`, `lambda_contrastive=0.0`.
- Contrastive only: `lambda_dsm=0.0`, `lambda_contrastive=1.0`.
- Training-weight combination: `lambda_dsm=0.1`, `lambda_contrastive=1.0`.
- Equal combination: `lambda_dsm=1.0`, `lambda_contrastive=1.0`.

## Test order

1. Run fixed-transition one-step recovery from `codex.run`.
2. Compare expert, zero, random, and optimized-action energies.
3. Record action MSE, recovery ratio, cosine similarity, next-state error, and action-bound saturation.
4. Compare Adam/ALM, Langevin, and random search at multiple optimization strengths.
5. Continue only when optimized actions consistently outperform zero action.
6. Run horizon-5 recovery with the expert latent trajectory fixed.
7. Compare fixed-latent, free-latent, transition-constrained, and dynamics-rollout modes.
8. Run MPC only after one-step and horizon-5 recovery pass.
9. In dynamics/MPC modes, sweep `lambda_goal` over `10, 30, 100, 300`.
10. Compare fixed oracle-video suffixes with re-anchored or regenerated targets.

## Langevin

- Run Langevin through `local/gvpwm`; do not use `extension/` as the experiment runner.
- Compare 3 and 8 starts, 100 and 300 steps, and a small step-size/temperature sweep.
- Treat incorrect actions with energy below the expert as energy-model exploits.

## Folder responsibilities

- `extension/feasibility2`: model architecture and checkpoint loading.
- `codex/`: training, fixed-transition evaluation, and controlled recovery tests.
- `local/gvpwm`: Langevin sampling, rollout planning, and MPC.

## Reporting

- Use at least three optimizer seeds and preferably all 50 held-out test episodes.
- Save energies, actions, predicted states, environment states, success, runtime, and GPU memory.
- Report mean, median, standard deviation, and 95% bootstrap confidence intervals.
- Keep validation-model selection separate from the final test-set report.

## Pass criteria

- Expert energy is below zero, random, and optimizer-generated action energies.
- Recovery MSE is lower than zero-action MSE and cosine similarity is positive.
- Horizon-5 errors do not grow uncontrollably.
- Predicted goal improvement corresponds to actual environment success.
