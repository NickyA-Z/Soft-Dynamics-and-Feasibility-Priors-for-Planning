from __future__ import annotations

import os
from typing import Any, Callable
import torch
import torch.nn as nn

from .config import PlannerConfig
from .interfaces import MPCResult, MPCStepResult, VideoPlan, VideoPlanSource, WorldModelAdapter
#from .solver import LatentCollocationSolver
from .solver_feas import LatentCollocationSolver
from .utils import ensure_history_length, shift_action_warm_start, shift_latent_warm_start
from .video import temporal_resample_sequence
from local.feasibility2.integrate import load_feasibility_scorer
from extension.feasibility2.langevin import langevin_action_search

VERBOSE_DIAGNOSTICS = os.environ.get("WALL_VERBOSE_DIAGNOSTICS", "0") == "1"


class GVPWMPlanner:
    def __init__(self, world_model: WorldModelAdapter, config: PlannerConfig) -> None:
        self.world_model = world_model
        self.config = config

        #feasibility integration related logic
        self.feasibility_model: nn.Module | None = None
        if self.config.feasibility.enabled:
            if self.config.feasibility.checkpoint_path is None:
                raise ValueError(
                    "config.feasibility.enabled=True, but checkpoint_path is None."
                )

            # load the model and set to eval mode
            self.feasibility_model = load_feasibility_scorer(
                self.config.feasibility.checkpoint_path,
                device=self.world_model.device,
                freeze=True, # ToDo need to look into
            )

        # added feasibility
        self.solver = LatentCollocationSolver(world_model=world_model, config=config.alm,
                                            feasibility_config=config.feasibility,
                                            feasibility_model=self.feasibility_model,
                                            langevin_config=config.langevin_action,
                                            )

    def _encode_video(self, video_plan: VideoPlan, horizon: int) -> torch.Tensor:
        if video_plan.encoded:
            encoded = torch.as_tensor(video_plan.data, dtype=torch.float32, device=self.world_model.device)
        else:
            encoded = self.world_model.encode_sequence(video_plan.data).to(self.world_model.device)
        return temporal_resample_sequence(encoded, target_length=horizon + 1)

    def _planner_cost_breakdown(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            # better have everything on same device 
            actions = actions.to(self.world_model.device)
            latent_context = latent_context.to(self.world_model.device)
            past_action_context = past_action_context.to(self.world_model.device)
            goal_latent = goal_latent.to(self.world_model.device)
            video_latents = video_latents.to(self.world_model.device)

            current_goal_cost = self.world_model.goal_loss(
                latent_context[-1],
                goal_latent,
            )
            raw_rollout_latents = self.world_model.rollout(
                latent_context=latent_context,
                past_action_context=past_action_context,
                planned_actions=actions,
            )
            if (
                raw_rollout_latents.shape[0] == actions.shape[0] + 1
                and torch.allclose(raw_rollout_latents[0], latent_context[-1], atol=1e-5, rtol=1e-5)
            ):
                candidate_latents = raw_rollout_latents
            else:
                candidate_latents = torch.cat([latent_context[-1:].clone(), raw_rollout_latents], dim=0)
            first_predicted_index = 1 if candidate_latents.shape[0] > 1 else 0
            first_step_goal_cost = self.world_model.goal_loss(
                candidate_latents[first_predicted_index],
                goal_latent,
            )
            final_goal_cost = self.world_model.goal_loss(
                candidate_latents[-1],
                goal_latent,
            )
            objective, pieces = self.solver._objective(
                latents=candidate_latents,
                actions=actions,
                goal_latent=goal_latent,
                video_latents=video_latents,
                action_prior=None,
            )
            feasibility_penalty, feasibility_pieces = self.solver._feasibility_penalty(
                latent_context=latent_context,
                candidate_latents=candidate_latents,
                candidate_actions=actions,
            )
            return {
                "current_goal_cost": float(current_goal_cost.detach().cpu()),
                "first_step_goal_cost": float(first_step_goal_cost.detach().cpu()),
                "goal_only_cost": float(final_goal_cost.detach().cpu()),
                "pred_first_goal_progress": float(
                    (current_goal_cost - first_step_goal_cost).detach().cpu()
                ),
                "pred_final_goal_progress": float(
                    (current_goal_cost - final_goal_cost).detach().cpu()
                ),
                "planner_base_cost": float(objective.detach().cpu()),
                "planner_total_cost": float((objective + feasibility_penalty).detach().cpu()),
                "weighted_feasibility": float(feasibility_penalty.detach().cpu()),
                "weighted_transition": float(
                    feasibility_pieces.get("weighted_transition_energy", candidate_latents.new_tensor(0.0))
                    .detach()
                    .cpu()
                ),
            }

    def _refine_actions(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.config.refinement.enabled or self.config.refinement.num_samples <= 0:
            return actions, {
                "refinement_enabled": 0.0,
                "refinement_base_cost": float("nan"),
                "refinement_best_cost": float("nan"),
                "refinement_selected_index": -1.0,
                "refinement_changed": 0.0,
                "refinement_delta_norm": 0.0,
            }
        device = actions.device
        candidates = [actions]
        for _ in range(self.config.refinement.num_samples):
            noisy = actions + torch.randn_like(actions) * (self.config.refinement.noise_variance ** 0.5)
            noisy = noisy.clamp(
                min=self.world_model.action_low,
                max=self.world_model.action_high,
            )
            candidates.append(noisy)

        best_actions = actions
        best_cost = None
        base_cost_value = float("nan")
        best_index = 0
        for candidate_index, candidate in enumerate(candidates):
            rollout = self.world_model.rollout(
                latent_context=latent_context,
                past_action_context=past_action_context,
                planned_actions=candidate,
            )
            cost = self.world_model.goal_loss(rollout[-1], goal_latent)
            if self.config.refinement.objective == "planner":
                video_cost = cost.new_tensor(0.0)
                if self.config.alm.use_video_loss and rollout.shape[0] > 2:
                    for index in range(1, rollout.shape[0] - 1):
                        video_cost = video_cost + self.world_model.video_alignment_loss(
                            rollout[index],
                            video_latents[index],
                        )
                action_cost = candidate.pow(2).sum()
                cost = (
                    self.config.alm.lambda_video * video_cost
                    + self.config.alm.lambda_goal * cost
                    + self.config.alm.lambda_action * action_cost
                )
            elif self.config.refinement.objective != "goal":
                raise ValueError(f"Unknown refinement objective: {self.config.refinement.objective}")
            if candidate_index == 0:
                base_cost_value = float(cost.detach().cpu())
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_actions = candidate
                best_index = candidate_index
        best_actions = best_actions.to(device)
        return best_actions, {
            "refinement_enabled": 1.0,
            "refinement_base_cost": base_cost_value,
            "refinement_best_cost": float(best_cost.detach().cpu()),
            "refinement_selected_index": float(best_index),
            "refinement_changed": float(best_index != 0),
            "refinement_delta_norm": float((best_actions - actions).norm().detach().cpu()),
        }

    def solve_once(
        self,
        observation_history: Any,
        goal_observation: Any,
        video_plan: VideoPlan,
        past_action_history: torch.Tensor | None = None,
        warm_start_latents: torch.Tensor | None = None,
        warm_start_actions: torch.Tensor | None = None,
    ):
        latent_context = self.world_model.encode_sequence(observation_history).to(self.world_model.device)
        if self.config.alm.pad_initial_history:
            latent_context = ensure_history_length(
                latent_context,
                self.world_model.history_length,
                pad_mode="repeat_first",
            )
        goal_latent = self.world_model.encode_observation(goal_observation).to(self.world_model.device)
        if past_action_history is None:
            initial_action_history = max(self.world_model.history_length - 1, 0)
            if not self.config.alm.pad_initial_history:
                initial_action_history = 0
            past_action_history = torch.zeros(
                initial_action_history,
                self.world_model.action_dim,
                device=self.world_model.device,
            )
        else:
            max_action_history = max(self.world_model.history_length - 1, 0)
            past_action_history = past_action_history.to(self.world_model.device)
            past_action_history = (
                past_action_history[-max_action_history:]
                if max_action_history > 0
                else past_action_history[:0]
            )
        encoded_video = self._encode_video(video_plan, self.config.mpc.horizon)
        return self.solver.solve(
            latent_context=latent_context,
            past_action_context=past_action_history,
            goal_latent=goal_latent,
            video_latents=encoded_video,
            warm_start_latents=warm_start_latents,
            warm_start_actions=warm_start_actions,
        )

    def run_mpc(
        self,
        observation_history: Any,
        goal_observation: Any,
        step_fn: Callable[[torch.Tensor], Any],
        video_source: VideoPlanSource | None = None,
        video_plan: VideoPlan | None = None,
        past_action_history: torch.Tensor | None = None,
        initial_warm_start_latents: torch.Tensor | None = None,
        initial_warm_start_actions: torch.Tensor | None = None,
        diagnostic_action_provider: Callable[[int, int, torch.device], torch.Tensor | None] | None = None,
    ) -> MPCResult:
        if video_plan is None:
            if video_source is None:
                raise ValueError("Either video_source or video_plan must be provided.")
            initial_observation = observation_history[-1]
            video_plan = video_source.generate(
                initial_observation=initial_observation,
                goal_observation=goal_observation,
                horizon=self.config.mpc.horizon,
            )

        latent_context = self.world_model.encode_sequence(observation_history).to(self.world_model.device)
        if self.config.alm.pad_initial_history:
            latent_context = ensure_history_length(
                latent_context,
                self.world_model.history_length,
                pad_mode="repeat_first",
            )
        goal_latent = self.world_model.encode_observation(goal_observation).to(self.world_model.device)
        encoded_video = self._encode_video(video_plan, self.config.mpc.horizon)

        if past_action_history is None:
            initial_action_history = max(self.world_model.history_length - 1, 0)
            if not self.config.alm.pad_initial_history:
                initial_action_history = 0
            past_action_history = torch.zeros(
                initial_action_history,
                self.world_model.action_dim,
                device=self.world_model.device,
            )
        else:
            max_action_history = max(self.world_model.history_length - 1, 0)
            past_action_history = past_action_history.to(self.world_model.device)
            past_action_history = (
                past_action_history[-max_action_history:]
                if max_action_history > 0
                else past_action_history[:0]
            )

        warm_start_latents = (
            initial_warm_start_latents.to(self.world_model.device)
            if initial_warm_start_latents is not None
            else None
        )
        warm_start_actions = (
            initial_warm_start_actions.to(self.world_model.device)
            if initial_warm_start_actions is not None
            else None
        )
        executed_actions = []
        executed_latents = [latent_context[-1]]
        steps: list[MPCStepResult] = []

        time_index = 0
        while time_index < self.config.mpc.horizon:
            remaining = self.config.mpc.horizon - time_index
            current_video = temporal_resample_sequence(
                encoded_video[time_index:],
                remaining + 1,
            )
            result = self.solver.solve(
                latent_context=latent_context,
                past_action_context=past_action_history,
                goal_latent=goal_latent,
                video_latents=current_video,
                warm_start_latents=warm_start_latents,
                warm_start_actions=warm_start_actions,
            )
            solver_actions = result.actions
            planned_actions, refinement_debug = self._refine_actions(
                latent_context=latent_context,
                past_action_context=past_action_history,
                goal_latent=goal_latent,
                video_latents=current_video,
                actions=solver_actions,
            )

            solver_first_norm = float(solver_actions[0].norm().detach().cpu())
            refined_first_norm = float(planned_actions[0].norm().detach().cpu())
            solver_seq_norm = float(solver_actions.norm().detach().cpu())
            refined_seq_norm = float(planned_actions.norm().detach().cpu())
            solver_to_refined_norm = float((planned_actions - solver_actions).norm().detach().cpu())
            solver_costs = self._planner_cost_breakdown(
                latent_context=latent_context,
                past_action_context=past_action_history,
                goal_latent=goal_latent,
                video_latents=current_video,
                actions=solver_actions,
            )
            refined_costs = self._planner_cost_breakdown(
                latent_context=latent_context,
                past_action_context=past_action_history,
                goal_latent=goal_latent,
                video_latents=current_video,
                actions=planned_actions,
            )
            expert_pred_text = ""
            if diagnostic_action_provider is not None:
                diagnostic_actions = diagnostic_action_provider(
                    time_index,
                    remaining,
                    self.world_model.device,
                )
                if diagnostic_actions is not None and diagnostic_actions.shape[0] == remaining:
                    diagnostic_actions = diagnostic_actions.to(self.world_model.device)
                    diagnostic_costs = self._planner_cost_breakdown(
                        latent_context=latent_context,
                        past_action_context=past_action_history,
                        goal_latent=goal_latent,
                        video_latents=current_video,
                        actions=diagnostic_actions,
                    )
                    expert_pred_text = (
                        f" expert_pred_first={diagnostic_costs['pred_first_goal_progress']:.4f}"
                        f" expert_pred_final={diagnostic_costs['pred_final_goal_progress']:.4f}"
                    )
            if warm_start_actions is not None and warm_start_actions.shape[0] >= remaining:
                current_warm_actions = warm_start_actions[:remaining]
                warm_to_solver_norm = float((solver_actions - current_warm_actions).norm().detach().cpu())
            else:
                warm_to_solver_norm = float("nan")
            print(
                f"[planner step {time_index}] "
                f"mode={self.config.alm.dynamics_mode} rem={remaining} "
                f"pred_first={refined_costs['pred_first_goal_progress']:.4f} "
                f"pred_final={refined_costs['pred_final_goal_progress']:.4f} "
                f"goal={refined_costs['goal_only_cost']:.4f} "
                f"planner={refined_costs['planner_base_cost']:.4f} "
                f"total={refined_costs['planner_total_cost']:.4f} "
                f"first_norm={refined_first_norm:.3f} seq_norm={refined_seq_norm:.3f} "
                f"solver_to_refine={solver_to_refined_norm:.3f} "
                f"warm_to_solver={warm_to_solver_norm:.3f} "
                f"refine={int(refinement_debug['refinement_changed'])}"
                f"{expert_pred_text}"
            )

            n_exec = min(self.config.mpc.execution_stride, remaining)
            executed_this_round = planned_actions[:n_exec]

            for offset in range(n_exec):
                action = executed_this_round[offset].detach()
                next_observation = step_fn(action)
                next_latent = self.world_model.encode_observation(next_observation).to(self.world_model.device)
                executed_actions.append(action)
                executed_latents.append(next_latent)
                latent_context = torch.cat([latent_context, next_latent.unsqueeze(0)], dim=0)[
                    -self.world_model.history_length :
                ]
                if self.world_model.history_length > 1:
                    past_action_history = torch.cat(
                        [past_action_history, action.unsqueeze(0)],
                        dim=0,
                    )[-(self.world_model.history_length - 1) :]
                time_index += 1

            steps.append(
                MPCStepResult(
                    step_index=time_index,
                    planned_actions=planned_actions.detach(),
                    executed_actions=executed_this_round.detach(),
                    collocation_objective=result.objective,
                    dynamics_residual_norm=result.dynamics_residual_norm,
                )
            )

            if self.config.mpc.warm_start:
                warm_start_latents = shift_latent_warm_start(result.latents, n_exec)
                warm_start_latents[0] = latent_context[-1]
                warm_start_actions = shift_action_warm_start(planned_actions, n_exec)
            else:
                warm_start_latents = None
                warm_start_actions = None

        return MPCResult(
            video_latents=encoded_video.detach(),
            executed_actions=torch.stack(executed_actions, dim=0),
            executed_latents=torch.stack(executed_latents, dim=0),
            steps=steps,
        )
