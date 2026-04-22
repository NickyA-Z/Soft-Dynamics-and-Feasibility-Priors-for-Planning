from __future__ import annotations

from typing import Any, Callable
from unittest import result

import torch

from .config import PlannerConfig
from .interfaces import MPCResult, MPCStepResult, VideoPlan, VideoPlanSource, WorldModelAdapter
from .losses import goal_mse
from .solver import LatentCollocationSolver
from .utils import ensure_history_length, shift_action_warm_start, shift_latent_warm_start
from .video import temporal_resample_sequence


class GVPWMPlanner:
    def __init__(self, world_model: WorldModelAdapter, config: PlannerConfig) -> None:
        self.world_model = world_model
        self.config = config
        self.solver = LatentCollocationSolver(world_model=world_model, config=config.alm)

    def _encode_video(self, video_plan: VideoPlan, horizon: int) -> torch.Tensor:
        if video_plan.encoded:
            encoded = torch.as_tensor(video_plan.data, dtype=torch.float32, device=self.world_model.device)
        else:
            encoded = self.world_model.encode_sequence(video_plan.data).to(self.world_model.device)
        return temporal_resample_sequence(encoded, target_length=horizon + 1)

    def _refine_actions(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.refinement.enabled or self.config.refinement.num_samples <= 0:
            return actions
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
        for candidate in candidates:
            rollout = self.world_model.rollout(
                latent_context=latent_context,
                past_action_context=past_action_context,
                planned_actions=candidate,
            )
            terminal_cost = self.world_model.goal_loss(rollout[-1], goal_latent)
            if best_cost is None or terminal_cost < best_cost:
                best_cost = terminal_cost
                best_actions = candidate
        return best_actions.to(device)

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
        latent_context = ensure_history_length(
            latent_context,
            self.world_model.history_length,
            pad_mode="repeat_first",
        )
        goal_latent = self.world_model.encode_observation(goal_observation).to(self.world_model.device)
        if past_action_history is None:
            past_action_history = torch.zeros(
                max(self.world_model.history_length - 1, 0),
                self.world_model.action_dim,
                device=self.world_model.device,
            )
        else:
            past_action_history = past_action_history.to(self.world_model.device)
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
        latent_context = ensure_history_length(
            latent_context,
            self.world_model.history_length,
            pad_mode="repeat_first",
        )
        goal_latent = self.world_model.encode_observation(goal_observation).to(self.world_model.device)
        encoded_video = self._encode_video(video_plan, self.config.mpc.horizon)

        if past_action_history is None:
            past_action_history = torch.zeros(
                max(self.world_model.history_length - 1, 0),
                self.world_model.action_dim,
                device=self.world_model.device,
            )
        else:
            past_action_history = past_action_history.to(self.world_model.device)

        warm_start_latents = None
        warm_start_actions = None
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
            planned_actions = self._refine_actions(
                latent_context=latent_context,
                past_action_context=past_action_history,
                goal_latent=goal_latent,
                actions=result.actions,
            )

            ##########DEBUG##########
            if time_index % 10 == 0 or remaining == 1:
                print(f"[mpc step {time_index}] solver diagnostics: {result.diagnostics}(planner DEBUG)")

            raw_first = result.actions[0].detach().cpu()
            ref_first = planned_actions[0].detach().cpu()

            print(
                f"[refine check {time_index}] "
                f"raw_norm_sum=({raw_first[0::2].sum():.3f},{raw_first[1::2].sum():.3f}) "
                f"ref_norm_sum=({ref_first[0::2].sum():.3f},{ref_first[1::2].sum():.3f}) "
                f"delta_norm={float((ref_first - raw_first).norm()):.4f}"
            )
            #########################
            
            n_exec = min(self.config.mpc.execution_stride, remaining)
            executed_this_round = planned_actions[:n_exec]

            for offset in range(n_exec):
                action = executed_this_round[offset].detach()
                next_observation = step_fn(action)
                
                ###################debug###################
                next_time_index = time_index + 1
                next_latent = self.world_model.encode_observation(next_observation).to(self.world_model.device)

                env_ref_loss = self.world_model.video_alignment_loss(
                    next_latent,
                    encoded_video[next_time_index],
                )

                print(
                    f"[env-ref loss {next_time_index}](planner DEBUG)"
                    f"{float(env_ref_loss.detach().cpu()):.6f}"
                )
                ###########################################
                
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
