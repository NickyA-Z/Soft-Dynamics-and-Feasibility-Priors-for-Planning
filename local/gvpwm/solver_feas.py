from __future__ import annotations

import os

import torch
import torch.nn as nn
from local.feasibility2.integrate import score_trajectory_feasibility

from .config import ALMConfig, FeasibilityConfig, LangevinActionConfig, ActionSearchConfig
from .interfaces import CollocationResult, WorldModelAdapter
from .losses import squared_norm
from .utils import ensure_history_length

from local.gvpwm.langevin import (
    ActionEvaluation,
    LangevinAdamConfig,
    MultiStartAdamConfig,
    make_initial_action_parameters,
    run_langevin_adam,
    run_multistart_adam,
)

VERBOSE_DIAGNOSTICS = os.environ.get("WALL_VERBOSE_DIAGNOSTICS", "0") == "1"

class LatentCollocationSolver:
    def __init__(self, world_model: WorldModelAdapter, config: ALMConfig,
                 feasibility_config: FeasibilityConfig | None = None, # feasibility integration
                 feasibility_model: nn.Module | None = None, # feasibility integration
                 langevin_config: LangevinActionConfig | None = None, # langevin integration 
                 ) -> None:
        self.world_model = world_model
        self.config = config
        self.feasibility_config = feasibility_config # feasibility integration
        self.feasibility_model = feasibility_model # feasibility integration
        self.langevin_config = (
            langevin_config or LangevinActionConfig(enabled=False)
        )
        self.action_search_config = (
            action_search_config or ActionSearchConfig()
        )

    def _actions_from_parameter(self, action_parameter: torch.Tensor) -> torch.Tensor:
        if not self.config.use_action_reparameterization:
            return torch.clamp(
                action_parameter,
                min=self.world_model.action_low,
                max=self.world_model.action_high,
            )
        low = self.world_model.action_low.to(action_parameter)
        high = self.world_model.action_high.to(action_parameter)
        scaled = torch.tanh(action_parameter)
        return low + 0.5 * (scaled + 1.0) * (high - low)

    def _raw_parameter_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.config.use_action_reparameterization:
            return actions.clone()
        low = self.world_model.action_low.to(actions)
        high = self.world_model.action_high.to(actions)
        scaled = 2.0 * (actions - low) / (high - low).clamp_min(1e-8) - 1.0
        scaled = scaled.clamp(-0.999999, 0.999999)
        return torch.atanh(scaled)

    def _initialize_latents(
        self,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        warm_start_latents: torch.Tensor | None,
    ) -> torch.Tensor:
        horizon = video_latents.shape[0] - 1
        if warm_start_latents is not None and warm_start_latents.shape[0] >= horizon + 1:
            latents = warm_start_latents[: horizon + 1].clone()
        elif self.config.use_video_init:
            latents = self.world_model.initialize_latents_from_video(
                current_latent=current_latent,
                video_latents=video_latents,
            )
        else:
            view_shape = (horizon + 1,) + (1,) * current_latent.ndim
            alpha = torch.linspace(
                0.0,
                1.0,
                horizon + 1,
                device=current_latent.device,
                dtype=current_latent.dtype,
            ).view(view_shape)
            latents = current_latent.unsqueeze(0) + alpha * (
                goal_latent.unsqueeze(0) - current_latent.unsqueeze(0)
            )
        latents[0] = current_latent
        return latents

    def _initialize_actions(
        self,
        horizon: int,
        warm_start_actions: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if warm_start_actions is not None and warm_start_actions.shape[0] >= horizon:
            return self._raw_parameter_from_actions(warm_start_actions[:horizon].to(device))
        """
        initial_actions = torch.zeros(
            horizon,
            self.world_model.action_dim,
            device=device,
            dtype=self.world_model.action_low.dtype,
        )
        initial_actions = torch.clamp(
            initial_actions,
            min=self.world_model.action_low.to(device),
            max=self.world_model.action_high.to(device),
        )"""

        # non zero start 14 juli
        initial_actions = 0.2 * torch.randn(
            horizon,
            self.world_model.action_dim,
            device=device,
            dtype=self.world_model.action_low.dtype,
        )

        initial_actions = torch.clamp(
            initial_actions,
            min=self.world_model.action_low.to(device),
            max=self.world_model.action_high.to(device),
        )

        return self._raw_parameter_from_actions(initial_actions)

    def _dynamics_residuals(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        candidate_latents: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.pad_initial_history:
            history = ensure_history_length(
                latent_context,
                self.world_model.history_length,
                pad_mode="repeat_first",
            )
        else:
            history = latent_context[-self.world_model.history_length :]
        if self.world_model.history_length > 1:
            action_context_len = self.world_model.history_length - 1
            action_context = past_action_context[-action_context_len:].clone()
            missing_actions = action_context_len - action_context.shape[0]
            if not self.config.pad_initial_history:
                missing_actions = max(history.shape[0] - 1 - action_context.shape[0], 0)
            if missing_actions > 0:
                if self.config.history_action_pad == "zeros":
                    pad = candidate_actions.new_zeros(missing_actions, self.world_model.action_dim)
                elif self.config.history_action_pad == "repeat_available":
                    if action_context.shape[0] > 0:
                        pad_value = action_context[:1]
                    else:
                        pad_value = candidate_actions[:1]
                    pad = pad_value.expand(missing_actions, -1)
                else:
                    raise ValueError(f"Unknown history_action_pad: {self.config.history_action_pad}")
                action_context = torch.cat([pad, action_context], dim=0)
        else:
            action_context = candidate_actions.new_zeros((0, self.world_model.action_dim))

        residuals = []
        for index in range(candidate_actions.shape[0]):
            state_window = torch.cat([history, candidate_latents[1 : index + 1]], dim=0)[
                -self.world_model.history_length :
            ]
            action_window_full = torch.cat(
                [action_context, candidate_actions[: index + 1]],
                dim=0,
            )
            action_window_len = (
                self.world_model.history_length
                if self.config.pad_initial_history
                else state_window.shape[0]
            )
            action_window = action_window_full[-action_window_len:]
            if action_window.shape[0] < state_window.shape[0]:
                pad = candidate_actions.new_zeros(
                    state_window.shape[0] - action_window.shape[0],
                    self.world_model.action_dim,
                )
                action_window = torch.cat([pad, action_window], dim=0)
            predicted_next = self.world_model.predict_next_latent(state_window, action_window)
            # constrain violation of world model dynamics to be small
            residuals.append(candidate_latents[index + 1] - predicted_next)
        return torch.stack(residuals, dim=0)

    # small test for personal understading
    def _dynamics_penalty(
        self,
        residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.residual_reduction == "mean":
            penalty = residuals.pow(2).mean()
        elif self.config.residual_reduction == "sum":
            penalty = residuals.pow(2).sum()
        else:
            raise ValueError(f"Unknown residual_reduction: {self.config.residual_reduction}")

        weighted_penalty = self.config.lambda_dynamics * penalty

        return weighted_penalty, {
            "dynamics_penalty": penalty,
            "weighted_dynamics_penalty": weighted_penalty,
        }
    
    # Langevin
    def _langevin_action_step(
        self,
        action_parameter: torch.nn.Parameter,
    ) -> None:
        grad = action_parameter.grad

        if grad is None:
            raise RuntimeError(
                "No action gradient was produced during rollout Langevin sampling."
            )

        with torch.no_grad():
            if self.langevin_config.grad_clip_norm is not None:
                grad_norm = grad.norm()
                max_norm = self.langevin_config.grad_clip_norm

                if grad_norm > max_norm:
                    grad = grad * (
                        max_norm / grad_norm.clamp_min(1e-12)
                    )

            step_size = self.langevin_config.step_size

            # Gradient drift.
            action_parameter.add_(grad, alpha=-step_size)

            # Langevin diffusion.
            if (
                self.langevin_config.add_noise
                and self.langevin_config.temperature > 0
            ):
                noise_std = (
                    2.0
                    * step_size
                    * self.langevin_config.temperature
                ) ** 0.5

                action_parameter.add_(
                    noise_std * torch.randn_like(action_parameter)
                )

            if not self.config.use_action_reparameterization:
                action_parameter.clamp_(
                    min=self.world_model.action_low.to(action_parameter),
                    max=self.world_model.action_high.to(action_parameter),
                )

    def _objective(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        action_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        video_loss = latents.new_tensor(0.0)
        if self.config.use_video_loss and latents.shape[0] > 2:
            for index in range(1, latents.shape[0] - 1):
                video_loss = video_loss + self.world_model.video_alignment_loss(
                    latents[index],
                    video_latents[index],
                )
        goal_loss = self.world_model.goal_loss(latents[-1], goal_latent)
        action_loss = actions.pow(2).sum()
        action_prior_loss = latents.new_tensor(0.0)
        if action_prior is not None and self.config.lambda_action_prior > 0:
            action_prior_loss = (actions - action_prior).pow(2).sum()
        #action_loss = actions.pow(2).sum(dim=-1).mean()

        # Anti-stillness: penalize actions whose norm is too small.
        action_norm = actions.reshape(actions.shape[0], -1).norm(dim=-1)
        anti_stillness_loss = torch.relu(
            self.config.min_action_norm - action_norm
        ).pow(2).mean()

        objective = (
            self.config.lambda_video * video_loss
            + self.config.lambda_goal * goal_loss
            + self.config.lambda_action * action_loss
            + self.config.lambda_action_prior * action_prior_loss
            + self.config.lambda_anti_stillness * anti_stillness_loss
        )

        return objective, {
            "video_loss": video_loss,
            "goal_loss": goal_loss,
            "action_loss": action_loss,
            "action_prior_loss": action_prior_loss,
            "anti_stillness_loss": anti_stillness_loss,
            "weighted_anti_stillness": self.config.lambda_anti_stillness * anti_stillness_loss,
        }
    
    def _rollout_world_model(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Generate candidate latents by recursively applying DINO-WM.

        Returns:
            candidate_latents: [H + 1, *latent_shape]
        """
        current_latent = latent_context[-1]
        generated_latents = [current_latent]

        if self.config.pad_initial_history:
            history = ensure_history_length(
                latent_context,
                self.world_model.history_length,
                pad_mode="repeat_first",
            )
        else:
            history = latent_context[-self.world_model.history_length :]

        if self.world_model.history_length > 1:
            action_context_len = self.world_model.history_length - 1
            action_context = past_action_context[-action_context_len:].clone()

            missing_actions = action_context_len - action_context.shape[0]
            if not self.config.pad_initial_history:
                missing_actions = max(
                    history.shape[0] - 1 - action_context.shape[0],
                    0,
                )

            if missing_actions > 0:
                if self.config.history_action_pad == "zeros":
                    pad = candidate_actions.new_zeros(
                        missing_actions,
                        self.world_model.action_dim,
                    )
                elif self.config.history_action_pad == "repeat_available":
                    if action_context.shape[0] > 0:
                        pad_value = action_context[:1]
                    else:
                        pad_value = candidate_actions[:1]
                    pad = pad_value.expand(missing_actions, -1)
                else:
                    raise ValueError(
                        f"Unknown history_action_pad: {self.config.history_action_pad}"
                    )

                action_context = torch.cat([pad, action_context], dim=0)
        else:
            action_context = candidate_actions.new_zeros(
                (0, self.world_model.action_dim)
            )

        for index in range(candidate_actions.shape[0]):
            if len(generated_latents) > 1:
                generated_so_far = torch.stack(generated_latents[1:], dim=0)
                state_source = torch.cat([history, generated_so_far], dim=0)
            else:
                state_source = history

            state_window = state_source[-self.world_model.history_length :]

            action_window_full = torch.cat(
                [action_context, candidate_actions[: index + 1]],
                dim=0,
            )

            action_window_len = (
                self.world_model.history_length
                if self.config.pad_initial_history
                else state_window.shape[0]
            )

            action_window = action_window_full[-action_window_len:]

            if action_window.shape[0] < state_window.shape[0]:
                pad = candidate_actions.new_zeros(
                    state_window.shape[0] - action_window.shape[0],
                    self.world_model.action_dim,
                )
                action_window = torch.cat([pad, action_window], dim=0)

            predicted_next = self.world_model.predict_next_latent(
                state_window,
                action_window,
            )

            generated_latents.append(predicted_next)

        return torch.stack(generated_latents, dim=0)

    def _feasibility_penalty(
        self,
        latent_context: torch.Tensor,
        candidate_latents: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            self.feasibility_config is None
            or not self.feasibility_config.enabled
            or self.feasibility_config.lambda_feasibility <= 0
            or self.feasibility_model is None
        ):
            zero = candidate_actions.new_tensor(0.0)
            #return zero, {"feasibility_loss": zero}
            return zero, {
                "feasibility_loss": zero,
                "dsm_energy": zero,
                "transition_energy": zero,
                "weighted_transition_energy": zero,
                "lambda_transition": 0.0,
            }

        penalty, debug = score_trajectory_feasibility(
            feasibility_model=self.feasibility_model,
            latent_context=latent_context,
            candidate_latents=candidate_latents,
            candidate_actions=candidate_actions,
            history_length=self.world_model.history_length,
            noise_level=self.feasibility_config.noise_level,

            reduction=self.feasibility_config.reduction,
            lambda_transition=self.feasibility_config.lambda_transition,
            lambda_action_consistency=self.feasibility_config.lambda_action_consistency,
            action_consistency_margin=self.feasibility_config.action_consistency_margin,
            latent_reduction=self.feasibility_config.latent_reduction,

            return_diagnostics=True,
        )
        pieces: dict[str, torch.Tensor | float] = {
            "feasibility_loss": penalty,
        }
        pieces.update(debug)
        pieces["configured_lambda_transition"] = float(
            self.feasibility_config.lambda_transition
        )

        weighted_penalty = self.feasibility_config.lambda_feasibility * penalty

        return weighted_penalty, pieces
        
    def solve(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        warm_start_latents: torch.Tensor | None = None,
        warm_start_actions: torch.Tensor | None = None,
    ) -> CollocationResult:
        latent_context = latent_context.to(self.world_model.device)
        past_action_context = past_action_context.to(self.world_model.device)
        goal_latent = goal_latent.to(self.world_model.device)
        video_latents = video_latents.to(self.world_model.device)

        current_latent = latent_context[-1]
        horizon = video_latents.shape[0] - 1

        initial_latents = self._initialize_latents(
            current_latent=current_latent,
            goal_latent=goal_latent,
            video_latents=video_latents,
            warm_start_latents=warm_start_latents,
        )

        action_prior = None
        if (
            warm_start_actions is not None
            and warm_start_actions.shape[0] >= horizon
        ):
            action_prior = warm_start_actions[:horizon].to(
                self.world_model.device
            )

        if self.config.dynamics_mode == "rollout":
            latent_parameter = None
        elif self.config.fix_states_to_video:
            latent_parameter = None
        else:
            latent_parameter = torch.nn.Parameter(
                initial_latents[1:].clone()
            )

        # Initialize actions exactly once.
        base_parameter = self._initialize_actions(
            horizon=horizon,
            warm_start_actions=warm_start_actions,
            device=self.world_model.device,
        )

        action_search_method = self.action_search_config.method

        # The new action-search algorithms optimize through a world-model rollout.
        if action_search_method in {
            "multistart_adam",
            "langevin_adam",
        }:
            if self.config.dynamics_mode != "rollout":
                raise ValueError(
                    f"Action search method {action_search_method!r} requires "
                    "dynamics_mode='rollout'."
                )

            initial_parameters = make_initial_action_parameters(
                base_parameter=base_parameter,
                num_starts=self.action_search_config.num_starts,
                noise_std=(
                    self.action_search_config.initialization_noise_std
                ),
                include_base=True,
                raw_action_limit=3.0,
            )

            print(
                f"[action search] method={action_search_method} "
                f"chains={len(initial_parameters)}"
            )

            evaluator = RolloutActionEvaluator(
                solver=self,
                latent_context=latent_context,
                past_action_context=past_action_context,
                goal_latent=goal_latent,
                video_latents=video_latents,
                action_prior=action_prior,
            )

            if action_search_method == "multistart_adam":
                search_result = run_multistart_adam(
                    initial_parameters=initial_parameters,
                    evaluate_actions=evaluator,
                    config=self.action_search_config.multistart_adam,
                )
            else:
                search_result = run_langevin_adam(
                    initial_parameters=initial_parameters,
                    evaluate_actions=evaluator,
                    config=self.action_search_config.langevin_adam,
                )

            residuals = torch.zeros(
                (horizon,) + tuple(current_latent.shape),
                device=current_latent.device,
                dtype=current_latent.dtype,
            )

            diagnostics = {
                **search_result.diagnostics,
                "search_chain": float(search_result.chain_index),
                "search_step": float(search_result.step_index),
                "num_search_chains": float(len(initial_parameters)),
            }

            return CollocationResult(
                latents=search_result.latents,
                actions=search_result.actions,
                objective=search_result.objective,
                augmented_lagrangian=search_result.objective,
                dynamics_residual_norm=0.0,
                multipliers=residuals,
                rho=0.0,
                diagnostics=diagnostics,
            )

        if action_search_method != "existing_alm":
            raise ValueError(
                f"Unknown action search method: {action_search_method!r}"
            )

        # Existing ALM implementation starts here.
        action_parameter = torch.nn.Parameter(base_parameter)

        initial_actions_for_debug = self._actions_from_parameter(
            action_parameter.detach()
        ).clone()

        parameters = [action_parameter]
        if latent_parameter is not None:
            parameters.insert(0, latent_parameter)

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
        )

        # Existing single-start Adam/ALM path continues here.
        # The multistart_adam and langevin_adam branches returned above.
        action_parameter = torch.nn.Parameter(base_parameter)

        parameters: list[torch.nn.Parameter] = [action_parameter]
        if latent_parameter is not None:
            parameters.insert(0, latent_parameter)

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
        )

        residual_shape = (horizon,) + tuple(current_latent.shape)
        multipliers = torch.zeros(
            residual_shape,
            device=self.world_model.device,
            dtype=current_latent.dtype,
        )
        rho = float(self.config.rho_init)

        final_residuals = torch.zeros_like(multipliers)

        for outer_index in range(self.config.outer_steps):
            for inner_index in range(self.config.inner_steps):
                optimizer.zero_grad(set_to_none=True)

                actions = self._actions_from_parameter(
                    action_parameter
                )

                if self.config.dynamics_mode == "rollout":
                    candidate_latents = self._rollout_world_model(
                        latent_context=latent_context,
                        past_action_context=past_action_context,
                        candidate_actions=actions,
                    )
                elif latent_parameter is None:
                    candidate_latents = initial_latents
                else:
                    candidate_latents = torch.cat(
                        [
                            current_latent.unsqueeze(0),
                            latent_parameter,
                        ],
                        dim=0,
                    )

                if self.config.dynamics_mode in {"alm", "soft"}:
                    residuals = self._dynamics_residuals(
                        latent_context=latent_context,
                        past_action_context=past_action_context,
                        candidate_latents=candidate_latents,
                        candidate_actions=actions,
                    )
                else:
                    residuals = candidate_latents.new_zeros(
                        (horizon,) + tuple(current_latent.shape)
                    )

                objective, pieces = self._objective(
                    latents=candidate_latents,
                    actions=actions,
                    goal_latent=goal_latent,
                    video_latents=video_latents,
                    action_prior=action_prior,
                )

                augmented = objective
                zero = objective.new_tensor(0.0)

                pieces.setdefault("dynamics_penalty", zero)
                pieces.setdefault("weighted_dynamics_penalty", zero)
                pieces.setdefault("feasibility_loss", zero)
                pieces.setdefault("weighted_feasibility", zero)
                pieces.setdefault("dsm_energy", zero)
                pieces.setdefault("transition_energy", zero)
                pieces.setdefault("weighted_transition_energy", zero)

                if self.config.dynamics_mode == "soft":
                    dynamics_penalty, dynamics_pieces = (
                        self._dynamics_penalty(residuals)
                    )
                    augmented = augmented + dynamics_penalty
                    pieces.update(dynamics_pieces)

                if (
                    self.feasibility_config is not None
                    and self.feasibility_config.enabled
                ):
                    feasibility_penalty, feasibility_pieces = (
                        self._feasibility_penalty(
                            latent_context=latent_context,
                            candidate_latents=candidate_latents,
                            candidate_actions=actions,
                        )
                    )
                    augmented = augmented + feasibility_penalty
                    pieces.update(feasibility_pieces)
                    pieces["weighted_feasibility"] = (
                        feasibility_penalty
                    )

                dual_term = objective.new_tensor(0.0)
                rho_penalty = objective.new_tensor(0.0)

                if self.config.dynamics_mode == "alm":
                    if self.config.residual_reduction == "mean":
                        penalty_scale = (
                            1.0 / float(residuals[0].numel())
                        )
                    else:
                        penalty_scale = 1.0

                    for index in range(residuals.shape[0]):
                        dual_piece = (
                            multipliers[index] * residuals[index]
                        ).sum()

                        penalty_piece = (
                            0.5
                            * rho
                            * penalty_scale
                            * squared_norm(residuals[index])
                        )

                        augmented = (
                            augmented
                            + dual_piece
                            + penalty_piece
                        )
                        dual_term = dual_term + dual_piece
                        rho_penalty = (
                            rho_penalty + penalty_piece
                        )

                if not torch.isfinite(augmented):
                    raise RuntimeError(
                        "Non-finite optimization objective: "
                        f"objective="
                        f"{float(objective.detach().cpu())}, "
                        f"augmented="
                        f"{float(augmented.detach().cpu())}"
                    )

                augmented.backward()

                if action_parameter.grad is None:
                    raise RuntimeError(
                        "No gradient reached the action parameters."
                    )

                if not torch.isfinite(
                    action_parameter.grad
                ).all():
                    raise RuntimeError(
                        "Non-finite action gradient."
                    )

                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        self.config.clip_grad_norm,
                    )

                if (
                    VERBOSE_DIAGNOSTICS
                    and inner_index % 10 == 0
                ):
                    print(
                        f"[adam {outer_index}:{inner_index}] "
                        f"cost="
                        f"{float(augmented.detach().cpu()):.6f} "
                        f"grad_norm="
                        f"{float(action_parameter.grad.norm().detach().cpu()):.6e} "
                        f"action_norm="
                        f"{float(actions.norm().detach().cpu()):.6f}"
                    )

                optimizer.step()

                if not self.config.use_action_reparameterization:
                    with torch.no_grad():
                        action_parameter.clamp_(
                            min=self.world_model.action_low.to(
                                action_parameter
                            ),
                            max=self.world_model.action_high.to(
                                action_parameter
                            ),
                        )

                final_residuals = residuals.detach()

            if self.config.dynamics_mode == "alm":
                with torch.no_grad():
                    if self.config.residual_reduction == "mean":
                        dual_scale = (
                            1.0
                            / float(final_residuals[0].numel())
                        )
                    else:
                        dual_scale = 1.0

                    multipliers = (
                        multipliers
                        + rho
                        * dual_scale
                        * final_residuals
                    )

                    rho = min(
                        rho * self.config.rho_growth,
                        self.config.rho_max,
                    )

            if self.config.diagnostic_outer:
                if self.config.dynamics_mode in {"alm", "soft"}:
                    outer_residual_norm = (
                        final_residuals
                        .reshape(final_residuals.shape[0], -1)
                        .norm(dim=1)
                        .mean()
                    )
                else:
                    outer_residual_norm = (
                        current_latent.new_tensor(0.0)
                    )

                print(
                    f"[solver outer {outer_index}] "
                    f"residual="
                    f"{float(outer_residual_norm.cpu()):.6f} "
                    f"next_rho={rho:.6f}"
                )

        # Re-evaluate the final parameters so the reported objective
        # corresponds to the returned actions and latents.
        with torch.no_grad():
            final_actions = self._actions_from_parameter(
                action_parameter
            ).detach()

            if self.config.dynamics_mode == "rollout":
                final_latents = self._rollout_world_model(
                    latent_context=latent_context,
                    past_action_context=past_action_context,
                    candidate_actions=final_actions,
                ).detach()
            elif latent_parameter is None:
                final_latents = initial_latents.detach()
            else:
                final_latents = torch.cat(
                    [
                        current_latent.unsqueeze(0),
                        latent_parameter.detach(),
                    ],
                    dim=0,
                )

            if self.config.dynamics_mode in {"alm", "soft"}:
                final_residuals = self._dynamics_residuals(
                    latent_context=latent_context,
                    past_action_context=past_action_context,
                    candidate_latents=final_latents,
                    candidate_actions=final_actions,
                ).detach()
            else:
                final_residuals = final_latents.new_zeros(
                    (horizon,) + tuple(current_latent.shape)
                )

            final_objective_tensor, final_pieces = (
                self._objective(
                    latents=final_latents,
                    actions=final_actions,
                    goal_latent=goal_latent,
                    video_latents=video_latents,
                    action_prior=action_prior,
                )
            )

            final_augmented_tensor = (
                final_objective_tensor.clone()
            )

            final_pieces.setdefault(
                "dynamics_penalty",
                final_objective_tensor.new_tensor(0.0),
            )
            final_pieces.setdefault(
                "weighted_dynamics_penalty",
                final_objective_tensor.new_tensor(0.0),
            )
            final_pieces.setdefault(
                "feasibility_loss",
                final_objective_tensor.new_tensor(0.0),
            )
            final_pieces.setdefault(
                "weighted_feasibility",
                final_objective_tensor.new_tensor(0.0),
            )

            if self.config.dynamics_mode == "soft":
                final_dynamics_penalty, dynamics_pieces = (
                    self._dynamics_penalty(final_residuals)
                )
                final_augmented_tensor = (
                    final_augmented_tensor
                    + final_dynamics_penalty
                )
                final_pieces.update(dynamics_pieces)

            if (
                self.feasibility_config is not None
                and self.feasibility_config.enabled
            ):
                final_feasibility, feasibility_pieces = (
                    self._feasibility_penalty(
                        latent_context=latent_context,
                        candidate_latents=final_latents,
                        candidate_actions=final_actions,
                    )
                )
                final_augmented_tensor = (
                    final_augmented_tensor
                    + final_feasibility
                )
                final_pieces.update(feasibility_pieces)
                final_pieces["weighted_feasibility"] = (
                    final_feasibility
                )

            final_dual_term = (
                final_objective_tensor.new_tensor(0.0)
            )
            final_rho_penalty = (
                final_objective_tensor.new_tensor(0.0)
            )

            if self.config.dynamics_mode == "alm":
                if self.config.residual_reduction == "mean":
                    penalty_scale = (
                        1.0 / float(final_residuals[0].numel())
                    )
                else:
                    penalty_scale = 1.0

                for index in range(final_residuals.shape[0]):
                    dual_piece = (
                        multipliers[index]
                        * final_residuals[index]
                    ).sum()

                    penalty_piece = (
                        0.5
                        * rho
                        * penalty_scale
                        * squared_norm(final_residuals[index])
                    )

                    final_augmented_tensor = (
                        final_augmented_tensor
                        + dual_piece
                        + penalty_piece
                    )
                    final_dual_term = (
                        final_dual_term + dual_piece
                    )
                    final_rho_penalty = (
                        final_rho_penalty + penalty_piece
                    )

            if self.config.dynamics_mode in {"alm", "soft"}:
                residual_norm_tensor = (
                    final_residuals
                    .reshape(final_residuals.shape[0], -1)
                    .norm(dim=1)
                    .mean()
                )
            else:
                residual_norm_tensor = (
                    current_latent.new_tensor(0.0)
                )

            diagnostics = {
                key: (
                    float(value.detach().cpu())
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in final_pieces.items()
            }

            diagnostics.update(
                {
                    "base_objective": float(
                        final_objective_tensor.cpu()
                    ),
                    "total_objective": float(
                        final_augmented_tensor.cpu()
                    ),
                    "dual_term": float(
                        final_dual_term.cpu()
                    ),
                    "rho_penalty": float(
                        final_rho_penalty.cpu()
                    ),
                    "rho": float(rho),
                    "residual_norm": float(
                        residual_norm_tensor.cpu()
                    ),
                    "final_action_norm": float(
                        final_actions.norm().cpu()
                    ),
                    "final_first_action_norm": float(
                        final_actions[0].norm().cpu()
                    ),
                }
            )

            final_objective = float(
                final_objective_tensor.cpu()
            )
            final_augmented = float(
                final_augmented_tensor.cpu()
            )
            residual_norm = float(
                residual_norm_tensor.cpu()
            )

        return CollocationResult(
            latents=final_latents,
            actions=final_actions,
            objective=final_objective,
            augmented_lagrangian=final_augmented,
            dynamics_residual_norm=residual_norm,
            multipliers=multipliers.detach(),
            rho=rho,
            diagnostics=diagnostics,
        )