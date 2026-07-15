from __future__ import annotations

import os

import torch
import torch.nn as nn
from local.feasibility2.integrate import score_trajectory_feasibility

from .config import ALMConfig, FeasibilityConfig, LangevinActionConfig
from .interfaces import CollocationResult, WorldModelAdapter
from .losses import squared_norm
from .utils import ensure_history_length

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



        # Existing ALM path continues here.
        action_parameter = torch.nn.Parameter(base_parameter)


        initial_actions_for_debug = self._actions_from_parameter(action_parameter.detach()).clone()
        parameters = [action_parameter]
        if latent_parameter is not None:
            parameters.insert(0, latent_parameter)

        # new optimiser for langevin 
        use_langevin = (
            self.config.dynamics_mode == "rollout"
            and self.langevin_config.enabled
        )

        optimizer = None

        if not use_langevin:
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                eps=self.config.adam_eps,
            )
        """
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
        )"""

        residual_shape = (horizon,) + tuple(current_latent.shape)
        multipliers = torch.zeros(
            residual_shape,
            device=self.world_model.device,
            dtype=current_latent.dtype,
        )
        rho = float(self.config.rho_init)

        diagnostics: dict[str, float] = {}
        final_objective = current_latent.new_tensor(0.0)
        final_augmented = current_latent.new_tensor(0.0)
        final_residuals = torch.zeros_like(multipliers)

        # for langevin 
        best_rollout_cost = float("inf")
        best_rollout_actions: torch.Tensor | None = None
        best_rollout_latents: torch.Tensor | None = None
        best_rollout_diagnostics: dict[str, float] | None = None
        best_rollout_base_cost = float("inf")

        for outer_index in range(self.config.outer_steps):
            for inner_index in range(self.config.inner_steps):
                #optimizer.zero_grad()
                # new optimiser langevin
                if optimizer is not None:
                    optimizer.zero_grad()
                else:
                    for parameter in parameters:
                        parameter.grad = None
                    
                actions = self._actions_from_parameter(action_parameter)
                if self.config.dynamics_mode == "rollout":
                    # generate latents from wm with the candidate actions.
                    candidate_latents = self._rollout_world_model(
                        latent_context=latent_context,
                        past_action_context=past_action_context,
                        candidate_actions=actions,
                    )
                elif latent_parameter is None:
                    candidate_latents = initial_latents
                else:
                    candidate_latents = torch.cat(
                        [current_latent.unsqueeze(0), latent_parameter],
                        dim=0,
                    )
                # normal ALM residuals
                if self.config.dynamics_mode in ("alm", "soft"):
                    # here actions and latents are tied together in the residuals, 
                    residuals = self._dynamics_residuals(
                        latent_context=latent_context,
                        past_action_context=past_action_context,
                        candidate_latents=candidate_latents,
                        candidate_actions=actions,
                    )
                else:
                    # feasibility only optimisation
                    residuals = candidate_latents.new_zeros(
                        (horizon,) + tuple(current_latent.shape)
                    )

                # compute collocation objective (video alignment, goal loss, action regularization)
                # compute collocation objective (video alignment, goal loss, action regularization)
                objective, pieces = self._objective(
                    latents=candidate_latents,
                    actions=actions,
                    goal_latent=goal_latent,
                    video_latents=video_latents,
                    action_prior=action_prior,
                )

                # extra testing
                augmented = objective
                zero = current_latent.new_tensor(0.0)

                # Defaults for diagnostics.
                pieces["dynamics_penalty"] = zero
                pieces["weighted_dynamics_penalty"] = zero
                pieces["feasibility_loss"] = zero
                pieces["weighted_feasibility"] = zero
                pieces["dsm_energy"] = zero
                pieces["transition_energy"] = zero
                pieces["weighted_transition_energy"] = zero
   
                # Soft (feasibility) world-model dynamics penalty
                if self.config.dynamics_mode == "soft":
                    dynamics_penalty, dynamics_pieces = self._dynamics_penalty(residuals)
                    augmented = augmented + dynamics_penalty
                    pieces.update(dynamics_pieces)

                # Learned feasibility penalty ONLY (ignoring dynamics constraints)
                if (
                    self.feasibility_config is not None
                    and self.feasibility_config.enabled
                ):
                    feasibility_penalty, feasibility_pieces = self._feasibility_penalty(
                        latent_context=latent_context,
                        candidate_latents=candidate_latents,
                        candidate_actions=actions,
                    )
                    # so unles we gate this, if ALM=true we do both alm+feasibility!
                    augmented = augmented + feasibility_penalty
                    pieces.update(feasibility_pieces)
                    pieces["weighted_feasibility"] = feasibility_penalty

                dual_term = current_latent.new_tensor(0.0)
                rho_penalty = current_latent.new_tensor(0.0)

                # for feasibility only optimisation
                # debug to check feasibility loss
                if VERBOSE_DIAGNOSTICS and outer_index == 0 and inner_index == 0:
                    debug_pieces = {}
                    for k, v in pieces.items():
                        if torch.is_tensor(v):
                            debug_pieces[k] = float(v.detach().cpu())
                        else:
                            debug_pieces[k] = float(v)

                    weighted_goal = self.config.lambda_goal * debug_pieces.get("goal_loss", 0.0)
                    weighted_video = self.config.lambda_video * debug_pieces.get("video_loss", 0.0)
                    weighted_action = self.config.lambda_action * debug_pieces.get("action_loss", 0.0)
                    weighted_transition = debug_pieces.get("weighted_transition_energy", 0.0)
                    weighted_feas = debug_pieces.get("weighted_feasibility", 0.0)
                    lambda_transition_cfg = debug_pieces.get(
                        "configured_lambda_transition",
                        0.0,
                    )

                    print(
                        f"[objective summary] "
                        f"w_goal={weighted_goal:.4f} "
                        f"w_video={weighted_video:.4f} "
                        f"w_action={weighted_action:.4f} "
                        f"w_feas={weighted_feas:.4f} "
                        f"w_transition={weighted_transition:.4f} "
                        f"lambda_transition_cfg={lambda_transition_cfg:.4f} "
                        f"dsm={debug_pieces.get('dsm_energy', 0.0):.4f} "
                        f"transition={debug_pieces.get('transition_energy', 0.0):.4f}"
                    )

                # ALM-specific terms
                if self.config.dynamics_mode == "alm":
                    if self.config.residual_reduction == "mean":
                        pen_scale = 1.0 / float(residuals[0].numel())
                    else:
                        pen_scale = 1.0
                    for index in range(residuals.shape[0]):
                        dual_piece = (multipliers[index] * residuals[index]).sum()
                        penalty_piece = 0.5 * rho * pen_scale * squared_norm(residuals[index])

                        augmented = augmented + dual_piece + penalty_piece
                        dual_term = dual_term + dual_piece
                        rho_penalty = rho_penalty + penalty_piece

                # main optimising loop? 
                augmented.backward()

                if inner_index % 10 == 0:
                    grad = action_parameter.grad
                    print(
                        f"[langevin {inner_index}] "
                        f"cost={float(augmented.detach().cpu()):.6f} "
                        f"grad_norm={float(grad.norm().detach().cpu()) if grad is not None else float('nan'):.6e} "
                        f"action_norm={float(actions.norm().detach().cpu()):.6f}"
                    )

                if VERBOSE_DIAGNOSTICS and self.config.dynamics_mode == "rollout" and (
                    (outer_index == 0 and inner_index < 3)
                    or (
                        outer_index == self.config.outer_steps - 1
                        and inner_index == self.config.inner_steps - 1
                    )
                ):
                    action_grad = action_parameter.grad
                    action_grad_norm = (
                        float(action_grad.norm().detach().cpu())
                        if action_grad is not None
                        else float("nan")
                    )
                    action_norm = float(actions.norm().detach().cpu())
                    first_action_norm = float(actions[0].norm().detach().cpu())
                    print(
                        f"[solver rollout inner {outer_index}:{inner_index}] "
                        f"action_norm={action_norm:.4f} "
                        f"first_action_norm={first_action_norm:.4f} "
                        f"action_grad_norm={action_grad_norm:.4e} "
                        f"objective={float(objective.detach().cpu()):.4f} "
                        f"augmented={float(augmented.detach().cpu()):.4f}"
                    )
                if (
                    self.config.diagnostic_grad_norms
                    and self.config.diagnostic_inner_interval is not None
                    and (
                        inner_index % self.config.diagnostic_inner_interval == 0
                        or inner_index == self.config.inner_steps - 1
                    )
                ):
                    a_grad = action_parameter.grad
                    if action_parameter.grad is not None:
                        print("action_grad_norm:", float(action_parameter.grad.norm().detach().cpu()))
                    else:
                        print("action_grad_norm: None")
                    a_norm = float(a_grad.norm().cpu()) if a_grad is not None else float("nan")
                    if latent_parameter is not None and latent_parameter.grad is not None:
                        l_grad = latent_parameter.grad
                        l_norm = float(l_grad.norm().cpu())
                        l_mean_abs = float(l_grad.abs().mean().cpu())
                    else:
                        l_norm = float("nan")
                        l_mean_abs = float("nan")
                    print(
                        f"[grad outer {outer_index} inner {inner_index}] "
                        f"action_grad_norm={a_norm:.4e} "
                        f"latent_grad_norm={l_norm:.4e} "
                        f"latent_grad_mean_abs={l_mean_abs:.4e}"
                    )
                """
                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.config.clip_grad_norm)
                """
                if use_langevin:
                    current_cost = float(augmented.detach().cpu())

                    if not torch.isfinite(augmented):
                        print(
                            "[langevin] non-finite cost:",
                            "objective=", float(objective.detach().cpu()),
                            "augmented=", current_cost,
                        )
                        raise RuntimeError("Non-finite Langevin objective.")

                    if best_rollout_actions is None or current_cost < best_rollout_cost:
                        best_rollout_cost = current_cost
                        best_rollout_base_cost = float(objective.detach().cpu())
                        best_rollout_actions = actions.detach().clone()
                        best_rollout_latents = candidate_latents.detach().clone()
                        best_rollout_diagnostics = {
                            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                            for key, value in pieces.items()
                        }

                    self._langevin_action_step(action_parameter)
                else:
                    if self.config.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            self.config.clip_grad_norm,
                        )
                    
                    assert optimizer is not None
                    optimizer.step()

                    # ony not use_langevin added
                    if not use_langevin and not self.config.use_action_reparameterization:
                        with torch.no_grad():
                            action_parameter.clamp_(
                                min=self.world_model.action_low,
                                max=self.world_model.action_high,
                            )

                #optimizer.step()


                final_objective = objective.detach() # video + goal + action + action_prior
                final_augmented = augmented.detach() # augmented = objective + feasibility_penalty
                final_residuals = residuals.detach()

                # more debug
                pieces["base_objective"] = objective.detach()
                pieces["total_objective"] = augmented.detach()
                #pieces["weighted_feasibility"] = feasibility_penalty.detach()
                pieces["dual_term"] = dual_term.detach()
                pieces["rho_penalty"] = rho_penalty.detach()

                diagnostics = {}
                for key, value in pieces.items():
                    if torch.is_tensor(value):
                        diagnostics[key] = float(value.detach().cpu())
                    else:
                        diagnostics[key] = float(value)
                if (
                    self.config.diagnostic_inner_interval is not None
                    and outer_index == self.config.outer_steps - 1
                    and inner_index == self.config.inner_steps - 1
                ):
                    inner_residual_norm = residuals.reshape(residuals.shape[0], -1).norm(dim=1).mean()

                    weighted_video = self.config.lambda_video * pieces["video_loss"]
                    weighted_goal = self.config.lambda_goal * pieces["goal_loss"]
                    weighted_action = self.config.lambda_action * pieces["action_loss"]
                    weighted_action_prior = (
                        self.config.lambda_action_prior * pieces["action_prior_loss"]
                    )
                    # diagnostic print for feasibility loss
                    lambda_feasibility = (
                        self.feasibility_config.lambda_feasibility
                        if self.feasibility_config is not None and self.feasibility_config.enabled
                        else 0.0
                    )
                    weighted_feasibility = lambda_feasibility * pieces["feasibility_loss"]

                    print(
                        f"\n[alm outer {outer_index} inner {inner_index}] "
                        f"video={diagnostics['video_loss']:.4f} "
                        f"goal={diagnostics['goal_loss']:.4f} "
                        f"action={diagnostics['action_loss']:.2f} "
                        f"action_prior={diagnostics['action_prior_loss']:.2f} "
                        f"w_video={float(weighted_video.detach().cpu()):.4f} "
                        f"w_goal={float(weighted_goal.detach().cpu()):.4f} "
                        f"w_action={float(weighted_action.detach().cpu()):.4f} "
                        f"w_action_prior={float(weighted_action_prior.detach().cpu()):.4f} "
                        f"obj={float(objective.detach().cpu()):.4f} "
                        f"dual={float(dual_term.detach().cpu()):.4f} "
                        f"rho_pen={float(rho_penalty.detach().cpu()):.4f} "
                        f"aug={float(augmented.detach().cpu()):.4f} "
                        f"residual={float(inner_residual_norm.detach().cpu()):.4f} "
                        f"rho={rho:.1f} "
                        f"feasibility={diagnostics['feasibility_loss']:.4f} "
                        f"w_feasibility={float(weighted_feasibility.detach().cpu()):.4f} "
                        f"dsm={diagnostics.get('dsm_energy', 0.0):.4f} "
                        f"transition={diagnostics.get('transition_energy', 0.0):.4f} "
                        f"w_transition={diagnostics.get('weighted_transition_energy', 0.0):.4f} "
                        f"dyn={diagnostics.get('dynamics_penalty', 0.0):.4f} "
                        f"w_dyn={diagnostics.get('weighted_dynamics_penalty', 0.0):.4f} "
                    )

            with torch.no_grad():
                # for ALM ONLY 
                if self.config.dynamics_mode == "alm":
                    if self.config.residual_reduction == "mean":
                        dual_scale = 1.0 / float(final_residuals[0].numel())
                    else:
                        dual_scale = 1.0

                    multipliers = multipliers + rho * dual_scale * final_residuals
                    rho = min(rho * self.config.rho_growth, self.config.rho_max)

                if self.config.diagnostic_outer:
                    if self.config.dynamics_mode in ("alm", "soft"):
                        outer_residual_norm = final_residuals.reshape(final_residuals.shape[0], -1).norm(dim=1).mean()
                    else:
                        outer_residual_norm = current_latent.new_tensor(0.0)
                    print(
                        f"[alm outer {outer_index} done] "
                        f"video={diagnostics['video_loss']:.6f} "
                        f"goal={diagnostics['goal_loss']:.6f} "
                        f"action={diagnostics['action_loss']:.6f} "
                        f"residual={float(outer_residual_norm.cpu()):.6f} "
                        f"next_rho={rho:.6f}"
                    )
           
        if use_langevin:
            if best_rollout_actions is None or best_rollout_latents is None:
                raise RuntimeError("Langevin rollout produced no valid sample.")

            final_actions = best_rollout_actions
            final_latents = best_rollout_latents
        else:
            final_actions = self._actions_from_parameter(action_parameter).detach()

        if VERBOSE_DIAGNOSTICS and self.config.dynamics_mode == "rollout":
            print(
                f"[solver rollout summary] "
                f"init_action_norm={float(initial_actions_for_debug.norm().detach().cpu()):.4f} "
                f"final_action_norm={float(final_actions.norm().detach().cpu()):.4f} "
                f"init_first_action_norm={float(initial_actions_for_debug[0].norm().detach().cpu()):.4f} "
                f"final_first_action_norm={float(final_actions[0].norm().detach().cpu()):.4f} "
                f"init_to_final_norm={float((final_actions - initial_actions_for_debug).norm().detach().cpu()):.4f}"
            )
        # more rollout 
        if self.config.dynamics_mode == "rollout":
            with torch.no_grad():
                final_latents = self._rollout_world_model(
                    latent_context=latent_context,
                    past_action_context=past_action_context,
                    candidate_actions=final_actions,
                ).detach()
        elif latent_parameter is None:
            final_latents = initial_latents.detach()
        else:
            final_latents = torch.cat(
                [current_latent.unsqueeze(0), latent_parameter.detach()],
                dim=0,
            )

        if self.config.dynamics_mode in ("alm", "soft"):
            residual_norm = final_residuals.reshape(final_residuals.shape[0], -1).norm(dim=1).mean()
        else:
            residual_norm = current_latent.new_tensor(0.0)

        diagnostics.update(
            {
                "rho": float(rho),
                "residual_norm": float(residual_norm.cpu()),
                "init_action_norm": float(initial_actions_for_debug.norm().detach().cpu()),
                "final_action_norm": float(final_actions.norm().detach().cpu()),
                "init_first_action_norm": float(
                    initial_actions_for_debug[0].norm().detach().cpu()
                ),
                "final_first_action_norm": float(final_actions[0].norm().detach().cpu()),
                "init_to_final_action_norm": float(
                    (final_actions - initial_actions_for_debug).norm().detach().cpu()
                ),
            }
        )
        if use_langevin:
            if (
                best_rollout_actions is None
                or best_rollout_latents is None
                or best_rollout_diagnostics is None
            ):
                raise RuntimeError("Langevin rollout produced no valid sample.")

            final_actions = best_rollout_actions
            final_latents = best_rollout_latents
            final_objective = best_rollout_base_cost
            final_augmented = best_rollout_cost
            diagnostics = best_rollout_diagnostics
        else:
            final_objective = float(final_objective.cpu())
            final_augmented = float(final_augmented.cpu())

        return CollocationResult(
            latents=final_latents,
            actions=final_actions,
            objective=float(final_objective),
            augmented_lagrangian=float(final_augmented),
            dynamics_residual_norm=float(residual_norm),
            multipliers=multipliers.detach(),
            rho=rho,
            diagnostics=diagnostics,
        )
