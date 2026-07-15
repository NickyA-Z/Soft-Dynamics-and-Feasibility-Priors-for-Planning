from __future__ import annotations

import torch
import torch.nn as nn
from local.feasibility2.integrate import score_trajectory_feasibility

from .config import ALMConfig, FeasibilityConfig
from .interfaces import CollocationResult, WorldModelAdapter
from .losses import squared_norm
from .utils import ensure_history_length

class LatentCollocationSolver:
    def __init__(self, world_model: WorldModelAdapter, config: ALMConfig,
                 feasibility_config: FeasibilityConfig | None = None, # feasibility integration
                 feasibility_model: nn.Module | None = None, # feasibility integration
                 ) -> None:
        self.world_model = world_model
        self.config = config
        self.feasibility_config = feasibility_config # feasibility integration
        self.feasibility_model = feasibility_model # feasibility integration

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

        objective = (
            self.config.lambda_video * video_loss
            + self.config.lambda_goal * goal_loss
            + self.config.lambda_action * action_loss
            + self.config.lambda_action_prior * action_prior_loss
        )

        return objective, {
            "video_loss": video_loss,
            "goal_loss": goal_loss,
            "action_loss": action_loss,
            "action_prior_loss": action_prior_loss,
        }

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
        if warm_start_actions is not None and warm_start_actions.shape[0] >= horizon:
            action_prior = warm_start_actions[:horizon].to(self.world_model.device)

        if self.config.fix_states_to_video:
            latent_parameter = None
        else:
            latent_parameter = torch.nn.Parameter(initial_latents[1:].clone())

        action_parameter = torch.nn.Parameter(
            self._initialize_actions(
                horizon=horizon,
                warm_start_actions=warm_start_actions,
                device=self.world_model.device,
            )
        )

        parameters = [action_parameter]
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

        diagnostics: dict[str, float] = {}
        final_objective = current_latent.new_tensor(0.0)
        final_augmented = current_latent.new_tensor(0.0)
        final_residuals = torch.zeros_like(multipliers)

        for outer_index in range(self.config.outer_steps):
            for inner_index in range(self.config.inner_steps):
                optimizer.zero_grad()
                actions = self._actions_from_parameter(action_parameter)
                if latent_parameter is None:
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
                if outer_index == 0 and inner_index == 0:
                    debug_pieces = {}
                    for k, v in pieces.items():
                        if torch.is_tensor(v):
                            debug_pieces[k] = float(v.detach().cpu())
                        else:
                            debug_pieces[k] = float(v)
                    print("objective pieces:", debug_pieces)


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


                augmented.backward()
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
                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.config.clip_grad_norm)
                optimizer.step()
                if not self.config.use_action_reparameterization:
                    with torch.no_grad():
                        action_parameter.clamp_(
                            min=self.world_model.action_low,
                            max=self.world_model.action_high,
                        )
                final_objective = objective.detach() # video + goal + action + action_prior
                final_augmented = augmented.detach() # augmented = objective + feasibility_penalty
                final_residuals = residuals.detach()

                # more debug
                pieces["base_objective"] = objective.detach()
                pieces["total_objective"] = augmented.detach()
                #pieces["weighted_feasibility"] = feasibility_penalty.detach()
                pieces["dual_term"] = dual_term.detach()
                pieces["rho_penalty"] = rho_penalty.detach()
                """
                diagnostics = {
                    key: float(value.detach().cpu())
                    for key, value in pieces.items()
                }
                """
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

        final_actions = self._actions_from_parameter(action_parameter).detach()
        if latent_parameter is None:
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
            }
        )
        return CollocationResult(
            latents=final_latents,
            actions=final_actions,
            objective=float(final_objective.cpu()),
            augmented_lagrangian=float(final_augmented.cpu()),
            dynamics_residual_norm=float(residual_norm.cpu()),
            multipliers=multipliers.detach(),
            rho=rho,
            diagnostics=diagnostics,
        )
